"""SSN-BLIP2-v5: v4 with box-constrained mask-mean regularizer.

Strategy
--------
We do NOT duplicate v4's ``forward`` (200+ lines). Instead, we inherit from
v4 and intercept the specific call that computes the mask regularizer.

In v4:
    l_mask_target = (M_act.mean() - 0.5).pow(2)
is buried inside ``forward``. To swap it without rewriting forward, we
override a tiny helper ``_mask_target_loss`` here and have v4's forward call
the helper. Since v4 doesn't already define such a helper, we apply a
monkey-patch at import time that replaces the literal expression with a
call to ``self._mask_target_loss``.

But monkey-patching forward source is fragile. The cleaner alternative used
below is to override ``forward`` once in v5 to forward through a wrapper
that injects the new constants. The actual computation is delegated to a
single helper ``_mask_regularizer`` defined here.

To keep things tractable we DO rewrite forward, but only by chaining:
v5.forward() = v4.forward() + post-hoc replacement of l_kl_action and total.

Implementation: we patch v4's ``forward`` to use our new helper by setting
two new attributes BEFORE super().forward(), and v4's existing forward
honours them. To enable that, v4 already reads ``self._mask_div_w`` and
``self._mask_target_*`` if they exist? — No, v4 hardcodes 0.1 and 0.5.

Conclusion: we have to override forward. To minimize duplication we
literally copy v4.forward() once into this file with only the three
relevant lines changed. That keeps v4 untouched and v5 self-contained.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict

import torch
import torch.nn.functional as F

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from models.ssn_blip2 import Model as BaselineModel  # noqa: E402
from SSN_v4.ssn_blip2_v4 import Model as V4Model     # noqa: E402


class Model(V4Model):
    """v4 + box-constrained mask-mean regularizer."""

    def __init__(self, *args, **kwargs):
        # v5-only kwargs (pop before passing to v4 init).
        # Defaults chosen from the v4 run1 post-mortem:
        #   * box [0.4, 0.7] covers the "healthy" range seen up to epoch 16;
        #   * weight 0.5 is 5x v4 to actually hold the mean inside.
        self._mask_target_lo = float(kwargs.pop("mask_target_lo", 0.4))
        self._mask_target_hi = float(kwargs.pop("mask_target_hi", 0.7))
        self._mask_target_w = float(kwargs.pop("mask_target_w", 0.5))
        self._mask_div_w = float(kwargs.pop("mask_div_w", 0.1))

        super().__init__(*args, **kwargs)

    def forward(
        self,
        samples: Dict[str, Any],
        warmup: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if warmup:
            return BaselineModel.forward(self, samples, warmup=True)

        self.graph_enabled = True

        image_embeds = samples["image"].to(self.device)
        target_embeds = samples["target"].to(self.device)
        captions = samples["text_input"]
        graphs = samples["graph"]

        with self.maybe_autocast():
            image_embeds_ln = self.ln_vision(image_embeds)
            target_embeds_ln = self.ln_vision(target_embeds)

        F_target = self._qformer_image_only(target_embeds_ln, self.query_tokens)
        z_tgt_tokens = F.normalize(self.vision_proj(F_target), dim=-1)

        # ── Positive query ──────────────────────────────────────────────
        self._external_M_token = None
        self._compute_blend = self.lambda_counterfact > 0.0
        try:
            out_pos = self._build_query(
                image_embeds_ln, captions, graphs,
                target_patches=target_embeds_ln[:, 1:, :],
            )
        finally:
            self._compute_blend = False
        F_pos = out_pos["final_query"]

        ramp_done = self.aux_ramp.item() >= 0.99
        beta_eff = (torch.clamp(self.beta, min=0.0, max=self.beta_max)
                    * self.aux_ramp).detach().abs().item()
        gamma_eff = (self.gamma * self.aux_ramp).detach().abs().item()
        action_ready = ramp_done and (beta_eff + gamma_eff > 0.2)
        F_blend = out_pos.get("final_blend", None) if action_ready else None
        F_action = out_pos.get("final_action", None) if action_ready else None

        q_tokens = out_pos["q_tokens"]
        Fq_u = out_pos["Fq_u"]
        text_embed = out_pos["text_embed"]
        ctx = out_pos.get("_action_ctx", None)
        M_act = ctx["M_act"] if ctx is not None else None

        # ── Negative query ──────────────────────────────────────────────
        if M_act is not None and not self.ablate_no_action_mask:
            B = M_act.shape[0]
            perm = torch.randperm(B, device=M_act.device)
            same = (perm == torch.arange(B, device=M_act.device))
            if same.any():
                perm = (perm + 1) % B
            M_neg = M_act[perm]
            self._external_M_token = M_neg
            out_neg = self._build_query(
                image_embeds_ln, captions, graphs,
                target_patches=target_embeds_ln[:, 1:, :],
            )
            F_neg = out_neg["final_query"]
        else:
            F_neg = None
        self._external_M_token = None

        # ── Main InfoNCE ────────────────────────────────────────────────
        self.temp.data = torch.clamp(self.temp.data, min=1e-3)
        scale = 1.0 / self.temp

        pos_sim_tokens = torch.einsum('bd,ntd->bnt', F_pos, z_tgt_tokens)
        pos_sim_mat = pos_sim_tokens.max(dim=-1).values
        pos_logits = scale * pos_sim_mat
        gt = torch.arange(pos_logits.shape[0], device=pos_logits.device)
        ls = self.label_smoothing
        l_pos = (F.cross_entropy(pos_logits, gt, label_smoothing=ls)
                 + F.cross_entropy(pos_logits.T, gt, label_smoothing=ls)) / 2

        # ── Aux head ────────────────────────────────────────────────────
        q_guided = self.aux_head(q_tokens, F_target.detach())
        aux_sim = scale * torch.einsum('bd,ntd->bnt',
                                       q_guided, z_tgt_tokens).max(dim=-1).values
        l_aux = (F.cross_entropy(aux_sim, gt) + F.cross_entropy(aux_sim.T, gt)) / 2

        # ── Group-aware hard negatives ──────────────────────────────────
        hard_embeds = samples.get("hard")
        hard_mask = samples.get("hard_mask")
        if hard_embeds is not None:
            hard_embeds = hard_embeds.to(self.device)
            B_h, K = hard_embeds.shape[:2]
            with self.maybe_autocast():
                hard_flat = hard_embeds.view(B_h * K, *hard_embeds.shape[2:])
                hard_embeds_ln = self.ln_vision(hard_flat)
            with torch.no_grad():
                F_hard = self._qformer_image_only(hard_embeds_ln, self.query_tokens)
                z_hard = F.normalize(self.vision_proj(F_hard), dim=-1)
            z_hard = z_hard.view(B_h, K, z_hard.shape[1], z_hard.shape[2]).detach()
            sim_hard = torch.einsum('bd,bktd->bkt', F_pos, z_hard).max(dim=-1).values
            sim_hard = sim_hard.masked_fill(~hard_mask.to(sim_hard.device), -1e4)
            pos_diag = pos_sim_mat.diag().unsqueeze(1)
            group_logits = torch.cat([pos_diag, sim_hard], dim=1) * self.group_scale
            group_gt = torch.zeros(B_h, dtype=torch.long, device=F_pos.device)
            l_group = F.cross_entropy(group_logits, group_gt)
        else:
            l_group = torch.zeros((), device=self.device)

        # ── Push-pull objective on (pos, neg) — v5 BOX CONSTRAINT ───────
        l_kl_action = torch.zeros((), device=self.device)
        m_act_mean_val = torch.zeros((), device=self.device)
        if F_neg is not None and M_act is not None:
            neg_sim_tokens = torch.einsum('bd,ntd->bnt', F_neg, z_tgt_tokens)
            neg_sim_mat = neg_sim_tokens.max(dim=-1).values
            neg_logits = scale * neg_sim_mat
            neg_softmax = F.softmax(neg_logits, dim=-1)
            neg_p_gt = neg_softmax.gather(1, gt.view(-1, 1)).squeeze(1)
            l_push = -torch.log((1.0 - neg_p_gt).clamp_min(1e-6)).mean()

            m_var = M_act.var(dim=-1).mean()
            l_mask_div = -m_var

            # ── v5 box constraint on mean(M_act) — PER-SAMPLE in runB.
            #    runA used the batch-wide mean (M_act.mean()), but in that
            #    case a few samples with mean ≈ 0.1 could pull the batch
            #    average to 0.30 while most samples actually sat inside
            #    [0.4, 0.7] — so the box never fired even when individual
            #    samples were collapsed. Computing per-sample means and
            #    summing their box penalties penalises each collapsing
            #    sample directly.
            m_per_sample = M_act.mean(dim=-1)        # [B]
            lo = self._mask_target_lo
            hi = self._mask_target_hi
            l_mask_target = (
                F.relu(lo - m_per_sample).pow(2)
                + F.relu(m_per_sample - hi).pow(2)
            ).mean()                                 # mean over batch

            l_kl_action = (
                l_push
                + self._mask_div_w * l_mask_div
                + self._mask_target_w * l_mask_target
            )
            m_act_mean_val = M_act.mean().detach()

        # ── Counter-factual margin loss (unchanged from v4) ──────────────
        l_counterfact = torch.zeros((), device=self.device)
        l_blend = torch.zeros((), device=self.device)
        l_action = torch.zeros((), device=self.device)
        if F_blend is not None and self.lambda_counterfact > 0.0:
            blend_sim_mat = torch.einsum(
                'bd,ntd->bnt', F_blend, z_tgt_tokens).max(dim=-1).values
            blend_logits = scale * blend_sim_mat
            l_blend = (F.cross_entropy(blend_logits, gt, label_smoothing=ls)
                       + F.cross_entropy(blend_logits.T, gt, label_smoothing=ls)) / 2

        if F_action is not None and self.lambda_counterfact > 0.0:
            action_sim_mat = torch.einsum(
                'bd,ntd->bnt', F_action, z_tgt_tokens).max(dim=-1).values
            action_logits = scale * action_sim_mat
            l_action = (F.cross_entropy(action_logits, gt, label_smoothing=ls)
                        + F.cross_entropy(action_logits.T, gt, label_smoothing=ls)) / 2
            l_counterfact = F.relu(
                l_action - l_blend.detach() + self.counterfact_margin
            )

        # ── Other v2-style auxiliary losses (unchanged) ──────────────────
        l_align = torch.zeros((), device=self.device)
        if self.lambda_align > 0.0:
            l_align = F.mse_loss(Fq_u, text_embed.detach())

        l_global = torch.zeros((), device=self.device)
        if self.lambda_global > 0.0:
            g_sim = scale * torch.einsum('bd,ntd->bnt',
                                         Fq_u, z_tgt_tokens).max(dim=-1).values
            l_global = (F.cross_entropy(g_sim, gt, label_smoothing=ls)
                        + F.cross_entropy(g_sim.T, gt, label_smoothing=ls)) / 2

        l_colbert = torch.zeros((), device=self.device)
        if self.lambda_colbert > 0.0:
            q_tok_proj = F.normalize(self.vision_proj(q_tokens), dim=-1)
            sim_qt = torch.einsum('bid,njd->bnij', q_tok_proj, z_tgt_tokens)
            cb_sim = sim_qt.max(dim=-1).values.mean(dim=-1)
            cb_logits = scale * cb_sim
            l_colbert = (F.cross_entropy(cb_logits, gt, label_smoothing=ls)
                         + F.cross_entropy(cb_logits.T, gt, label_smoothing=ls)) / 2

        total = (
            l_pos
            + self.lambda_aux * l_aux
            + self.lambda_group * l_group
            + self.lambda_kl_action * l_kl_action
            + self.lambda_counterfact * l_counterfact
            + self.lambda_align * l_align
            + self.lambda_global * l_global
            + self.lambda_colbert * l_colbert
        )

        return {
            "l_main": l_pos,
            "l_aux": l_aux,
            "l_group": l_group,
            "l_kl_action": l_kl_action,
            "l_blend": l_blend,
            "l_counterfact": l_counterfact,
            "l_align": l_align,
            "l_global": l_global,
            "l_colbert": l_colbert,
            "m_act_mean": m_act_mean_val,
            "alpha_inject": self.semantic_injector.alpha_inject.detach(),
            "loss": total,
        }
