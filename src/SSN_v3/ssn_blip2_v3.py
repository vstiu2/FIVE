"""SSN-BLIP2-v3: action-conditioned token mask + push-pull KL.

Subclasses the v2 ``Model`` and:
  * registers an ``ActionMaskHead`` that produces M_act on reference patches;
  * overrides ``_build_query`` to optionally accept an external token mask;
  * overrides ``forward`` to compute the push-pull KL objective on
    (positive_query masked by M_act, negative_query masked by 1 − M_act).

At inference, ``inference`` is also overridden to apply ``M_act`` before
calling ``_build_query`` — the ablation flag ``ablate_no_action_mask``
collapses it back to all-ones, restoring v2 behavior bit-identically.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from models.ssn_blip2 import Model as BaselineModel       # noqa: E402
from model.action_transitions import action_to_category   # noqa: E402
from model.object_localizer import aggregate_mask         # noqa: E402

from SSN_v3.action_mask import ActionMaskHead             # noqa: E402


class Model(BaselineModel):
    """SSN-BLIP2-v3 with action-conditioned token mask + push-pull KL."""

    def __init__(self, *args, **kwargs):
        # New v3-only kwargs (pop before passing to v2 init)
        mask_hidden_dim = int(kwargs.pop("mask_hidden_dim", 512))
        mask_dropout = float(kwargs.pop("mask_dropout", 0.1))
        # bias_init=0 → sigmoid(0)=0.5: unbiased start, so the head doesn't
        # default to "keep everything" (which lets the model ignore action
        # and produce a v2-equivalent solution — observed in run1).
        mask_bias_init = float(kwargs.pop("mask_bias_init", 0.0))
        self.lambda_kl_action = float(kwargs.pop("lambda_kl_action", 0.1))

        super().__init__(*args, **kwargs)

        vit_dim = self.visual_encoder.embed_dim
        qformer_h = self.Qformer.config.hidden_size

        self.action_mask_head = ActionMaskHead(
            vit_dim=vit_dim,
            node_dim=qformer_h,
            hidden_dim=mask_hidden_dim,
            dropout=mask_dropout,
            bias_init=mask_bias_init,
        )

        # Extra ablation switch.
        self.ablate_no_action_mask: bool = False

        # Cached during forward so _build_query_masked can read it without
        # re-running ObjectLocalizer / action encoders.
        self._cached_M_act: torch.Tensor | None = None
        self._external_M_token: torch.Tensor | None = None

    # ── Ablation -----------------------------------------------------------
    def set_ablation(
        self,
        no_trans: bool = False,
        no_fl: bool = False,
        no_ff: bool = False,
        force_route: str = "",
        no_action_mask: bool = False,
    ) -> None:
        super().set_ablation(
            no_trans=no_trans,
            no_fl=no_fl,
            no_ff=no_ff,
            force_route=force_route,
        )
        self.ablate_no_action_mask = bool(no_action_mask)

    # ── Helpers ------------------------------------------------------------
    def _compute_action_mask(
        self,
        image_embeds_ln: torch.Tensor,
        graphs: Sequence[Any],
    ) -> torch.Tensor:
        """Return M_act ∈ [0, 1]^N. All-ones if ablated.

        Side effect: caches ``self._cached_localizer_logits`` — the raw
        [B, M, N] output from ObjectLocalizer on the ORIGINAL unmasked
        patches. v2's ``_build_query`` is then hijacked (see
        ``_build_query`` below) to skip the redundant ObjectLocalizer call
        and reuse these logits, avoiding double computation AND avoiding
        the corruption that would result from running the (frozen,
        unmasked-trained) localizer on the masked patches.
        """
        B = image_embeds_ln.shape[0]
        patch_tokens = image_embeds_ln[:, 1:, :]
        N = patch_tokens.shape[1]

        # ObjectLocalizer on RAW patches.
        loc_nodes = self._encode_object_for_localizer(graphs)
        with torch.no_grad():
            logits = self.object_localizer(
                patch_tokens.float(),
                loc_nodes["node_features"].float(),
                loc_nodes["node_mask"],
            )
            M_obj_raw = aggregate_mask(logits)
        # Cache logits so v2's _build_query can reuse them.
        self._cached_localizer_logits = logits

        # NOTE: v2's _build_query handles the "Add" → uniform mask post-
        # processing AFTER aggregate_mask, so we leave M_obj_raw untouched
        # here — the cached logits feed back into v2 unchanged, and v2
        # applies the Add overwrite once at the right place.

        if self.ablate_no_action_mask:
            return torch.ones(B, N, device=patch_tokens.device,
                              dtype=patch_tokens.dtype)

        # For the mask head we DO want the "Add → uniform" variant in
        # M_obj, because Add samples have no object in the ref so the raw
        # localizer mask is meaningless. We apply it to a LOCAL copy only.
        parsed = [self._parse_graph(g) for g in graphs]
        cats = [action_to_category(p.get("action")) for p in parsed]
        if self.ablate_force_route:
            cats = [self.ablate_force_route] * len(cats)
        is_add = torch.tensor([c == "Add" for c in cats], device=M_obj_raw.device)
        M_obj_for_head = M_obj_raw
        if is_add.any():
            M_obj_for_head = M_obj_raw.clone()
            M_obj_for_head[is_add] = 1.0

        action_feat = self._encode_field(graphs, "action")    # [B, H]

        # obj_feat: masked mean over object/location nodes (same as v2)
        obj_node_out = self._encode_obj_nodes(graphs)
        nm = obj_node_out["node_mask"].float().unsqueeze(-1)
        obj_feat = ((obj_node_out["node_features"] * nm).sum(1)
                    / nm.sum(1).clamp_min(1e-6))

        return self.action_mask_head(
            patches=patch_tokens, action_feat=action_feat,
            M_obj=M_obj_for_head, obj_feat=obj_feat,
        )

    # ── Overridden _build_query --------------------------------------------
    def _build_query(
        self,
        image_embeds_ln: torch.Tensor,
        captions: List[str],
        graphs: Sequence[Any],
        target_patches: Any = None,
    ) -> Dict[str, torch.Tensor]:
        # When graph branches are off (warmup or inference with
        # graph_enabled=False), we must NOT apply M_act — Q-Former would
        # see patches multiplied by an untrained (random) mask, which
        # destroys the BLIP-2 visual prior. Falling back to the un-masked
        # v2 path keeps the warmup phase bit-identical to v2.
        if not self.graph_enabled:
            return BaselineModel._build_query(
                self, image_embeds_ln, captions, graphs, target_patches
            )

        # Apply external mask if one is set (used inside forward()) — otherwise
        # we compute M_act ourselves so single-pass inference still benefits
        # from the action conditioning.
        if self._external_M_token is not None:
            M_token = self._external_M_token
        else:
            M_token = self._compute_action_mask(image_embeds_ln, graphs)

        # Ramp in: blend toward all-ones during early graph epochs to avoid
        # injecting random-mask noise the moment graph branches turn on. The
        # ramp coefficient ``self.aux_ramp`` is shared with β/γ, so M_act
        # is fully active only after `aux_ramp_epochs` graph epochs.
        # Effective mask = aux_ramp * M_act + (1 - aux_ramp) * 1.0
        r = self.aux_ramp.to(M_token.dtype)
        M_token_eff = r * M_token + (1.0 - r)

        # Cache for downstream introspection.
        self._cached_M_act = M_token_eff

        # Multiply patches by M_token_eff; keep CLS intact.
        cls = image_embeds_ln[:, :1, :]
        patches = image_embeds_ln[:, 1:, :] * M_token_eff.unsqueeze(-1).to(image_embeds_ln.dtype)
        image_embeds_masked = torch.cat([cls, patches], dim=1)

        # ── Critical: hijack ObjectLocalizer so v2's _build_query receives
        # the M_obj computed on the ORIGINAL (unmasked) patches. Without
        # this, v2 would call ObjectLocalizer on the MASKED patches and
        # get a corrupted mask (the selector was trained on raw patches,
        # not multiplied by an arbitrary [0, 1] field). This corrupted
        # M_obj would then pollute the Fl / Ff branches and the action
        # router. We cached the raw logits in _compute_action_mask; here
        # we wrap the localizer with a stub that just returns them, so
        # the downstream ``aggregate_mask(...)`` call produces the correct
        # M_obj_raw (and v2's "Add" → uniform mask post-processing still
        # runs as normal).
        #
        # Implementation detail: we swap by reaching into ``_modules``
        # directly to avoid re-registering a non-Module into nn.Module's
        # attribute machinery.
        cached_logits = getattr(self, "_cached_localizer_logits", None)
        hijacked = cached_logits is not None
        original_localizer = self._modules.get("object_localizer")

        if hijacked:
            class _StubLocalizer:
                """Callable returning the cached [B, M, N] logits unchanged."""
                def __init__(self, logits):
                    self.logits = logits
                def __call__(self, *a, **k):
                    return self.logits
            # Remove the original Module entry, install the stub as a
            # plain attribute. We restore both in ``finally``.
            del self._modules["object_localizer"]
            object.__setattr__(self, "object_localizer",
                               _StubLocalizer(cached_logits))

        prev_external = self._external_M_token
        self._external_M_token = None  # baseline _build_query expects no mask
        try:
            out = BaselineModel._build_query(
                self, image_embeds_masked, captions, graphs, target_patches
            )
        finally:
            self._external_M_token = prev_external
            if hijacked:
                # Drop the stub attribute and put the original Module back.
                try:
                    del self.__dict__["object_localizer"]
                except KeyError:
                    pass
                self._modules["object_localizer"] = original_localizer
        return out

    # ── Inference (apply M_act once, like v2 single-pass) ------------------
    @torch.no_grad()
    def inference(
        self,
        image_embeds_ln: torch.Tensor,
        F_target_all: torch.Tensor,
        captions: List[str],
        graphs: Sequence[Any],
        graph_enabled: bool = True,
    ) -> torch.Tensor:
        prev = self.graph_enabled
        self.graph_enabled = graph_enabled
        try:
            # Single-pass: M_act is auto-computed inside _build_query when
            # _external_M_token is None, which is the default at inference.
            out = self._build_query(image_embeds_ln, captions, graphs)
            final_query = out["final_query"]
        finally:
            self.graph_enabled = prev
        z_tgt_tokens = F.normalize(self.vision_proj(F_target_all), dim=-1)
        sim = torch.einsum('bd,ntd->bnt', final_query, z_tgt_tokens).max(dim=-1).values
        return sim

    # ── Forward (training): push-pull KL on (M_act, 1 − M_act) -------------
    def forward(
        self,
        samples: Dict[str, Any],
        warmup: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Warmup → exactly v2 behavior (no graph branches, no KL).
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

        # ── Compute M_act ONCE on the original (un-masked) input. ──────────
        M_act = self._compute_action_mask(image_embeds_ln, graphs)

        # ── Target encoding (shared, doesn't use M_act). ───────────────────
        F_target = self._qformer_image_only(target_embeds_ln, self.query_tokens)
        z_tgt_tokens = F.normalize(self.vision_proj(F_target), dim=-1)   # [B, 32, 256]

        # ── Two queries: positive (M_act) and negative (1 − M_act). ────────
        self._external_M_token = M_act
        out_pos = self._build_query(image_embeds_ln, captions, graphs,
                                    target_patches=target_embeds_ln[:, 1:, :])
        F_pos = out_pos["final_query"]
        q_tokens = out_pos["q_tokens"]
        Fq_u = out_pos["Fq_u"]
        text_embed = out_pos["text_embed"]

        # Negative query: instead of using (1 − M_act) — which collapses to
        # an all-zero patch tensor when the head trivially outputs M_act ≡ 1,
        # giving the model a degenerate shortcut — we use a SHUFFLED M_act
        # from the same batch. F_neg therefore looks at the SAME ref image
        # but through the action-mask of ANOTHER (random) sample. If the
        # mask truly carries action-conditioned information, swapping it
        # should hurt retrieval. The negative query stays well-defined for
        # any M_act distribution, so the optimizer can't cheat by collapsing.
        B = M_act.shape[0]
        perm = torch.randperm(B, device=M_act.device)
        # Make sure no sample gets its own mask back.
        same = (perm == torch.arange(B, device=M_act.device))
        if same.any():
            perm = (perm + 1) % B
        M_neg = M_act[perm]
        self._external_M_token = M_neg
        out_neg = self._build_query(image_embeds_ln, captions, graphs,
                                    target_patches=target_embeds_ln[:, 1:, :])
        F_neg = out_neg["final_query"]
        self._external_M_token = None  # clear

        # ── Main InfoNCE (positive query only, identical to v2). ────────────
        self.temp.data = torch.clamp(self.temp.data, min=1e-3)
        scale = 1.0 / self.temp

        pos_sim_tokens = torch.einsum('bd,ntd->bnt', F_pos, z_tgt_tokens)
        pos_sim_mat = pos_sim_tokens.max(dim=-1).values         # [B, B]
        pos_logits = scale * pos_sim_mat
        gt = torch.arange(pos_logits.shape[0], device=pos_logits.device)
        ls = self.label_smoothing
        l_pos = (F.cross_entropy(pos_logits, gt, label_smoothing=ls)
                 + F.cross_entropy(pos_logits.T, gt, label_smoothing=ls)) / 2

        # ── Aux head (target-aware), same as v2. ────────────────────────────
        q_guided = self.aux_head(q_tokens, F_target.detach())
        aux_sim = scale * torch.einsum('bd,ntd->bnt',
                                       q_guided, z_tgt_tokens).max(dim=-1).values
        l_aux = (F.cross_entropy(aux_sim, gt) + F.cross_entropy(aux_sim.T, gt)) / 2

        # ── Group-aware hard negatives, same as v2 (skip for simplicity if
        #    no hard provided). We re-compute under M_act-masked anchors so
        #    the gradient is consistent.
        hard_embeds = samples.get("hard")
        hard_mask = samples.get("hard_mask")
        if hard_embeds is not None:
            hard_embeds = hard_embeds.to(self.device)
            B, K = hard_embeds.shape[:2]
            with self.maybe_autocast():
                hard_flat = hard_embeds.view(B * K, *hard_embeds.shape[2:])
                hard_embeds_ln = self.ln_vision(hard_flat)
            with torch.no_grad():
                F_hard = self._qformer_image_only(hard_embeds_ln, self.query_tokens)
                z_hard = F.normalize(self.vision_proj(F_hard), dim=-1)
            z_hard = z_hard.view(B, K, z_hard.shape[1], z_hard.shape[2]).detach()
            sim_hard = torch.einsum('bd,bktd->bkt', F_pos, z_hard).max(dim=-1).values
            sim_hard = sim_hard.masked_fill(~hard_mask.to(sim_hard.device), -1e4)
            pos_diag = pos_sim_mat.diag().unsqueeze(1)
            group_logits = torch.cat([pos_diag, sim_hard], dim=1) * self.group_scale
            group_gt = torch.zeros(B, dtype=torch.long, device=F_pos.device)
            l_group = F.cross_entropy(group_logits, group_gt)
        else:
            l_group = torch.zeros((), device=self.device)

        # ── Push-pull objective on (pos, neg) — the core v3 supervision. ────
        # ⚠ Earlier version used `kl_pos - kl_neg_to_pos`, which has no lower
        # bound: the model can drive kl_neg_to_pos → +∞ by making M_act ≡ 1
        # (then 1−M_act ≡ 0, F_neg loses all visual signal, neg softmax
        # becomes near-uniform, KL(near-uniform || pos) blows up).
        # That degenerate solution was observed in run1 (m_act_mean = 1.0,
        # l_kl_action ≈ −260, model ≈ v2 baseline).
        #
        # New formulation:
        #   l_push = cross_entropy(neg_logits, gt)
        #     "F_neg should ALSO be able to retrieve its target" was wrong —
        #     we want the OPPOSITE. So we maximize CE = minimize -CE, equivalent
        #     to forcing the negative query to be a BAD retriever. To keep it
        #     bounded, we use −log(1 − softmax_neg[gt]) instead of raw -CE.
        #
        # i.e. for each sample i, neg_p_i = softmax(neg_logits[i])[i]. We want
        # neg_p_i → 0, so loss = −log(1 − neg_p_i + eps). This is bounded in
        # [0, +∞) and saturates as neg_p_i → 0 (gradient stops pushing M_act
        # to extremes once the mask is sufficiently discriminative).
        neg_sim_tokens = torch.einsum('bd,ntd->bnt', F_neg, z_tgt_tokens)
        neg_sim_mat = neg_sim_tokens.max(dim=-1).values
        neg_logits = scale * neg_sim_mat
        neg_softmax = F.softmax(neg_logits, dim=-1)         # [B, B]
        neg_p_gt = neg_softmax.gather(1, gt.view(-1, 1)).squeeze(1)  # [B]
        l_push = -torch.log((1.0 - neg_p_gt).clamp_min(1e-6)).mean()

        # ── Mask diversity regularizer: penalize M_act collapsing to a
        #    constant. We use −Var(M_act) as the loss term (small weight),
        #    so the optimizer is encouraged to produce non-trivial masks.
        m_var = M_act.var(dim=-1).mean()                     # scalar
        l_mask_div = -m_var

        # ── Mask sparsity prior (gentle): keep average retention near 0.5
        #    so the head doesn't trivially output near-1 (≈ v2) or near-0
        #    (drops all info). This is a soft target, not a hard constraint.
        l_mask_target = (M_act.mean() - 0.5).pow(2)

        # Total auxiliary action loss
        l_kl_action = l_push + 0.1 * l_mask_div + 0.1 * l_mask_target

        # ── Other v2-style auxiliary losses (align / global / colbert). ─────
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
            + self.lambda_align * l_align
            + self.lambda_global * l_global
            + self.lambda_colbert * l_colbert
        )

        return {
            "l_main": l_pos,
            "l_aux": l_aux,
            "l_group": l_group,
            "l_kl_action": l_kl_action,
            "l_align": l_align,
            "l_global": l_global,
            "l_colbert": l_colbert,
            # Diagnostic: average mask retention rate (lower = more selective)
            "m_act_mean": M_act.mean().detach(),
            # Effective mask after ramp-in: this is what Q-Former actually
            # gets multiplied by. = aux_ramp * M_act + (1 - aux_ramp).
            "m_eff_mean": (
                self.aux_ramp * M_act.mean() + (1.0 - self.aux_ramp)
            ).detach(),
            "loss": total,
        }
