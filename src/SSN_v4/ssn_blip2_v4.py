"""SSN-BLIP2-v4: action-only mask + semantic injection.

Design summary
--------------

In v3 the action mask was multiplied into the Q-Former input, which
contaminated the main retrieval branch (Fq_u / ref_embed / text_embed)
and made test-time ablation impossible to interpret. v4 isolates the
action signal:

    patches_main  = patches                                  (raw)
    patches_final = patches × M_act + α · M_loc · obj_proto  (action-side)

Where:
  * Main retrieval branch (Q-Former image+text fusion, Q-Former image-only,
    text_proj) operates on ``patches_main`` (raw, identical to baseline).
  * Action-side branches (ActionRouter for F_trans, Fl, Ff, Fg in
    token_fusion → Fq_fused) operate on ``patches_final``.

So M_act only affects what flows into β·Fq_fused + γ·F_trans, never the
Q-Former main path. Ablating M_act and α_inject is now a clean test of
the action branch's marginal contribution — no OOD shift.

Loss design follows v3: push-pull on (F_pos with own M_act,
F_neg with a batch-shuffled M_act) plus a small mask-diversity prior.

We inherit from v2's ``Model`` directly (not v3) to avoid double-hijacking.
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

from models.ssn_blip2 import Model as BaselineModel        # noqa: E402
from model.action_transitions import action_to_category    # noqa: E402
from model.object_localizer import aggregate_mask          # noqa: E402

from SSN_v3.action_mask import ActionMaskHead              # noqa: E402  (reuse)
from SSN_v4.semantic_injector import SemanticInjector      # noqa: E402


class Model(BaselineModel):
    """SSN-BLIP2-v4: M_act on action-side branches only, plus semantic injection."""

    def __init__(self, *args, **kwargs):
        # v4-only kwargs (pop before passing to baseline init)
        mask_hidden_dim = int(kwargs.pop("mask_hidden_dim", 512))
        mask_dropout = float(kwargs.pop("mask_dropout", 0.1))
        mask_bias_init = float(kwargs.pop("mask_bias_init", 0.0))
        self.lambda_kl_action = float(kwargs.pop("lambda_kl_action", 0.1))

        inject_hidden_dim = int(kwargs.pop("inject_hidden_dim", 1024))
        inject_dropout = float(kwargs.pop("inject_dropout", 0.1))
        alpha_inject_init = float(kwargs.pop("alpha_inject_init", 0.05))
        alpha_inject_max = float(kwargs.pop("alpha_inject_max", 0.5))

        # ── Anti-shortcut machinery (forces the optimizer to actually
        # train Fq_fused / F_trans instead of riding the SSN blend) ────
        # Counterfactual loss: requires L(full_query) < L(blend_only) − margin.
        self.lambda_counterfact = float(kwargs.pop("lambda_counterfact", 0.5))
        self.counterfact_margin = float(kwargs.pop("counterfact_margin", 0.05))
        # Main-path dropout: with this probability per sample, zero out the
        # SSN blend at training time so the optimizer must rely on the
        # action branch.
        self._blend_dropout_p = float(kwargs.pop("blend_dropout", 0.2))

        super().__init__(*args, **kwargs)

        vit_dim = self.visual_encoder.embed_dim       # 1408
        qformer_h = self.Qformer.config.hidden_size   # 768

        # ── New modules ──────────────────────────────────────────────────
        self.action_mask_head = ActionMaskHead(
            vit_dim=vit_dim, node_dim=qformer_h,
            hidden_dim=mask_hidden_dim, dropout=mask_dropout,
            bias_init=mask_bias_init,
        )
        self.semantic_injector = SemanticInjector(
            node_dim=qformer_h, vit_dim=vit_dim,
            hidden_dim=inject_hidden_dim, dropout=inject_dropout,
            alpha_init=alpha_inject_init, alpha_max=alpha_inject_max,
        )

        # ── Extra ablation switches ──────────────────────────────────────
        self.ablate_no_action_mask: bool = False
        self.ablate_no_semantic_inject: bool = False

        # Used to share M_act / patches_final between pos- and neg-passes
        # in forward() without recomputing.
        self._external_M_token: torch.Tensor | None = None
        # Signals _build_query to also emit the counter-factual ``final_blend``
        # query (only needed during training).
        self._compute_blend: bool = False

    # ── Ablation ----------------------------------------------------------
    def set_ablation(
        self,
        no_trans: bool = False,
        no_fl: bool = False,
        no_ff: bool = False,
        force_route: str = "",
        no_action_mask: bool = False,
        no_semantic_inject: bool = False,
    ) -> None:
        super().set_ablation(
            no_trans=no_trans, no_fl=no_fl, no_ff=no_ff,
            force_route=force_route,
        )
        self.ablate_no_action_mask = bool(no_action_mask)
        self.ablate_no_semantic_inject = bool(no_semantic_inject)

    # ── Helpers -----------------------------------------------------------
    def _encode_field_safe(self, graphs: Sequence[Any], field: str) -> torch.Tensor:
        """Convenience wrapper. Same as baseline ``_encode_field``."""
        return self._encode_field(graphs, field)

    def _build_action_inputs(
        self,
        image_embeds_ln: torch.Tensor,   # [B, 1+N, 1408]
        graphs: Sequence[Any],
    ) -> Dict[str, torch.Tensor]:
        """Compute (M_act, M_obj, M_loc, obj_feat, loc_feat, action_feat,
        action_labels) once per build. These feed both the action-side
        ``patches_final`` and the rest of the v2-style action branch.
        """
        patch_tokens = image_embeds_ln[:, 1:, :]      # [B, N, 1408]
        B, N, _ = patch_tokens.shape
        device = patch_tokens.device

        # ── Graph encoding (BERT, no_grad inside) ─────────────────────────
        # obj_feat (multi-node mean, used as M_act head context):
        obj_node_out = self._encode_obj_nodes(graphs)
        nm = obj_node_out["node_mask"].float().unsqueeze(-1)
        obj_feat = ((obj_node_out["node_features"] * nm).sum(1)
                    / nm.sum(1).clamp_min(1e-6))      # [B, H]
        action_feat = self._encode_field(graphs, "action")        # [B, H]
        loc_feat = self._encode_field(graphs, "result_location")  # [B, H]
        # ``object_feat``: pure object semantics (used by SemanticInjector).
        # ``obj_feat`` above is a mean over object + location + result_location
        # nodes; for injection we want ONLY the object semantics so the
        # "flag" in "add a flag on the road" is what gets injected, not the
        # mean of "flag + road".
        object_feat = self._encode_field(graphs, "object")        # [B, H]

        # Action labels and the per-sample Add flag.
        parsed = [self._parse_graph(g) for g in graphs]
        cats = [action_to_category(p.get("action")) for p in parsed]
        if self.ablate_force_route:
            cats = [self.ablate_force_route] * len(cats)
        is_add = torch.tensor([c == "Add" for c in cats], device=device)
        is_remove = torch.tensor([c == "Remove" for c in cats], device=device)

        # ── ObjectLocalizer on RAW patches: object node → M_obj_ref ───────
        # (used by Fl/Ff weighted mean and ActionRouter region pooling.)
        loc_nodes_obj = self._encode_object_for_localizer(graphs)
        with torch.no_grad():
            M_obj_ref = aggregate_mask(self.object_localizer(
                patch_tokens.float(),
                loc_nodes_obj["node_features"].float(),
                loc_nodes_obj["node_mask"],
            ))
        # Plan-A: Add samples → uniform M_obj.
        if is_add.any():
            M_obj_ref = M_obj_ref.clone()
            M_obj_ref[is_add] = 1.0

        # ── ObjectLocalizer for result_location → M_loc (where to inject) ─
        # M_loc is only used by SemanticInjector for non-Remove samples.
        # Remove samples have meaningless result_location (fallback to the
        # being-removed object), so we just give them an all-zero M_loc.
        if (~is_remove).any():
            loc_nodes_dst = self._encode_field_as_node(graphs, "result_location")
            with torch.no_grad():
                M_loc = aggregate_mask(self.object_localizer(
                    patch_tokens.float(),
                    loc_nodes_dst["node_features"].float(),
                    loc_nodes_dst["node_mask"],
                ))                                            # [B, N] in (0, 1)
            # Zero out Remove samples explicitly — the injection enable
            # mask in _compose_patches already handles this, but keeping
            # M_loc zero there makes the diagnostics interpretable.
            if is_remove.any():
                M_loc = M_loc.clone()
                M_loc[is_remove] = 0.0
        else:
            M_loc = torch.zeros(B, N, device=device, dtype=patch_tokens.dtype)

        # ── ActionMaskHead (or all-ones if ablated) ───────────────────────
        if self.ablate_no_action_mask:
            M_act = torch.ones(B, N, device=device, dtype=patch_tokens.dtype)
        else:
            M_act = self.action_mask_head(
                patches=patch_tokens, action_feat=action_feat,
                M_obj=M_obj_ref, obj_feat=obj_feat,
            )

        return {
            "patch_tokens": patch_tokens,
            "M_act": M_act,
            "M_obj_ref": M_obj_ref,
            "M_loc": M_loc,
            "obj_feat": obj_feat,             # multi-node mean (for action_router)
            "object_feat": object_feat,       # pure object semantics (for injector)
            "loc_feat": loc_feat,
            "action_feat": action_feat,
            "action_labels": cats,
            "is_add": is_add,
            "is_remove": is_remove,
        }

    def _encode_field_as_node(
        self, graphs: Sequence[Any], field: str
    ) -> Dict[str, torch.Tensor]:
        """Encode a single graph field as a one-node ObjectLocalizer query
        (matches the 'graph node: {label}' format the selector trained on)."""
        from model.graph_utils import encode_graph_node_features
        mini = []
        for g in graphs:
            g_parsed = self._parse_graph(g)
            v = g_parsed.get(field)
            if not v or str(v).strip().lower() == "unknown":
                # Fall back to the object field, then to a placeholder.
                v = g_parsed.get("object") or "unknown"
            mini.append({"nodes": [{"label": v}]})
        return encode_graph_node_features(
            tokenizer=self.tokenizer, bert_model=self.Qformer.bert,
            graphs=mini, device=self.device, max_txt_len=self.max_txt_len,
        )

    def _compose_patches(
        self,
        ctx: Dict[str, torch.Tensor],
        M_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build patches_final = patches × M_act + α · M_loc · obj_proto.

        ``M_token`` lets the caller override M_act (used by the negative
        pass: M_token = M_act[perm]). When None, we use ctx["M_act"].
        """
        patches = ctx["patch_tokens"]
        M_act = M_token if M_token is not None else ctx["M_act"]

        # Ramp-in: blend toward all-ones during early graph epochs.
        r = self.aux_ramp.to(M_act.dtype)
        M_eff = r * M_act + (1.0 - r)

        patches_final = patches * M_eff.unsqueeze(-1).to(patches.dtype)

        if not self.ablate_no_semantic_inject:
            # Injection enabled where the new object is meaningful — Add
            # always; Move sometimes brings a new attribute too. Remove
            # never injects (object is being removed).
            enable = ~ctx["is_remove"]   # [B] bool
            inject = self.semantic_injector(
                obj_feat=ctx["object_feat"], M_loc=ctx["M_loc"], enable=enable,
            )                            # [B, N, 1408]
            # Also fade in via aux_ramp (we don't want random injection
            # during the first graph epochs).
            patches_final = patches_final + r * inject.to(patches.dtype)

        return patches_final

    # ── Overridden _build_query ------------------------------------------
    def _build_query(
        self,
        image_embeds_ln: torch.Tensor,
        captions: List[str],
        graphs: Sequence[Any],
        target_patches: Any = None,
    ) -> Dict[str, torch.Tensor]:
        # ── Main retrieval branch (RAW patches; identical to v2) ─────────
        text_tokens = self._tokenize(captions)
        Fq_u, q_tokens, text_cls = self._qformer_fusion(image_embeds_ln, text_tokens)

        F_img_only = self._qformer_image_only(image_embeds_ln, self.query_tokens)
        ref_embed = F.normalize(self.vision_proj(F_img_only).mean(dim=1), dim=-1)
        text_embed = F.normalize(self.text_proj(text_cls), dim=-1)

        raw_combined = torch.cat([text_embed, ref_embed], dim=-1)
        combined = F.relu(self.combiner_layer(raw_combined))
        alpha = torch.clamp(self.alpha, min=0.0, max=self.alpha_max)

        out: Dict[str, torch.Tensor] = {
            "q_tokens": q_tokens, "Fq_u": Fq_u, "text_embed": text_embed,
        }

        if not self.graph_enabled:
            # Warmup path: SSN blend only.
            scalar_input = torch.cat(
                [text_embed, ref_embed, torch.zeros_like(ref_embed)], dim=-1)
            dynamic_s = self.dynamic_scalar(scalar_input)
            final_query = (
                alpha * self.combiner_output(combined)
                + dynamic_s * text_embed
                + (1 - dynamic_s) * ref_embed
            )
            out["final_query"] = F.normalize(final_query, dim=-1)
            return out

        # ── Action-side: compose patches_final and run action branches ────
        ctx = self._build_action_inputs(image_embeds_ln, graphs)
        # external_M_token lets forward() override M_act for the negative pass.
        patches_action = self._compose_patches(ctx, M_token=self._external_M_token)

        # ActionRouter on composed patches (region pooling uses M_obj_ref).
        router_out = self.action_router(
            ctx["action_labels"], patches_action, ctx["M_obj_ref"],
            ctx["obj_feat"], ctx["loc_feat"], ctx["action_feat"],
        )
        F_trans = router_out["F_trans"]

        # Visual branches (Fg/Fl/Ff) on composed patches.
        # NOTE: _visual_branches expects ``patch_tokens_ln`` of shape [B, N, vit_dim]
        # and an obj_mask [B, N]. We feed patches_action and the (raw) M_obj_ref.
        # Pass a dummy obj_node_out (its consumer-side use was removed in v2).
        vis = self._visual_branches(
            patches_action, ctx["M_obj_ref"], {"node_features": None, "node_mask": None}, graphs,
        )
        Fq_fused = self.token_fusion(Fq_u, vis["Fg"], vis["Fl"], vis["Ff"])

        # SSN-style linear addition (same as v2 schema).
        scalar_input = torch.cat([text_embed, ref_embed, Fq_fused], dim=-1)
        dynamic_s = self.dynamic_scalar(scalar_input)
        beta = torch.clamp(self.beta, min=0.0, max=self.beta_max) * self.aux_ramp
        gamma = self.gamma * self.aux_ramp
        if self.ablate_no_trans:
            gamma = gamma * 0.0
        F_trans_n = F.normalize(F_trans, dim=-1).to(Fq_fused.dtype)

        # ── Main-path dropout: with probability ``blend_dropout`` during
        # training, zero out the SSN blend (s·text + (1-s)·ref) so the
        # optimizer cannot solve the retrieval task with the blend alone
        # and is forced to train Fq_fused / F_trans through β / γ.
        #
        # IMPORTANT: Two gates before activating dropout —
        #   (1) ``aux_ramp`` must be done (β/γ have been allowed to flow);
        #   (2) ``β + γ`` must be large enough that final_query without the
        #       blend still has non-trivial magnitude. Otherwise dropped
        #       samples produce a near-zero query, normalize amplifies noise,
        #       and the resulting bad InfoNCE actively HURTS training.
        eff_dropout_p = self._blend_dropout_p
        ramp_done_flag = self.aux_ramp.item() >= 0.99
        # β and γ each clamped to [0, 0.5]; require their sum > 0.2 (i.e.
        # the action branch contributes ≥ 20% as much energy as the blend).
        branch_strength = (
            beta.detach().abs().item() + gamma.detach().abs().item()
        )
        if not (ramp_done_flag and branch_strength > 0.2):
            eff_dropout_p = 0.0

        if self.training and eff_dropout_p > 0.0:
            keep = (torch.rand(Fq_fused.shape[0], 1, device=Fq_fused.device)
                    > eff_dropout_p).to(Fq_fused.dtype)
            blend_scale = keep
        else:
            blend_scale = 1.0

        final_query_full = (
            alpha * self.combiner_output(combined)
            + beta * Fq_fused
            + gamma * F_trans_n
            + blend_scale * (dynamic_s * text_embed + (1 - dynamic_s) * ref_embed)
        )
        out["final_query"] = F.normalize(final_query_full, dim=-1)

        # ── Counter-factual queries (training-only diagnostics):
        # - ``final_blend``  : SSN blend ONLY (no β·Fq_fused, no γ·F_trans).
        #                      The reference baseline: "what can the model do
        #                      using just text+ref blending?"
        # - ``final_action`` : action branch ONLY (no SSN blend).
        #                      Used by the counter-factual loss to force the
        #                      action branch to be a competent retriever on
        #                      its own — otherwise its gradient is dwarfed
        #                      by the blend.
        # IMPORTANT: ``final_blend`` uses dynamic_s_blend computed WITHOUT
        # Fq_fused in its input, so it is a clean baseline. Otherwise the
        # blend would inherit action info via dynamic_s, defeating the
        # counter-factual comparison.
        if getattr(self, "_compute_blend", False):
            # Re-run dynamic_scalar with Fq_fused replaced by zeros — this
            # gives a "blend-only" interpolation weight.
            scalar_input_blend = torch.cat(
                [text_embed, ref_embed, torch.zeros_like(Fq_fused)], dim=-1)
            dynamic_s_blend = self.dynamic_scalar(scalar_input_blend)

            final_query_blend = (
                alpha * self.combiner_output(combined)
                + dynamic_s_blend * text_embed
                + (1 - dynamic_s_blend) * ref_embed
            )
            out["final_blend"] = F.normalize(final_query_blend, dim=-1)

            # ── Action-only direction: the L2-normalized direction of
            # β·Fq_fused + γ·F_trans. Note: we DO normalize this because
            # InfoNCE expects unit vectors against ``z_tgt_tokens`` which
            # are also normalized. To avoid the "tiny vector → noise after
            # normalize" problem when β/γ are small, the caller (forward())
            # only uses ``final_action`` after ``branch_strength`` > 0.2.
            action_dir = beta * Fq_fused + gamma * F_trans_n
            action_norm = action_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            out["final_action"] = action_dir / action_norm

        # Stash ctx so forward() can reuse M_act for the negative pass.
        out["_action_ctx"] = ctx
        return out

    # ── Inference --------------------------------------------------------
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
            self._external_M_token = None
            out = self._build_query(image_embeds_ln, captions, graphs)
            final_query = out["final_query"]
        finally:
            self.graph_enabled = prev

        z_tgt_tokens = F.normalize(self.vision_proj(F_target_all), dim=-1)
        sim = torch.einsum('bd,ntd->bnt', final_query, z_tgt_tokens).max(dim=-1).values
        return sim

    # ── Forward (training) ------------------------------------------------
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

        # ── Positive query: use the head's own M_act. Also request the
        # counter-factual blend-only query so we can compute L_full vs
        # L_blend in one pass.
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
        # Counter-factual queries are only USED when the head has had time
        # to learn AND β/γ are non-trivial — see the same gate inside
        # _build_query for blend_dropout. If we activate the hinge before
        # β/γ have grown, the action-only query is essentially zero, gets
        # normalized to noise, and the hinge pushes β/γ in arbitrary
        # directions, hurting training.
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

        # ── Negative query: shuffled M_act (only if M_act is active). ────
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

        # ── Main InfoNCE (positive query only) ───────────────────────────
        self.temp.data = torch.clamp(self.temp.data, min=1e-3)
        scale = 1.0 / self.temp

        pos_sim_tokens = torch.einsum('bd,ntd->bnt', F_pos, z_tgt_tokens)
        pos_sim_mat = pos_sim_tokens.max(dim=-1).values
        pos_logits = scale * pos_sim_mat
        gt = torch.arange(pos_logits.shape[0], device=pos_logits.device)
        ls = self.label_smoothing
        l_pos = (F.cross_entropy(pos_logits, gt, label_smoothing=ls)
                 + F.cross_entropy(pos_logits.T, gt, label_smoothing=ls)) / 2

        # ── Aux head (same as v2/v3) ─────────────────────────────────────
        q_guided = self.aux_head(q_tokens, F_target.detach())
        aux_sim = scale * torch.einsum('bd,ntd->bnt',
                                       q_guided, z_tgt_tokens).max(dim=-1).values
        l_aux = (F.cross_entropy(aux_sim, gt) + F.cross_entropy(aux_sim.T, gt)) / 2

        # ── Group-aware hard negatives (same as v2) ──────────────────────
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

        # ── Push-pull objective on (pos, neg) ────────────────────────────
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
            l_mask_target = (M_act.mean() - 0.5).pow(2)

            l_kl_action = l_push + 0.1 * l_mask_div + 0.1 * l_mask_target
            m_act_mean_val = M_act.mean().detach()

        # ── Counter-factual margin loss ──────────────────────────────────
        # We want: the ACTION-ONLY query (β·Fq_fused + γ·F_trans, normalized
        # as a direction) should retrieve better than the blend-only query,
        # otherwise the action branch is contributing nothing.
        #
        # Concretely we ask
        #     L(F_action) ≤ L(F_blend) − margin
        # via a ReLU hinge.
        #
        # Why F_action (not F_pos)? If we hinge L(F_pos) vs L(F_blend), the
        # gradient flows through every component of final_query — including
        # the blend itself — so the optimizer can satisfy the hinge by
        # making the blend better, not by training the action branch. By
        # using F_action (which is exactly β·Fq_fused + γ·F_trans, no blend),
        # the hinge gradient goes ONLY into action-branch parameters.
        #
        # ``l_blend.detach()`` ensures the optimizer can't satisfy the
        # hinge by making the blend deliberately worse.
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
            # Hinge: action-only should beat blend-only by ``margin``.
            l_counterfact = F.relu(
                l_action - l_blend.detach() + self.counterfact_margin
            )

        # ── Other v2-style auxiliary losses ──────────────────────────────
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
