"""SSN-BLIP2-v9: F_trans 进 dynamic_scalar 影响 s.

继承 v8. 核心改动:
  1. __init__: 用新 dynamic_scalar_v9 (输入 1024 维) 替换继承的 dynamic_scalar
     (输入 768 维). 多吃一个 F_trans 输入.
  2. _build_query: 把 F_trans concat 到 dynamic_scalar 输入里.

v8 / v7 / v6 / 共享模块 (token_fusion / reference_fft) 全部不动.
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
from SSN_v8.ssn_blip2_v8 import Model as V8Model           # noqa: E402


class Model(V8Model):
    """v9 = v8 + dynamic_scalar 多吃 F_trans 输入."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ── 替换 dynamic_scalar: 768 → 1024 (多 256 维 F_trans 输入) ──
        # 原 v2 baseline 的 dynamic_scalar 是 nn.Sequential(
        #     Linear(768, 1024), ReLU, Linear(1024, 1), Sigmoid
        # )
        # v9 把第一个 Linear 改成 1024 → 1024.
        proj_dim = self._proj_dim
        scalar_input_dim_v9 = proj_dim * 4   # 256 × 4 = 1024
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(scalar_input_dim_v9, proj_dim * 4),
            nn.ReLU(),
            nn.Linear(proj_dim * 4, 1),
            nn.Sigmoid(),
        )

    # ── Override _build_query ────────────────────────────────────────────
    def _build_query(
        self,
        image_embeds_ln: torch.Tensor,
        captions: List[str],
        graphs: Sequence[Any],
        target_patches: Any = None,
    ) -> Dict[str, torch.Tensor]:
        # Main path (raw patches) - 跟 v7/v8 一致
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
            # Warmup: dynamic_scalar 看不到 Fq_fused 也看不到 F_trans →
            # 用 zeros 填占位.
            scalar_input = torch.cat(
                [text_embed, ref_embed,
                 torch.zeros_like(ref_embed),       # Fq_fused 占位
                 torch.zeros_like(ref_embed)],      # F_trans 占位
                dim=-1)
            dynamic_s = self.dynamic_scalar(scalar_input)
            final_query = (
                alpha * self.combiner_output(combined)
                + dynamic_s * text_embed
                + (1 - dynamic_s) * ref_embed
            )
            out["final_query"] = F.normalize(final_query, dim=-1)
            return out

        ctx = self._build_action_inputs(image_embeds_ln, graphs)
        patches_action = self._compose_patches(ctx, M_token=self._external_M_token)

        obj_proto = self.semantic_injector.proto_mlp(
            ctx["object_feat"].to(self.semantic_injector.proto_mlp[-1].weight.dtype)
        )

        router_out = self.action_router(
            action_labels=ctx["action_labels"],
            patches=patches_action,
            M_obj_ref=ctx["M_obj_ref"],
            obj_feat=ctx["obj_feat"],
            loc_feat=ctx["loc_feat"],
            action_feat=ctx["action_feat"],
            obj_proto=obj_proto,
        )
        F_trans = router_out["F_trans"]

        vis = self._visual_branches(
            patches_action, ctx["M_obj_ref"],
            {"node_features": None, "node_mask": None}, graphs,
        )
        Fg, Fl, Ff = vis["Fg"], vis["Fl"], vis["Ff"]

        F_trans_n = F.normalize(F_trans, dim=-1).to(Fq_u.dtype)
        if self.f_trans_via_fusion:
            Fq_fused = self.token_fusion(Fq_u, Fg, Fl, Ff, F_trans_n)
        else:
            Fq_fused = self.token_fusion(Fq_u, Fg, Fl, Ff)

        # ── v9 改动: scalar_input 多吃一个 F_trans_n ────────────────────
        scalar_input = torch.cat(
            [text_embed, ref_embed, Fq_fused, F_trans_n], dim=-1)   # [B, 1024]
        dynamic_s = self.dynamic_scalar(scalar_input)

        beta = torch.clamp(self.beta, min=0.0, max=self.beta_max) * self.aux_ramp
        gamma = self.gamma * self.aux_ramp
        if self.ablate_no_trans:
            gamma = gamma * 0.0
        gamma_eff = gamma if self.f_trans_via_gamma else torch.zeros_like(gamma)

        # blend dropout
        if self.training and self._blend_dropout_p > 0.0:
            ramp_done_flag = self.aux_ramp.item() >= 0.99
            branch_strength = (
                beta.detach().abs().item() + gamma.detach().abs().item()
            )
            if ramp_done_flag and branch_strength > 0.2:
                keep = (torch.rand(Fq_fused.shape[0], 1, device=Fq_fused.device)
                        > self._blend_dropout_p).to(Fq_fused.dtype)
                blend_scale = keep
            else:
                blend_scale = 1.0
        else:
            blend_scale = 1.0

        final_query_full = (
            alpha * self.combiner_output(combined)
            + beta * Fq_fused
            + gamma_eff * F_trans_n
            + blend_scale * (dynamic_s * text_embed + (1 - dynamic_s) * ref_embed)
        )
        out["final_query"] = F.normalize(final_query_full, dim=-1)
        out["_action_ctx"] = ctx
        out["F_trans_raw"] = F_trans
        out["F_trans_norm"] = F_trans_n
        # Diagnostic: average s, can plot per-action distribution later.
        out["dynamic_s_mean"] = dynamic_s.mean().detach()

        # Counter-factual queries (training-only, used by l_trans_align /
        # counterfact). For blend-only we use zeros for both Fq_fused and
        # F_trans so the blend has no action signal.
        if getattr(self, "_compute_blend", False):
            scalar_input_blend = torch.cat(
                [text_embed, ref_embed,
                 torch.zeros_like(Fq_fused),
                 torch.zeros_like(F_trans_n)],
                dim=-1)
            dynamic_s_blend = self.dynamic_scalar(scalar_input_blend)
            final_query_blend = (
                alpha * self.combiner_output(combined)
                + dynamic_s_blend * text_embed
                + (1 - dynamic_s_blend) * ref_embed
            )
            out["final_blend"] = F.normalize(final_query_blend, dim=-1)
            action_dir = beta * Fq_fused + gamma_eff * F_trans_n
            action_norm = action_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            out["final_action"] = action_dir / action_norm

        return out
