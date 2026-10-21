"""SSN-BLIP2-v10 主模型: per-class action visual branch + drop F_trans/Fg.

继承 v8. 改动:
  1. __init__: 加 PerClassActionBranch (top-K=8 视觉聚合).
  2. _build_query: 删 Fg, 删 F_trans 末项, 删 F_trans 进 token_fusion;
     新 F_action 进 token_fusion 当 K/V (8 个 token).
  3. _visual_branches override: 不再产 Fg.
  4. dynamic_scalar 输入退回 v8 风格 768 维 (没 F_trans 进 scalar).
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

from SSN_v10.per_class_action import (                     # noqa: E402
    PerClassActionBranch, action_to_category_v10,
)


class Model(V8Model):
    """v10 = v8 - F_trans MLP - Fg + per-class mask-based F_action."""

    def __init__(self, *args, **kwargs):
        # v10-only kwargs
        self._top_k_action = int(kwargs.pop("top_k_action", 8))
        self._action_branch_hidden = int(kwargs.pop("action_branch_hidden", 512))

        super().__init__(*args, **kwargs)

        vit_dim = self.visual_encoder.embed_dim       # 1408
        qformer_h = self.Qformer.config.hidden_size   # 768

        # 新增 per-class action branch
        self.per_class_action = PerClassActionBranch(
            vit_dim=vit_dim,
            node_dim=qformer_h,
            embed_dim=self.embed_dim,
            hidden_dim=self._action_branch_hidden,
            top_k=self._top_k_action,
            dropout=0.1,
        )

        # 因为继承的 v8 在 v9 路径下 dynamic_scalar 是 1024 输入, 但 v10
        # 不用 F_trans 进 scalar, 所以要把 dynamic_scalar 重设回 768 输入.
        # 直接继承 v2 baseline 的设计: proj_dim * 3 = 768.
        proj_dim = self._proj_dim
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim * 4),
            nn.ReLU(),
            nn.Linear(proj_dim * 4, 1),
            nn.Sigmoid(),
        )

    # ── Override _visual_branches: 不再产 Fg ────────────────────────────
    def _visual_branches(
        self,
        patch_tokens_ln: torch.Tensor,
        obj_mask: torch.Tensor,
        obj_node_out: Dict[str, torch.Tensor],
        graphs: Sequence[Any],
    ) -> Dict[str, torch.Tensor]:
        """跟 v8 几乎一样, 但返回 Fg = None (token_fusion 不再用 Fg)."""
        vis = super()._visual_branches(
            patch_tokens_ln, obj_mask, obj_node_out, graphs
        )
        # 把 Fg 置 None: token_fusion 看到 None 会跳过它
        # (token_fusion dual-mode 已经支持 None 输入)
        vis["Fg"] = None
        return vis

    # ── Override _build_query ────────────────────────────────────────────
    def _build_query(
        self,
        image_embeds_ln: torch.Tensor,
        captions: List[str],
        graphs: Sequence[Any],
        target_patches: Any = None,
    ) -> Dict[str, torch.Tensor]:
        # Q-Former 主路 (raw)
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

        # action 侧 (跟 v8 一致 + 新增 per-class action branch)
        ctx = self._build_action_inputs(image_embeds_ln, graphs)
        patches_action = self._compose_patches(ctx, M_token=self._external_M_token)

        # Visual branches (Fg=None, Fl/Ff 仍 multi-token 见 v8)
        vis = self._visual_branches(
            patches_action, ctx["M_obj_ref"],
            {"node_features": None, "node_mask": None}, graphs,
        )
        Fl, Ff = vis["Fl"], vis["Ff"]   # Fg 已被置 None

        # === v10 新设计: PerClassActionBranch ===
        # 用 v10 4-class 标签 (v8 是 v6 的 4 类, 完全兼容)
        v10_labels = [action_to_category_v10(
            self._parse_graph(g).get("action")
        ) for g in graphs]
        action_out = self.per_class_action(
            patches=patches_action,
            obj_feat=ctx["obj_feat"],
            action_feat=ctx["action_feat"],
            action_labels=v10_labels,
        )
        F_action = action_out["F_action"]   # [B, K=8, embed_dim]
        out["M_cls"] = action_out["M_cls"].detach()
        out["m_cls_mean"] = action_out["M_cls"].mean().detach()

        # token_fusion: K/V = Fl(4) + Ff(4) + F_action(8) = 16 tokens
        # token_fusion.py 已经 dual-mode + 接受 None
        Fq_fused = self.token_fusion(Fq_u, None, Fl, Ff, F_action)

        # dynamic_scalar (退回 v8 风格 768)
        scalar_input = torch.cat([text_embed, ref_embed, Fq_fused], dim=-1)
        dynamic_s = self.dynamic_scalar(scalar_input)
        beta = torch.clamp(self.beta, min=0.0, max=self.beta_max) * self.aux_ramp

        # blend dropout (沿用 v8/v9)
        if self.training and self._blend_dropout_p > 0.0:
            ramp_done_flag = self.aux_ramp.item() >= 0.99
            # v10 没有 gamma 作用了 (F_trans 末项删除), 用 beta 单独判断
            branch_strength = beta.detach().abs().item()
            if ramp_done_flag and branch_strength > 0.1:
                keep = (torch.rand(Fq_fused.shape[0], 1, device=Fq_fused.device)
                        > self._blend_dropout_p).to(Fq_fused.dtype)
                blend_scale = keep
            else:
                blend_scale = 1.0
        else:
            blend_scale = 1.0

        # final_query: 没有 γ·F_trans 末项
        final_query_full = (
            alpha * self.combiner_output(combined)
            + beta * Fq_fused
            + blend_scale * (dynamic_s * text_embed + (1 - dynamic_s) * ref_embed)
        )
        out["final_query"] = F.normalize(final_query_full, dim=-1)
        out["_action_ctx"] = ctx
        # 为兼容 v8 forward 的 l_trans_align 等, 仍要返回 F_trans_norm.
        # v10 没有 F_trans, 我们用 F_action 的 mean 作为替代 (语义上是
        # "transition direction" 的近似), 这样 v8 的 l_trans_align 仍能跑.
        # 不想要这条监督的话训练时设 lambda_trans_align=0.
        F_action_pooled = F_action.mean(dim=1)                  # [B, embed_dim]
        out["F_trans_raw"] = F_action_pooled
        out["F_trans_norm"] = F.normalize(F_action_pooled, dim=-1)

        # Counter-factual (跟 v8 一致, 用 zeros 占位 Fq_fused)
        if getattr(self, "_compute_blend", False):
            scalar_input_blend = torch.cat(
                [text_embed, ref_embed, torch.zeros_like(Fq_fused)], dim=-1)
            dynamic_s_blend = self.dynamic_scalar(scalar_input_blend)
            final_query_blend = (
                alpha * self.combiner_output(combined)
                + dynamic_s_blend * text_embed
                + (1 - dynamic_s_blend) * ref_embed
            )
            out["final_blend"] = F.normalize(final_query_blend, dim=-1)
            # action-only: 只剩 beta·Fq_fused
            action_dir = beta * Fq_fused
            action_norm = action_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            out["final_action"] = action_dir / action_norm

        return out
