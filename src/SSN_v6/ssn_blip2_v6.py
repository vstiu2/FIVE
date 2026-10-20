"""SSN-BLIP2-v6.

继承 v5 (它继承 v4)，做四件事：
1. 把 self.action_router 换成 v6 版本 (4-class, 修复 Add/Move region, LayerNorm 尾部)。
2. _build_query 里把 F_trans 同时送进 token_fusion (第 4 个 K/V) 和
   保留 γ·F_trans 项 (双出口)。
3. AddTransition 需要 obj_proto，所以从 SemanticInjector.proto_mlp 抽出来传入。
4. forward 末尾加 l_trans_align = 1 - cos(F_trans, target_embed - ref_embed)。
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

from models.ssn_blip2 import Model as BaselineModel  # noqa: E402
from SSN_v5.ssn_blip2_v5 import Model as V5Model     # noqa: E402

from SSN_v6.action_transitions_v6 import (             # noqa: E402
    ActionRouterV6, action_to_category_v6,
)


class Model(V5Model):
    """v6 = v5 + ActionRouter v6 + F_trans 进 token_fusion + l_trans_align."""

    def __init__(self, *args, **kwargs):
        # v6-only kwargs
        self.lambda_trans_align = float(kwargs.pop("lambda_trans_align", 0.1))
        self.f_trans_via_fusion = bool(kwargs.pop("f_trans_via_fusion", True))
        self.f_trans_via_gamma = bool(kwargs.pop("f_trans_via_gamma", True))

        super().__init__(*args, **kwargs)

        # 换成 v6 ActionRouter。先把 v2 留下来的 self.action_router 删掉
        # 再重新注册，避免 nn.Module 注册表里有重复 key。
        vit_dim = self.visual_encoder.embed_dim
        qformer_h = self.Qformer.config.hidden_size
        # v2/v5 baseline 通过 selector_hidden_dim 参数初始化 ActionRouter；
        # 我们沿用同一个 hidden_dim 以保持参数量近似可比。
        hidden_dim = self.action_router.remove_t.trans_mlp[1].out_features // 2  # _trans_mlp 内部是 hidden_dim*2
        dropout = 0.1

        # 删旧 + 装新
        del self._modules["action_router"]
        self.action_router = ActionRouterV6(
            vit_dim=vit_dim, node_dim=qformer_h,
            hidden_dim=hidden_dim, proj_dim=self.embed_dim, dropout=dropout,
        )

    # ── 重写动作分类，使用 4 类 ─────────────────────────────────────────────
    def _action_labels(self, graphs: Sequence[Any]) -> List[str]:
        parsed = [self._parse_graph(g) for g in graphs]
        return [action_to_category_v6(p.get("action")) for p in parsed]

    # ── 重写 _build_action_inputs，但只换 action_labels 的算法 ────────────
    def _build_action_inputs(
        self,
        image_embeds_ln: torch.Tensor,
        graphs: Sequence[Any],
    ) -> Dict[str, torch.Tensor]:
        ctx = super()._build_action_inputs(image_embeds_ln, graphs)
        # 父类把 action_labels 设成 v2 的 3 类，这里覆盖为 v6 的 4 类。
        ctx["action_labels"] = self._action_labels(graphs)
        # 同时维护 is_add / is_remove (4 类下含义不变)
        device = ctx["patch_tokens"].device
        cats = ctx["action_labels"]
        ctx["is_add"] = torch.tensor([c == "Add" for c in cats], device=device)
        ctx["is_remove"] = torch.tensor([c == "Remove" for c in cats], device=device)
        return ctx

    # ── 重写 _build_query ───────────────────────────────────────────────────
    def _build_query(
        self,
        image_embeds_ln: torch.Tensor,
        captions: List[str],
        graphs: Sequence[Any],
        target_patches: Any = None,
    ) -> Dict[str, torch.Tensor]:
        # Q-Former 主路 (raw patches): 同 v4/v5
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

        # action 侧
        ctx = self._build_action_inputs(image_embeds_ln, graphs)
        patches_action = self._compose_patches(ctx, M_token=self._external_M_token)

        # obj_proto 给 AddTransition 用 (从 SemanticInjector 的 proto_mlp 取)
        # 注意:  semantic_injector.proto_mlp 直接吃 BERT-768 的 obj_feat
        # 输出 ViT 维度的视觉原型;  我们用 ctx["object_feat"] (纯 object 节点
        # 的 BERT CLS) 作为输入。
        obj_proto = self.semantic_injector.proto_mlp(
            ctx["object_feat"].to(self.semantic_injector.proto_mlp[-1].weight.dtype)
        )                                                  # [B, vit_dim]

        # ActionRouter v6
        router_out = self.action_router(
            action_labels=ctx["action_labels"],
            patches=patches_action,
            M_obj_ref=ctx["M_obj_ref"],
            M_loc=ctx["M_loc"],
            obj_feat=ctx["obj_feat"],
            loc_feat=ctx["loc_feat"],
            action_feat=ctx["action_feat"],
            obj_proto=obj_proto,
        )
        F_trans = router_out["F_trans"]                    # [B, proj_dim]

        # 多尺度视觉分支
        vis = self._visual_branches(
            patches_action, ctx["M_obj_ref"],
            {"node_features": None, "node_mask": None}, graphs,
        )
        Fg, Fl, Ff = vis["Fg"], vis["Fl"], vis["Ff"]

        # F_trans 出口 1: 进 token_fusion 作为第 4 个 K/V
        F_trans_n = F.normalize(F_trans, dim=-1).to(Fq_u.dtype)
        if self.f_trans_via_fusion:
            Fq_fused = self.token_fusion(Fq_u, Fg, Fl, Ff, F_trans_n)
        else:
            Fq_fused = self.token_fusion(Fq_u, Fg, Fl, Ff)

        # SSN-style 线性组合
        scalar_input = torch.cat([text_embed, ref_embed, Fq_fused], dim=-1)
        dynamic_s = self.dynamic_scalar(scalar_input)
        beta = torch.clamp(self.beta, min=0.0, max=self.beta_max) * self.aux_ramp
        gamma = self.gamma * self.aux_ramp
        if self.ablate_no_trans:
            gamma = gamma * 0.0
        # F_trans 出口 2: γ·F_trans  (可关掉以做单出口消融)
        gamma_eff = gamma if self.f_trans_via_gamma else torch.zeros_like(gamma)

        # blend dropout (跟 v4/v5 一致, 双门控)
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
        out["F_trans_raw"] = F_trans          # 训练时拿来算 l_trans_align
        out["F_trans_norm"] = F_trans_n       # 同上 (向后兼容)

        # 反事实 (跟 v4/v5 完全一致)
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

            action_dir = beta * Fq_fused + gamma_eff * F_trans_n
            action_norm = action_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            out["final_action"] = action_dir / action_norm

        return out

    # ── 重写 forward, 在 v5 末尾追加 l_trans_align ──────────────────────────
    def forward(
        self,
        samples: Dict[str, Any],
        warmup: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if warmup:
            return BaselineModel.forward(self, samples, warmup=True)

        # 走 v5 forward 拿到 baseline losses + final_query 等
        out = V5Model.forward(self, samples, warmup=False)
        if self.lambda_trans_align <= 0.0:
            return out

        # 重新计算 ref_embed / target_embed / F_trans, 拿来做对齐监督。
        # 为避免重复跑 Q-Former, 我们重新走一次必要的前向 (在 no_grad 下拿
        # 锚点; F_trans 这条带 grad).
        image_embeds = samples["image"].to(self.device)
        target_embeds = samples["target"].to(self.device)
        captions = samples["text_input"]
        graphs = samples["graph"]

        with self.maybe_autocast():
            image_embeds_ln = self.ln_vision(image_embeds)
            target_embeds_ln = self.ln_vision(target_embeds)

        # ref_embed (raw patches, no grad: 它已经在 v5 forward 中被 InfoNCE
        # 训练过, 这里作为对齐 target 不带梯度)
        with torch.no_grad():
            F_img_only_ref = self._qformer_image_only(image_embeds_ln, self.query_tokens)
            ref_anchor = F.normalize(
                self.vision_proj(F_img_only_ref).mean(dim=1), dim=-1
            )                                              # [B, D]

            F_img_only_tgt = self._qformer_image_only(target_embeds_ln, self.query_tokens)
            tgt_anchor = F.normalize(
                self.vision_proj(F_img_only_tgt).mean(dim=1), dim=-1
            )                                              # [B, D]

        transition_target = tgt_anchor - ref_anchor        # [B, D]
        norm = transition_target.norm(dim=-1, keepdim=True)
        # 对小编辑(几乎没动) 不监督
        valid = (norm.squeeze(-1) > 1e-3)                  # [B]
        transition_target = F.normalize(
            transition_target, dim=-1, eps=1e-6
        )

        # F_trans  (要带梯度)
        self._external_M_token = None
        self._compute_blend = False
        self.graph_enabled = True
        out_q = self._build_query(
            image_embeds_ln, captions, graphs,
            target_patches=target_embeds_ln[:, 1:, :],
        )
        F_trans_n = out_q.get("F_trans_norm")
        if F_trans_n is None or valid.sum() == 0:
            return out

        # cosine alignment, 只在 valid 样本上算
        cos = (F_trans_n * transition_target).sum(dim=-1)          # [B]
        l_trans_align = (1.0 - cos[valid]).mean()

        out["l_trans_align"] = l_trans_align.detach()
        out["loss"] = out["loss"] + self.lambda_trans_align * l_trans_align
        return out
