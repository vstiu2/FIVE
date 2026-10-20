"""SSN-BLIP2-v7 主模型: drop M_loc + 用 object_core 做多 node grounding.

继承 v6_stage, override 4 处:
  1. ``_encode_object_for_localizer``: 用 graph.object_core (list) 多 node
     grounding 取代 v2 baseline 的单 string.
  2. ``_build_action_inputs``: 不算 M_loc, ctx 里 M_loc = None.
  3. ``_compose_patches``: SemanticInjector 改成全图均匀注入 obj_proto
     (不依赖 M_loc).
  4. ``__init__``: action_router 替换成 ActionRouterV7.

v4 / v5 / v6 / v6_stage / model/*.py 全部不动.
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
from model.object_localizer import aggregate_mask         # noqa: E402
from model.graph_utils import encode_graph_node_features  # noqa: E402

from SSN_v6.ssn_blip2_v6_stage import Model as V6StageModel       # noqa: E402
from SSN_v7.action_transitions_v7 import (                         # noqa: E402
    ActionRouterV7, action_to_category_v7,
)


class Model(V6StageModel):
    """v7 = v6_stage + object_core grounding + drop M_loc."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 把 v6 的 action_router 换成 v7 版本.
        vit_dim = self.visual_encoder.embed_dim
        qformer_h = self.Qformer.config.hidden_size
        # 沿用 v2 默认 hidden_dim = 512 (matches selector_hidden_dim)
        hidden_dim = 512
        dropout = 0.1

        # 删旧 + 装新 (跟 v6 model 的做法一致)
        del self._modules["action_router"]
        self.action_router = ActionRouterV7(
            vit_dim=vit_dim, node_dim=qformer_h,
            hidden_dim=hidden_dim, proj_dim=self.embed_dim, dropout=dropout,
        )

    # ── Override 1: 用 graph.object_core 多 node grounding ──────────────
    def _encode_object_for_localizer(self, graphs: Sequence[Any]) -> Dict[str, torch.Tensor]:
        """v7: prefer ``graph.object_core`` (list of head nouns) over the
        full ``graph.object`` string. Falls back to v2-style single string
        when object_core is missing (so v7 still works on old data).
        """
        mini = []
        for g in graphs:
            g_parsed = self._parse_graph(g)
            cores = g_parsed.get("object_core")
            if isinstance(cores, list) and cores:
                # multi-node grounding query
                labels = [str(c).strip() for c in cores if str(c).strip()]
                if not labels:
                    labels = ["unknown"]
            else:
                # fallback: v2-style single object string
                obj = g_parsed.get("object")
                if not obj or str(obj).strip().lower() == "unknown":
                    nodes = g_parsed.get("nodes") or []
                    obj = (nodes[0].get("label")
                           if nodes and isinstance(nodes[0], dict) else None)
                labels = [str(obj or "unknown").strip()]
            mini.append({"nodes": [{"label": x} for x in labels]})
        return encode_graph_node_features(
            tokenizer=self.tokenizer, bert_model=self.Qformer.bert,
            graphs=mini, device=self.device, max_txt_len=self.max_txt_len,
        )

    # ── v7 helper: BERT(simplified object string) for SemanticInjector ──
    def _encode_object_simplified(self, graphs: Sequence[Any]) -> torch.Tensor:
        """Encode a simplified-object string via Q-Former BERT, returning
        the [B, H] CLS embedding.

        For each graph, build the input text as " ".join(object_core). If
        object_core is missing, fall back to graph.object. If both missing,
        use "unknown". This mirrors ``_encode_field("object")`` but with
        spaCy-simplified content, so SemanticInjector's obj_proto is built
        on the same simplified noun(s) as the selector grounding query.
        """
        import json
        texts: List[str] = []
        for g in graphs:
            g_parsed = self._parse_graph(g)
            cores = g_parsed.get("object_core")
            if isinstance(cores, list) and cores:
                txt = " ".join(str(c).strip() for c in cores if str(c).strip())
            else:
                txt = g_parsed.get("object") or "unknown"
            txt = str(txt).strip() or "unknown"
            texts.append(f"graph object: {txt}")
        tokens = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.max_txt_len, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.Qformer.bert(
                tokens.input_ids, attention_mask=tokens.attention_mask,
                return_dict=True,
            )
        return out.last_hidden_state[:, 0, :]   # CLS, [B, H]

    # ── Override 2: _build_action_inputs 不算 M_loc ──────────────────────
    def _build_action_inputs(
        self,
        image_embeds_ln: torch.Tensor,
        graphs: Sequence[Any],
    ) -> Dict[str, torch.Tensor]:
        """同 v4/v5/v6 流程, 但跳过 M_loc 计算."""
        patch_tokens = image_embeds_ln[:, 1:, :]
        B, N, _ = patch_tokens.shape
        device = patch_tokens.device

        # Graph encoding
        obj_node_out = self._encode_obj_nodes(graphs)
        nm = obj_node_out["node_mask"].float().unsqueeze(-1)
        obj_feat = ((obj_node_out["node_features"] * nm).sum(1)
                    / nm.sum(1).clamp_min(1e-6))
        action_feat = self._encode_field(graphs, "action")
        loc_feat = self._encode_field(graphs, "result_location")
        # v7: object_feat 用 object_core 拼出的简化字符串 ("long boat-filled
        # dock" → "dock", "road and trees" → "road tree") 喂给 BERT, 而不是
        # 原始 graph.object 长串. 这样 SemanticInjector 的 obj_proto 也用
        # 简化语义, 跟 selector 路径一致.
        object_feat = self._encode_object_simplified(graphs)

        # Action 4-class labels (v7 sets unknown actions to Attr by default)
        parsed = [self._parse_graph(g) for g in graphs]
        cats = [action_to_category_v7(p.get("action")) for p in parsed]
        if self.ablate_force_route:
            cats = [self.ablate_force_route] * len(cats)
        is_add = torch.tensor([c == "Add" for c in cats], device=device)
        is_remove = torch.tensor([c == "Remove" for c in cats], device=device)

        # ObjectLocalizer on RAW patches (with v7's object_core multi-node)
        loc_nodes_obj = self._encode_object_for_localizer(graphs)
        with torch.no_grad():
            logits = self.object_localizer(
                patch_tokens.float(),
                loc_nodes_obj["node_features"].float(),
                loc_nodes_obj["node_mask"],
            )
            M_obj_raw = aggregate_mask(logits)
        self._cached_localizer_logits = logits

        # Add 类强制 M_obj=1 (Add 的物体不在 ref, mask 无意义)
        if is_add.any():
            M_obj_raw = M_obj_raw.clone()
            M_obj_raw[is_add] = 1.0

        # ActionMaskHead (或 ablated 全 1)
        if self.ablate_no_action_mask:
            M_act = torch.ones(B, N, device=device, dtype=patch_tokens.dtype)
        else:
            M_obj_for_head = M_obj_raw
            M_act = self.action_mask_head(
                patches=patch_tokens, action_feat=action_feat,
                M_obj=M_obj_for_head, obj_feat=obj_feat,
            )

        # ⚠ v7: NO M_loc.  put None so anything downstream that tries to
        # use it fails loudly instead of silently using a noisy mask.
        return {
            "patch_tokens": patch_tokens,
            "M_act": M_act,
            "M_obj_ref": M_obj_raw,
            "M_loc": None,                  # ← v7 不算
            "obj_feat": obj_feat,
            "object_feat": object_feat,
            "loc_feat": loc_feat,
            "action_feat": action_feat,
            "action_labels": cats,
            "is_add": is_add,
            "is_remove": is_remove,
        }

    # ── Override 3: SemanticInjector 改成全图均匀 (不用 M_loc) ───────────
    def _compose_patches(
        self,
        ctx: Dict[str, torch.Tensor],
        M_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """v7: patches_final = patches × M_act + α · obj_proto (broadcast 全图).

        与 v4/v5/v6 不同, 注入项不再被 M_loc 加权 (M_loc 在 v7 中根本不算);
        改成把 obj_proto 当成一个"全图位置无关"的语义偏置加到每个 patch 上.
        """
        patches = ctx["patch_tokens"]
        M_act = M_token if M_token is not None else ctx["M_act"]

        r = self.aux_ramp.to(M_act.dtype)
        M_eff = r * M_act + (1.0 - r)

        patches_final = patches * M_eff.unsqueeze(-1).to(patches.dtype)

        if not self.ablate_no_semantic_inject:
            # v7 改动: 全图均匀注入 (不再用 M_loc).
            # obj_proto = SemanticInjector.proto_mlp(object_feat)  [B, vit_dim]
            obj_feat_for_proto = ctx["object_feat"].to(
                self.semantic_injector.proto_mlp[-1].weight.dtype
            )
            obj_proto = self.semantic_injector.proto_mlp(obj_feat_for_proto)
            # 不限制 Add/Move only — 全图均匀注入对 Remove 类的"被删物体"
            # 实际上是温和的负样本提示, 不会破坏检索. 想限制可以加 enable.
            enable = (~ctx["is_remove"]).to(patches.dtype).view(-1, 1, 1)
            alpha = self.semantic_injector.alpha_inject.to(patches.dtype)
            inject = alpha * enable * obj_proto.unsqueeze(1)   # [B, 1, vit_dim] broadcast
            patches_final = patches_final + r * inject.to(patches.dtype)

        return patches_final

    # ── Override 4: _build_query 里调 ActionRouter 不传 M_loc ─────────────
    def _build_query(
        self,
        image_embeds_ln: torch.Tensor,
        captions: List[str],
        graphs: Sequence[Any],
        target_patches: Any = None,
    ) -> Dict[str, torch.Tensor]:
        # Main path (raw patches) - 跟 v6 一致
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

        ctx = self._build_action_inputs(image_embeds_ln, graphs)
        patches_action = self._compose_patches(ctx, M_token=self._external_M_token)

        # obj_proto for AddTransitionV7 (BERT(object) -> ViT space)
        obj_proto = self.semantic_injector.proto_mlp(
            ctx["object_feat"].to(self.semantic_injector.proto_mlp[-1].weight.dtype)
        )

        # v7 ActionRouter 不再接收 M_loc
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

        scalar_input = torch.cat([text_embed, ref_embed, Fq_fused], dim=-1)
        dynamic_s = self.dynamic_scalar(scalar_input)
        beta = torch.clamp(self.beta, min=0.0, max=self.beta_max) * self.aux_ramp
        gamma = self.gamma * self.aux_ramp
        if self.ablate_no_trans:
            gamma = gamma * 0.0
        gamma_eff = gamma if self.f_trans_via_gamma else torch.zeros_like(gamma)

        # blend dropout 跟 v6 一致
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
