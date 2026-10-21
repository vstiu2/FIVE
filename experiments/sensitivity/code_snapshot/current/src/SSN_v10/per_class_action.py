"""PerClassActionBranch: per-class mask + top-K visual aggregation.

替代 v8 的 ActionRouter + F_trans 抽象路径. 输入是 v8 的 patches_action
(已经被共享 M_act 过滤过), 输出 [B, K=8, embed_dim] 的多 token 视觉
特征, 直接给 token_fusion 当 K/V.

设计原则
--------
* **共享 backbone**: 所有 4 类样本都过同一个 MLP 学通用视觉/动作理解.
  缓解 Move/Attr 数据稀疏问题 (Move 类 8%, Attr 类 13%).
* **末层 per-class**: 4 个 Linear head, 每类一个, 学 "我这类该把高分
  给哪种 patch" 的差异化决策.
* **top-K aggregation**: 不用 mean pool, 而是取 mask 最高分的 K 个 patch,
  每个 patch 作为独立 token. 保留细粒度空间信息, 让 token_fusion 的
  attention 区分各个 patch 的贡献.
* **mask 加权**: top-K patch 在 gather 后乘以对应 mask 分数, 让 mask
  分数有梯度反传到 mask head.

参数量预算
----------
* backbone: (vit_dim + 2*node_dim) → hidden → hidden ≈ 1.2M 参数
* 4 个 per-class head: 4 × (hidden + 1) ≈ 2K 参数
* feat_proj: vit_dim → embed_dim ≈ 360K 参数
* 总: ~1.6M 参数 (跟 v8 ActionRouter ~3M 比, 反而更小)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# 4 类 action 映射 (跟 v6/v7/v8 一致)
ACTION_TO_CATEGORY_V10: Dict[str, str] = {
    "remove": "Remove", "replace": "Remove",
    "add": "Add", "attach": "Add", "surround": "Add", "fill": "Add",
    "complete": "Add", "cast": "Add", "pack": "Add", "place": "Add",
    "put": "Add", "reveal": "Add",
    "add two more piers": "Add", "add three more houses": "Add",
    "move": "Move", "shift": "Move", "swap": "Move", "rearrange": "Move",
    "center": "Move",
    "rotate and move": "Move", "rotate and shift": "Move",
    "rotate": "Attr", "rotate clockwise": "Attr",
    "rotate slightly clockwise": "Attr",
    "rotate clockwise into diagonal position": "Attr",
    "turn": "Attr",
    "enlarge": "Attr", "reduce": "Attr", "increase": "Attr", "shrink": "Attr",
    "zoom in": "Attr", "zoom out": "Attr", "zoom": "Attr",
    "expand": "Attr", "widen": "Attr", "extend": "Attr", "spread": "Attr",
    "make smaller": "Attr", "make larger": "Attr",
    "make much larger and more visible": "Attr",
    "change": "Attr", "change color": "Attr", "repaint": "Attr",
    "recolor": "Attr", "paint": "Attr", "darken": "Attr", "brighten": "Attr",
    "make": "Attr", "make darker": "Attr", "make lighter": "Attr",
    "make much darker": "Attr",
    "make denser and greener": "Attr",
    "make darker and more cracked": "Attr",
}

CATEGORIES_V10 = ("Remove", "Add", "Move", "Attr")


def action_to_category_v10(action: Optional[str]) -> str:
    if not isinstance(action, str):
        return "Attr"
    return ACTION_TO_CATEGORY_V10.get(action.strip().lower(), "Attr")


class PerClassActionBranch(nn.Module):
    """共享 backbone + 末层 per-class + top-K 视觉聚合."""

    def __init__(
        self,
        vit_dim: int = 1408,
        node_dim: int = 768,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        top_k: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vit_dim = vit_dim
        self.embed_dim = embed_dim
        self.top_k = top_k

        # 共享 backbone: 对每个 patch + 上下文 (obj_feat + action_feat) 编码
        in_dim = vit_dim + node_dim * 2     # patch + obj + action
        self.backbone = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # 末层 per-class: 4 个 Linear (hidden → 1) 各自输出 patch 分数
        self.head_remove = nn.Linear(hidden_dim, 1)
        self.head_add = nn.Linear(hidden_dim, 1)
        self.head_move = nn.Linear(hidden_dim, 1)
        self.head_attr = nn.Linear(hidden_dim, 1)

        # 视觉聚合: vit_dim → embed_dim
        self.feat_proj = nn.Linear(vit_dim, embed_dim)

    def _route_head(self, cat: str) -> nn.Linear:
        return {
            "Remove": self.head_remove,
            "Add":    self.head_add,
            "Move":   self.head_move,
            "Attr":   self.head_attr,
        }[cat]

    def forward(
        self,
        patches: torch.Tensor,       # [B, N, vit_dim]
        obj_feat: torch.Tensor,      # [B, node_dim]
        action_feat: torch.Tensor,   # [B, node_dim]
        action_labels: List[str],    # len B, each in CATEGORIES_V10
    ) -> Dict[str, torch.Tensor]:
        """Return F_action [B, K, embed_dim] + M_cls [B, N] for diagnostics."""
        B, N, _ = patches.shape
        device = patches.device

        # Cast inputs to backbone dtype
        pdtype = self.backbone[1].weight.dtype
        patches_f = patches.to(pdtype)
        obj_feat_f = obj_feat.to(pdtype)
        action_feat_f = action_feat.to(pdtype)

        # Broadcast上下文到每个 patch
        obj_ctx = obj_feat_f.unsqueeze(1).expand(B, N, -1)         # [B, N, node]
        action_ctx = action_feat_f.unsqueeze(1).expand(B, N, -1)   # [B, N, node]
        x = torch.cat([patches_f, obj_ctx, action_ctx], dim=-1)    # [B, N, vit+2*node]

        # 共享 backbone
        feat = self.backbone(x)                                     # [B, N, hidden]

        # Per-class head: hard route
        # 性能优化: 按类分桶, 同类样本批量过 head, 避免逐样本循环.
        scores = torch.zeros(B, N, device=device, dtype=feat.dtype)
        cat_to_idx: Dict[str, List[int]] = {c: [] for c in CATEGORIES_V10}
        for i, cat in enumerate(action_labels):
            cat_to_idx[cat if cat in cat_to_idx else "Attr"].append(i)

        for cat, idx_list in cat_to_idx.items():
            if not idx_list:
                continue
            idx = torch.tensor(idx_list, device=device, dtype=torch.long)
            head = self._route_head(cat)
            sub_scores = head(feat[idx]).squeeze(-1)                # [|idx|, N]
            scores[idx] = sub_scores

        M_cls = torch.sigmoid(scores)                               # [B, N] in (0,1)

        # Top-K 聚合: 取 mask 最高的 K 个 patch
        K = min(self.top_k, N)
        top_vals, top_idx = M_cls.topk(K, dim=1)                    # [B, K]

        # Gather 高分 patch 的原始视觉向量 (不是 backbone feat, 是原 patches)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, self.vit_dim)  # [B, K, vit_dim]
        top_patches = patches.gather(1, gather_idx)                 # [B, K, vit_dim]

        # 用 mask 分数加权 patch 特征 (让 mask 分数有梯度反传)
        # 这里 top_vals 是 mask sigmoid 后的分数 (0~1), 乘进去自然加权.
        weighted = top_patches * top_vals.unsqueeze(-1).to(top_patches.dtype)

        # 投影到 embed_dim 并 normalize
        F_action = self.feat_proj(weighted)                         # [B, K, embed_dim]
        F_action = F.normalize(F_action, dim=-1)

        return {
            "F_action": F_action,                                   # [B, K, embed_dim]
            "M_cls": M_cls,                                         # [B, N]
            "top_idx": top_idx,                                     # [B, K]
            "top_vals": top_vals,                                   # [B, K]
        }
