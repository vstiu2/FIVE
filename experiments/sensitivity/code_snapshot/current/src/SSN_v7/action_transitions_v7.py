"""ActionRouter v7: drop M_loc, AddTransition 4-input.

Differences from action_transitions_v6.py
------------------------------------------
* AddTransitionV7: 4-element input  [obj_proto, obj_feat, loc_feat, action_feat]
  (no region; the previous region(M_loc) was noise and Add's M_obj = uniform
  duplicates Fg anyway).
* MoveTransitionV7: 4-element input  [region(M_obj), obj_feat, loc_feat, action_feat]
  (drops region_dst(M_loc); reverts to v5-style single-region pooling on
  source location).
* RemoveTransitionV7 / AttrTransitionV7: unchanged from v6.

Hard routing (4 classes) and trans_mlp tail LayerNorm are inherited from v6.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 v6 的 4-类映射和 helpers
from SSN_v6.action_transitions_v6 import (  # noqa: E402
    CATEGORIES_V6 as CATEGORIES_V7,
    action_to_category_v6 as action_to_category_v7,
    _masked_mean,
    _trans_mlp,
)


class RemoveTransitionV7(nn.Module):
    """Same as RemoveTransitionV6: region(M_obj) + obj_feat + action_feat."""

    def __init__(self, vit_dim: int, node_dim: int, hidden_dim: int,
                 proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.region_proj = nn.Linear(vit_dim, node_dim)
        self.trans_mlp = _trans_mlp(node_dim * 3, hidden_dim, proj_dim, dropout)

    def forward(self, patches: torch.Tensor, M_obj: torch.Tensor,
                obj_feat: torch.Tensor, loc_feat: torch.Tensor,
                action_feat: torch.Tensor,
                obj_proto: Optional[torch.Tensor] = None,
                **kw) -> Dict[str, torch.Tensor]:
        region = self.region_proj(_masked_mean(patches, M_obj))
        F_trans = self.trans_mlp(
            torch.cat([region, obj_feat, action_feat], dim=-1))
        return {"F_trans": F_trans, "F_obj_region": region}


class AddTransitionV7(nn.Module):
    """Add: 4-input [obj_proto, obj_feat, loc_feat, action_feat].

    Compared to v6:
      * No region(M_loc): M_loc was unreliable for free-form location words.
      * No region(M_obj): Add's M_obj is forced uniform → equals Fg → redundant
        with token_fusion's Fg input.
      * Visual info enters F_trans purely via obj_proto (BERT(object) →
        ViT-space projection from the SemanticInjector pipeline).
    """

    def __init__(self, vit_dim: int, node_dim: int, hidden_dim: int,
                 proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proto_proj = nn.Linear(vit_dim, node_dim)
        # obj_proto + obj_feat + loc_feat + action_feat = 4 × node_dim
        self.trans_mlp = _trans_mlp(node_dim * 4, hidden_dim, proj_dim, dropout)

    def forward(self, patches: torch.Tensor, M_obj: torch.Tensor,
                obj_feat: torch.Tensor, loc_feat: torch.Tensor,
                action_feat: torch.Tensor,
                obj_proto: Optional[torch.Tensor] = None,
                **kw) -> Dict[str, torch.Tensor]:
        if obj_proto is None:
            # Fall back to zeros so the module still runs even when
            # SemanticInjector hasn't supplied the projected prototype.
            proto = torch.zeros(patches.shape[0], self.proto_proj.out_features,
                                device=patches.device, dtype=patches.dtype)
        else:
            proto = self.proto_proj(obj_proto.to(patches.dtype))
        F_trans = self.trans_mlp(
            torch.cat([proto, obj_feat, loc_feat, action_feat], dim=-1))
        # Return zero-region for API compatibility (Remove/Move emit
        # F_obj_region; some downstream code may look at it).
        return {
            "F_trans": F_trans,
            "F_obj_region": torch.zeros_like(proto),
        }


class MoveTransitionV7(nn.Module):
    """Move: 4-input [region(M_obj), obj_feat, loc_feat, action_feat].

    Reverts to v5-style single-region. Without M_loc the dst region is
    inaccessible; we settle for keeping the source-region (where the
    object currently is) and trust loc_feat (BERT text embedding) to
    carry the destination semantic.
    """

    def __init__(self, vit_dim: int, node_dim: int, hidden_dim: int,
                 proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.region_proj = nn.Linear(vit_dim, node_dim)
        # region + obj + loc + action = 4 × node_dim
        self.trans_mlp = _trans_mlp(node_dim * 4, hidden_dim, proj_dim, dropout)

    def forward(self, patches: torch.Tensor, M_obj: torch.Tensor,
                obj_feat: torch.Tensor, loc_feat: torch.Tensor,
                action_feat: torch.Tensor,
                obj_proto: Optional[torch.Tensor] = None,
                **kw) -> Dict[str, torch.Tensor]:
        region = self.region_proj(_masked_mean(patches, M_obj))
        F_trans = self.trans_mlp(
            torch.cat([region, obj_feat, loc_feat, action_feat], dim=-1))
        return {"F_trans": F_trans, "F_obj_region": region}


class AttrTransitionV7(nn.Module):
    """Same as AttrTransitionV6: region(M_obj) + obj_feat + action_feat."""

    def __init__(self, vit_dim: int, node_dim: int, hidden_dim: int,
                 proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.region_proj = nn.Linear(vit_dim, node_dim)
        self.trans_mlp = _trans_mlp(node_dim * 3, hidden_dim, proj_dim, dropout)

    def forward(self, patches: torch.Tensor, M_obj: torch.Tensor,
                obj_feat: torch.Tensor, loc_feat: torch.Tensor,
                action_feat: torch.Tensor,
                obj_proto: Optional[torch.Tensor] = None,
                **kw) -> Dict[str, torch.Tensor]:
        region = self.region_proj(_masked_mean(patches, M_obj))
        F_trans = self.trans_mlp(
            torch.cat([region, obj_feat, action_feat], dim=-1))
        return {"F_trans": F_trans, "F_obj_region": region}


class ActionRouterV7(nn.Module):
    """4-way hard-routing dispatcher: Remove / Add / Move / Attr (v7 modules)."""

    def __init__(self, vit_dim: int, node_dim: int, hidden_dim: int,
                 proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj_dim = proj_dim
        self.node_dim = node_dim
        self.remove_t = RemoveTransitionV7(vit_dim, node_dim, hidden_dim, proj_dim, dropout)
        self.add_t = AddTransitionV7(vit_dim, node_dim, hidden_dim, proj_dim, dropout)
        self.move_t = MoveTransitionV7(vit_dim, node_dim, hidden_dim, proj_dim, dropout)
        self.attr_t = AttrTransitionV7(vit_dim, node_dim, hidden_dim, proj_dim, dropout)

    @staticmethod
    def _bucket(action_labels: List[str]) -> Dict[str, List[int]]:
        cat_idx: Dict[str, List[int]] = {c: [] for c in CATEGORIES_V7}
        for i, c in enumerate(action_labels):
            cat_idx[c if c in cat_idx else "Attr"].append(i)
        return cat_idx

    def forward(
        self,
        action_labels: List[str],
        patches: torch.Tensor,
        M_obj_ref: torch.Tensor,
        obj_feat: torch.Tensor,
        loc_feat: torch.Tensor,
        action_feat: torch.Tensor,
        obj_proto: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pdtype = self.remove_t.region_proj.weight.dtype if hasattr(self.remove_t, 'region_proj') \
                 else next(self.parameters()).dtype
        patches = patches.to(pdtype)
        M_obj_ref = M_obj_ref.to(pdtype)
        obj_feat = obj_feat.to(pdtype)
        loc_feat = loc_feat.to(pdtype)
        action_feat = action_feat.to(pdtype)
        if obj_proto is not None:
            obj_proto = obj_proto.to(pdtype)

        B = patches.shape[0]
        device, dtype = patches.device, patches.dtype
        F_trans = torch.zeros(B, self.proj_dim, device=device, dtype=dtype)
        cat_idx = self._bucket(action_labels)

        def put(buf, idx, src):
            buf[idx] = src.to(buf.dtype)

        for cat_name, module in [("Remove", self.remove_t),
                                 ("Add",    self.add_t),
                                 ("Move",   self.move_t),
                                 ("Attr",   self.attr_t)]:
            if not cat_idx[cat_name]:
                continue
            idx = torch.tensor(cat_idx[cat_name], device=device, dtype=torch.long)
            sub_obj_proto = obj_proto[idx] if obj_proto is not None else None
            out = module(
                patches=patches[idx], M_obj=M_obj_ref[idx],
                obj_feat=obj_feat[idx], loc_feat=loc_feat[idx],
                action_feat=action_feat[idx],
                obj_proto=sub_obj_proto,
            )
            put(F_trans, idx, out["F_trans"])

        return {"F_trans": F_trans, "cat_idx": cat_idx}
