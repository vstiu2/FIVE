"""Action-specific transition modules (selector-driven version).

Object localization is provided EXTERNALLY by a grounding-trained ObjectLocalizer
(see `model/object_localizer.py`). Each transition module is now a thin
projector: it takes an object mask M_obj (already accurate) and produces a
per-action transition feature F_trans for final_query.

Routing: hard, by graph.action → Remove / Add / Move (see ACTION_TO_CATEGORY).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Action label → category mapping (covers all 53 observed actions; default Move)
# ─────────────────────────────────────────────────────────────────────────────

ACTION_TO_CATEGORY: Dict[str, str] = {
    # ── Remove (4209, 55.3%)  incl. replace: graph.object = OLD object in ref ──
    "remove": "Remove", "replace": "Remove",
    # ── Add (1804, 23.7%)  graph.object = NEW object present in target ──
    "add": "Add",
    "attach": "Add", "surround": "Add", "fill": "Add", "complete": "Add",
    "cast": "Add", "pack": "Add", "place": "Add", "put": "Add", "reveal": "Add",
    "add two more piers": "Add", "add three more houses": "Add",
    # ── Move (1602, 21.0%)  obj identity kept, other change ──
    "move": "Move", "shift": "Move", "swap": "Move", "rearrange": "Move", "center": "Move",
    "rotate": "Move", "rotate clockwise": "Move", "rotate slightly clockwise": "Move",
    "rotate clockwise into diagonal position": "Move", "rotate and move": "Move",
    "rotate and shift": "Move", "turn": "Move",
    "enlarge": "Move", "reduce": "Move", "increase": "Move", "shrink": "Move",
    "zoom in": "Move", "zoom out": "Move", "zoom": "Move",
    "expand": "Move", "widen": "Move", "extend": "Move", "spread": "Move",
    "make smaller": "Move", "make larger": "Move", "make much larger and more visible": "Move",
    "change": "Move", "change color": "Move", "repaint": "Move", "recolor": "Move",
    "paint": "Move", "darken": "Move", "brighten": "Move",
    "make": "Move", "make darker": "Move", "make lighter": "Move",
    "make much darker": "Move", "make denser and greener": "Move",
    "make darker and more cracked": "Move",
}

CATEGORIES = ("Remove", "Add", "Move")


def action_to_category(action: Optional[str]) -> str:
    if not isinstance(action, str):
        return "Move"
    return ACTION_TO_CATEGORY.get(action.strip().lower(), "Move")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _masked_mean(patches: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """patches [B,N,D], mask [B,N] → [B,D] weighted mean."""
    w = mask.unsqueeze(-1)
    return (patches * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)


def _trans_mlp(in_dim: int, hidden_dim: int, proj_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim * 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim * 2, proj_dim),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Transition modules — all take an externally-provided M_obj
# ─────────────────────────────────────────────────────────────────────────────

class RemoveTransition(nn.Module):
    """obj region (already located) → transition feature for removal."""

    def __init__(self, vit_dim: int, node_dim: int, hidden_dim: int,
                 proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.region_proj = nn.Linear(vit_dim, node_dim)
        # region + obj + action  (all node_dim)
        self.trans_mlp = _trans_mlp(node_dim * 3, hidden_dim, proj_dim, dropout)

    def forward(self, patches: torch.Tensor, M_obj: torch.Tensor,
                obj_feat: torch.Tensor, action_feat: torch.Tensor,
                **kw) -> Dict[str, torch.Tensor]:
        region = self.region_proj(_masked_mean(patches, M_obj))         # [B, node_dim]
        F_trans = self.trans_mlp(
            torch.cat([region, obj_feat, action_feat], dim=-1))
        return {"F_trans": F_trans, "F_obj_region": region}


class AddTransition(nn.Module):
    """obj region + dst location + action → transition feature for insertion."""

    def __init__(self, vit_dim: int, node_dim: int, hidden_dim: int,
                 proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.region_proj = nn.Linear(vit_dim, node_dim)
        # region + obj + loc + action  (all node_dim)
        self.trans_mlp = _trans_mlp(node_dim * 4, hidden_dim, proj_dim, dropout)

    def forward(self, patches: torch.Tensor, M_obj: torch.Tensor,
                obj_feat: torch.Tensor, loc_feat: torch.Tensor,
                action_feat: torch.Tensor, **kw) -> Dict[str, torch.Tensor]:
        region = self.region_proj(_masked_mean(patches, M_obj))
        F_trans = self.trans_mlp(
            torch.cat([region, obj_feat, loc_feat, action_feat], dim=-1))
        return {"F_trans": F_trans, "F_obj_region": region}


class MoveTransition(nn.Module):
    """obj region (located by selector on ref/tgt) + dst loc + action → transition.

    The selector already finds the object wherever it sits in the image (src in ref,
    dst in target), so we no longer need separate src/dst scorers.
    """

    def __init__(self, vit_dim: int, node_dim: int, hidden_dim: int,
                 proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.region_proj = nn.Linear(vit_dim, node_dim)
        # region + obj + loc(dst) + action  (all node_dim)
        self.trans_mlp = _trans_mlp(node_dim * 4, hidden_dim, proj_dim, dropout)

    def forward(self, patches: torch.Tensor, M_obj: torch.Tensor,
                obj_feat: torch.Tensor, loc_feat: torch.Tensor,
                action_feat: torch.Tensor, **kw) -> Dict[str, torch.Tensor]:
        region = self.region_proj(_masked_mean(patches, M_obj))
        F_trans = self.trans_mlp(
            torch.cat([region, obj_feat, loc_feat, action_feat], dim=-1))
        return {"F_trans": F_trans, "F_obj_region": region}


# ─────────────────────────────────────────────────────────────────────────────
# Hard router
# ─────────────────────────────────────────────────────────────────────────────

class ActionRouter(nn.Module):
    """Hard-routing dispatcher. Object mask M_obj is provided externally."""

    def __init__(self, vit_dim: int, node_dim: int, hidden_dim: int,
                 proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj_dim = proj_dim
        self.node_dim = node_dim
        self.remove_t = RemoveTransition(vit_dim, node_dim, hidden_dim, proj_dim, dropout)
        self.add_t = AddTransition(vit_dim, node_dim, hidden_dim, proj_dim, dropout)
        self.move_t = MoveTransition(vit_dim, node_dim, hidden_dim, proj_dim, dropout)

    @staticmethod
    def _bucket(action_labels: List[str]) -> Dict[str, List[int]]:
        cat_idx: Dict[str, List[int]] = {c: [] for c in CATEGORIES}
        for i, c in enumerate(action_labels):
            cat_idx[c if c in cat_idx else "Move"].append(i)
        return cat_idx

    def forward(
        self,
        action_labels: List[str],          # len B, each in CATEGORIES
        patches: torch.Tensor,             # [B,N,vit_dim]  reference patches
        M_obj_ref: torch.Tensor,           # [B,N]  obj mask on reference (uniform for Add)
        obj_feat: torch.Tensor,            # [B,node_dim]
        loc_feat: torch.Tensor,            # [B,node_dim]  result_location
        action_feat: torch.Tensor,         # [B,node_dim]
    ) -> Dict[str, torch.Tensor]:
        # Cast inputs to module param dtype (autocast may produce fp16).
        pdtype = self.remove_t.region_proj.weight.dtype
        patches = patches.to(pdtype)
        M_obj_ref = M_obj_ref.to(pdtype)
        obj_feat, loc_feat = obj_feat.to(pdtype), loc_feat.to(pdtype)
        action_feat = action_feat.to(pdtype)

        B = patches.shape[0]
        device, dtype = patches.device, patches.dtype
        F_trans = torch.zeros(B, self.proj_dim, device=device, dtype=dtype)
        cat_idx = self._bucket(action_labels)

        def put(buf, idx, src):
            buf[idx] = src.to(buf.dtype)

        for cat_name, module in [("Remove", self.remove_t),
                                 ("Add",    self.add_t),
                                 ("Move",   self.move_t)]:
            if not cat_idx[cat_name]:
                continue
            idx = torch.tensor(cat_idx[cat_name], device=device, dtype=torch.long)
            out = module(
                patches[idx], M_obj_ref[idx],
                obj_feat=obj_feat[idx], loc_feat=loc_feat[idx],
                action_feat=action_feat[idx],
            )
            put(F_trans, idx, out["F_trans"])

        return {"F_trans": F_trans, "cat_idx": cat_idx}
