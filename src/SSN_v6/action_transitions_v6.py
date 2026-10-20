"""ActionRouter v6: 4 categories + fixed Add/Move region inputs + LayerNorm tail.

Differences from src/model/action_transitions.py (v2-v5 version)
----------------------------------------------------------------

* CATEGORIES = ("Remove", "Add", "Move", "Attr"). "Attr" absorbs all the
  attribute-change actions previously dumped into Move: rotate / resize /
  recolor. This keeps Move pure ("position change") and gives the
  attribute branch its own MLP that sees ~9% of samples (4-5% per old
  sub-category × 3 ≈ 13%; coarse-grained so each batch still sees enough).

* AddTransition takes ``M_loc`` (where the new object should appear) and
  ``obj_proto`` (BERT(object) projected to ViT space, supplied externally
  by the caller — usually SemanticInjector.proto_mlp's output) instead of
  a region pooled over the whole image. This fixes the "Add region = Fg"
  bug.

* MoveTransition takes both ``region_src`` (M_obj-pooled patches) AND
  ``region_dst`` (M_loc-pooled patches), so F_trans can express "from
  source location → destination location" instead of only the source.

* All trans_mlp end with LayerNorm so F_trans output magnitude is bounded.

* Hard routing kept (per-class MLPs); the bottleneck of hard routing is
  partly mitigated by funnelling F_trans into token_fusion's soft
  attention downstream (see ssn_blip2_v6.py).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Action label → category mapping (4 categories, refined from v2-v5)
# ─────────────────────────────────────────────────────────────────────────────

ACTION_TO_CATEGORY_V6: Dict[str, str] = {
    # Remove (~55%)
    "remove": "Remove", "replace": "Remove",

    # Add (~24%)
    "add": "Add",
    "attach": "Add", "surround": "Add", "fill": "Add", "complete": "Add",
    "cast": "Add", "pack": "Add", "place": "Add", "put": "Add", "reveal": "Add",
    "add two more piers": "Add", "add three more houses": "Add",

    # Move: 真正的位置变化 (~8%)
    "move": "Move", "shift": "Move", "swap": "Move", "rearrange": "Move",
    "center": "Move",
    "rotate and move": "Move", "rotate and shift": "Move",

    # Attr (~13%) — 属性变化: 物体本身的旋转/缩放/颜色等
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

CATEGORIES_V6 = ("Remove", "Add", "Move", "Attr")


def action_to_category_v6(action: Optional[str]) -> str:
    if not isinstance(action, str):
        return "Attr"   # safer fallback than Move (won't pollute pure position)
    return ACTION_TO_CATEGORY_V6.get(action.strip().lower(), "Attr")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _masked_mean(patches: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """patches [B,N,D], mask [B,N] → [B,D] weighted mean."""
    w = mask.unsqueeze(-1)
    return (patches * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)


def _trans_mlp(in_dim: int, hidden_dim: int, proj_dim: int, dropout: float) -> nn.Sequential:
    """v6: ends with LayerNorm so F_trans magnitude is bounded."""
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim * 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim * 2, proj_dim),
        nn.LayerNorm(proj_dim),       # ← v6 addition
    )


# ─────────────────────────────────────────────────────────────────────────────
# Transition modules
# ─────────────────────────────────────────────────────────────────────────────

class RemoveTransitionV6(nn.Module):
    """Remove: region(M_obj) + obj_feat + action_feat (same as v5)."""

    def __init__(self, vit_dim: int, node_dim: int, hidden_dim: int,
                 proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.region_proj = nn.Linear(vit_dim, node_dim)
        self.trans_mlp = _trans_mlp(node_dim * 3, hidden_dim, proj_dim, dropout)

    def forward(self, patches: torch.Tensor, M_obj: torch.Tensor,
                obj_feat: torch.Tensor, loc_feat: torch.Tensor,
                action_feat: torch.Tensor, M_loc: torch.Tensor,
                obj_proto: Optional[torch.Tensor] = None,
                **kw) -> Dict[str, torch.Tensor]:
        region = self.region_proj(_masked_mean(patches, M_obj))   # [B, node_dim]
        F_trans = self.trans_mlp(
            torch.cat([region, obj_feat, action_feat], dim=-1))
        return {"F_trans": F_trans, "F_obj_region": region}


class AddTransitionV6(nn.Module):
    """Add: region(M_loc) + obj_proto + loc_feat + action_feat.

    region is pooled at the DESTINATION (M_loc), not over the whole image.
    obj_proto is the BERT(object) projected to ViT space — supplied
    externally (e.g. from SemanticInjector.proto_mlp's first stage), giving
    F_trans access to "what the new object's visual concept looks like".
    """

    def __init__(self, vit_dim: int, node_dim: int, hidden_dim: int,
                 proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.region_proj = nn.Linear(vit_dim, node_dim)
        self.proto_proj = nn.Linear(vit_dim, node_dim)
        # region + proto + obj + loc + action
        self.trans_mlp = _trans_mlp(node_dim * 5, hidden_dim, proj_dim, dropout)

    def forward(self, patches: torch.Tensor, M_obj: torch.Tensor,
                obj_feat: torch.Tensor, loc_feat: torch.Tensor,
                action_feat: torch.Tensor, M_loc: torch.Tensor,
                obj_proto: Optional[torch.Tensor] = None,
                **kw) -> Dict[str, torch.Tensor]:
        # Visual: pool patches at destination location (M_loc).
        region = self.region_proj(_masked_mean(patches, M_loc))
        # Object's visual concept (from BERT projection); fall back to zeros
        # if not supplied so the module still runs (e.g. at smoke-test time).
        if obj_proto is None:
            proto = torch.zeros_like(region)
        else:
            proto = self.proto_proj(obj_proto.to(patches.dtype))
        F_trans = self.trans_mlp(
            torch.cat([region, proto, obj_feat, loc_feat, action_feat], dim=-1))
        return {"F_trans": F_trans, "F_obj_region": region}


class MoveTransitionV6(nn.Module):
    """Move: region_src(M_obj) + region_dst(M_loc) + obj/loc/action_feat.

    Now expresses the source → destination shift explicitly.
    """

    def __init__(self, vit_dim: int, node_dim: int, hidden_dim: int,
                 proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.region_src_proj = nn.Linear(vit_dim, node_dim)
        self.region_dst_proj = nn.Linear(vit_dim, node_dim)
        # region_src + region_dst + obj + loc + action
        self.trans_mlp = _trans_mlp(node_dim * 5, hidden_dim, proj_dim, dropout)

    def forward(self, patches: torch.Tensor, M_obj: torch.Tensor,
                obj_feat: torch.Tensor, loc_feat: torch.Tensor,
                action_feat: torch.Tensor, M_loc: torch.Tensor,
                obj_proto: Optional[torch.Tensor] = None,
                **kw) -> Dict[str, torch.Tensor]:
        region_src = self.region_src_proj(_masked_mean(patches, M_obj))
        region_dst = self.region_dst_proj(_masked_mean(patches, M_loc))
        F_trans = self.trans_mlp(
            torch.cat([region_src, region_dst, obj_feat, loc_feat, action_feat],
                      dim=-1))
        return {"F_trans": F_trans, "F_obj_region": region_src}


class AttrTransitionV6(nn.Module):
    """Attr (rotate / resize / recolor): region(M_obj) + obj/action_feat.

    Object stays in the same location, so we don't need M_loc. action_feat
    (BERT-encoded "rotate clockwise" / "darken" / etc.) carries the
    attribute-specific information.
    """

    def __init__(self, vit_dim: int, node_dim: int, hidden_dim: int,
                 proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.region_proj = nn.Linear(vit_dim, node_dim)
        # region + obj + action
        self.trans_mlp = _trans_mlp(node_dim * 3, hidden_dim, proj_dim, dropout)

    def forward(self, patches: torch.Tensor, M_obj: torch.Tensor,
                obj_feat: torch.Tensor, loc_feat: torch.Tensor,
                action_feat: torch.Tensor, M_loc: torch.Tensor,
                obj_proto: Optional[torch.Tensor] = None,
                **kw) -> Dict[str, torch.Tensor]:
        region = self.region_proj(_masked_mean(patches, M_obj))
        F_trans = self.trans_mlp(
            torch.cat([region, obj_feat, action_feat], dim=-1))
        return {"F_trans": F_trans, "F_obj_region": region}


# ─────────────────────────────────────────────────────────────────────────────
# Hard router v6
# ─────────────────────────────────────────────────────────────────────────────

class ActionRouterV6(nn.Module):
    """4-way hard-routing dispatcher: Remove / Add / Move / Attr."""

    def __init__(self, vit_dim: int, node_dim: int, hidden_dim: int,
                 proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj_dim = proj_dim
        self.node_dim = node_dim
        self.remove_t = RemoveTransitionV6(vit_dim, node_dim, hidden_dim, proj_dim, dropout)
        self.add_t = AddTransitionV6(vit_dim, node_dim, hidden_dim, proj_dim, dropout)
        self.move_t = MoveTransitionV6(vit_dim, node_dim, hidden_dim, proj_dim, dropout)
        self.attr_t = AttrTransitionV6(vit_dim, node_dim, hidden_dim, proj_dim, dropout)

    @staticmethod
    def _bucket(action_labels: List[str]) -> Dict[str, List[int]]:
        cat_idx: Dict[str, List[int]] = {c: [] for c in CATEGORIES_V6}
        for i, c in enumerate(action_labels):
            cat_idx[c if c in cat_idx else "Attr"].append(i)
        return cat_idx

    def forward(
        self,
        action_labels: List[str],            # len B, each in CATEGORIES_V6
        patches: torch.Tensor,               # [B, N, vit_dim]
        M_obj_ref: torch.Tensor,             # [B, N]
        M_loc: torch.Tensor,                 # [B, N]  result_location mask
        obj_feat: torch.Tensor,              # [B, node_dim]
        loc_feat: torch.Tensor,              # [B, node_dim]
        action_feat: torch.Tensor,           # [B, node_dim]
        obj_proto: Optional[torch.Tensor] = None,   # [B, vit_dim] — for Add
    ) -> Dict[str, torch.Tensor]:
        pdtype = self.remove_t.region_proj.weight.dtype
        patches = patches.to(pdtype)
        M_obj_ref = M_obj_ref.to(pdtype)
        M_loc = M_loc.to(pdtype)
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
                action_feat=action_feat[idx], M_loc=M_loc[idx],
                obj_proto=sub_obj_proto,
            )
            put(F_trans, idx, out["F_trans"])

        return {"F_trans": F_trans, "cat_idx": cat_idx}
