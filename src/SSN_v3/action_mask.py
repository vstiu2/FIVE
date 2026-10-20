"""ActionMaskHead: produces a per-patch soft mask conditioned on the action
embedding and the (frozen) ObjectLocalizer prior M_obj.

Design notes
------------
* Input layout:
    - patches    : [B, N, vit_dim]   reference ViT patches (post-ln_vision)
    - action_feat: [B, node_dim]     action label encoded by Q-Former BERT
    - M_obj      : [B, N]            object prior from frozen ObjectLocalizer
                                     (uniform 1.0 for "Add" samples)
    - obj_feat   : [B, node_dim]     optional, mean of object node embeddings
                                     (when None, replaced by action_feat).

* Output:
    - M_act : [B, N]  in (0, 1) — soft retention weight for each patch.

* Architecture:
    The head sees, for each patch i:
        x_i  = [patches_i, action_feat_broadcast, obj_feat_broadcast,
                M_obj_i (scalar)]
    runs a 2-layer MLP per token, and ends with sigmoid. The M_obj scalar
    is injected as an extra feature so the head can "respect" or "override"
    the grounding prior — different actions (Remove vs Add) want different
    behaviors near the object region.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionMaskHead(nn.Module):
    """Per-token mask head conditioned on (patch, action, M_obj, obj)."""

    def __init__(
        self,
        vit_dim: int = 1408,
        node_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        # bias_init shifts the pre-sigmoid logit. We want M_act ≈ 1 at init
        # (preserve baseline behavior before training kicks in), so we add
        # a positive bias to the final linear.
        bias_init: float = 2.0,
    ) -> None:
        super().__init__()
        in_dim = vit_dim + node_dim + node_dim + 1  # patch + action + obj + M_obj scalar

        self.norm_patch = nn.LayerNorm(vit_dim)
        self.norm_action = nn.LayerNorm(node_dim)
        self.norm_obj = nn.LayerNorm(node_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Initialize the final linear to output bias_init → sigmoid ≈ 0.88,
        # so M_act starts close to 1.0. Training pulls it down where needed.
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.01)   # near-zero contribution at init
            self.mlp[-1].bias.fill_(bias_init)

    def forward(
        self,
        patches: torch.Tensor,        # [B, N, vit_dim]
        action_feat: torch.Tensor,    # [B, node_dim]
        M_obj: torch.Tensor,          # [B, N]
        obj_feat: torch.Tensor | None = None,  # [B, node_dim] or None
    ) -> torch.Tensor:                # [B, N] in (0, 1)
        B, N, _ = patches.shape

        pdtype = self.mlp[0].weight.dtype
        patches = patches.to(pdtype)
        action_feat = action_feat.to(pdtype)
        M_obj = M_obj.to(pdtype)
        if obj_feat is None:
            obj_feat = action_feat
        else:
            obj_feat = obj_feat.to(pdtype)

        p = self.norm_patch(patches)                          # [B, N, vit_dim]
        a = self.norm_action(action_feat).unsqueeze(1).expand(B, N, -1)
        o = self.norm_obj(obj_feat).unsqueeze(1).expand(B, N, -1)
        m = M_obj.unsqueeze(-1)                               # [B, N, 1]

        joint = torch.cat([p, a, o, m], dim=-1)               # [B, N, in_dim]
        logits = self.mlp(joint).squeeze(-1)                  # [B, N]
        return torch.sigmoid(logits)
