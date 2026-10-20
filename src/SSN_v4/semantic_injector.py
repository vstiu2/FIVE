"""SemanticInjector: inject caption-side object semantics into the patch
field, at the location indicated by ``result_location``.

The unified edit equation for v4 is:

    patches_final = patches × M_act + α_inject · M_loc · obj_proto

This module produces the second term:

    inject = M_loc[..., None] * obj_proto[..., None, :]

shape: [B, N, vit_dim]

where
  * ``M_loc``     [B, N]      — soft localization of the result_location
                                node (frozen ObjectLocalizer);
  * ``obj_proto`` [B, vit_dim]— BERT(object) projected into ViT space.

Why this matters
----------------
For Add edits the object is NOT in the reference image, so multiplying ref
patches by a mask alone (M_act → 1) is not enough — the query needs the
*new* object's semantics somewhere. We materialize them at the location
the caption asks for, then let Q-Former merge them with the rest of the
ref via its cross-attention.

Design choices
--------------
* obj_proto goes through ``LayerNorm → Linear → GELU → Linear`` to
  re-project BERT-768 into ViT-1408 space.
* The last linear is initialized near-zero so at the start of training the
  injection is effectively off (training has a chance to learn whether,
  where and how much to inject).
* ``alpha_inject`` is parameterized as ``alpha_max · sigmoid(raw)`` — a
  bounded, never-zero-gradient scalar in (0, alpha_max).
* Hard gating: callers pass an ``enable_mask`` [B] boolean per sample so
  Remove samples receive α=0 (no injection — they're not adding anything).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticInjector(nn.Module):
    """BERT(object) → obj_proto in ViT space; combine with M_loc."""

    def __init__(
        self,
        node_dim: int = 768,
        vit_dim: int = 1408,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        alpha_init: float = 0.05,
        alpha_max: float = 0.5,
    ) -> None:
        super().__init__()

        self.proto_mlp = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vit_dim),
        )
        # Near-zero last layer so the injection starts off and training
        # can decide where/how much to inject without first having to
        # un-learn random noise.
        with torch.no_grad():
            self.proto_mlp[-1].weight.mul_(0.01)
            self.proto_mlp[-1].bias.zero_()

        # alpha_inject = alpha_max * sigmoid(raw); never zero-grad.
        a0 = min(max(alpha_init, 1e-4), alpha_max - 1e-4)
        self.alpha_max = float(alpha_max)
        self.raw_alpha = nn.Parameter(
            torch.tensor(math.log(a0 / (alpha_max - a0)))
        )

    @property
    def alpha_inject(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.raw_alpha)

    def forward(
        self,
        obj_feat: torch.Tensor,        # [B, node_dim]  — BERT(object) CLS
        M_loc: torch.Tensor,           # [B, N]         — ObjectLocalizer mask
        enable: Optional[torch.Tensor] = None,   # [B] bool, default all True
    ) -> torch.Tensor:                 # [B, N, vit_dim]
        """Return the additive injection term (already multiplied by α_inject).

        Caller is expected to ADD this to the (M_act-gated) patches:
            patches_final = patches * M_act[..., None] + injector(...)
        """
        pdtype = self.proto_mlp[-1].weight.dtype
        obj_feat = obj_feat.to(pdtype)
        M_loc = M_loc.to(pdtype)

        obj_proto = self.proto_mlp(obj_feat)            # [B, vit_dim]
        # [B, N, 1] * [B, 1, vit_dim] → [B, N, vit_dim]
        inject = M_loc.unsqueeze(-1) * obj_proto.unsqueeze(1)
        # Scale by α_inject and (optionally) gate by per-sample enable flag.
        alpha = self.alpha_inject.to(pdtype)
        if enable is not None:
            gate = enable.to(pdtype).view(-1, 1, 1)     # [B, 1, 1]
            alpha = alpha * gate
        return alpha * inject
