"""Object localizer: graph-grounding-trained patch scorer (EVA-ViT-G + BERT space).

Trained with real grounding masks (BCE + Dice) in train_selector.py, then frozen
and used inside SSN-BLIP2 to produce an accurate object mask M_obj for the
action-transition branch (replacing the from-scratch scorers that failed to learn
localization from the weak contrastive signal).

Architecture mirrors the original CLIP selector's scoring core
(token_proj / node_proj / score_mlp over [tok, node, tok*node]), with dims adapted
to EVA-ViT-G patches (1408) and Q-Former BERT node features (768).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ObjectLocalizer(nn.Module):
    """Score each ViT patch against each graph node → per-node patch logits."""

    def __init__(self, vit_dim: int = 1408, node_dim: int = 768,
                 hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.token_proj = nn.Linear(vit_dim, hidden_dim)
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.score_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,   # [B, N, vit_dim]
        node_features: torch.Tensor,  # [B, M, node_dim]
        node_mask: torch.Tensor,      # [B, M]  bool, True = valid node
    ) -> torch.Tensor:                # [B, M, N]  per-node patch logits
        tok = self.token_proj(patch_tokens)               # [B, N, H]
        nod = self.node_proj(node_features)                # [B, M, H]
        B, M, H = nod.shape
        N = tok.shape[1]
        pair_tok = tok.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, N, H]
        pair_nod = nod.unsqueeze(2).expand(-1, -1, N, -1)  # [B, M, N, H]
        pair = torch.cat([pair_tok, pair_nod, pair_tok * pair_nod], dim=-1)
        scores = self.score_mlp(pair).squeeze(-1)          # [B, M, N]  logits
        scores = scores.masked_fill(~node_mask.unsqueeze(-1), -1e4)
        return scores


def aggregate_mask(node_token_scores: torch.Tensor) -> torch.Tensor:
    """[B, M, N] logits → [B, N] object mask in (0,1) via max-over-nodes + sigmoid.

    Invalid nodes are already masked to -1e4 by ObjectLocalizer, so the max picks
    the most-activating valid node per patch.
    """
    token_scores = node_token_scores.max(dim=1).values     # [B, N]
    return torch.sigmoid(token_scores)
