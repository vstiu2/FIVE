"""Graph-guided local token selector over ViT patch tokens."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class GraphGuidedLocalTokenSelector(nn.Module):
    """Select/weight patch tokens conditioned on graph node embeddings.

    For each sample, the selector computes per-node token scores, takes the
    max over valid nodes to get a single token importance map, then produces
    a soft-attention-weighted local feature.
    """

    def __init__(
        self,
        vit_embed_dim: int,       # ViT output dim (e.g. 1408 for EVA-ViT-G)
        node_embed_dim: int,      # BERT hidden size (768)
        hidden_dim: int = 512,
        top_k: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.top_k = top_k
        # Project ViT tokens and node features into a shared hidden space
        self.token_proj = nn.Linear(vit_embed_dim, hidden_dim)
        self.node_proj = nn.Linear(node_embed_dim, hidden_dim)
        self.score_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Project the weighted local feature to a fixed output dim
        self.out_proj = nn.Linear(vit_embed_dim, vit_embed_dim)

    def compute_node_token_scores(
        self,
        patch_tokens: torch.Tensor,   # [B, N, vit_dim]
        node_features: torch.Tensor,  # [B, M, node_dim]
        node_mask: torch.Tensor,      # [B, M] bool
    ) -> torch.Tensor:                # [B, M, N]
        token_ctx = self.token_proj(patch_tokens)   # [B, N, H]
        node_ctx = self.node_proj(node_features)    # [B, M, H]
        B, M, H = node_ctx.shape
        N = token_ctx.shape[1]
        pair_token = token_ctx.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, N, H]
        pair_node = node_ctx.unsqueeze(2).expand(-1, -1, N, -1)    # [B, M, N, H]
        pair = torch.cat([pair_token, pair_node, pair_token * pair_node], dim=-1)
        scores = self.score_mlp(pair).squeeze(-1)  # [B, M, N]
        scores = scores.masked_fill(~node_mask.unsqueeze(-1), -1e4)
        return scores

    def forward(
        self,
        patch_tokens: torch.Tensor,   # [B, N, vit_dim]
        node_features: torch.Tensor,  # [B, M, node_dim]
        node_mask: torch.Tensor,      # [B, M]
        token_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        node_token_scores = self.compute_node_token_scores(
            patch_tokens, node_features, node_mask
        )
        token_scores = node_token_scores.max(dim=1).values  # [B, N]

        # Soft region mask for aggregation (sigmoid, not softmax, to avoid
        # over-concentration on a single peak patch)
        region_weights = torch.sigmoid(token_scores)
        region_weights = region_weights / region_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        local_feature = (patch_tokens * region_weights.unsqueeze(-1)).sum(dim=1)  # [B, vit_dim]
        local_feature = self.out_proj(local_feature)

        k = min(self.top_k, patch_tokens.shape[1])
        _, selected_idx = torch.topk(token_scores, k=k, dim=1)

        return {
            "local_feature": local_feature,
            "token_scores": token_scores,
            "node_token_scores": node_token_scores,
            "region_weights": region_weights,
            "selected_idx": selected_idx,
            "score_maps": token_scores.view(token_scores.shape[0], token_hw[0], token_hw[1]),
            "token_hw": token_hw,
        }
