"""Token-level fusion modules.

TokenFusionBlock: z_rm (1 token) cross-attends to [Fg, Fl, Ff] (3 tokens).
TgtAwareAuxHead:  training-only head, q_tokens cross-attend to tgt_tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenFusionBlock(nn.Module):
    """Cross-attention fusion: z_rm query attends to visual condition tokens.

    z_rm  : [B, D]  — single vector from TME encode_fusion
    Fg/Fl/Ff : each [B, D]

    They are treated as a sequence of 3 condition tokens [B, 3, D].
    z_rm is the single query token [B, 1, D].
    Output: [B, D] fused query vector.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        z_rm: torch.Tensor,                       # [B, D]
        Fg: torch.Tensor | None,                  # [B, D] or [B, K, D] or None (v10+)
        Fl: torch.Tensor | None,                  # [B, D] or [B, K, D] or None
        Ff: torch.Tensor | None,                  # [B, D] or [B, K, D] or None
        F_trans: torch.Tensor | None = None,      # [B, D] or [B, K, D] or None
    ) -> torch.Tensor:
        """Dual-mode: each K/V input can be:
          * a single token [B, D] (v2-v7),
          * a multi-token sequence [B, K, D] (v8+ Fl/Ff),
          * or None (v10+ drops Fg / drops F_trans).

        Internal stack concatenates available K/V along the K/V dimension,
        so attention sees any number of tokens (>=1).
        """
        dtype = z_rm.dtype

        def _to_3d(x: torch.Tensor) -> torch.Tensor:
            # [B, D] → [B, 1, D]; [B, K, D] passes through
            return x.unsqueeze(1) if x.dim() == 2 else x

        kv_parts = []
        for x in (Fg, Fl, Ff, F_trans):
            if x is not None:
                kv_parts.append(_to_3d(x))
        if not kv_parts:
            raise ValueError("token_fusion needs at least one non-None K/V input")
        kv = torch.cat(kv_parts, dim=1).to(dtype)              # [B, K_total, D]
        q = z_rm.unsqueeze(1).to(dtype)                        # [B, 1, D]

        q_n = self.q_norm(q)
        kv_n = self.kv_norm(kv)
        attn_out, _ = self.cross_attn(q_n, kv_n, kv_n)   # [B, 1, D]
        q = q + attn_out
        q = q + self.ffn(q)                                # FFN residual
        out = self.out_norm(q).squeeze(1)                  # [B, D]
        return out


class TgtAwareAuxHead(nn.Module):
    """Training-only auxiliary head.

    q_tokens  : [B, 32, H]  — Q-Former output tokens (query side)
    tgt_tokens: [B, 32, H]  — Q-Former output tokens (positive target)

    The head cross-attends q_tokens <- tgt_tokens, pools the result,
    projects to embed_dim, and returns a contrastive-ready vector.
    """

    def __init__(self, qformer_hidden: int, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=qformer_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.q_norm = nn.LayerNorm(qformer_hidden)
        self.kv_norm = nn.LayerNorm(qformer_hidden)
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(qformer_hidden),
            nn.Linear(qformer_hidden, embed_dim),
        )

    def forward(
        self,
        q_tokens: torch.Tensor,    # [B, 32, H]
        tgt_tokens: torch.Tensor,  # [B, 32, H]  — detached outside
    ) -> torch.Tensor:             # [B, embed_dim]
        q_n = self.q_norm(q_tokens)
        kv_n = self.kv_norm(tgt_tokens)
        guided, _ = self.cross_attn(q_n, kv_n, kv_n)  # [B, 32, H]
        guided = q_tokens + guided                      # residual
        pooled = guided.mean(dim=1)                     # [B, H]
        return F.normalize(self.pool_proj(pooled), dim=-1)
