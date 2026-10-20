"""FFT branch: frequency-domain feature over graph-guided local region."""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn


def _parse_band_edges(value: str | Sequence[float]) -> list[float]:
    if isinstance(value, str):
        edges = [float(x.strip()) for x in value.split(",") if x.strip()]
    else:
        edges = list(value)
    assert len(edges) >= 2, "Need at least 2 band edges."
    return edges


class FixedBoundaryFFTBranch(nn.Module):
    """Radial-band FFT over graph-guided local region.

    Band boundaries are fixed; per-band weights are learnable.
    Input: ViT patch tokens + token_scores from the selector.
    Output: frequency feature vector in ViT embed space.
    """

    def __init__(
        self,
        embed_dim: int,
        band_edges: str | Sequence[float] = "0,0.15,0.35,0.6,1.01",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.band_edges = _parse_band_edges(band_edges)
        num_bands = len(self.band_edges) - 1
        prior = torch.tensor([0.65, 1.0, 1.2, 0.85, 0.55], dtype=torch.float32)
        init = prior[:num_bands] if num_bands <= prior.numel() else torch.cat(
            [prior, torch.ones(num_bands - prior.numel())]
        )
        self.band_logits = nn.Parameter(init.log())
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _band_masks(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        yy = torch.linspace(-1.0, 1.0, h, device=device)
        xx = torch.linspace(-1.0, 1.0, w, device=device)
        try:
            gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        except TypeError:
            gy, gx = torch.meshgrid(yy, xx)
        r = (gx.square() + gy.square()).sqrt()
        r = r / r.max().clamp_min(1e-6)
        masks = []
        for i, (lo, hi) in enumerate(zip(self.band_edges[:-1], self.band_edges[1:])):
            last = i == len(self.band_edges) - 2
            masks.append(((r >= lo) & (r <= hi if last else r < hi)).float())
        return torch.stack(masks, dim=0)  # [K, H, W]

    def forward(
        self,
        patch_tokens: torch.Tensor,   # [B, N, embed_dim]
        token_scores: torch.Tensor,   # [B, N]  expected to already be in (0, 1)
        token_hw: Tuple[int, int],
        return_per_band: bool = False,   # v8: return per-band features [B, K, D]
    ) -> torch.Tensor:
        """Forward with optional per-band output.

        * return_per_band=False (default, v2-v7 behavior): collapse the K
          frequency bands via learned softmax weights into a single [B, D]
          feature vector.
        * return_per_band=True (v8): skip the softmax aggregation and
          return each band as its own projected token [B, K, D]. Lets
          downstream token_fusion attend to individual bands.
        """
        B, N, D = patch_tokens.shape
        H, W = token_hw
        assert N == H * W

        # Caller passes a probability-style mask in (0,1); just renormalize.
        local_mask = token_scores.float().clamp(min=0.0)
        local_mask = local_mask / local_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        local_mask = local_mask.view(B, 1, H, W)

        token_grid = patch_tokens.float().transpose(1, 2).reshape(B, D, H, W)
        local_grid = token_grid * local_mask  # weight by region importance

        spectrum = torch.fft.fftshift(
            torch.fft.fft2(local_grid, dim=(-2, -1), norm="ortho"), dim=(-2, -1)
        )
        magnitude = torch.log1p(spectrum.abs())  # [B, D, H, W]

        masks = self._band_masks(H, W, patch_tokens.device).to(magnitude.dtype)  # [K, H, W]
        band_mass = masks.flatten(1).sum(dim=1).clamp_min(1.0)  # [K]
        band_feats = (magnitude.unsqueeze(1) * masks.view(1, -1, 1, H, W)).sum(dim=(-2, -1))
        band_feats = band_feats / band_mass.view(1, -1, 1)       # [B, K, D]

        proj_dtype = next(self.proj.parameters()).dtype

        if return_per_band:
            # Project each band token independently. self.proj is a
            # Linear(D, embed_dim), so it operates per-token.
            B_, K, D_ = band_feats.shape
            band_proj = self.proj(band_feats.to(proj_dtype).view(B_ * K, D_))
            return band_proj.view(B_, K, -1).to(patch_tokens.dtype)  # [B, K, embed_dim]

        weights = torch.softmax(self.band_logits.float(), dim=0).to(band_feats.dtype)
        freq_feat = (band_feats * weights.view(1, -1, 1)).sum(dim=1)  # [B, D]
        return self.proj(freq_feat.to(proj_dtype)).to(patch_tokens.dtype)
