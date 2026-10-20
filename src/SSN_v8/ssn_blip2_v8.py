"""SSN-BLIP2-v8 主模型: Fl/Ff 多 token + Fl/Ff 直接监督.

继承 v7, 不动 v2-v7 / baseline.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from models.ssn_blip2 import Model as BaselineModel       # noqa: E402
from SSN_v7.ssn_blip2_v7 import Model as V7Model           # noqa: E402


class Model(V7Model):
    """v8 = v7 + Fl/Ff multi-token + Fl/Ff retrieval supervision."""

    def __init__(self, *args, **kwargs):
        # v8-only kwargs
        self._top_k_fl = int(kwargs.pop("top_k_fl", 4))
        self.lambda_fl = float(kwargs.pop("lambda_fl", 0.1))
        self.lambda_ff = float(kwargs.pop("lambda_ff", 0.1))
        # multi-token toggle (allow ablation back to single-token v7 behavior)
        self._fl_multi = bool(kwargs.pop("fl_multi", True))
        self._ff_multi = bool(kwargs.pop("ff_multi", True))

        super().__init__(*args, **kwargs)

        # Retrieval heads for direct supervision on Fl/Ff.
        # Both pool the (possibly multi-token) feature into a single
        # 256-d retrieval query.
        # Fl_pool_raw is in selector_hidden_dim space (512); we reuse the
        # existing self.local_proj (512→256) before this head.
        # Ff_pool_raw is the FFT branch's already-projected output, which
        # lives in selector_hidden_dim space (512) — we need self.freq_proj
        # (512→256) before this head as well.
        embed_dim = self.embed_dim
        self.fl_retrieval_head = nn.Linear(embed_dim, embed_dim)
        self.ff_retrieval_head = nn.Linear(embed_dim, embed_dim)

    # ── Override _visual_branches: emit multi-token Fl/Ff ───────────────
    def _visual_branches(
        self,
        patch_tokens_ln: torch.Tensor,   # [B, N, 1408]  (already action-composed)
        obj_mask: torch.Tensor,          # [B, N]  M_obj
        obj_node_out: Dict[str, torch.Tensor],
        graphs: Sequence[Any],
    ) -> Dict[str, torch.Tensor]:
        """v8 differences vs v2 baseline:
          * Fl is [B, K, embed_dim] (top-K patches under M_obj projected
            independently) instead of [B, embed_dim] (mean-pool result).
          * Ff is [B, 4, embed_dim] (per-band) instead of [B, embed_dim]
            (softmax-aggregated).
          * Fg is unchanged ([B, embed_dim]).
        Falls back to v7 single-token behavior when fl_multi / ff_multi
        are False (ablation).
        """
        sel_tokens = self.visual_to_selector(patch_tokens_ln)   # [B, N, 512]
        grid_size = int(patch_tokens_ln.shape[1] ** 0.5)
        w = obj_mask.to(sel_tokens.dtype)                       # [B, N]

        # ── Fg: global mean-pool (unchanged) ─────────────────────────────
        Fg_raw = sel_tokens.mean(dim=1)                         # [B, 512]
        Fg_out = F.normalize(self.global_proj(Fg_raw), dim=-1)  # [B, embed_dim]

        # ── Fl: top-K M_obj-weighted patches → multi-token ───────────────
        if self._fl_multi:
            B, N, D = sel_tokens.shape
            K = min(self._top_k_fl, N)
            # M_obj scores per patch
            w_scores = w.detach()                               # [B, N]
            # Pick top-K patch indices per sample
            top_vals, top_idx = w_scores.topk(K, dim=1)          # [B, K]
            # Gather corresponding sel_tokens
            gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, D) # [B, K, D]
            Fl_raw_multi = sel_tokens.gather(1, gather_idx)      # [B, K, D=512]
            # Project each token to embed_dim
            Fl_proj = self.local_proj(
                Fl_raw_multi.reshape(B * K, D)
            ).view(B, K, -1)                                     # [B, K, embed_dim]
            Fl_out = F.normalize(Fl_proj, dim=-1)
            # Also build a pooled version for the retrieval supervision head:
            #   weighted-mean using top-K weights as soft attention.
            w_norm = (top_vals / top_vals.sum(dim=1, keepdim=True)
                       .clamp_min(1e-6)).unsqueeze(-1)            # [B, K, 1]
            Fl_pool_raw = (Fl_raw_multi * w_norm).sum(dim=1)     # [B, D=512]
        else:
            # v7 fallback: single-token mean
            w_unsq = w.unsqueeze(-1)
            Fl_raw_single = (sel_tokens * w_unsq).sum(dim=1) / \
                            w_unsq.sum(dim=1).clamp_min(1e-6)    # [B, 512]
            Fl_out = F.normalize(self.local_proj(Fl_raw_single), dim=-1)  # [B, embed_dim]
            Fl_pool_raw = Fl_raw_single

        # ── Ff: per-band frequency features → multi-token ────────────────
        if self._ff_multi:
            Ff_raw_multi = self.fft_branch(
                sel_tokens, obj_mask, (grid_size, grid_size),
                return_per_band=True,
            )                                                    # [B, K_band, embed_dim]
            Ff_out = F.normalize(self.freq_proj(Ff_raw_multi), dim=-1)
            # Pooled version for retrieval supervision head: simple mean.
            Ff_pool_raw = self.fft_branch(
                sel_tokens, obj_mask, (grid_size, grid_size),
                return_per_band=False,
            )                                                    # [B, embed_dim]
        else:
            Ff_raw_single = self.fft_branch(
                sel_tokens, obj_mask, (grid_size, grid_size),
            )                                                    # [B, embed_dim]
            Ff_out = F.normalize(self.freq_proj(Ff_raw_single), dim=-1)
            Ff_pool_raw = Ff_raw_single

        # Ablation switches inherited from baseline (no_fl / no_ff zero out).
        if self.ablate_no_fl:
            Fl_out = torch.zeros_like(Fl_out)
        if self.ablate_no_ff:
            Ff_out = torch.zeros_like(Ff_out)

        return {
            "Fg": Fg_out,                        # [B, embed_dim]
            "Fl": Fl_out,                        # [B, K, embed_dim] (or [B, embed_dim] if not multi)
            "Ff": Ff_out,                        # [B, 4, embed_dim] (or [B, embed_dim])
            # Pooled single-token versions for the retrieval supervision heads:
            "Fl_pool_raw": Fl_pool_raw,          # [B, 512]
            "Ff_pool_raw": Ff_pool_raw,          # [B, embed_dim]
        }

    # ── Override forward: add l_fl / l_ff retrieval supervision ──────────
    def forward(
        self,
        samples: Dict[str, Any],
        warmup: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if warmup:
            return BaselineModel.forward(self, samples, warmup=True)

        # Run v7 forward to get all v7 losses + final_query + ctx
        out = V7Model.forward(self, samples, warmup=False)

        # Need to re-compute Fl_pool_raw / Ff_pool_raw and z_tgt_tokens for
        # the retrieval supervision. We can't easily fish them out of v7's
        # forward, so re-run the cheap parts here under no_grad for target.
        if self.lambda_fl <= 0 and self.lambda_ff <= 0:
            return out

        image_embeds = samples["image"].to(self.device)
        target_embeds = samples["target"].to(self.device)
        captions = samples["text_input"]
        graphs = samples["graph"]

        with self.maybe_autocast():
            image_embeds_ln = self.ln_vision(image_embeds)
            target_embeds_ln = self.ln_vision(target_embeds)

        # Target tokens (no grad, frozen target encoding path)
        with torch.no_grad():
            F_target = self._qformer_image_only(target_embeds_ln, self.query_tokens)
            z_tgt_tokens = F.normalize(self.vision_proj(F_target), dim=-1)   # [B, 32, 256]

        # Re-run action ctx + visual branches (with grad) to get pooled Fl/Ff
        self._external_M_token = None
        self.graph_enabled = True
        ctx = self._build_action_inputs(image_embeds_ln, graphs)
        patches_action = self._compose_patches(ctx, M_token=None)
        vis = self._visual_branches(
            patches_action, ctx["M_obj_ref"],
            {"node_features": None, "node_mask": None}, graphs,
        )

        # Project pooled features → retrieval queries
        self.temp.data = torch.clamp(self.temp.data, min=1e-3)
        scale = 1.0 / self.temp
        gt = torch.arange(z_tgt_tokens.shape[0], device=z_tgt_tokens.device)
        ls = self.label_smoothing

        l_fl = torch.zeros((), device=self.device)
        if self.lambda_fl > 0.0:
            Fl_pool_raw = vis["Fl_pool_raw"]                            # [B, 512]
            # local_proj is [512 → embed_dim], reuse it then through retrieval head
            Fl_pooled = self.local_proj(Fl_pool_raw)                    # [B, embed_dim]
            Fl_query = F.normalize(self.fl_retrieval_head(Fl_pooled), dim=-1)
            sim_fl = torch.einsum('bd,ntd->bnt', Fl_query, z_tgt_tokens).max(-1).values
            logits_fl = scale * sim_fl
            l_fl = (F.cross_entropy(logits_fl, gt, label_smoothing=ls)
                    + F.cross_entropy(logits_fl.T, gt, label_smoothing=ls)) / 2

        l_ff = torch.zeros((), device=self.device)
        if self.lambda_ff > 0.0:
            Ff_pool_raw = vis["Ff_pool_raw"]                            # [B, embed_dim_inner]
            # FFTBranch.proj outputs in selector_hidden_dim space (512), not
            # the final embed_dim (256). Run through self.freq_proj first.
            Ff_pooled = self.freq_proj(Ff_pool_raw)                     # [B, embed_dim=256]
            Ff_query = F.normalize(self.ff_retrieval_head(Ff_pooled), dim=-1)
            sim_ff = torch.einsum('bd,ntd->bnt', Ff_query, z_tgt_tokens).max(-1).values
            logits_ff = scale * sim_ff
            l_ff = (F.cross_entropy(logits_ff, gt, label_smoothing=ls)
                    + F.cross_entropy(logits_ff.T, gt, label_smoothing=ls)) / 2

        out["l_fl"] = l_fl.detach()
        out["l_ff"] = l_ff.detach()
        out["loss"] = (
            out["loss"]
            + self.lambda_fl * l_fl
            + self.lambda_ff * l_ff
        )
        return out
