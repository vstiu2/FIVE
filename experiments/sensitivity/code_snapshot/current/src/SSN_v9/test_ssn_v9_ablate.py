"""v9 test entry with additional ablations that don't have CLI flags:

  --ablate_no_mobj   : run FFTBranch with a uniform mask (== w/o M_obj)
  --ablate_no_graph  : force inference(graph_enabled=False) (== w/o G)

All existing SSN_v9 test_ssn_v9.py flags remain available.
This file does NOT modify anything under models/ or SSN_v9/ modules --
it patches at import time so future users can still reproduce v9 exactly.

Note: w/o Gate Fusion is done by retraining (scripts/train_wo_gate.sh),
not by test-time surgery, so it is intentionally NOT provided here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch                                                    # noqa: E402
import models.ssn_blip2 as ssn_blip2_module                     # noqa: E402
import test_cirr_blip2 as tcb_test                              # noqa: E402
from data_utils import CustomCIRRDataset                        # noqa: E402
from model.reference_fft import FixedBoundaryFFTBranch          # noqa: E402
from SSN_v9.ssn_blip2_v9 import Model as V9Model                # noqa: E402
import json                                                     # noqa: E402

def _custom_cirr_root() -> Path:
    packaged = Path(__file__).resolve().parents[2] / "data" / "CustomCIRR_graph"
    default = (packaged if packaged.exists()
               else Path("/data/lj/CIR/datasets/merged_query_text_with_graph_allclass_normalized"))
    return Path(os.environ.get("CUSTOM_CIRR_ROOT", default))


CUSTOM_CIRR_ROOT = _custom_cirr_root()

_EXTRA_FLAGS = {
    "--ablate_no_action_mask",
    "--ablate_no_semantic_inject",
    "--ablate_no_mobj",
    "--ablate_no_graph",
    "--ablate_no_kv",
    "--ablate_no_gate",
}


def _intercept(flags: set) -> dict:
    out = {k: False for k in flags}
    cleaned = []
    i = 0
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok in flags:
            out[tok] = True
            i += 1
            continue
        if any(tok.startswith(k + "=") for k in flags):
            key, val = tok.split("=", 1)
            out[key] = val.strip().lower() in ("1", "true", "yes", "y")
            i += 1
            continue
        cleaned.append(tok)
        i += 1
    sys.argv = cleaned
    return out


_original_init = CustomCIRRDataset.__init__


def _patched_init(self, split, mode, preprocess):
    _original_init(self, split, mode, preprocess)
    src = CUSTOM_CIRR_ROOT / split / f"{split}_query_text_with_graph_object_core.json"
    if src.exists() and hasattr(self, "triplets"):
        with open(src) as f:
            new_triplets = json.load(f)
        if len(new_triplets) == len(self.triplets):
            self.triplets = new_triplets
            n_with_core = sum(
                1 for t in new_triplets if t.get("graph", {}).get("object_core")
            )
            print(f"[v9 test] {split}: object_core injected "
                  f"({n_with_core}/{len(new_triplets)})")


def _patch_model_class() -> None:
    ssn_blip2_module.Model = V9Model


def _wrap_set_ablation(flags: dict) -> None:
    original = V9Model.set_ablation

    def _wrapped(self, no_trans=False, no_fl=False, no_ff=False, force_route=""):
        original(self, no_trans=no_trans, no_fl=no_fl, no_ff=no_ff,
                 force_route=force_route,
                 no_action_mask=flags.get("--ablate_no_action_mask", False),
                 no_semantic_inject=flags.get("--ablate_no_semantic_inject", False))

    V9Model.set_ablation = _wrapped   # type: ignore[assignment]


# ── w/o M_obj: replace token_scores with uniform mask inside FFT ───────────
def _patch_no_mobj() -> None:
    original_forward = FixedBoundaryFFTBranch.forward

    def _forward_uniform(self,
                         patch_tokens: torch.Tensor,
                         token_scores: torch.Tensor,
                         token_hw,
                         return_per_band: bool = False,
                         ) -> torch.Tensor:
        uniform = torch.ones_like(token_scores)
        return original_forward(self, patch_tokens, uniform, token_hw,
                                return_per_band=return_per_band)

    FixedBoundaryFFTBranch.forward = _forward_uniform  # type: ignore[assignment]
    print("[ablate_no_mobj] FFTBranch uses uniform mask (no M_obj weighting).")


# ── w/o G: force inference() to run with graph_enabled=False ───────────────
def _patch_no_graph() -> None:
    original_inference = V9Model.inference

    def _inf_no_graph(self, image_embeds_ln, gallery_embed_ln, captions,
                      graphs, graph_enabled: bool = True):
        return original_inference(self, image_embeds_ln, gallery_embed_ln,
                                  captions, graphs, graph_enabled=False)

    V9Model.inference = _inf_no_graph  # type: ignore[assignment]
    print("[ablate_no_graph] Q-Former warmup path forced at inference.")


# ── w/o KV: replace token_fusion(...) with mean-pooled Fq_u ────────────────
def _patch_no_kv() -> None:
    """Skip cross-attention over external KV (Fg/Fl/Ff/F_trans).
    Fq_fused degenerates to a simple mean-pool of the Q-Former query tokens.
    """
    from model import token_fusion as _tf_mod

    def _forward_noop(self, z_rm, Fg, Fl, Ff, F_trans=None):
        # z_rm is the pooled Fq_u [B, D]; there is no way to recover the
        # original 32 tokens here, so we simply return z_rm untouched
        # (identity). This drops all external KV information.
        return z_rm

    _tf_mod.TokenFusionBlock.forward = _forward_noop  # type: ignore[assignment]
    print("[ablate_no_kv] TokenFusionBlock returns z_rm identity "
          "(no cross-attention over external KV).")


# ── w/o Gate Fusion: force blend term to zero at inference ─────────────────
def _patch_no_gate() -> None:
    """Test-time surgery to remove the (s*text + (1-s)*ref) blend term
    from final_query. Achieved by forcing self._blend_dropout_p = 1.0
    and self.training = True so blend_scale becomes exactly 0 inside
    _build_query. NOTE: proper w/o Gate should be retrained; this
    test-only variant preserves the training-time-trained weights but
    strips the gate contribution at eval.
    """
    original_build = V9Model._build_query

    def _build_no_gate(self, image_embeds_ln, captions, graphs,
                       target_patches=None):
        # Force blend_scale=0 by flipping the two toggles read on line
        # ssn_blip2_v9.py:130. We restore afterwards so other calls
        # (e.g. mid-forward auxiliary branches) aren't affected.
        prev_training = self.training
        prev_p = self._blend_dropout_p
        prev_ramp = self.aux_ramp.detach().clone()

        # 1. self.training must be True to enter the dropout branch
        object.__setattr__(self, 'training', True)
        # 2. _blend_dropout_p must be 1.0 so random > 1.0 always False
        self._blend_dropout_p = 1.0
        # 3. aux_ramp must be >= 0.99 for the ramp_done_flag gate
        with torch.no_grad():
            self.aux_ramp.fill_(1.0)

        # But branch_strength gate needs beta+gamma > 0.2; guarantee by
        # temporarily raising beta/gamma clamped values. Instead of
        # touching parameters, just enforce blend_scale=0 by monkey-
        # patching torch.rand to always return 0 (< 1.0 => keep=0).
        orig_rand = torch.rand

        def _rand_zero(*args, **kwargs):
            return orig_rand(*args, **kwargs) * 0.0

        torch.rand = _rand_zero
        try:
            out = original_build(self, image_embeds_ln, captions, graphs,
                                 target_patches=target_patches)
        finally:
            torch.rand = orig_rand
            object.__setattr__(self, 'training', prev_training)
            self._blend_dropout_p = prev_p
            with torch.no_grad():
                self.aux_ramp.copy_(prev_ramp)
        return out

    V9Model._build_query = _build_no_gate  # type: ignore[assignment]
    print("[ablate_no_gate] Gate-fusion blend term forced to zero at eval "
          "(test-time surgery; not the same as retraining with blend=0).")


if __name__ == "__main__":
    flags = _intercept(_EXTRA_FLAGS)
    CustomCIRRDataset.__init__ = _patched_init
    _patch_model_class()
    _wrap_set_ablation(flags)

    if flags["--ablate_no_mobj"]:
        _patch_no_mobj()
    if flags["--ablate_no_graph"]:
        _patch_no_graph()
    if flags["--ablate_no_kv"]:
        _patch_no_kv()
    if flags["--ablate_no_gate"]:
        _patch_no_gate()

    tcb_test.main()
