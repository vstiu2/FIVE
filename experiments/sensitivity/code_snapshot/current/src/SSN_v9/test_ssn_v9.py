"""Test entry point for SSN-BLIP2-v9."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import models.ssn_blip2 as ssn_blip2_module  # noqa: E402
import test_cirr_blip2 as tcb_test           # noqa: E402
from data_utils import CustomCIRRDataset     # noqa: E402
from model.reference_fft import make_fft_band_edges  # noqa: E402
from SSN_v9.ssn_blip2_v9 import Model as V9Model  # noqa: E402

def _custom_cirr_root() -> Path:
    packaged = Path(__file__).resolve().parents[2] / "data" / "CustomCIRR_graph"
    default = (packaged if packaged.exists()
               else Path("/data/lj/CIR/datasets/merged_query_text_with_graph_allclass_normalized"))
    return Path(os.environ.get("CUSTOM_CIRR_ROOT", default))


CUSTOM_CIRR_ROOT = _custom_cirr_root()


def _patch_model_class(fft_num_bands: int) -> None:
    fft_band_edges = make_fft_band_edges(fft_num_bands)

    def _model_factory():
        return V9Model(fft_band_edges=fft_band_edges)

    ssn_blip2_module.Model = _model_factory


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


def _intercept(flags: set) -> dict:
    out = {k: False for k in flags}
    cleaned = []
    i = 0
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok in flags:
            out[tok] = True; i += 1; continue
        if any(tok.startswith(k + "=") for k in flags):
            key, val = tok.split("=", 1)
            out[key] = val.strip().lower() in ("1","true","yes","y")
            i += 1; continue
        cleaned.append(tok); i += 1
    sys.argv = cleaned
    return out


def _intercept_value(flag: str, default, value_type):
    value = default
    cleaned = []
    i = 0
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok == flag:
            if i + 1 >= len(sys.argv):
                raise ValueError(f"{flag} requires a value")
            value = value_type(sys.argv[i + 1])
            i += 2
            continue
        if tok.startswith(flag + "="):
            value = value_type(tok.split("=", 1)[1])
            i += 1
            continue
        cleaned.append(tok)
        i += 1
    sys.argv = cleaned
    return value


def _wrap_set_ablation(flags: dict) -> None:
    original = V9Model.set_ablation
    def _wrapped(self, no_trans=False, no_fl=False, no_ff=False, force_route=""):
        original(self, no_trans=no_trans, no_fl=no_fl, no_ff=no_ff,
                 force_route=force_route,
                 no_action_mask=flags.get("--ablate_no_action_mask", False),
                 no_semantic_inject=flags.get("--ablate_no_semantic_inject", False))
    V9Model.set_ablation = _wrapped  # type: ignore[assignment]


if __name__ == '__main__':
    flags = _intercept({"--ablate_no_action_mask", "--ablate_no_semantic_inject"})
    fft_num_bands = _intercept_value("--fft_num_bands", 4, int)
    CustomCIRRDataset.__init__ = _patched_init
    _patch_model_class(fft_num_bands)
    _wrap_set_ablation(flags)
    tcb_test.main()
