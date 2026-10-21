"""Test entry point for SSN-BLIP2-v4.

Thin wrapper around src/test_cirr_blip2.py:
  * Patches ``models.ssn_blip2.Model`` to be the v4 subclass.
  * Adds ``--ablate_no_action_mask`` and ``--ablate_no_semantic_inject`` and
    wires them through ``model.set_ablation``.
"""
from __future__ import annotations

import os
import sys

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import models.ssn_blip2 as ssn_blip2_module  # noqa: E402
import test_cirr_blip2 as tcb_test           # noqa: E402
from SSN_v4.ssn_blip2_v4 import Model as V4Model  # noqa: E402


def _patch_model_class() -> None:
    ssn_blip2_module.Model = V4Model


def _intercept(flags: dict) -> dict:
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


def _wrap_set_ablation(flags: dict) -> None:
    original = V4Model.set_ablation

    def _wrapped(self, no_trans=False, no_fl=False, no_ff=False, force_route=""):
        original(self, no_trans=no_trans, no_fl=no_fl, no_ff=no_ff,
                 force_route=force_route,
                 no_action_mask=flags.get("--ablate_no_action_mask", False),
                 no_semantic_inject=flags.get("--ablate_no_semantic_inject", False))
    V4Model.set_ablation = _wrapped  # type: ignore[assignment]


if __name__ == '__main__':
    flags = _intercept({"--ablate_no_action_mask", "--ablate_no_semantic_inject"})
    _patch_model_class()
    _wrap_set_ablation(flags)
    tcb_test.main()
