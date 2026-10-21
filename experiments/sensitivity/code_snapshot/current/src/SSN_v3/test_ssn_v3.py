"""Test entry point for SSN-BLIP2-v3.

Thin wrapper around ``src/test_cirr_blip2.py``:
  * Patches ``models.ssn_blip2.Model`` to be the v3 subclass.
  * Adds ``--ablate_no_action_mask`` to the CLI and wires it through
    ``model.set_ablation``.

Usage
-----
    python src/SSN_v3/test_ssn_v3.py \\
        --model_path logs/SSN_BLIP2/CustomCIRR/ssn_v3/run1/best_arithmetic.pth \\
        --dataset CustomCIRR --split test \\
        --feature_dir features/CustomCIRR --gpu 0

    # Ablate: M_act ≡ 1, equivalent to v2 inference path
    python src/SSN_v3/test_ssn_v3.py \\
        --model_path logs/SSN_BLIP2/CustomCIRR/ssn_v3/run1/best_arithmetic.pth \\
        ... --ablate_no_action_mask
"""
from __future__ import annotations

import os
import sys

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import models.ssn_blip2 as ssn_blip2_module  # noqa: E402
import test_cirr_blip2 as tcb_test           # noqa: E402
from SSN_v3.ssn_blip2_v3 import Model as V3Model  # noqa: E402


def _patch_model_class() -> None:
    ssn_blip2_module.Model = V3Model


def _intercept_no_action_mask() -> bool:
    no_act = False
    cleaned = []
    i = 0
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok == '--ablate_no_action_mask':
            no_act = True
            i += 1
            continue
        if tok.startswith('--ablate_no_action_mask='):
            no_act = tok.split('=', 1)[1].strip().lower() in ('1', 'true', 'yes', 'y')
            i += 1
            continue
        cleaned.append(tok)
        i += 1
    sys.argv = cleaned
    return no_act


def _wrap_set_ablation(no_action_mask: bool) -> None:
    """Wrap V3Model.set_ablation at the class level so that whatever the
    baseline ``main()`` calls with its 4-arg signature still produces a
    correct configuration for v3."""
    original = V3Model.set_ablation

    def _wrapped(self, no_trans=False, no_fl=False, no_ff=False, force_route=""):
        original(self, no_trans=no_trans, no_fl=no_fl, no_ff=no_ff,
                 force_route=force_route, no_action_mask=no_action_mask)
    V3Model.set_ablation = _wrapped  # type: ignore[assignment]


if __name__ == '__main__':
    no_act = _intercept_no_action_mask()
    _patch_model_class()
    _wrap_set_ablation(no_act)
    tcb_test.main()
