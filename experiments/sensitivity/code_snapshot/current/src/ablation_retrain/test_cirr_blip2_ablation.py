"""Test entry point for the Fl/Fg/Ff retrain ablation.

Thin wrapper around ``src/test_cirr_blip2.py``:

* Loads the ablation subclass (so ``ablate_no_fg`` exists on the model).
* Adds ``--ablate_no_fg`` to the CLI.
* Wires the new flag through to ``model.set_ablation`` AFTER the baseline
  test script has set the rest of the switches.

Everything else (dataset loading, gallery encoding, metrics) is reused
verbatim from the baseline test script.

Usage example
-------------
    # Run that trained with --ablate_no_fl --ablate_no_ff
    python src/ablation_retrain/test_cirr_blip2_ablation.py \\
        --model_path logs/SSN_BLIP2/CustomCIRR/ablation_retrain/no_fl_ff/best_arithmetic.pth \\
        --dataset CustomCIRR --split test \\
        --feature_dir features/CustomCIRR --gpu 0 \\
        --ablate_no_fl --ablate_no_ff

    # Run that trained with --ablate_no_fg
    python src/ablation_retrain/test_cirr_blip2_ablation.py \\
        --model_path logs/SSN_BLIP2/CustomCIRR/ablation_retrain/no_fg/best_arithmetic.pth \\
        --dataset CustomCIRR --split test \\
        --feature_dir features/CustomCIRR --gpu 0 \\
        --ablate_no_fg
"""
from __future__ import annotations

import os
import sys

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import models.ssn_blip2 as ssn_blip2_module  # baseline module  # noqa: E402
import test_cirr_blip2 as tcb_test           # baseline test script  # noqa: E402
from ablation_retrain.ssn_blip2_ablation import Model as AblationModel  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Patch the baseline module so test_cirr_blip2.main() instantiates our
# ablation subclass (it does `from models.ssn_blip2 import Model` inside
# main(), so we override the module attribute before main runs).
# ─────────────────────────────────────────────────────────────────────────────


def _patch_model_class() -> None:
    ssn_blip2_module.Model = AblationModel


def _patch_argparser_and_dispatch() -> None:
    """Intercept --ablate_no_fg before delegating to the baseline test main.

    The baseline ``main()`` builds its own ArgumentParser, so we can't add a
    new flag to it directly. Strategy:
      1. Pull --ablate_no_fg off sys.argv (remember its value).
      2. Monkey-patch ``model.set_ablation`` after model construction so the
         baseline code still calls a 4-arg signature but our injected
         no_fg flag is applied too.

    We do (2) by wrapping ``AblationModel.set_ablation`` at the class level
    just before main() runs — main() then calls it via the wrapped method.
    """
    show_help = ('--help' in sys.argv) or ('-h' in sys.argv)
    no_fg = False
    cleaned = []
    i = 0
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok == '--ablate_no_fg':
            no_fg = True
            i += 1
            continue
        if tok.startswith('--ablate_no_fg='):
            val = tok.split('=', 1)[1].strip().lower()
            no_fg = val in ('1', 'true', 'yes', 'y')
            i += 1
            continue
        cleaned.append(tok)
        i += 1
    sys.argv = cleaned

    if show_help:
        # Run the baseline help, then append our extra flag info.
        # main() will sys.exit after printing — print our note first.
        print("Extra ablation_retrain flag:")
        print("  --ablate_no_fg        Zero out the Fg (global mean-pool) branch in vis dict.\n")
        return  # let baseline main print its own help and exit

    # Wrap the class method so any call inside baseline main() picks up no_fg.
    original_cls_set_ablation = AblationModel.set_ablation

    def _wrapped_set_ablation(
        self,
        no_trans: bool = False,
        no_fl: bool = False,
        no_ff: bool = False,
        force_route: str = "",
    ) -> None:
        original_cls_set_ablation(
            self,
            no_trans=no_trans,
            no_fl=no_fl,
            no_fg=no_fg,
            no_ff=no_ff,
            force_route=force_route,
        )

    AblationModel.set_ablation = _wrapped_set_ablation  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _patch_model_class()
    _patch_argparser_and_dispatch()
    tcb_test.main()
