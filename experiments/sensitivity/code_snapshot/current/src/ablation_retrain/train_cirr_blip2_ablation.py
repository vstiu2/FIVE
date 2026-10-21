"""Training entry point for the Fl/Fg/Ff retrain ablation.

Thin wrapper around ``src/train_cirr_blip2.py``:

* Swaps the model class for the subclass in ``ssn_blip2_ablation.py``.
* Adds ``--ablate_no_fg`` to the CLI.
* Wires the new flag through to ``model.set_ablation`` so the existing
  training loop doesn't need to know about it.

Everything else (datasets, optimizer, scheduler, SWA, validation, logging,
checkpoint format) is reused verbatim from the baseline script.

Usage example
-------------
    python src/ablation_retrain/train_cirr_blip2_ablation.py \\
        --dataset CustomCIRR --exp_name ablation_no_fl \\
        --feature_dir features/CustomCIRR --gpu 0 \\
        --batch_size 128 --num_epochs 30 --warmup_epoch 3 --lr 1e-5 \\
        --save_training --ablate_no_fl
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# ── Make local src importable ─────────────────────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch  # noqa: E402

import train_cirr_blip2 as tcb  # baseline training script (reused)  # noqa: E402
from ablation_retrain.ssn_blip2_ablation import Model as AblationModel  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Patched build_model: uses the ablation subclass and wires --ablate_no_fg
# into model.set_ablation so the original train() loop doesn't have to know
# about the new flag.
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(args, device):
    model = AblationModel(
        lambda_align=args.lambda_align,
        lambda_global=args.lambda_global,
        lambda_colbert=args.lambda_colbert,
        gamma_init=args.gamma_init,
        gamma_max=args.gamma_max,
        selector_ckpt=args.selector_ckpt,
        label_smoothing=args.label_smoothing,
        alpha_init=args.alpha_init,
        beta_init=args.beta_init,
        alpha_max=args.alpha_max,
        beta_max=args.beta_max,
        max_txt_len=args.max_txt_len,
    )
    pretrained = 'weight/blip2_pretrained.pth'
    if os.path.isfile(pretrained):
        model.load_from_pretrained(pretrained)
        logging.info(f'Loaded BLIP2 pretrained weights from {pretrained}')
    else:
        logging.warning(
            f'Pretrained weight not found at {pretrained}, starting from scratch.'
        )

    # The baseline train() calls model.set_ablation(no_trans=..., no_fl=...,
    # no_ff=..., force_route=...) but knows nothing about no_fg. We wrap the
    # bound method so the new flag is always applied without touching train().
    original_set_ablation = model.set_ablation
    no_fg_flag = bool(args.ablate_no_fg)

    def _wrapped_set_ablation(
        no_trans: bool = False,
        no_fl: bool = False,
        no_ff: bool = False,
        force_route: str = "",
    ) -> None:
        original_set_ablation(
            no_trans=no_trans,
            no_fl=no_fl,
            no_fg=no_fg_flag,
            no_ff=no_ff,
            force_route=force_route,
        )

    model.set_ablation = _wrapped_set_ablation  # type: ignore[assignment]
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# CLI: extends the baseline parser with --ablate_no_fg
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    # Intercept --ablate_no_fg / --help before delegating to the baseline
    # parser (argparse doesn't let us re-open an existing parser to add a
    # new flag). We also pre-parse --help so users see the new flag listed.
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

    saved_argv = sys.argv
    try:
        sys.argv = cleaned
        args = tcb.parse_args()
    finally:
        sys.argv = saved_argv
    args.ablate_no_fg = no_fg

    if show_help:
        print("\nExtra ablation_retrain flags:")
        print("  --ablate_no_fg        Zero out the Fg (global mean-pool) branch in vis dict.")

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Monkey-patch the baseline module so its train() picks up our model
    # builder. Done before train() is called.
    tcb.build_model = _build_model
    args = parse_args()
    tcb.train(args)
