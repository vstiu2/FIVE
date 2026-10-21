"""Training entry point for SSN-BLIP2-v3 (action-mask + push-pull KL).

Thin wrapper around ``src/train_cirr_blip2.py``:
  * Swaps the model class for ``SSN_v3.ssn_blip2_v3.Model``.
  * Adds ``--lambda_kl_action`` and ``--ablate_no_action_mask`` to the CLI.
  * Wires the new flags through to the model.

Everything else (datasets, optimizer, scheduler, SWA, validation, logging,
checkpoint format) is reused verbatim from the v2 training script.

Usage
-----
    python src/SSN_v3/train_ssn_v3.py \\
        --dataset CustomCIRR --exp_name ssn_v3/run1 \\
        --feature_dir features/CustomCIRR --gpu 0 \\
        --batch_size 64 --num_epochs 30 --warmup_epoch 3 --lr 1e-5 \\
        --lambda_align 0.4 --gamma_init 0.05 --beta_init 0.05 \\
        --patch_dropout 0.15 --save_training \\
        --lambda_kl_action 0.1

Logs land under ``logs/SSN_BLIP2/<dataset>/<exp_name>/``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch  # noqa: E402

import train_cirr_blip2 as tcb  # noqa: E402
from SSN_v3.ssn_blip2_v3 import Model as V3Model  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Patched build_model: uses the v3 subclass and forwards new kwargs.
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(args, device):
    model = V3Model(
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
        # v3-specific
        lambda_kl_action=args.lambda_kl_action,
        mask_hidden_dim=args.mask_hidden_dim,
        mask_dropout=args.mask_dropout,
        mask_bias_init=args.mask_bias_init,
    )
    pretrained = 'weight/blip2_pretrained.pth'
    if os.path.isfile(pretrained):
        model.load_from_pretrained(pretrained)
        logging.info(f'Loaded BLIP2 pretrained weights from {pretrained}')
    else:
        logging.warning(f'Pretrained weight not found at {pretrained}.')

    # Wrap set_ablation so the baseline trainer can call the 4-arg form
    # and the new no_action_mask flag is still applied.
    original_set_ablation = model.set_ablation
    no_action_mask_flag = bool(args.ablate_no_action_mask)

    def _wrapped(no_trans=False, no_fl=False, no_ff=False, force_route=""):
        original_set_ablation(
            no_trans=no_trans, no_fl=no_fl, no_ff=no_ff,
            force_route=force_route, no_action_mask=no_action_mask_flag,
        )
    model.set_ablation = _wrapped  # type: ignore[assignment]
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Extend the baseline CLI with --lambda_kl_action / --ablate_no_action_mask
# and a few mask_head knobs.
# ─────────────────────────────────────────────────────────────────────────────

_V3_FLAGS = {
    "--lambda_kl_action": dict(type=float, default=0.1,
        help="Weight of the action-conditioned push-pull KL. 0 disables it "
             "→ behavior approaches v2 (mask still applied to patches, "
             "but no extra supervision)."),
    "--mask_hidden_dim": dict(type=int, default=512,
        help="Hidden dim of ActionMaskHead MLP."),
    "--mask_dropout": dict(type=float, default=0.1,
        help="Dropout in ActionMaskHead."),
    "--mask_bias_init": dict(type=float, default=0.0,
        help="Bias of the final linear in ActionMaskHead; "
             "controls the initial M_act value (sigmoid(bias)). "
             "Default 0 → M_act ≈ 0.5 (unbiased start)."),
    "--ablate_no_action_mask": dict(action="store_true",
        help="Force M_act = 1 → bit-identical to v2 inference."),
}


def parse_args() -> argparse.Namespace:
    show_help = ('--help' in sys.argv) or ('-h' in sys.argv)

    # If --help is requested, print v3 flags FIRST (baseline parser exits via
    # sys.exit after printing its own help).
    if show_help:
        print("SSN_v3-only flags (in addition to all baseline flags below):")
        for flag, spec in _V3_FLAGS.items():
            default = spec.get("default", "")
            print(f"  {flag:<28} (default={default!r})  {spec.get('help', '')}")
        print()

    # Pre-strip our flags so the baseline parser doesn't choke on them.
    extra_values: dict = {}
    cleaned: list[str] = []
    i = 0
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok in _V3_FLAGS:
            spec = _V3_FLAGS[tok]
            if spec.get("action") == "store_true":
                extra_values[tok] = True
                i += 1
                continue
            extra_values[tok] = spec["type"](sys.argv[i + 1])
            i += 2
            continue
        if any(tok.startswith(k + "=") for k in _V3_FLAGS):
            key, val = tok.split("=", 1)
            spec = _V3_FLAGS[key]
            if spec.get("action") == "store_true":
                extra_values[key] = val.strip().lower() in ("1", "true", "yes", "y")
            else:
                extra_values[key] = spec["type"](val)
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

    # Apply defaults + overrides.
    for flag, spec in _V3_FLAGS.items():
        name = flag.lstrip("-")
        default = spec.get("default", False if spec.get("action") == "store_true" else None)
        setattr(args, name, extra_values.get(flag, default))

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Entry point: patch the baseline module's build_model, then run train().
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tcb.build_model = _build_model
    args = parse_args()
    tcb.train(args)
