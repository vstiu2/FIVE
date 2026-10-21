"""Training entry point for SSN-BLIP2-v5.

Thin wrapper around src/train_cirr_blip2.py:
  * Swaps the model class to SSN_v5.ssn_blip2_v5.Model.
  * Re-exports all v4 CLI flags (ablation, mask, injector, counterfact,
    blend_dropout) and adds v5-only mask-box flags.

Usage
-----
    # Fine-tune from v4 best_arithmetic with the v5 box constraint
    python src/SSN_v5/train_ssn_v5.py \\
        --resume logs/SSN_BLIP2/CustomCIRR/ssn_v4/run1/best_arithmetic.pth \\
        --exp_name ssn_v5/runA \\
        --warmup_epoch 0 --num_epochs 10 --lr 2e-6 \\
        --lambda_kl_action 0.05 \\
        --lambda_counterfact 0 --blend_dropout 0 \\
        --mask_target_lo 0.4 --mask_target_hi 0.7 --mask_target_w 0.5
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

import train_cirr_blip2 as tcb                       # noqa: E402
from SSN_v5.ssn_blip2_v5 import Model as V5Model     # noqa: E402


def _build_model(args, device):
    model = V5Model(
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
        # v4-inherited
        lambda_kl_action=args.lambda_kl_action,
        mask_hidden_dim=args.mask_hidden_dim,
        mask_dropout=args.mask_dropout,
        mask_bias_init=args.mask_bias_init,
        inject_hidden_dim=args.inject_hidden_dim,
        inject_dropout=args.inject_dropout,
        alpha_inject_init=args.alpha_inject_init,
        alpha_inject_max=args.alpha_inject_max,
        lambda_counterfact=args.lambda_counterfact,
        counterfact_margin=args.counterfact_margin,
        blend_dropout=args.blend_dropout,
        # v5-specific
        mask_target_lo=args.mask_target_lo,
        mask_target_hi=args.mask_target_hi,
        mask_target_w=args.mask_target_w,
        mask_div_w=args.mask_div_w,
    )
    pretrained = 'weight/blip2_pretrained.pth'
    if os.path.isfile(pretrained):
        model.load_from_pretrained(pretrained)
        logging.info(f'Loaded BLIP2 pretrained weights from {pretrained}')
    else:
        logging.warning(f'Pretrained weight not found at {pretrained}.')

    # Wrap set_ablation so the baseline trainer's 4-arg call still applies
    # the v4/v5-specific ablation flags.
    original = model.set_ablation
    no_mask_flag = bool(args.ablate_no_action_mask)
    no_inject_flag = bool(args.ablate_no_semantic_inject)

    def _wrapped(no_trans=False, no_fl=False, no_ff=False, force_route=""):
        original(no_trans=no_trans, no_fl=no_fl, no_ff=no_ff,
                 force_route=force_route,
                 no_action_mask=no_mask_flag,
                 no_semantic_inject=no_inject_flag)
    model.set_ablation = _wrapped  # type: ignore[assignment]
    return model.to(device)


# v4 flags inherited + v5-specific flags.
_V5_FLAGS = {
    # v4-inherited
    "--lambda_kl_action": dict(type=float, default=0.1,
        help="Weight of the push-pull action loss. 0 disables it."),
    "--mask_hidden_dim": dict(type=int, default=512,
        help="Hidden dim of ActionMaskHead MLP."),
    "--mask_dropout": dict(type=float, default=0.1,
        help="Dropout in ActionMaskHead."),
    "--mask_bias_init": dict(type=float, default=0.0,
        help="Pre-sigmoid bias of the final mask logit; 0 → M_act ≈ 0.5."),
    "--inject_hidden_dim": dict(type=int, default=1024,
        help="Hidden dim of SemanticInjector projection MLP."),
    "--inject_dropout": dict(type=float, default=0.1,
        help="Dropout in SemanticInjector."),
    "--alpha_inject_init": dict(type=float, default=0.2,
        help="Initial value of α_inject."),
    "--alpha_inject_max": dict(type=float, default=0.5,
        help="Upper bound of α_inject."),
    "--lambda_counterfact": dict(type=float, default=0.5,
        help="Weight of the counter-factual hinge loss."),
    "--counterfact_margin": dict(type=float, default=0.05,
        help="Margin for the counter-factual hinge."),
    "--blend_dropout": dict(type=float, default=0.2,
        help="Probability of zeroing the SSN blend at training time."),
    "--ablate_no_action_mask": dict(action="store_true",
        help="Force M_act ≡ 1 (no action gating)."),
    "--ablate_no_semantic_inject": dict(action="store_true",
        help="Force α_inject · M_loc · obj_proto = 0 (no injection)."),

    # v5-specific
    "--mask_target_lo": dict(type=float, default=0.4,
        help="[v5] Lower bound of the box constraint on mean(M_act). "
             "No penalty as long as mean ≥ lo."),
    "--mask_target_hi": dict(type=float, default=0.7,
        help="[v5] Upper bound of the box constraint on mean(M_act). "
             "No penalty as long as mean ≤ hi."),
    "--mask_target_w": dict(type=float, default=0.5,
        help="[v5] Weight of the box-constraint penalty. v4 used 0.1 with "
             "a (mean-0.5)² target; v5 uses 0.5 with a one-sided box."),
    "--mask_div_w": dict(type=float, default=0.1,
        help="[v5] Weight of the -Var(M_act) regularizer (same role as v4)."),
}


def parse_args() -> argparse.Namespace:
    show_help = ('--help' in sys.argv) or ('-h' in sys.argv)
    if show_help:
        print("SSN_v5 flags (in addition to all baseline flags below):")
        for flag, spec in _V5_FLAGS.items():
            default = spec.get("default", "")
            print(f"  {flag:<32} (default={default!r})  {spec.get('help', '')}")
        print()

    extra_values: dict = {}
    cleaned: list[str] = []
    i = 0
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok in _V5_FLAGS:
            spec = _V5_FLAGS[tok]
            if spec.get("action") == "store_true":
                extra_values[tok] = True
                i += 1
                continue
            extra_values[tok] = spec["type"](sys.argv[i + 1])
            i += 2
            continue
        if any(tok.startswith(k + "=") for k in _V5_FLAGS):
            key, val = tok.split("=", 1)
            spec = _V5_FLAGS[key]
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

    for flag, spec in _V5_FLAGS.items():
        name = flag.lstrip("-")
        default = spec.get("default", False if spec.get("action") == "store_true" else None)
        setattr(args, name, extra_values.get(flag, default))
    return args


if __name__ == '__main__':
    tcb.build_model = _build_model
    args = parse_args()
    tcb.train(args)
