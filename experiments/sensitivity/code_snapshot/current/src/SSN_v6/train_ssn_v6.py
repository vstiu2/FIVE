"""Training entry point for SSN-BLIP2-v6.

Thin wrapper around src/train_cirr_blip2.py:
  * Swaps the model class to SSN_v6.ssn_blip2_v6.Model.
  * Re-exports all v5 CLI flags + v6-only flags
    (--lambda_trans_align / --f_trans_via_fusion / --f_trans_via_gamma).
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
from SSN_v6.ssn_blip2_v6 import Model as V6Model     # noqa: E402


def _build_model(args, device):
    model = V6Model(
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
        # v5
        mask_target_lo=args.mask_target_lo,
        mask_target_hi=args.mask_target_hi,
        mask_target_w=args.mask_target_w,
        mask_div_w=args.mask_div_w,
        # v6
        lambda_trans_align=args.lambda_trans_align,
        f_trans_via_fusion=args.f_trans_via_fusion,
        f_trans_via_gamma=args.f_trans_via_gamma,
    )
    pretrained = 'weight/blip2_pretrained.pth'
    if os.path.isfile(pretrained):
        model.load_from_pretrained(pretrained)
        logging.info(f'Loaded BLIP2 pretrained weights from {pretrained}')
    else:
        logging.warning(f'Pretrained weight not found at {pretrained}.')

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


_V6_FLAGS = {
    # v4-inherited
    "--lambda_kl_action": dict(type=float, default=0.1),
    "--mask_hidden_dim": dict(type=int, default=512),
    "--mask_dropout": dict(type=float, default=0.1),
    "--mask_bias_init": dict(type=float, default=0.0),
    "--inject_hidden_dim": dict(type=int, default=1024),
    "--inject_dropout": dict(type=float, default=0.1),
    "--alpha_inject_init": dict(type=float, default=0.2),
    "--alpha_inject_max": dict(type=float, default=0.5),
    "--lambda_counterfact": dict(type=float, default=0.0),
    "--counterfact_margin": dict(type=float, default=0.05),
    "--blend_dropout": dict(type=float, default=0.0),
    "--ablate_no_action_mask": dict(action="store_true"),
    "--ablate_no_semantic_inject": dict(action="store_true"),
    # v5
    "--mask_target_lo": dict(type=float, default=0.4),
    "--mask_target_hi": dict(type=float, default=0.7),
    "--mask_target_w": dict(type=float, default=1.0),
    "--mask_div_w": dict(type=float, default=0.1),
    # v6
    "--lambda_trans_align": dict(type=float, default=0.1,
        help="[v6] Weight of cosine(F_trans, target_embed - ref_embed). "
             "0 disables the F_trans direct supervision."),
    "--f_trans_via_fusion": dict(type=lambda x: x.lower() in ("1","true","yes","y"),
        default=True,
        help="[v6] Send F_trans into token_fusion as the 4th K/V token."),
    "--f_trans_via_gamma": dict(type=lambda x: x.lower() in ("1","true","yes","y"),
        default=True,
        help="[v6] Keep the γ·F_trans additive tail (legacy v5 path)."),
}


def parse_args() -> argparse.Namespace:
    show_help = ('--help' in sys.argv) or ('-h' in sys.argv)
    if show_help:
        print("SSN_v6 flags:")
        for flag, spec in _V6_FLAGS.items():
            default = spec.get("default", "")
            print(f"  {flag:<32} (default={default!r})  {spec.get('help', '')}")
        print()

    extra_values: dict = {}
    cleaned: list[str] = []
    i = 0
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok in _V6_FLAGS:
            spec = _V6_FLAGS[tok]
            if spec.get("action") == "store_true":
                extra_values[tok] = True
                i += 1
                continue
            extra_values[tok] = spec["type"](sys.argv[i + 1])
            i += 2
            continue
        if any(tok.startswith(k + "=") for k in _V6_FLAGS):
            key, val = tok.split("=", 1)
            spec = _V6_FLAGS[key]
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

    for flag, spec in _V6_FLAGS.items():
        name = flag.lstrip("-")
        default = spec.get("default", False if spec.get("action") == "store_true" else None)
        setattr(args, name, extra_values.get(flag, default))
    return args


if __name__ == '__main__':
    tcb.build_model = _build_model
    args = parse_args()
    tcb.train(args)
