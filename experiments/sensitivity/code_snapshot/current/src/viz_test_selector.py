"""Visualize test-time selector outputs (M_obj / M_loc / M_act) on CustomCIRR test.

Usage:
    # 12 random test samples, default save to selector_test_check.png
    CUDA_VISIBLE_DEVICES=2 python src/viz_test_selector.py \\
        --n 12 \\
        --model_path logs/SSN_BLIP2/CustomCIRR/ssn_v6_stage/runStage2/best_arithmetic.pth

    # specific samples by index
    CUDA_VISIBLE_DEVICES=2 python src/viz_test_selector.py \\
        --indices 0,10,42,100 \\
        --model_path logs/SSN_BLIP2/CustomCIRR/ssn_v6_stage/runStage2/best_arithmetic.pth

For each sample shows 4 columns:
  1. ref image (the actual 224x224 the encoder sees, with caption + graph metadata)
  2. M_obj heatmap (graph.object grounding via frozen ObjectLocalizer)
  3. M_loc heatmap (graph.result_location grounding via the SAME ObjectLocalizer)
  4. M_act heatmap (per-patch retention weight from v6 ActionMaskHead)

So you can compare:
  * Is M_obj accurate? (the object should be highlighted in red)
  * Is M_loc accurate? (the destination region should be highlighted)
  * Is M_act doing something sensible? (action-conditioned token gating)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data_utils import TargetPad  # noqa: E402
import models.ssn_blip2 as ssn_blip2_module  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

TEST_JSON = "/data/lj/CIR/datasets/merged_query_text_with_graph_allclass_normalized/test/test_query_text_with_graph.json"

# 224×224 image the encoder actually sees (no normalize, for plotting)
def disp_image(path: str, dim: int = 224) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    im = TargetPad(1.25, dim)(im)
    im = TF.resize(im, dim, interpolation=InterpolationMode.BICUBIC)
    im = TF.center_crop(im, dim)
    return np.array(im)


# Encoder-side preprocessing (matches data_utils.squarepad_transform)
def build_img_transform(dim: int = 224):
    return transforms.Compose([
        transforms.Lambda(TargetPad(1.25, dim)),
        transforms.Resize(dim, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(dim),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])


def grid_from_mask(mask_1d: np.ndarray) -> np.ndarray:
    """[N] → [H, H] grid (assumes square patch grid)."""
    n = mask_1d.shape[0]
    h = int(n ** 0.5)
    assert h * h == n, f"non-square patch count: {n}"
    return mask_1d.reshape(h, h)


def heat_overlay(disp_rgb: np.ndarray, mask_grid: np.ndarray) -> np.ndarray:
    """Resize [h, h] mask to display size for jet overlay."""
    g = mask_grid
    gmin, gmax = g.min(), g.max()
    if gmax - gmin < 1e-9:
        g_norm = np.zeros_like(g)
    else:
        g_norm = (g - gmin) / (gmax - gmin)
    heat = np.array(
        Image.fromarray((255 * g_norm).astype("uint8"))
        .resize((disp_rgb.shape[1], disp_rgb.shape[0]), Image.BILINEAR)
    )
    return heat


def focus_stats(grid: np.ndarray):
    m = grid.flatten()
    p = m / (m.sum() + 1e-9)
    ent = -(p * np.log(p + 1e-12)).sum() / np.log(len(m))   # normalized 0~1, 1=均匀
    return ent, m.max() / (m.mean() + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
        help="path to v6/v6-stage best_arithmetic.pth")
    ap.add_argument("--n", type=int, default=12,
        help="number of test samples to visualize")
    ap.add_argument("--indices", type=str, default="",
        help="comma-separated test triplet indices (overrides --n)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="selector_test_check.png")
    ap.add_argument("--alpha", type=float, default=0.5,
        help="heatmap overlay opacity")
    ap.add_argument("--model_variant", choices=["v6", "v6_stage"], default="v6_stage",
        help="which Model class to instantiate (affects ckpt load)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build model ────────────────────────────────────────────────────
    if args.model_variant == "v6_stage":
        from SSN_v6.ssn_blip2_v6_stage import Model as VModel
    else:
        from SSN_v6.ssn_blip2_v6 import Model as VModel

    print(f"Building {args.model_variant} model …")
    model = VModel()  # default kwargs are fine; we'll load ckpt anyway
    pretrained = "weight/blip2_pretrained.pth"
    if os.path.isfile(pretrained):
        model.load_from_pretrained(pretrained)

    print(f"Loading checkpoint: {args.model_path}")
    state = torch.load(args.model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    model = model.to(device).eval()
    model.graph_enabled = True

    # ── Load test triplets ─────────────────────────────────────────────
    print(f"Loading test triplets from {TEST_JSON}")
    triplets = json.loads(open(TEST_JSON).read())
    print(f"  total: {len(triplets)}")

    if args.indices:
        idxs = [int(x) for x in args.indices.split(",") if x.strip()]
    else:
        rng = random.Random(args.seed)
        idxs = rng.sample(range(len(triplets)), min(args.n, len(triplets)))

    # ── Sanity-check that the picked triplets have readable images ─────
    img_tf = build_img_transform(224)

    # ── For each triplet, run ViT → ln_vision → masks ──────────────────
    print(f"\nVisualizing {len(idxs)} samples …\n")
    n = len(idxs)
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for r, idx in enumerate(idxs):
        entry = triplets[idx]
        ref_path = entry["ref_image"]
        graph = entry.get("graph", {})
        caption = entry.get("query_text", "")
        obj = graph.get("object", "unknown")
        loc = graph.get("result_location", "unknown")
        action = graph.get("action", "unknown")

        if not os.path.exists(ref_path):
            print(f"  [skip idx={idx}] ref_image not found: {ref_path}")
            continue

        # Encode image to ViT patches → ln_vision → [1, 1+N, 1408]
        pil = Image.open(ref_path).convert("RGB")
        x = img_tf(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                vit_out = model.visual_encoder(x)            # [1, 1+N, vit_dim]
                image_embeds_ln = model.ln_vision(vit_out)   # [1, 1+N, 1408]

            # Run v6's _build_action_inputs to get M_obj, M_loc, M_act all at once
            ctx = model._build_action_inputs(image_embeds_ln, [graph])

        # All masks shape: [1, N=256]
        M_obj = ctx["M_obj_ref"][0].float().cpu().numpy()
        M_loc = ctx["M_loc"][0].float().cpu().numpy()
        M_act = ctx["M_act"][0].float().cpu().numpy()

        g_obj = grid_from_mask(M_obj)
        g_loc = grid_from_mask(M_loc)
        g_act = grid_from_mask(M_act)

        disp = disp_image(ref_path)

        # Column 0: ref image + caption + metadata
        axes[r, 0].imshow(disp)
        axes[r, 0].axis("off")
        meta = (f"idx={idx} action={action}\n"
                f"obj={obj}\n"
                f"loc={loc}\n"
                f"caption={caption[:60]}")
        axes[r, 0].set_title(meta, fontsize=7, loc="left")

        # Column 1: M_obj
        e_obj, p_obj = focus_stats(g_obj)
        axes[r, 1].imshow(disp)
        axes[r, 1].imshow(heat_overlay(disp, g_obj), cmap="jet", alpha=args.alpha)
        axes[r, 1].axis("off")
        axes[r, 1].set_title(
            f"M_obj ({obj})\nent={e_obj:.2f}  pk/mean={p_obj:.1f}",
            fontsize=8,
        )

        # Column 2: M_loc
        e_loc, p_loc = focus_stats(g_loc)
        axes[r, 2].imshow(disp)
        axes[r, 2].imshow(heat_overlay(disp, g_loc), cmap="jet", alpha=args.alpha)
        axes[r, 2].axis("off")
        axes[r, 2].set_title(
            f"M_loc ({loc})\nent={e_loc:.2f}  pk/mean={p_loc:.1f}",
            fontsize=8,
        )

        # Column 3: M_act
        e_act, p_act = focus_stats(g_act)
        axes[r, 3].imshow(disp)
        axes[r, 3].imshow(heat_overlay(disp, g_act), cmap="jet", alpha=args.alpha)
        axes[r, 3].axis("off")
        axes[r, 3].set_title(
            f"M_act (action={action})\nmean={M_act.mean():.2f}  ent={e_act:.2f}  pk/mean={p_act:.1f}",
            fontsize=8,
        )

        print(f"  idx={idx:5d} action={action:8s} obj={obj[:15]:15s} loc={loc[:15]:15s}  "
              f"M_obj[pk/mean={p_obj:.1f}]  M_loc[pk/mean={p_loc:.1f}]  M_act[mean={M_act.mean():.2f}]")

    plt.tight_layout()
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
