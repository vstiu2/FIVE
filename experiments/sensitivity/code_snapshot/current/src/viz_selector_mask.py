"""Standalone visualizer: where does the trained selector locate an object?

Runs the SAME pipeline the selector was trained with:
  image --targetpad(1.25,224)--> EVA-ViT-G (frozen) --> patch tokens [256,1408]
  object text --> Q-Former BERT --> node feature [768]
  ObjectLocalizer(patches, node) --> per-patch logits --> sigmoid --> M_obj [16x16]
Overlays M_obj on the (targetpad-preprocessed) image the encoder actually saw.

Examples:
  # single image + one or more objects
  python src/viz_selector_mask.py --image /path/to/img.png --object "airplane,taxiway"

  # sample N rows from a CIRR/grounding json (uses each sample's graph object)
  python src/viz_selector_mask.py --json /data/lj/CIR/datasets/merged_query_text_with_graph_allclass_normalized/val/val_query_text_with_graph.json --n 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data_utils import TargetPad
from model.graph_utils import encode_graph_node_features
from model.object_localizer import ObjectLocalizer, aggregate_mask
from train_selector import FrozenBackbone


def disp_image(path: str, dim: int = 224) -> np.ndarray:
    """The exact 224x224 image EVA saw (targetpad geometry, no normalize)."""
    im = Image.open(path).convert("RGB")
    im = TargetPad(1.25, dim)(im)
    im = TF.resize(im, dim, interpolation=InterpolationMode.BICUBIC)
    im = TF.center_crop(im, dim)
    return np.array(im)


@torch.no_grad()
def object_mask(backbone: FrozenBackbone, localizer: ObjectLocalizer,
                image_tensor: torch.Tensor, object_text: str) -> np.ndarray:
    """image_tensor [1,3,224,224] → M_obj [16,16] for the given object text."""
    patches, hw = backbone.extract_patches(image_tensor)         # [1, N, 1408]
    graph = {"nodes": [{"label": object_text}]}
    node_out = encode_graph_node_features(
        tokenizer=backbone.tokenizer, bert_model=backbone.Qformer.bert,
        graphs=[graph], device=backbone.device, max_txt_len=backbone.max_txt_len,
    )
    M = aggregate_mask(localizer(
        patches.float(), node_out["node_features"].float(), node_out["node_mask"]))
    h, w = hw
    return M[0].float().cpu().numpy().reshape(h, w)


def focus_stats(grid: np.ndarray):
    m = grid.flatten()
    p = m / (m.sum() + 1e-9)
    ent = -(p * np.log(p + 1e-12)).sum() / np.log(len(m))
    return ent, m.max() / (m.mean() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector_ckpt", default="weight/selector_eva.pt")
    ap.add_argument("--image", default="", help="single image path")
    ap.add_argument("--object", default="", help="comma-separated object text(s)")
    ap.add_argument("--json", default="", help="sample rows from this json (graph.object)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="selector_mask_check.png")
    ap.add_argument("--gpu", default="6")
    ap.add_argument("--alpha", type=float, default=0.5, help="heatmap overlay opacity")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = FrozenBackbone(img_size=224).to(device).eval()
    if os.path.isfile("weight/blip2_pretrained.pth"):
        backbone.load_from_pretrained("weight/blip2_pretrained.pth")
    localizer = ObjectLocalizer(
        vit_dim=1408, node_dim=backbone.Qformer.config.hidden_size, hidden_dim=512,
    ).to(device).eval()
    ck = torch.load(args.selector_ckpt, map_location="cpu")
    localizer.load_state_dict(ck.get("model", ck), strict=False)
    print(f"loaded selector {args.selector_ckpt} (epoch {ck.get('epoch')}, "
          f"val_loss {ck.get('val_loss')})")

    from torchvision import transforms
    img_tf = transforms.Compose([
        transforms.Lambda(TargetPad(1.25, 224)),
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

    # Build (image_path, object_text) jobs
    jobs = []
    if args.image:
        objs = [o.strip() for o in args.object.split(",") if o.strip()] or ["object"]
        for o in objs:
            jobs.append((args.image, o))
    elif args.json:
        rows = json.loads(open(args.json).read())
        for x in rows:
            g = x.get("graph") or {}
            obj = g.get("object") or (g.get("nodes", [{}])[0].get("label") if g.get("nodes") else None)
            path = x.get("ref_image") or x.get("image")
            if obj and path and os.path.exists(path):
                jobs.append((path, obj))
            if len(jobs) >= args.n:
                break
    else:
        raise SystemExit("provide --image (+--object) or --json")

    n = len(jobs)
    fig, axes = plt.subplots(n, 2, figsize=(7, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    for r, (path, obj) in enumerate(jobs):
        pil = Image.open(path).convert("RGB")
        x = img_tf(pil).unsqueeze(0).to(device)
        with torch.cuda.amp.autocast():
            grid = object_mask(backbone, localizer, x, obj)
        ent, pk = focus_stats(grid)
        disp = disp_image(path)
        heat = np.array(Image.fromarray(
            (255 * (grid - grid.min()) / (np.ptp(grid) + 1e-9)).astype("uint8")
        ).resize((disp.shape[1], disp.shape[0]), Image.BILINEAR))
        axes[r, 0].imshow(disp); axes[r, 0].axis("off")
        axes[r, 0].set_title(f"obj={obj}", fontsize=8)
        axes[r, 1].imshow(disp); axes[r, 1].imshow(heat, cmap="jet", alpha=args.alpha)
        axes[r, 1].axis("off")
        axes[r, 1].set_title(f"M_obj  entropy={ent:.2f} peak/mean={pk:.1f}", fontsize=8)
        print(f"  {os.path.basename(path)[:32]:32s} obj={obj[:18]:18s} "
              f"entropy={ent:.3f} peak/mean={pk:.1f}")
    plt.tight_layout()
    plt.savefig(args.out, dpi=95, bbox_inches="tight")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
