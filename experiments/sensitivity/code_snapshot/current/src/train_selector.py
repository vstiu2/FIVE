"""Train ObjectLocalizer on graph-grounding masks in the EVA-ViT-G + BERT space.

Produces weight/selector_eva.pt that SSN-BLIP2 loads (frozen) as its localization
backbone. Mask geometry MUST follow the same TargetPad+Resize+CenterCrop chain
as the EVA feature precomputation, otherwise the supervision and the patch grid
won't align.

Usage:
    python src/train_selector.py \
        --train_json /data/lj/CIR/datasets/selector_dataset/train/train_allclass_grounding.json \
        --val_json   /data/lj/CIR/datasets/selector_dataset/val/val_allclass_grounding.json \
        --gpu 6 --epochs 20 --batch_size 32 --out weight/selector_eva.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data_utils import TargetPad
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from model.graph_utils import encode_graph_node_features, graph_to_node_texts
from model.object_localizer import ObjectLocalizer


# ─────────────────────────────────────────────────────────────────────────────
# Mask transform — geometrically identical to the image transform used for EVA
# features (TargetPad(1.25) → Resize → CenterCrop), no normalization, NEAREST.
# ─────────────────────────────────────────────────────────────────────────────

class MaskTargetPadResize:
    def __init__(self, target_ratio: float = 1.25, dim: int = 224):
        self.pad = TargetPad(target_ratio, dim)
        self.dim = dim

    def __call__(self, mask_pil: Image.Image) -> torch.Tensor:
        padded = self.pad(mask_pil)
        resized = TF.resize(padded, self.dim, interpolation=InterpolationMode.NEAREST)
        cropped = TF.center_crop(resized, self.dim)
        return TF.pil_to_tensor(cropped).squeeze(0).float() / 255.0   # [H, W] in {0,1}


def downsample_mask_to_grid(mask: torch.Tensor, token_hw: Tuple[int, int]) -> torch.Tensor:
    """[H, W] → [h, w] via NEAREST (preserves coverage of small objects better than
    max-pool of a soft mask)."""
    return F.interpolate(
        mask.unsqueeze(0).unsqueeze(0), size=token_hw, mode="nearest",
    ).squeeze(0).squeeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class GraphGroundingDataset(Dataset):
    """Each sample: one image + its graph + per-node grounding regions (mask/bbox)."""

    def __init__(self, json_paths: List[str], image_size: int = 224):
        self.samples: List[Dict[str, Any]] = []
        for jp in json_paths:
            rows = json.loads(Path(jp).read_text(encoding="utf-8"))
            ann_root = str(Path(jp).resolve().parent)
            for row in rows:
                row = dict(row); row["_annotation_root"] = ann_root
                self.samples.append(row)
        self.image_size = image_size
        # Same geometry as targetpad_transform(1.25, 224) but for PIL images.
        self.pad = TargetPad(1.25, image_size)
        self.image_tf = transforms.Compose([
            transforms.Lambda(self.pad),
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        self.mask_tf = MaskTargetPadResize(1.25, image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve(self, p: str, ann_root: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        return (Path(ann_root) / path).resolve()

    def _load_mask(self, region: Dict[str, Any], orig_size: Tuple[int, int],
                   ann_root: str) -> Tuple[Image.Image, str]:
        w, h = orig_size
        if "mask" in region:
            mp = self._resolve(region["mask"], ann_root)
            return Image.open(mp).convert("L").resize((w, h), Image.NEAREST), "mask"
        if "bbox" in region:
            x1, y1, x2, y2 = region["bbox"]
            m = Image.new("L", (w, h), 0)
            ImageDraw.Draw(m).rectangle([x1, y1, x2, y2], fill=255)
            return m, "bbox"
        raise ValueError("region needs 'mask' or 'bbox'")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        ann_root = s["_annotation_root"]
        img_path = self._resolve(s.get("image") or s.get("ref_image"), ann_root)
        img_pil = Image.open(img_path).convert("RGB")
        orig = img_pil.size
        image = self.image_tf(img_pil)                       # [3, 224, 224]
        regions = []
        for region in s.get("regions", []):
            mask_pil, src = self._load_mask(region, orig, ann_root)
            mask_t = self.mask_tf(mask_pil)                  # [224, 224]
            regions.append({
                "node_id": int(region["node_id"]),
                "mask": mask_t,
                "source_type": src,
            })
        return {"image": image, "graph": s["graph"], "regions": regions,
                "image_path": str(img_path)}


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": torch.stack([b["image"] for b in batch], dim=0),
        "graphs": [b["graph"] for b in batch],
        "regions": [b["regions"] for b in batch],
        "paths": [b["image_path"] for b in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        num = 2.0 * (probs * targets).sum(dim=1)
        den = probs.sum(dim=1) + targets.sum(dim=1) + 1e-6
        return 1.0 - (num + 1e-6) / den


def build_supervision(node_scores: torch.Tensor,
                      regions: List[Dict[str, Any]],
                      token_hw: Tuple[int, int]
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """node_scores: [M, h*w] logits → reshaped [M, h, w] target maps + valid mask.

    Each region's full-res mask is downsampled to (h, w) via NEAREST.
    """
    M = node_scores.shape[0]
    h, w = token_hw
    targets = torch.zeros(M, h, w, device=node_scores.device, dtype=torch.float)
    valid = torch.zeros(M, dtype=torch.bool, device=node_scores.device)
    for r in regions:
        nid = int(r["node_id"])
        if not (0 <= nid < M):
            continue
        m = r["mask"].to(node_scores.device, dtype=torch.float)
        targets[nid] = torch.maximum(targets[nid], downsample_mask_to_grid(m, token_hw))
        valid[nid] = True
    return targets, valid


# ─────────────────────────────────────────────────────────────────────────────
# Backbone wrapper: EVA-ViT-G (frozen) + Q-Former BERT (frozen, text-only path)
# ─────────────────────────────────────────────────────────────────────────────

class FrozenBackbone(Blip2Base):
    """Minimal Blip2Base subclass that only initialises the bits we need."""

    def __init__(self, img_size: int = 224, vit_precision: str = "fp16",
                 max_txt_len: int = 32):
        super().__init__()
        self.max_txt_len = max_txt_len
        self.tokenizer = self.init_tokenizer()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            "eva_clip_g", img_size, drop_path_rate=0.0,
            use_grad_checkpoint=False, precision=vit_precision,
        )
        # init_Qformer signature: (num_query_token, vision_width, cross_attention_freq)
        self.Qformer, self.query_tokens = self.init_Qformer(
            32, self.visual_encoder.num_features, 2,
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        # Freeze everything (we only train the localizer).
        for p in self.visual_encoder.parameters():
            p.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()
        self.visual_encoder.train = disabled_train
        for p in self.Qformer.parameters():
            p.requires_grad = False
        self.Qformer = self.Qformer.eval()
        self.Qformer.train = disabled_train

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def extract_patches(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        with self.maybe_autocast():
            tokens = self.visual_encoder(images)              # [B, 1+N, 1408]
            ln = self.ln_vision(tokens)
        patches = ln[:, 1:, :].float()                         # [B, N, 1408]
        N = patches.shape[1]
        h = int(N ** 0.5)
        return patches, (h, h)

    @torch.no_grad()
    def encode_nodes(self, graphs):
        """Same call as SSN_v1's _encode_obj_nodes equivalent, via shared util."""
        return encode_graph_node_features(
            tokenizer=self.tokenizer,
            bert_model=self.Qformer.bert,
            graphs=graphs,
            device=self.device,
            max_txt_len=self.max_txt_len,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(backbone: FrozenBackbone, localizer: ObjectLocalizer,
              loader: DataLoader, bce: nn.Module, dice: nn.Module,
              bce_w: float, dice_w: float, device: torch.device,
              optimizer=None) -> Dict[str, float]:
    train = optimizer is not None
    localizer.train(train)
    running = {"loss": 0.0, "bce": 0.0, "dice": 0.0,
               "valid_samples": 0, "supervised_nodes": 0, "batches": 0}
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        patches, hw = backbone.extract_patches(images)         # no_grad
        node_out = backbone.encode_nodes(batch["graphs"])       # no_grad
        node_feats = node_out["node_features"].to(device)
        node_mask = node_out["node_mask"].to(device)
        scores = localizer(patches, node_feats, node_mask)      # [B, M, N] logits
        h, w = hw

        sample_losses = []
        for b_idx, regions in enumerate(batch["regions"]):
            n_nodes = int(node_mask[b_idx].sum().item())
            if n_nodes == 0 or not regions:
                continue
            ns = scores[b_idx, :n_nodes].view(n_nodes, h, w)    # [M', h, w]
            target, valid = build_supervision(ns, regions, (h, w))
            if not valid.any():
                continue
            pred = ns[valid]
            tgt = target[valid]
            l_bce = bce(pred, tgt)
            l_dice = dice(pred, tgt).mean()
            sample_losses.append(bce_w * l_bce + dice_w * l_dice)
            running["bce"] += float(l_bce.detach())
            running["dice"] += float(l_dice.detach())
            running["valid_samples"] += 1
            running["supervised_nodes"] += int(valid.sum().item())

        if not sample_losses:
            continue
        loss = torch.stack(sample_losses).mean()
        running["loss"] += float(loss.detach())
        running["batches"] += 1
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(localizer.parameters(), 1.0)
            optimizer.step()
    denom = max(running["batches"], 1)
    return {
        "loss": running["loss"] / denom,
        "bce":  running["bce"]  / max(running["valid_samples"], 1),
        "dice": running["dice"] / max(running["valid_samples"], 1),
        "valid_samples_per_batch": running["valid_samples"] / denom,
        "supervised_nodes_per_batch": running["supervised_nodes"] / denom,
    }


def parse_args():
    p = argparse.ArgumentParser()
    DR = "/data/lj/CIR/datasets/selector_dataset"
    p.add_argument("--train_json", default=f"{DR}/train/train_allclass_grounding.json")
    p.add_argument("--val_json",   default=f"{DR}/val/val_allclass_grounding.json")
    p.add_argument("--out",        default="weight/selector_eva.pt")
    p.add_argument("--gpu",        default="0")
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers",    type=int, default=6)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--bce_weight",  type=float, default=1.0)
    p.add_argument("--dice_weight", type=float, default=1.0)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dropout",   type=float, default=0.1)
    p.add_argument("--max_txt_len", type=int, default=32)
    p.add_argument("--seed",      type=int, default=3407)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out_path.parent / f"selector_train_log_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_dir / "train.log"),
                  logging.StreamHandler(sys.stdout)],
    )
    logging.info(f"args: {vars(args)}")

    train_ds = GraphGroundingDataset([args.train_json])
    val_ds   = GraphGroundingDataset([args.val_json])
    logging.info(f"train samples: {len(train_ds)} | val samples: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True, collate_fn=collate)

    backbone = FrozenBackbone(img_size=224, max_txt_len=args.max_txt_len).to(device)
    if os.path.isfile("weight/blip2_pretrained.pth"):
        backbone.load_from_pretrained("weight/blip2_pretrained.pth")
        logging.info("loaded weight/blip2_pretrained.pth (ViT + Q-Former BERT)")
    localizer = ObjectLocalizer(
        vit_dim=1408, node_dim=backbone.Qformer.config.hidden_size,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(localizer.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    best_val = float("inf")
    for epoch in range(args.epochs):
        tr = run_epoch(backbone, localizer, train_loader, bce, dice,
                       args.bce_weight, args.dice_weight, device, optimizer)
        va = run_epoch(backbone, localizer, val_loader, bce, dice,
                       args.bce_weight, args.dice_weight, device, None)
        logging.info(
            f"epoch {epoch+1}/{args.epochs}  "
            f"train loss={tr['loss']:.4f} bce={tr['bce']:.4f} dice={tr['dice']:.4f}  "
            f"val loss={va['loss']:.4f} bce={va['bce']:.4f} dice={va['dice']:.4f}  "
            f"sup_nodes/batch={tr['supervised_nodes_per_batch']:.1f}"
        )
        if va["loss"] < best_val:
            best_val = va["loss"]
            ckpt = {
                "model": localizer.state_dict(),
                "epoch": epoch + 1,
                "val_loss": va["loss"],
                "config": vars(args),
            }
            torch.save(ckpt, out_path)
            logging.info(f"saved best -> {out_path} (val loss {va['loss']:.4f})")
    logging.info(f"done. best val loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
