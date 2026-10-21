"""Precompute BLIP2 ViT features for the PatternNet (patterncir) dataset.

train_cirr_blip2.py / test_cirr_blip2.py consume precomputed ViT features
(features/<dataset>/<split>_images.pt + <split>_name_to_idx.json) instead of
raw images. CustomCIRR features already exist; this script generates the
equivalent for patterncir.

The ViT backbone and transform must match what training expects:
  - backbone: the same visual_encoder used by models.ssn_blip2.Model, loaded
    from weight/blip2_pretrained.pth (so stored features == visual_encoder
    output, BEFORE ln_vision, exactly as the trainer assumes).
  - transform: targetpad_transform(1.25, 224), the same one used to build the
    CustomCIRR features.

Run from the SSN_v1 repo root, e.g.:
    CUDA_VISIBLE_DEVICES=6 python src/precompute_patterncir.py \\
        --splits train val test1 --batch_size 128 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data_utils import PatternCIRDataset, targetpad_transform


def collate(batch):
    batch = [b for b in batch if b is not None]
    names = [b[0] for b in batch]
    images = torch.stack([b[1] for b in batch], dim=0)
    return names, images


def build_vit(device):
    from models.ssn_blip2 import Model
    model = Model()
    pretrained = 'weight/blip2_pretrained.pth'
    if os.path.isfile(pretrained):
        model.load_from_pretrained(pretrained)
        logging.info(f'Loaded BLIP2 pretrained weights from {pretrained}')
    else:
        raise FileNotFoundError(
            f'Pretrained weight not found at {pretrained}. Run from the repo '
            f'root so the relative path resolves.'
        )
    model.eval()
    return model.to(device)


@torch.no_grad()
def compute_split(model, split, batch_size, num_workers, device):
    preprocess = targetpad_transform(1.25, 224)
    dataset = PatternCIRDataset(split, 'classic', lambda im: {'pixel_values': preprocess(im)})
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True, drop_last=False, shuffle=False, collate_fn=collate)

    logging.info(f'[{split}] total gallery images: {len(dataset)}')
    feats = []
    names = []
    for batch_names, images in tqdm(loader, ncols=100, desc=split):
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            emb = model.visual_encoder(images)
        feats.append(emb.cpu())
        names.extend(batch_names)
    feats = torch.cat(feats, dim=0)
    name_to_idx = {n: i for i, n in enumerate(names)}
    logging.info(f'[{split}] feature shape={tuple(feats.shape)} dtype={feats.dtype}')
    return feats, name_to_idx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test1'],
                        choices=['train', 'val', 'test1'])
    parser.add_argument('--feature_dir', type=str, default='features/patterncir')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        stream=sys.stdout)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    model = build_vit(device)
    os.makedirs(args.feature_dir, exist_ok=True)
    for split in args.splits:
        feats, name_to_idx = compute_split(model, split, args.batch_size,
                                           args.num_workers, device)
        torch.save(feats, os.path.join(args.feature_dir, f'{split}_images.pt'))
        with open(os.path.join(args.feature_dir, f'{split}_name_to_idx.json'), 'w') as f:
            json.dump(name_to_idx, f)
        logging.info(f'[{split}] saved to {args.feature_dir}')
    logging.info('Precomputation done')
