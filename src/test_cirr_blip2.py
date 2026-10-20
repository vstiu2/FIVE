"""Test script for SSN-BLIP2 on CustomCIRR.

Usage:
    CUDA_VISIBLE_DEVICES=1 python src/test_cirr_blip2.py \
        --model_path logs/SSN_BLIP2/CustomCIRR/ssn_blip2_v4/best_model.pth \
        --split val

    CUDA_VISIBLE_DEVICES=1 python src/test_cirr_blip2.py \
        --model_path logs/SSN_BLIP2/CustomCIRR/ssn_blip2_v4/best_model.pth \
        --split test
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data_utils import CustomCIRRDataset, CIRRDataset, PatternCIRDataset, squarepad_transform
from train_cirr_blip2 import FeatureDataset, cir_collate_fn


def _get_gallery_names(dataset):
    if hasattr(dataset, 'image_paths'):          # CustomCIRRDataset
        return list(dataset.image_paths)
    if hasattr(dataset, 'name_to_relpath'):      # CIRRDataset
        return list(dataset.name_to_relpath.keys())
    if hasattr(dataset, 'image_names'):          # PatternCIRDataset / FashionIQDataset
        return list(dataset.image_names)
    raise AttributeError(f"Cannot find gallery names on {type(dataset)}")


def run_eval(model, dataset: FeatureDataset, device, split: str):
    """Compute R@K metrics.  Works for both val and test splits."""
    model.eval()
    batch_sz = 256

    # ── Gallery embeddings ────────────────────────────────────────────────────
    all_names = _get_gallery_names(dataset.dataset)
    F_gallery = []
    for i in range(0, len(all_names), batch_sz):
        names_batch = all_names[i: i + batch_sz]
        raw = dataset.get_image_features(names_batch).to(device)
        with torch.no_grad():
            raw_f = raw.to(next(model.parameters()).dtype)
            emb_ln = model.ln_vision(raw_f)
            F_gallery.append(model._qformer_image_only(emb_ln, model.query_tokens))
    F_gallery = torch.cat(F_gallery, dim=0)       # [N, 32, H]
    name_to_gidx = {n: i for i, n in enumerate(all_names)}

    # ── Query dataset with graphs ─────────────────────────────────────────────
    inner_ds = dataset.dataset

    class QueryDataset(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            item = self.ds[idx]         # (ref_path, tgt_path, caption, candidates)
            graph = {}
            if hasattr(self.ds, 'triplets') and idx < len(self.ds.triplets):
                graph = self.ds.triplets[idx].get('graph', {})
            return item[0], item[1], item[2], graph, item[3]

    query_ds = QueryDataset(inner_ds)
    loader = torch.utils.data.DataLoader(
        query_ds, batch_size=32, shuffle=False,
        num_workers=4, collate_fn=cir_collate_fn,
    )

    distances = []
    ref_names_all = []
    tgt_names_all = []
    group_members_all = []

    for batch in tqdm(loader, ncols=120, desc=f'{split}'):
        if batch is None:
            continue
        ref_names, tgt_names, captions, graphs, group_members = batch

        raw = dataset.get_image_features(ref_names).to(device)
        with torch.no_grad():
            raw_f = raw.to(next(model.parameters()).dtype)
            emb_ln = model.ln_vision(raw_f)
            dist = model.inference(emb_ln, F_gallery, list(captions), graphs)
        distances.append(dist.cpu())
        ref_names_all.extend(ref_names)
        tgt_names_all.extend(tgt_names)
        if isinstance(group_members[0], (list, tuple)):
            group_members_all.extend([list(g) for g in group_members])
        else:
            group_members_all.extend([[g] for g in group_members])

    distances = torch.cat(distances, dim=0)   # [Q, N]

    # Mask reference image
    for i, rn in enumerate(ref_names_all):
        if rn in name_to_gidx:
            distances[i, name_to_gidx[rn]] = -1e10

    sorted_idx = torch.argsort(distances, dim=-1, descending=True).cpu()
    sorted_names = np.array(all_names)[sorted_idx.numpy()]

    labels = (sorted_names == np.array(tgt_names_all)[:, None])
    r1  = float(labels[:, :1].sum()  / len(labels)) * 100
    r5  = float(labels[:, :5].sum()  / len(labels)) * 100
    r10 = float(labels[:, :10].sum() / len(labels)) * 100
    r50 = float(labels[:, :50].sum() / len(labels)) * 100

    # Group recall (subset accuracy): target must be rank-1 among group_members only
    group_hits1 = group_hits2 = group_hits3 = 0
    for i, (tgt, members) in enumerate(zip(tgt_names_all, group_members_all)):
        member_set = set(members)
        group_sorted = [n for n in sorted_names[i] if n in member_set]
        if len(group_sorted) == 0:
            continue
        if group_sorted[0] == tgt:
            group_hits1 += 1
        if tgt in group_sorted[:2]:
            group_hits2 += 1
        if tgt in group_sorted[:3]:
            group_hits3 += 1
    n = len(tgt_names_all)
    gs1 = group_hits1 / n * 100
    gs2 = group_hits2 / n * 100
    gs3 = group_hits3 / n * 100

    return {
        'R@1': r1, 'R@5': r5, 'R@10': r10, 'R@50': r50,
        'acc': (r1 + r5 + r10 + r50) / 4,
        'GS@1': gs1, 'GS@2': gs2, 'GS@3': gs3,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='CustomCIRR',
                        choices=['CustomCIRR', 'CIRR', 'patterncir'])
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test', 'test1'])
    parser.add_argument('--feature_dir', type=str, default='features/CustomCIRR')
    parser.add_argument('--gpu', type=str, default='0')

    # ── Ablation switches (all default OFF → bit-identical to baseline) ───
    parser.add_argument('--ablate_no_trans', action='store_true')
    parser.add_argument('--ablate_no_fl', action='store_true')
    parser.add_argument('--ablate_no_ff', action='store_true')
    parser.add_argument('--ablate_force_route', type=str, default='',
                        choices=['', 'Remove', 'Add', 'Move'])
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    log_dir = Path(args.output_dir) if args.output_dir else Path(args.model_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'test_{args.split}.log'),
            logging.StreamHandler(sys.stdout),
        ]
    )

    # ── Load model ────────────────────────────────────────────────────────────
    from models.ssn_blip2 import Model
    model = Model()
    pretrained = os.environ.get('FIVE_BLIP2_PRETRAINED', 'weight/blip2_pretrained.pth')
    if os.path.isfile(pretrained):
        model.load_from_pretrained(pretrained)
    state = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state)
    model = model.to(device)
    model.set_ablation(
        no_trans=args.ablate_no_trans,
        no_fl=args.ablate_no_fl,
        no_ff=args.ablate_no_ff,
        force_route=args.ablate_force_route,
    )
    logging.info(f'Loaded checkpoint: {args.model_path}')

    # ── Dataset ───────────────────────────────────────────────────────────────
    preprocess = squarepad_transform(224)
    if args.dataset == 'CustomCIRR':
        inner = CustomCIRRDataset(args.split, 'relative', preprocess)
    elif args.dataset == 'CIRR':
        inner = CIRRDataset(args.split, 'relative', preprocess)
    elif args.dataset == 'patterncir':
        inner = PatternCIRDataset(args.split, 'relative', preprocess, with_graph=True)
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')
    ds = FeatureDataset(inner)
    ds.load_image_features(args.feature_dir)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = run_eval(model, ds, device, args.split)
    metrics['mean_r5_gs1'] = (metrics['R@5'] + metrics['GS@1']) / 2

    logging.info('\n' + json.dumps(metrics, indent=2))

    result_file = log_dir / f'test_results_{args.split}.json'
    with open(result_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f'Results saved to {result_file}')


if __name__ == '__main__':
    main()
