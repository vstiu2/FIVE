"""Training script for SSN-BLIP2 on CustomCIRR / CIRR / FashionIQ.

Usage:
    python train_cirr_blip2.py --dataset CustomCIRR --exp_name my_run \\
        --num_epochs 30 --warmup_epoch 3 --batch_size 128 --lr 1e-5 --gpu 0

Warmup strategy:
    Epochs [0, warmup_epoch):  Q-Former fusion only (no graph decomposition,
                               no multi-scale branches, no aux loss).
    Epochs [warmup_epoch, end): Full model with graph branches enabled.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import random
from statistics import mean
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from pathlib import Path

# ── Make local src importable ─────────────────────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ── Dataset & evaluation from TME (copied into SSN_v1/src/) ──────────────────
from data_utils import CustomCIRRDataset, CIRRDataset, PatternCIRDataset


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper: adds precomputed ViT feature loading
# We reuse CustomCIRRDataset but add feature loading on top.
# ─────────────────────────────────────────────────────────────────────────────

class FeatureDataset:
    """Thin wrapper around a dataset that provides precomputed ViT features."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.images = None          # [N, 1+patch, 1408]
        self.name_to_idx = None

    def load_image_features(self, feature_dir: str):
        split = self.dataset.split if hasattr(self.dataset, 'split') else 'train'
        feat_path = Path(feature_dir) / f'{split}_images.pt'
        idx_path  = Path(feature_dir) / f'{split}_name_to_idx.json'
        logging.info(f'Loading features from {feat_path}')
        self.images = torch.load(feat_path, map_location='cpu')
        with open(idx_path) as f:
            self.name_to_idx = json.load(f)
        logging.info(f'Loaded {self.images.shape[0]} image features.')

    def get_image_features(self, names):
        indices = [self.name_to_idx[n] for n in names]
        return self.images[indices]

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


# ─────────────────────────────────────────────────────────────────────────────
# collate_fn
# ─────────────────────────────────────────────────────────────────────────────

def cir_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    transposed = list(zip(*batch))
    result = []
    for col in transposed:
        first = col[0]
        # dicts (graphs), strings, lists (group_members / candidate paths) → keep as list
        if isinstance(first, (dict, list)):
            result.append(list(col))
        else:
            try:
                result.append(default_collate(list(col)))
            except Exception:
                result.append(list(col))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def _get_gallery_names(dataset):
    """Return the list of gallery image names for any supported dataset type."""
    if hasattr(dataset, 'image_paths'):          # CustomCIRRDataset
        return list(dataset.image_paths)
    if hasattr(dataset, 'name_to_relpath'):      # CIRRDataset
        return list(dataset.name_to_relpath.keys())
    if hasattr(dataset, 'image_names'):          # FashionIQDataset
        return list(dataset.image_names)
    raise AttributeError(f"Cannot find gallery names on {type(dataset)}")


def compute_val_metrics(model, val_dataset: FeatureDataset, device, graph_enabled: bool = True):
    """Recall + group-recall computation for CustomCIRR / CIRR validation sets."""
    model.eval()

    # ── Build gallery Q-Former tokens ────────────────────────────────────────
    batch_sz = 256
    all_names = _get_gallery_names(val_dataset.dataset)
    F_gallery = []
    for i in range(0, len(all_names), batch_sz):
        names_batch = all_names[i: i + batch_sz]
        raw = val_dataset.get_image_features(names_batch).to(device)
        with torch.no_grad():
            raw_f = raw.to(next(model.parameters()).dtype)
            emb_ln = model.ln_vision(raw_f)
            F_gallery.append(model._qformer_image_only(emb_ln, model.query_tokens))
    F_gallery = torch.cat(F_gallery, dim=0)   # [N, 32, H]
    name_to_gidx = {n: i for i, n in enumerate(all_names)}

    # ── Build a val loader that includes graphs from triplets ─────────────────
    inner_ds = val_dataset.dataset

    class ValWithGraphs(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
            self.triplets = ds.triplets if hasattr(ds, 'triplets') else None

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            item = self.ds[idx]       # (ref, tgt, caption, candidates)
            graph = {}
            if self.triplets is not None and idx < len(self.triplets):
                graph = self.triplets[idx].get('graph', {})
            return item[0], item[1], item[2], graph, item[3]

    val_graph_ds = ValWithGraphs(inner_ds)
    val_loader = torch.utils.data.DataLoader(
        val_graph_ds, batch_size=32, shuffle=False,
        num_workers=4, collate_fn=cir_collate_fn,
    )

    # ── Query loop ────────────────────────────────────────────────────────────
    distances = []
    reference_names_all = []
    target_names_all = []
    group_members_all = []

    for batch in tqdm(val_loader, ncols=120, desc='val'):
        if batch is None:
            continue
        # ValWithGraphs → (ref, tgt, caption, graph, candidates)
        ref_names, tgt_names, captions, graphs, group_members = batch

        raw = val_dataset.get_image_features(ref_names).to(device)
        with torch.no_grad():
            raw_f = raw.to(next(model.parameters()).dtype)
            emb_ln = model.ln_vision(raw_f)
            dist = model.inference(emb_ln, F_gallery, list(captions), graphs,
                                   graph_enabled=graph_enabled)  # [B, N]
        distances.append(dist.cpu())
        reference_names_all.extend(ref_names)
        target_names_all.extend(tgt_names)
        if isinstance(group_members[0], (list, tuple)):
            group_members_all.extend([list(g) for g in group_members])
        else:
            group_members_all.extend([[g] for g in group_members])

    distances = torch.cat(distances, dim=0)   # [Q, N]

    # Mask out reference image from results
    for i, rn in enumerate(reference_names_all):
        if rn in name_to_gidx:
            distances[i, name_to_gidx[rn]] = -1e10

    sorted_idx = torch.argsort(distances, dim=-1, descending=True).cpu()
    sorted_names = np.array(all_names)[sorted_idx.numpy()]   # [Q, N]

    labels = (sorted_names == np.array(target_names_all)[:, None])
    r1  = float(labels[:, :1].sum() / len(labels))  * 100
    r5  = float(labels[:, :5].sum() / len(labels))  * 100
    r10 = float(labels[:, :10].sum() / len(labels)) * 100
    r50 = float(labels[:, :50].sum() / len(labels)) * 100

    # Group recall (subset accuracy): target ranking within the 6 candidate
    # group members of each query. Mirrors the test-time GS@K computation so
    # the best-model selector can optimize the same signal as test.
    group_hits1 = group_hits2 = group_hits3 = 0
    n_groups = 0
    for i, (tgt, members) in enumerate(zip(target_names_all, group_members_all)):
        member_set = set(members)
        group_sorted = [n for n in sorted_names[i] if n in member_set]
        if len(group_sorted) == 0:
            continue
        n_groups += 1
        if group_sorted[0] == tgt:
            group_hits1 += 1
        if tgt in group_sorted[:2]:
            group_hits2 += 1
        if tgt in group_sorted[:3]:
            group_hits3 += 1
    n_groups = max(n_groups, 1)
    gs1 = group_hits1 / n_groups * 100
    gs2 = group_hits2 / n_groups * 100
    gs3 = group_hits3 / n_groups * 100

    r_acc = (r1 + r5 + r10 + r50) / 4
    return {'R@1': r1, 'R@5': r5, 'R@10': r10, 'R@50': r50,
            'acc': r_acc,
            'GS@1': gs1, 'GS@2': gs2, 'GS@3': gs3,
            # Composite score used to pick the best checkpoint; balances
            # full-gallery recall with within-group ranking which is what
            # the test set actually measures.
            'best_score': 0.7 * r_acc + 0.3 * gs1}


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(args, device):
    """Instantiate SSN-BLIP2 model with BLIP2 pretrained weights."""
    # Import here to ensure sys.path is already set up
    from models.ssn_blip2 import Model
    model = Model(
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
    # Load BLIP2 pretrained weights (ViT + QFormer)
    pretrained = os.environ.get('FIVE_BLIP2_PRETRAINED', 'weight/blip2_pretrained.pth')
    if os.path.isfile(pretrained):
        model.load_from_pretrained(pretrained)
        logging.info(f'Loaded BLIP2 pretrained weights from {pretrained}')
    else:
        logging.warning(f'Pretrained weight not found at {pretrained}, starting from scratch.')
    return model.to(device)


def train(args):
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = Path('logs') / 'SSN_BLIP2' / args.dataset / (args.exp_name or timestamp)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(
        level=logging.INFO, format=log_format,
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler(sys.stdout),
        ]
    )

    with open(log_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    set_seed(args.seed)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args, device)

    # ── Ablation switches (no-op when all defaults) ──────────────────────────
    model.set_ablation(
        no_trans=args.ablate_no_trans,
        no_fl=args.ablate_no_fl,
        no_ff=args.ablate_no_ff,
        force_route=args.ablate_force_route,
    )

    # ── Optional: resume from a saved checkpoint (weights only) ──────────────
    # Loads model state_dict from the given path. Optimizer / scheduler /
    # epoch counter are NOT restored — they restart fresh. This is the
    # standard "fine-tune from a previously trained model" workflow, since
    # v12_2's checkpoint only contains model weights.
    #
    # Recommended companion flags when resuming a fully-converged model:
    #   --warmup_epoch 0   (skip warmup; graph branches are already trained)
    #   --lr 2e-6          (lower LR for fine-tuning; default 1e-5 is too hot)
    #   --num_epochs 10    (shorter continuation run)
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f'--resume path not found: {args.resume}')
        logging.info(f'Resuming weights from {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        # Allow both raw state_dict and dicts containing it under common keys.
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing:
            logging.info(f'  missing keys: {len(missing)}  (first 5: {missing[:5]})')
        if unexpected:
            logging.info(f'  unexpected keys: {len(unexpected)}  (first 5: {unexpected[:5]})')
        logging.info('Resume load complete.')

    # ── Datasets ──────────────────────────────────────────────────────────────
    from data_utils import squarepad_transform
    preprocess = squarepad_transform(224)

    if args.dataset == 'CustomCIRR':
        train_inner = CustomCIRRDataset('train', 'relative', preprocess)
        val_inner   = CustomCIRRDataset('val',   'relative', preprocess)
    elif args.dataset == 'CIRR':
        train_inner = CIRRDataset('train', 'relative', preprocess)
        val_inner   = CIRRDataset('val',   'relative', preprocess)
    elif args.dataset == 'patterncir':
        train_inner = PatternCIRDataset('train', 'relative', preprocess, with_graph=True)
        val_inner   = PatternCIRDataset('val',   'relative', preprocess, with_graph=True)
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')

    train_ds = FeatureDataset(train_inner)
    val_ds   = FeatureDataset(val_inner)
    train_ds.load_image_features(args.feature_dir)
    val_ds.load_image_features(args.feature_dir)

    # ── DataLoader  ───────────────────────────────────────────────────────────
    # Build a stable training dataset that always yields:
    # (ref_name, tgt_name, caption, graph, group_member_names)
    class TrainPairsDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            if hasattr(dataset, 'triplets'):              # CustomCIRRDataset
                self.mode = 'custom'
            elif hasattr(dataset, 'triplets'):            # fallback
                self.mode = 'custom'
            else:
                self.mode = 'generic'

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            # CustomCIRRDataset structure
            if hasattr(self.dataset, 'triplets'):
                entry = self.dataset.triplets[index]
                ref_name = entry.get('ref_image', entry.get('reference'))
                tgt_name = entry.get('tgt_image', entry.get('target_hard'))
                caption = entry.get('query_text', entry.get('caption'))
                graph = entry.get('graph', {})
                # Hard negatives = the other group members besides ref & tgt.
                # Group typically has 6 entries; we keep all non-ref/non-tgt ones.
                candidate = entry.get('candidate', [])
                if not candidate:
                    # patterncir stores the candidate pool under img_set.members
                    img_set = entry.get('img_set', {})
                    if isinstance(img_set, dict):
                        candidate = img_set.get('members', []) or []
                group_hard = [c for c in candidate
                              if c != ref_name and c != tgt_name]
                return ref_name, tgt_name, caption, graph, group_hard

            # Generic fallback: expect native dataset already returns names/caption
            item = self.dataset[index]
            if len(item) >= 3:
                ref_name, tgt_name, caption = item[:3]
                graph = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
                return ref_name, tgt_name, caption, graph, []
            raise ValueError('Unsupported training item format')

    train_pairs = TrainPairsDataset(train_inner)

    train_loader = torch.utils.data.DataLoader(
        train_pairs,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=cir_collate_fn,
    )

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    # Two param groups so the randomly-initialized modules (decomposer,
    # selector, fft, fusion, etc.) can train at a higher peak LR than the
    # already-pretrained Qformer/projection heads.
    pretrained_prefixes = (
        'Qformer.', 'visual_encoder.', 'ln_vision.',
        'vision_proj.', 'text_proj.', 'query_tokens', 'temp',
    )
    pretrained_params, new_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(n == k or n.startswith(k) for k in pretrained_prefixes):
            pretrained_params.append(p)
        else:
            new_params.append(p)
    optimizer = torch.optim.AdamW(
        [
            {'params': pretrained_params, 'lr': args.lr},
            {'params': new_params,        'lr': args.lr * 1.5},
        ],
        betas=(0.9, 0.98),
        eps=1e-7,
        weight_decay=args.weight_decay,
    )
    logging.info(
        f'Param groups: pretrained={sum(p.numel() for p in pretrained_params):,}  '
        f'new={sum(p.numel() for p in new_params):,}'
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.lr, args.lr * 1.5],
        pct_start=min(0.3, 1.5 / args.num_epochs),
        div_factor=100.0,
        steps_per_epoch=len(train_loader),
        epochs=args.num_epochs,
    )
    scaler = torch.cuda.amp.GradScaler()

    # ── SSN-style best tracking (arithmetic only) ───────────────────────────
    best_arithmetic = 0.0

    early_stop_patience = args.early_stop_patience
    early_stop_best = 0.0
    early_stop_counter = 0

    # ── SWA (Stochastic Weight Averaging) state ────────────────────────────
    # Maintain running mean of state_dict over the last (num_epochs - swa_start)
    # epochs. After training, we save a single SWA checkpoint.
    swa_state = None
    swa_count = 0
    swa_start = (args.num_epochs - args.swa_last_n) if args.swa_last_n > 0 else args.num_epochs

    log_frame = pd.DataFrame()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.num_epochs):
        warmup = epoch < args.warmup_epoch
        # Ramp β/γ (the aux visual/transition branch weights) linearly from
        # 0 -> 1 over the first `aux_ramp_epochs` graph epochs after warmup.
        # During warmup the value is irrelevant (graph branches are off).
        if warmup or args.aux_ramp_epochs <= 0:
            ramp = 1.0
        else:
            graph_epoch = epoch - args.warmup_epoch
            ramp = min(1.0, (graph_epoch + 1) / float(args.aux_ramp_epochs))
        model.aux_ramp.fill_(ramp)
        model.train()
        running = {'images': 0}

        # In a TTY tqdm overwrites one line in place (mininterval small). When
        # output is redirected to a file/tee (non-TTY), carriage-return overwrite
        # is not rendered, so refresh rarely to avoid one log line per second.
        bar = tqdm(train_loader, dynamic_ncols=True, leave=True,
                   mininterval=(1.0 if sys.stdout.isatty() else 60.0))
        for batch in bar:
            if batch is None:
                continue
            ref_names, tgt_names, captions, graphs, group_hards = batch

            ref_feats = train_ds.get_image_features(ref_names).to(device, non_blocking=True)
            tgt_feats = train_ds.get_image_features(tgt_names).to(device, non_blocking=True)

            # ── Patch token dropout (training-time pseudo-augmentation) ────────
            # Since we use precomputed ViT features, the standard image-space
            # augmentations (RandomCrop, Flip, ColorJitter) are unavailable —
            # every epoch sees the IDENTICAL feature tensor per image, so the
            # Q-Former tends to memorize specific patch configurations.
            #
            # Patch dropout randomly zeroes a fraction of the 256 patch tokens
            # (the [CLS] token at index 0 is preserved). This forces the model
            # to rely on the FULL set of patches rather than a few salient
            # ones, mimicking the regularization effect of spatial crops.
            #
            # Applied ONLY to the reference image during training. Target and
            # hard-neg features are kept intact so the contrastive target
            # distribution is unchanged.
            if args.patch_dropout > 0.0 and model.training:
                p = args.patch_dropout
                cls = ref_feats[:, :1, :]
                patches = ref_feats[:, 1:, :]
                # Per-sample, per-patch Bernoulli mask
                keep = (torch.rand(patches.shape[0], patches.shape[1], 1,
                                    device=patches.device) > p).to(patches.dtype)
                # Scale kept patches to preserve the mean magnitude (inverted dropout)
                patches = patches * keep / (1.0 - p)
                ref_feats = torch.cat([cls, patches], dim=1)

            # Group-aware hard negatives: each anchor's same-group candidates
            # (excluding ref & tgt). Pad to K to allow stacking; padded entries
            # are masked out in the loss.
            K = 4
            B = len(ref_names)
            hard_names = []
            hard_mask = torch.zeros(B, K, dtype=torch.bool)
            for i, gh in enumerate(group_hards):
                gh = list(gh)[:K]
                for j, name in enumerate(gh):
                    hard_names.append(name)
                    hard_mask[i, j] = True
                # pad with ref name (will be masked out)
                for _ in range(K - len(gh)):
                    hard_names.append(ref_names[i])
            hard_feats = train_ds.get_image_features(hard_names).to(device, non_blocking=True)
            # [B*K, 1+N, 1408] → [B, K, 1+N, 1408]
            hard_feats = hard_feats.view(B, K, *hard_feats.shape[1:])
            hard_mask = hard_mask.to(device, non_blocking=True)

            samples = {
                'image':      ref_feats,
                'target':     tgt_feats,
                'text_input': list(captions),
                'graph':      list(graphs),
                'hard':       hard_feats,
                'hard_mask':  hard_mask,
            }

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss_dict = model(samples, warmup=warmup)

            scaler.scale(loss_dict['loss']).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = ref_feats.shape[0]
            running['images'] += bs
            for k, v in loss_dict.items():
                running.setdefault(k, 0.0)
                running[k] += v.item() * bs

            desc_parts = [
                f"{k}: {running[k] / running['images']:.3f}"
                for k in loss_dict
            ]
            bar.set_description(f'[{epoch+1}/{args.num_epochs}] warm={warmup}')
            bar.set_postfix_str('  '.join(desc_parts))

        # epoch log
        row = {'epoch': epoch, 'warmup': warmup}
        for k in loss_dict:
            row[k] = running[k] / max(running['images'], 1)
        row['gamma'] = model.gamma.item()   # effective transition weight (health check)
        row['aux_ramp'] = float(model.aux_ramp.item())
        log_frame = pd.concat([log_frame, pd.DataFrame([row])])
        log_frame.to_csv(log_dir / 'train_metrics.csv', index=False)

        cur_lr = scheduler.get_last_lr()[0]
        train_summary = '  '.join(f'{k}={row[k]:.4f}' for k in loss_dict)
        train_summary += f'  gamma={row["gamma"]:.4f}'
        logging.info(
            f'[{epoch+1}/{args.num_epochs}] warm={warmup}  {train_summary}  lr={cur_lr:.2e}'
        )

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_metrics = compute_val_metrics(
                model, val_ds, device, graph_enabled=(not warmup)
            )
        logging.info(
            f'Epoch {epoch+1}  '
            + '  '.join(f'{k}={v:.2f}' for k, v in val_metrics.items())
        )

        # ── SSN-style best save (arithmetic only) ────────────────────────────
        # Save the checkpoint that maximizes the arithmetic mean of the 7 metrics.
        seven_metrics = [
            val_metrics['GS@1'], val_metrics['GS@2'], val_metrics['GS@3'],
            val_metrics['R@1'],  val_metrics['R@5'],
            val_metrics['R@10'], val_metrics['R@50'],
        ]
        m_arith = mean(seven_metrics)
        logging.info(f'Epoch {epoch+1}  arithmetic={m_arith:.4f}')

        if args.save_training:
            if m_arith > best_arithmetic:
                best_arithmetic = m_arith
                torch.save(model.state_dict(), log_dir / 'best_arithmetic.pth')
                logging.info(f'Saved best_arithmetic.pth  (arith={m_arith:.4f})')

        if m_arith > early_stop_best:
            early_stop_best = m_arith
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            logging.info(f'Early stopping counter: {early_stop_counter}/{early_stop_patience} '
                         f'(best arithmetic={early_stop_best:.4f})')
            if early_stop_counter >= early_stop_patience:
                logging.info(f'Early stopping triggered at epoch {epoch+1}: '
                             f'no improvement on arithmetic for {early_stop_patience} validations.')
                break

        # ── SWA state update ─────────────────────────────────────────────────
        # After the LR has decayed, accumulate a running average of the
        # last N epochs' weights. The final averaged checkpoint sits in a
        # wider/flatter region of the loss surface and tends to generalize
        # better than any single late-epoch snapshot.
        if epoch >= swa_start and args.save_training:
            if swa_state is None:
                swa_state = {
                    k: v.detach().to('cpu').float().clone()
                    for k, v in model.state_dict().items()
                }
                swa_count = 1
            else:
                swa_count += 1
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        v_cpu = v.detach().to('cpu').float()
                        # Incremental mean: avg += (new - avg) / n
                        swa_state[k].add_((v_cpu - swa_state[k]) / swa_count)
            logging.info(f'SWA: epoch {epoch+1} folded in (count={swa_count})')

    # ── Save SWA checkpoint at end of training ──────────────────────────────
    if args.save_training and swa_state is not None:
        # Cast back to the dtype of each parameter in the live model so
        # state_dict load will work without dtype mismatch.
        live_state = model.state_dict()
        swa_save = {
            k: swa_state[k].to(dtype=live_state[k].dtype)
            for k in swa_state
        }
        torch.save(swa_save, log_dir / 'swa_model.pth')
        logging.info(f'Saved swa_model.pth  (averaged over {swa_count} epochs)')

    if args.save_training:
        torch.save(model.state_dict(), log_dir / 'last_model.pth')

    logging.info(
        f'Training done.  best_arithmetic={best_arithmetic:.4f}'
    )
    return best_arithmetic


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',      type=str,   default='CustomCIRR',
                        choices=['CustomCIRR', 'CIRR', 'FashionIQ', 'patterncir'])
    parser.add_argument('--exp_name',     type=str,   default='')
    parser.add_argument('--feature_dir',  type=str,   default='./features/CustomCIRR',
                        help='Directory with precomputed ViT features (.pt + .json)')
    parser.add_argument('--gpu',          type=str,   default='0')
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--num_workers',  type=int,   default=8)
    parser.add_argument('--batch_size',   type=int,   default=128)
    parser.add_argument('--num_epochs',   type=int,   default=30)
    parser.add_argument('--warmup_epoch', type=int,   default=3,
                        help='Epochs with Q-Former-only (no graph, no aux loss)')
    parser.add_argument('--lr',           type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lambda_align', type=float, default=0.0,
                        help='SPRC-style align loss weight: MSE(Fq_u, text_embed.detach()). '
                             '0.0 disables it (= v11 behavior). SPRC uses 0.4.')
    parser.add_argument('--gamma_init', type=float, default=0.1,
                        help='Initial value of gamma (γ·F_trans contribution to final_query).')
    parser.add_argument('--gamma_max', type=float, default=0.5,
                        help='Upper bound of gamma (sigmoid-parameterized: γ=gamma_max·σ(raw)).')
    parser.add_argument('--aux_ramp_epochs', type=int, default=5,
                        help='Linearly ramp β/γ from 0 to 1 over this many graph epochs '
                             'after warmup, to avoid injecting random-MLP noise the moment '
                             'graph branches turn on. 0 disables the ramp (β/γ active in full '
                             'from the first graph epoch).')
    parser.add_argument('--selector_ckpt', type=str, default='weight/selector_eva.pt',
                        help='Path to grounding-trained ObjectLocalizer checkpoint. '
                             'Loaded and frozen at init. If missing, localizer runs with '
                             'random weights (localization will be unreliable).')
    parser.add_argument('--lambda_global', type=float, default=0.0,
                        help='ComqueryFormer-style multi-scale global contrastive loss '
                             'weight: InfoNCE(Fq_u, z_tgt). 0.0 disables (= v12 behavior).')
    parser.add_argument('--lambda_colbert', type=float, default=0.0,
                        help='ColBERT-style multi-vector q_tokens↔z_tgt_tokens InfoNCE. '
                             '0.0 disables (= v12 behavior). Recommended starting value: 0.3.')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing eps for InfoNCE losses. 0.0 = v12 behavior. '
                             'Recommended: 0.1 for reducing test/val gap.')
    parser.add_argument('--alpha_init', type=float, default=0.01,
                        help='Init value for alpha (combiner branch scale). v12=0.01.')
    parser.add_argument('--beta_init',  type=float, default=0.01,
                        help='Init value for beta (Fq_fused branch scale). v12=0.01.')
    parser.add_argument('--alpha_max',  type=float, default=0.5,
                        help='Clamp upper bound for alpha. v12=0.5.')
    parser.add_argument('--beta_max',   type=float, default=0.5,
                        help='Clamp upper bound for beta. v12=0.5.')
    parser.add_argument('--max_txt_len', type=int, default=32,
                        help='Max BERT token length for captions. v12 default = 32. '
                             'CIRR caption max observed = 44; setting 48 covers full '
                             'distribution but did not improve metrics in v15.')
    parser.add_argument('--swa_last_n',  type=int, default=5,
                        help='Apply Stochastic Weight Averaging to the last N '
                             'epochs and save as swa_model.pth. 0 disables SWA. '
                             'Default 5 covers the typical late-training plateau.')
    parser.add_argument('--resume',     type=str,   default='',
                        help='Path to a checkpoint (.pth) whose model state_dict '
                             'will be loaded BEFORE training. Optimizer / '
                             'scheduler / epoch counter are NOT restored. Use '
                             'with --warmup_epoch 0 and a lower --lr for '
                             'fine-tuning a converged model.')
    parser.add_argument('--patch_dropout', type=float, default=0.0,
                        help='Training-time dropout probability applied per '
                             'reference-image patch token (CLS preserved). '
                             'Acts as a pseudo data-augmentation since we use '
                             'precomputed ViT features. 0.0 = v12 behavior. '
                             'Recommended starting value: 0.15.')
    parser.add_argument('--save_training', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=7,
                        help='Stop training if arithmetic mean does not improve for '
                             'this many consecutive validations.')

    # ── Ablation switches (all default OFF → bit-identical to baseline) ───
    parser.add_argument('--ablate_no_trans', action='store_true',
                        help='A1: drop the F_trans branch (force γ=0 inside _build_query).')
    parser.add_argument('--ablate_no_fl', action='store_true',
                        help='A3: zero out the Fl (graph-guided local) branch in vis dict.')
    parser.add_argument('--ablate_no_ff', action='store_true',
                        help='A4: zero out the Ff (FFT) branch in vis dict.')
    parser.add_argument('--ablate_force_route', type=str, default='',
                        choices=['', 'Remove', 'Add', 'Move'],
                        help='A2: force every sample to a single transition category '
                             '(keeps the branch active but removes the routing signal).')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
