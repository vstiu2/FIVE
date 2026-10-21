"""Training entry point for SSN-BLIP2-v8.

跟 v7 几乎一致, 只把模型类换成 V8Model, 加 4 个 v8 CLI flag:
  --top_k_fl / --lambda_fl / --lambda_ff / --fl_multi / --ff_multi.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import train_cirr_blip2 as tcb                                  # noqa: E402
from train_cirr_blip2 import (                                  # noqa: E402
    FeatureDataset, cir_collate_fn, compute_val_metrics, set_seed,
)
from data_utils import CustomCIRRDataset, CIRRDataset, PatternCIRDataset  # noqa: E402
from SSN_v8.ssn_blip2_v8 import Model as V8Model                # noqa: E402


def _build_model(args, device):
    model = V8Model(
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
        # v6-stage
        stage_switch_epoch=args.stage_switch_epoch,
        stage2_mask_target_w=args.stage2_mask_target_w,
        stage2_lambda_kl_action=args.stage2_lambda_kl_action,
        # v8
        top_k_fl=args.top_k_fl,
        lambda_fl=args.lambda_fl,
        lambda_ff=args.lambda_ff,
        fl_multi=args.fl_multi,
        ff_multi=args.ff_multi,
    )
    pretrained = 'weight/blip2_pretrained.pth'
    if os.path.isfile(pretrained):
        model.load_from_pretrained(pretrained)
        logging.info(f'Loaded BLIP2 pretrained weights from {pretrained}')
    else:
        logging.warning(f'Pretrained weight not found at {pretrained}.')

    original_set_ablation = model.set_ablation
    no_mask_flag = bool(args.ablate_no_action_mask)
    no_inject_flag = bool(args.ablate_no_semantic_inject)

    def _wrapped(no_trans=False, no_fl=False, no_ff=False, force_route=""):
        original_set_ablation(
            no_trans=no_trans, no_fl=no_fl, no_ff=no_ff,
            force_route=force_route,
            no_action_mask=no_mask_flag,
            no_semantic_inject=no_inject_flag,
        )
    model.set_ablation = _wrapped  # type: ignore[assignment]
    return model.to(device)


CUSTOM_CIRR_ROOT = Path(
    "/data/lj/CIR/datasets/merged_query_text_with_graph_allclass_normalized"
)


def _inject_object_core(inner_ds, split: str) -> None:
    if not hasattr(inner_ds, "triplets"):
        logging.warning(f"[v8] dataset has no `triplets` attribute; skip injection")
        return
    src = CUSTOM_CIRR_ROOT / split / f"{split}_query_text_with_graph_object_core.json"
    if not src.exists():
        logging.warning(f"[v8] {src} not found")
        return
    with open(src) as f:
        new_triplets = json.load(f)
    if len(new_triplets) != len(inner_ds.triplets):
        logging.warning(f"[v8] length mismatch on {split}, skip")
        return
    inner_ds.triplets = new_triplets
    n_with_core = sum(1 for t in new_triplets if t.get("graph", {}).get("object_core"))
    logging.info(f"[v8] {split}: injected object_core into "
                 f"{n_with_core}/{len(new_triplets)} triplets")


def train(args):
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = Path('logs') / 'SSN_BLIP2' / args.dataset / (args.exp_name or timestamp)
    log_dir.mkdir(parents=True, exist_ok=True)

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

    model = _build_model(args, device)
    model.set_ablation(
        no_trans=args.ablate_no_trans,
        no_fl=args.ablate_no_fl,
        no_ff=args.ablate_no_ff,
        force_route=args.ablate_force_route,
    )

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f'--resume path not found: {args.resume}')
        logging.info(f'Resuming weights from {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing:
            logging.info(f'  missing keys: {len(missing)}  (first 5: {missing[:5]})')
        if unexpected:
            logging.info(f'  unexpected keys: {len(unexpected)}  (first 5: {unexpected[:5]})')
        logging.info('Resume load complete.')

    from data_utils import squarepad_transform
    preprocess = squarepad_transform(224)

    if args.dataset == 'CustomCIRR':
        train_inner = CustomCIRRDataset('train', 'relative', preprocess)
        val_inner   = CustomCIRRDataset('val',   'relative', preprocess)
        _inject_object_core(train_inner, 'train')
        _inject_object_core(val_inner,   'val')
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

    class TrainPairsDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            if hasattr(self.dataset, 'triplets'):
                entry = self.dataset.triplets[index]
                ref_name = entry.get('ref_image', entry.get('reference'))
                tgt_name = entry.get('tgt_image', entry.get('target_hard'))
                caption = entry.get('query_text', entry.get('caption'))
                graph = entry.get('graph', {})
                candidate = entry.get('candidate', [])
                if not candidate:
                    img_set = entry.get('img_set', {})
                    if isinstance(img_set, dict):
                        candidate = img_set.get('members', []) or []
                group_hard = [c for c in candidate if c != ref_name and c != tgt_name]
                return ref_name, tgt_name, caption, graph, group_hard
            item = self.dataset[index]
            if len(item) >= 3:
                ref_name, tgt_name, caption = item[:3]
                graph = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
                return ref_name, tgt_name, caption, graph, []
            raise ValueError('Unsupported training item format')

    train_pairs = TrainPairsDataset(train_inner)
    train_loader = torch.utils.data.DataLoader(
        train_pairs, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        collate_fn=cir_collate_fn,
    )

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
        betas=(0.9, 0.98), eps=1e-7, weight_decay=args.weight_decay,
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

    best_arithmetic = 0.0
    early_stop_patience = args.early_stop_patience
    early_stop_best = 0.0
    early_stop_counter = 0
    swa_state = None
    swa_count = 0
    swa_start = (args.num_epochs - args.swa_last_n) if args.swa_last_n > 0 else args.num_epochs

    log_frame = pd.DataFrame()

    for epoch in range(args.num_epochs):
        model._current_epoch.fill_(epoch)
        stage = "stage2" if epoch >= args.stage_switch_epoch else "stage1"
        if epoch == 0 or epoch == args.stage_switch_epoch:
            mw = (args.stage2_mask_target_w if stage == 'stage2'
                  else args.mask_target_w)
            kw = (args.stage2_lambda_kl_action if stage == 'stage2'
                  else args.lambda_kl_action)
            logging.info(
                f'[v8-stage] epoch={epoch} entering {stage} '
                f'(mask_target_w={mw}, lambda_kl_action={kw})'
            )

        warmup = epoch < args.warmup_epoch
        if warmup or args.aux_ramp_epochs <= 0:
            ramp = 1.0
        else:
            graph_epoch = epoch - args.warmup_epoch
            ramp = min(1.0, (graph_epoch + 1) / float(args.aux_ramp_epochs))
        model.aux_ramp.fill_(ramp)
        model.train()
        running = {'images': 0}

        bar = tqdm(train_loader, dynamic_ncols=True, leave=True,
                   mininterval=(1.0 if sys.stdout.isatty() else 60.0))
        for batch in bar:
            if batch is None:
                continue
            ref_names, tgt_names, captions, graphs, group_hards = batch

            ref_feats = train_ds.get_image_features(ref_names).to(device, non_blocking=True)
            tgt_feats = train_ds.get_image_features(tgt_names).to(device, non_blocking=True)

            if args.patch_dropout > 0.0 and model.training:
                p = args.patch_dropout
                cls = ref_feats[:, :1, :]
                patches = ref_feats[:, 1:, :]
                keep = (torch.rand(patches.shape[0], patches.shape[1], 1,
                                    device=patches.device) > p).to(patches.dtype)
                patches = patches * keep / (1.0 - p)
                ref_feats = torch.cat([cls, patches], dim=1)

            K = 4
            B = len(ref_names)
            hard_names = []
            hard_mask = torch.zeros(B, K, dtype=torch.bool)
            for i, gh in enumerate(group_hards):
                gh = list(gh)[:K]
                for j, name in enumerate(gh):
                    hard_names.append(name)
                    hard_mask[i, j] = True
                for _ in range(K - len(gh)):
                    hard_names.append(ref_names[i])
            hard_feats = train_ds.get_image_features(hard_names).to(device, non_blocking=True)
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
            bar.set_description(f'[{epoch+1}/{args.num_epochs}] warm={warmup} {stage}')
            bar.set_postfix_str('  '.join(desc_parts))

        row = {'epoch': epoch, 'warmup': warmup, 'stage': stage}
        for k in loss_dict:
            row[k] = running[k] / max(running['images'], 1)
        row['gamma'] = model.gamma.item()
        row['aux_ramp'] = float(model.aux_ramp.item())
        log_frame = pd.concat([log_frame, pd.DataFrame([row])])
        log_frame.to_csv(log_dir / 'train_metrics.csv', index=False)

        cur_lr = scheduler.get_last_lr()[0]
        train_summary = '  '.join(f'{k}={row[k]:.4f}' for k in loss_dict)
        train_summary += f'  gamma={row["gamma"]:.4f}'
        logging.info(
            f'[{epoch+1}/{args.num_epochs}] warm={warmup} {stage}  {train_summary}  lr={cur_lr:.2e}'
        )

        model.eval()
        with torch.no_grad():
            val_metrics = compute_val_metrics(
                model, val_ds, device, graph_enabled=(not warmup)
            )
        logging.info(
            f'Epoch {epoch+1}  '
            + '  '.join(f'{k}={v:.2f}' for k, v in val_metrics.items())
        )

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
                logging.info(f'Early stopping triggered at epoch {epoch+1}.')
                break

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
                        swa_state[k].add_((v_cpu - swa_state[k]) / swa_count)
            logging.info(f'SWA: epoch {epoch+1} folded in (count={swa_count})')

    if args.save_training and swa_state is not None:
        live_state = model.state_dict()
        swa_save = {
            k: swa_state[k].to(dtype=live_state[k].dtype)
            for k in swa_state
        }
        torch.save(swa_save, log_dir / 'swa_model.pth')
        logging.info(f'Saved swa_model.pth  (averaged over {swa_count} epochs)')

    if args.save_training:
        torch.save(model.state_dict(), log_dir / 'last_model.pth')

    logging.info(f'Training done.  best_arithmetic={best_arithmetic:.4f}')
    return best_arithmetic


_V8_FLAGS = {
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
    "--mask_target_lo": dict(type=float, default=0.4),
    "--mask_target_hi": dict(type=float, default=0.7),
    "--mask_target_w": dict(type=float, default=1.0),
    "--mask_div_w": dict(type=float, default=0.1),
    "--lambda_trans_align": dict(type=float, default=0.1),
    "--f_trans_via_fusion": dict(
        type=lambda x: x.lower() in ("1", "true", "yes", "y"), default=True),
    "--f_trans_via_gamma": dict(
        type=lambda x: x.lower() in ("1", "true", "yes", "y"), default=True),
    "--stage_switch_epoch": dict(type=int, default=25),
    "--stage2_mask_target_w": dict(type=float, default=2.0),
    "--stage2_lambda_kl_action": dict(type=float, default=0.05),
    # v8
    "--top_k_fl": dict(type=int, default=4,
        help="[v8] Number of top-K patches kept as separate Fl tokens."),
    "--lambda_fl": dict(type=float, default=0.1,
        help="[v8] Weight of Fl direct retrieval InfoNCE supervision."),
    "--lambda_ff": dict(type=float, default=0.1,
        help="[v8] Weight of Ff direct retrieval InfoNCE supervision."),
    "--fl_multi": dict(
        type=lambda x: x.lower() in ("1", "true", "yes", "y"), default=True,
        help="[v8] Use multi-token Fl output. False = single-token v7 behavior."),
    "--ff_multi": dict(
        type=lambda x: x.lower() in ("1", "true", "yes", "y"), default=True,
        help="[v8] Use multi-token (per-band) Ff output. False = single-token."),
}


def parse_args() -> argparse.Namespace:
    show_help = ('--help' in sys.argv) or ('-h' in sys.argv)
    if show_help:
        print("SSN_v8 flags:")
        for flag, spec in _V8_FLAGS.items():
            default = spec.get("default", "")
            print(f"  {flag:<32} (default={default!r})")
        print()

    extra_values: dict = {}
    cleaned: list[str] = []
    i = 0
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok in _V8_FLAGS:
            spec = _V8_FLAGS[tok]
            if spec.get("action") == "store_true":
                extra_values[tok] = True
                i += 1
                continue
            extra_values[tok] = spec["type"](sys.argv[i + 1])
            i += 2
            continue
        if any(tok.startswith(k + "=") for k in _V8_FLAGS):
            key, val = tok.split("=", 1)
            spec = _V8_FLAGS[key]
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

    for flag, spec in _V8_FLAGS.items():
        name = flag.lstrip("-")
        default = spec.get("default", False if spec.get("action") == "store_true" else None)
        setattr(args, name, extra_values.get(flag, default))
    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)
