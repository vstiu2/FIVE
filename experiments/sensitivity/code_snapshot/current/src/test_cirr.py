import json
import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, harmonic_mean, geometric_mean

import torch

import models
from data_utils import CIRRDataset, CustomCIRRDataset, PatternCIRDataset
from utils import device
from validate import compute_cirr_val_metrics

logs_root = Path('/data/lj/SSN/logs')


def test(args):
    # Load training args to ensure model architecture matches checkpoint
    run_dir = Path(args.model_path).parent.parent
    args_file = run_dir / 'args.json'
    if args_file.exists():
        with open(args_file) as f:
            train_args = json.load(f)
        for k, v in train_args.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)
        # always override arch-related params from training args
        for k in ('projection_dim', 'hidden_dim', 'kl_weight', 'pretrained_model_name'):
            if k in train_args:
                setattr(args, k, train_args[k])
        print(f"Loaded training args from {args_file}")
    else:
        print(f"Warning: {args_file} not found, using command-line args (may cause size mismatch)")

    model = models.setup(args).to(device, non_blocking=True)

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model_name = model.__class__.__name__
    model.load_state_dict(checkpoint[model_name])
    model.eval()

    # Setup logging
    log_dir = Path(args.model_path).parent.parent  # .../logs/<dataset>/<run>/saved_models -> <run>/
    log_file = log_dir / 'test.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    log = logging.getLogger()
    log.info(f"Loaded checkpoint from {args.model_path}")

    def preprocess(im):
        return model.processor(images=im, return_tensors='pt')

    if args.dataset == 'CustomCIRR':
        classic_dataset = CustomCIRRDataset(args.split, 'classic', preprocess)
        relative_dataset = CustomCIRRDataset(args.split, 'relative', preprocess)
    elif args.dataset == 'patterncir':
        classic_dataset = PatternCIRDataset(args.split, 'classic', preprocess)
        relative_dataset = PatternCIRDataset(args.split, 'relative', preprocess)
    else:
        classic_dataset = CIRRDataset(args.split, 'classic', preprocess)
        relative_dataset = CIRRDataset(args.split, 'relative', preprocess)

    results = compute_cirr_val_metrics(classic_dataset, relative_dataset, model)
    group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results

    results_dict = {
        'group_recall_at1': group_recall_at1,
        'group_recall_at2': group_recall_at2,
        'group_recall_at3': group_recall_at3,
        'recall_at1': recall_at1,
        'recall_at5': recall_at5,
        'recall_at10': recall_at10,
        'recall_at50': recall_at50,
        'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
        'arithmetic_mean': mean(results),
        'harmonic_mean': harmonic_mean(results),
        'geometric_mean': geometric_mean(results),
    }

    log.info('\n' + json.dumps(results_dict, indent=4))

    # Save results to json
    result_file = log_dir / f"test_results_{args.split}_{Path(args.model_path).stem}.json"
    with open(result_file, 'w') as f:
        json.dump(results_dict, f, indent=4)
    log.info(f"Results saved to {result_file}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='CustomCIRR')
    parser.add_argument("--split", type=str, default='test', choices=['val', 'test', 'test1'])
    parser.add_argument("--model", type=str, default='ssn')
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .pt checkpoint file")
    parser.add_argument("--pretrained_model_name", default="/data/lj/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", type=str)
    parser.add_argument("--projection_dim", default=1024, type=int)
    parser.add_argument("--hidden_dim", default=2048, type=int)
    parser.add_argument("--kl_weight", default=1.0, type=float)

    args = parser.parse_args()
    test(args)
