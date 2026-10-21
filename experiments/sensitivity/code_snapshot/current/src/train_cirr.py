from comet_ml import Experiment
import json
import logging
import multiprocessing
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, harmonic_mean, geometric_mean
from typing import List
import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import models

from data_utils import base_path, CIRRDataset, CustomCIRRDataset, PatternCIRDataset
from utils import collate_fn, update_train_running_results, set_train_bar_description, save_model, device
from validate import compute_cirr_val_metrics


def combiner_training_cirr(args):
    """
    Train the Combiner on CIRR dataset
    """
    # Make training folder
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        '/data/lj/SSN/logs' / Path(f"{args.dataset}") / f"{args.model}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save training args for reproducibility and test-time loading
    with open(training_path / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(training_path / 'train.log'),
            logging.StreamHandler(),
        ]
    )
    log = logging.getLogger()
    # redirect print to log so all existing print() calls are captured
    import builtins
    _original_print = builtins.print
    def _print_to_log(*args, **kwargs):
        _original_print(*args, **kwargs)
        log.info(' '.join(str(a) for a in args))
    builtins.print = _print_to_log

    model = models.setup(args).to(device, non_blocking=True)

    def preprocess(im):
        return model.processor(images=im, return_tensors='pt')

    # Define the validation datasets and extract the validation index features
    if args.dataset == 'CustomCIRR':
        DatasetClass = CustomCIRRDataset
        classic_val_dataset = DatasetClass('val', 'classic', preprocess)
        relative_val_dataset = DatasetClass('val', 'relative', preprocess)
        relative_train_dataset = DatasetClass('train', 'relative', preprocess)
    elif args.dataset == 'patterncir':
        classic_val_dataset = PatternCIRDataset('val', 'classic', preprocess)
        relative_val_dataset = PatternCIRDataset('val', 'relative', preprocess)
        relative_train_dataset = PatternCIRDataset('train', 'relative', preprocess)
    else:
        classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
        relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
        relative_train_dataset = CIRRDataset('train', 'relative', preprocess)
    # results = compute_cirr_val_metrics(classic_val_dataset, relative_val_dataset, model)

    # Define the model and the train dataset
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=args.batch_size,
                                       num_workers=4,
                                       pin_memory=True, collate_fn=collate_fn, drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    clip_params, proj_params, modality_params, fusion_params, head_params = [], [], [], [], []
    for name, para in model.named_parameters():
        if 'clip' in name:
            clip_params += [para]
        elif 'token_proj' in name:
            proj_params += [para]
        elif 'token_type' in name:
            modality_params += [para]
        elif 'fusion' in name or 'token_selection' in name:
            fusion_params += [para]
        else:
            head_params += [para]
    optimizer = optim.Adam([{'params': head_params},
                            {'params': fusion_params, 'lr': args.lr * args.lr_ratio * 4},
                            {'params': modality_params, 'lr': args.lr * args.lr_ratio * 4},
                            {'params': proj_params, 'lr': args.lr * args.lr_ratio * 2},
                            {'params': clip_params, 'lr': args.lr * args.lr_ratio}
                            ], lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best results to zero
    if args.save_best:
        best_arithmetic = 0

    early_stop_patience = args.early_stop_patience
    early_stop_best = 0.0
    early_stop_counter = 0

    # Start with the training loop
    print('Training loop started')
    for epoch in range(args.num_epochs):
        with experiment.train():
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            model.train()
            train_bar = tqdm(relative_train_loader, ncols=150)
            for idx, (reference_images, target_images, texts) in enumerate(train_bar):  # Load a batch of triplets
                step = len(train_bar) * epoch + idx
                num_samples = len(texts)

                optimizer.zero_grad()

                reference_images = reference_images.to(device)
                target_images = target_images.to(device)
                text_inputs = model.tokenizer(texts, padding=True, return_tensors='pt').to(device)

                # Compute the logits and loss
                with torch.cuda.amp.autocast():
                    ground_truth = torch.arange(num_samples, dtype=torch.long, device=device)
                    loss = model(reference_images, text_inputs, target_images, ground_truth)

                # Backpropagate and update the weights
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, num_samples)
                set_train_bar_description(train_bar, epoch, args.num_epochs, train_running_results)

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            scheduler.step()
            experiment.log_metric('learning_rate', scheduler.get_last_lr()[0], epoch=epoch)

        if epoch % args.validation_frequency == 0:
            with experiment.validate():
                model.eval()

                # Compute and log validation metrics
                results = compute_cirr_val_metrics(classic_val_dataset, relative_val_dataset, model)
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
                    'geometric_mean': geometric_mean(results)
                }

                print(json.dumps(results_dict, indent=4))
                experiment.log_metrics(
                    results_dict,
                    epoch=epoch
                )

                # Save model
                if args.save_training:
                    if args.save_best and results_dict['arithmetic_mean'] > best_arithmetic:
                        best_arithmetic = results_dict['arithmetic_mean']
                        save_model('combiner_arithmetic', epoch, model, training_path)
                    if not args.save_best:
                        save_model(f'combiner_{epoch}', epoch, model, training_path)

                if results_dict['arithmetic_mean'] > early_stop_best:
                    early_stop_best = results_dict['arithmetic_mean']
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    print(f'Early stopping counter: {early_stop_counter}/{early_stop_patience} '
                          f'(best arithmetic_mean={early_stop_best:.4f})')
                    if early_stop_counter >= early_stop_patience:
                        print(f'Early stopping triggered at epoch {epoch}: '
                              f'no improvement on arithmetic_mean for {early_stop_patience} validations.')
                        break


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--model", type=str, required=True, help="could be 'FusionModel'")
    parser.add_argument("--clip_model_name", default="openai/clip-vit-base-patch32", type=str, help="CLIP model to use")
    parser.add_argument("--pretrained_model_name", default="/data/lj/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", type=str, help="Pretrained CLIP model name")
    parser.add_argument("--kl_weight", default=1.0, type=float, help="Weight for KL divergence loss")

    parser.add_argument("--projection_dim", default=1024, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden_dim", default=2048, type=int, help="Combiner hidden dim")

    parser.add_argument("--lr", default=4e-5, type=float, help="Init learning rate")
    parser.add_argument("--lr_step_size", default=5, type=int)
    parser.add_argument("--lr_gamma", default=0.1, type=float)
    parser.add_argument("--lr_ratio", default=0.2, type=float)

    parser.add_argument("--num_epochs", default=100, type=int, help="number training epochs")
    parser.add_argument("--batch_size", default=384, type=int, help="Batch size of the Combiner training")
    parser.add_argument("--validation_frequency", default=1, type=int, help="Validation frequency expressed in epochs")

    parser.add_argument("--save_training", action='store_true', help="Whether save the training model")
    parser.add_argument("--save_best", action='store_true', help="Save only the best model during training")
    parser.add_argument("--early_stop_patience", default=7, type=int,
                        help="Stop training if arithmetic_mean does not improve for this many consecutive validations")

    # comet.ml
    parser.add_argument("--api_key", type=str, default='xxx', help="api for Comet logging")
    parser.add_argument("--workspace", type=str, help="workspace of Comet logging")
    parser.add_argument("--project_name", default="xxx", type=str, help="name of the project on Comet")

    args = parser.parse_args()

    if args.api_key and args.workspace:
        print("Comet logging ENABLED")
        experiment = Experiment(
            api_key=args.api_key,
            project_name=args.project_name,
            workspace=args.workspace,
            disabled=False
        )
    else:
        print("Comet loging DISABLED, to enable it provide an api key and a workspace")
        experiment = Experiment(
            api_key="",
            project_name="",
            workspace="",
            disabled=True
        )

    experiment.log_code(folder=str(base_path / 'code'))
    experiment.log_parameters(args)

    combiner_training_cirr(args)

