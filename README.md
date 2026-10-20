# FIVE

Training and retrieval evaluation code for the saved FIVE models on **PatternCIRRS** and **PatternCIR**. The implementation retains the original SSN v9 model and its inherited components. `PatternCIRRS` is called `CustomCIRR` inside the original code; `PatternCIR` uses the internal name `patterncir`.

## Recorded results

The numbers below come from the checkpoint-associated evaluation logs. All metrics are percentages. **Mean = (R@5 + GS@1) / 2**; the historical JSON field `acc` is a different metric and must not be used as this table's Mean.

| Dataset | Evaluation split | R@1 | R@5 | R@10 | R@50 | GS@1 | GS@2 | GS@3 | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PatternCIRRS | test | 29.74 | 61.62 | 75.10 | 96.29 | 66.23 | 85.47 | 93.66 | 63.93 |
| PatternCIR | val | 47.11 | 74.22 | 84.46 | 97.24 | 72.40 | 86.73 | 93.99 | 73.31 |

Full precision results are in [`results/PatternCIRRS_recorded.json`](results/PatternCIRRS_recorded.json) and [`results/PatternCIR_recorded.json`](results/PatternCIR_recorded.json).

**PatternCIR table reconciliation:** the supplied manuscript row is `47.10 / 74.21 / 84.46 / 97.24 / 72.40 / 86.72 / 93.89 / 73.31`. The associated checkpoint's saved log records `GS@3 = 93.98865784499056`, which rounds to **93.99**, not 93.89. R@1, R@5 and GS@2 also differ under ordinary rounding versus truncation. This repository reports the saved values rather than silently changing them to match the manuscript. A checkpoint producing exactly GS@3 = 93.89 has not been established.

PatternCIR uses its labeled `val` split for local evaluation. The public `test1` split is a submission split and is not interchangeable with these labeled validation results.

## Code layout

```text
configs/                       Original training arguments for each checkpoint
results/                       Full precision recorded metrics
src/SSN_v9/train_ssn_v9.py      Training entry point
src/SSN_v9/test_ssn_v9.py       Retrieval evaluation entry point
src/SSN_v3 ... SSN_v8/          Model components inherited by v9
src/model/                     Localization, graph, fusion and FFT components
src/models/                    Shared SSN BLIP2 implementation
src/lavis/                     Vendored LAVIS dependencies and configurations
src/precompute_patterncir.py   PatternCIR image feature extraction
tools/run.py                   Dataset-specific launcher and resource checks
tools/verify_weights.py        Streaming SHA-256 verification
```

The v3-v8 modules are required by the v9 inheritance chain. They are not separate checkpoints for the result table. The separate Q-Former baseline, migration scripts, unrelated training sweeps and server credentials are not part of this release.

## Environment

The source server uses Linux, Python 3.12.3, PyTorch 2.3.0 with CUDA 12.1, torchvision 0.18.0 and transformers 4.40.2. Use a CUDA-capable Linux environment or WSL2 for training and evaluation. The Windows directory described below is the local asset destination.

```bash
conda create -n five python=3.12 -y
conda activate five
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`configs/requirements.server-snapshot.txt` preserves the original environment export, including machine-specific `file:///` entries; do not install that file directly. `requirements.txt` selects the relevant PyPI dependencies and retains NumPy 1.26.4 from that historical export. Installing this environment from scratch is separate from validating the saved model results.

## Weights and local assets

The requested Windows destination is `E:\ICLR2027-FIVE\code`. The final asset layout is:

```text
code/
  weights/
    FIVE_PatternCIRRS.pth
    FIVE_PatternCIR.pth
  weight/
    eva_vit_g.pth
    blip2_pretrained.pth
    selector_eva.pt
  bert-base-uncased/
    config.json
    vocab.txt
    pytorch_model.bin
    ...
  download_manifest.json
```

Trained checkpoint checksums:

| File | Size in bytes | SHA-256 |
| --- | --- | --- |
| `weights/FIVE_PatternCIRRS.pth` | 2842846488 | `4c5ccfb6a802b76ac293a6c24f12fbab9080c0610917217be26f02775cb4f67a` |
| `weights/FIVE_PatternCIR.pth` | 2842846488 | `0da1d080b231de43bc47607cfdf24cef5e54e199ecfa97f827aef2184034bec1` |

The trained weights are kept locally and excluded from Git. There is no public model download URL in this repository. EVA and BERT assets are used during model construction; BLIP2 initialization and the selector are also needed for training. Use `--assets-root` to point to the directory containing `weights/`, `weight/` and `bert-base-uncased/`.

Verify the trained files without loading a model:

```bash
python tools/verify_weights.py --assets-root /path/to/code
```

## Dataset and feature preparation

Annotations, images and precomputed feature tensors are external assets and are not included in Git. Use the original normalized graphs and feature mappings for checkpoint evaluation. The graph `action`, object and location fields are model inputs.

For PatternCIRRS, `--data-root` must contain:

```text
CustomCIRR_graph/
  train/train_query_text_with_graph.json
  train/train_split_image.json
  val/val_query_text_with_graph.json
  val/val_split_image.json
  test/test_query_text_with_graph.json
  test/test_split_image.json
```

Retain the corresponding `<split>_query_text_with_graph_object_core.json` files where supplied: the v9 entry points inject these annotations when their lengths match. Preserve image identifier strings in annotations and `<split>_name_to_idx.json`; changing only one side breaks feature lookup.

For PatternCIR, `--data-root` points to the normalized graph root containing `<split>/cap.rc2.<split>.json`. `--images-root` points to the PatternNet root containing `image_splits/split.rc2.train.json`, `split.rc2.val.json`, and `split.rc2.test1.json`, plus the original split image folders.

Both datasets use a feature directory containing, for every required split:

```text
<split>_images.pt
<split>_name_to_idx.json
```

Feature tensors are the original EVA visual encoder outputs **before** `ln_vision`, with shape `[N, 1 + number_of_patches, 1408]`. Training needs `train` and `val`; table evaluation needs `test` for PatternCIRRS and `val` for PatternCIR. The original PatternCIR extractor is provided in `src/precompute_patterncir.py`; its paths can be set with `PATTERNNET_ROOT`, `BERT_ROOT`, `FIVE_EVA_CKPT`, and `FIVE_BLIP2_PRETRAINED`.

## Evaluate the saved checkpoints

Run from the repository root. Replace example paths with your mounted dataset, feature and asset directories.

```bash
python tools/run.py eval --dataset PatternCIRRS \
  --assets-root /path/to/code \
  --data-root /path/to/CustomCIRR_graph \
  --feature-dir /path/to/features/CustomCIRR \
  --gpu 0

python tools/run.py eval --dataset PatternCIR \
  --assets-root /path/to/code \
  --data-root /path/to/PatternCIR_graph \
  --images-root /path/to/PatternNet \
  --feature-dir /path/to/features/patterncir \
  --gpu 0
```

For WSL2, the Windows asset directory typically corresponds to `/mnt/e/ICLR2027-FIVE/code`. The launcher chooses the correct checkpoint and labeled split. Use `--checkpoint` for an alternative file and `--output-dir` for an alternative result directory. By default, results go to `outputs/<dataset>/test_results_<split>.json`, including the explicitly named `mean_r5_gs1` metric.

Append `--check` to check required resource files without constructing a model, or `--dry-run` to inspect the generated command without requiring local resources.

## Train

The launcher uses the checkpoint-associated training configurations in `configs/PatternCIRRS.json` and `configs/PatternCIR.json`.

```bash
python tools/run.py train --dataset PatternCIRRS \
  --assets-root /path/to/code \
  --data-root /path/to/CustomCIRR_graph \
  --feature-dir /path/to/features/CustomCIRR \
  --gpu 0

python tools/run.py train --dataset PatternCIR \
  --assets-root /path/to/code \
  --data-root /path/to/PatternCIR_graph \
  --images-root /path/to/PatternNet \
  --feature-dir /path/to/features/patterncir \
  --gpu 0
```

Both runs use seed 42, batch size 64 and 35 epochs. PatternCIRRS uses learning rate `1e-5`, three warmup epochs and a stage switch at epoch 25. PatternCIR uses learning rate `1.3e-5`, five warmup epochs and a stage switch at epoch 35. All remaining settings are preserved in the JSON files.

New runs write into `logs/SSN_BLIP2/<internal-dataset>/ssn_v9/reproduce_<dataset>/`. Choose `--exp-name` for a different run directory. Extra original training options may be passed after `--`, for example `-- --num_workers 4`.

The original checkpoint names were `CustomCIRR/ssn_v9/runA/best_arithmetic.pth` and `patterncir/ssn_v9/patterncir_stage1_ep35/best_arithmetic.pth`. Retraining is stochastic and is not a promise of bit-identical results.

## Source and validation

Model forward computation, the training objectives and the original retrieval metric calculations are retained. Packaging changes add configurable dataset/asset paths, an evaluation output directory, launch/resource checks, and the separately named table Mean. See `docs/VALIDATION.md` for the checks performed on this release.

Third-party copyright headers are retained. LAVIS's BSD 3-Clause license is included in `licenses/LAVIS-LICENSE.txt`; see `THIRD_PARTY.md` for upstream attribution. No new license is assigned here to the original FIVE research code.
