# Parameter sensitivity experiment archive

This directory preserves the available sensitivity records for FIVE on **CustomCIRR / PatternCIRRS**. The saved sweep evaluations use the labeled **val** split. These are archived results, not new evaluations or the manuscript test scores.

## Contents

| File or directory | Contents |
| --- | --- |
| `run_index.csv` | 32 checkpoint-bearing parameter settings, with recorded seed and artifact availability |
| `runs/<parameter>/<value>/` | Original configuration, training history and evaluation files where available |
| `results.csv` | Validation metrics and explicitly computed `mean_r5_gs1` |
| `training_history.csv` | 857 recorded per-epoch training rows from 27 runs |
| `figures/` | Training loss and validation sensitivity curves for each parameter group |
| `original_summaries/` | The original sweep summary tables, preserved separately |
| `baseline/runA/` | Existing main-run configuration, history and validation reference |
| `code_snapshot/` | Available first-party source snapshots, sweep scripts and SHA-256 manifest |
| `DATA_VERSION.json` | SHA-256, size and modification time for 17 current annotation/feature files |
| `checkpoint_sizes.csv` | Every inventoried sweep checkpoint's size; no checkpoint bytes |
| `checkpoint_group_sizes.json` | Storage accounting by hyperparameter |
| `artifact_manifest.csv` | Original per-run artifact hashes |

The 27 available configurations record **seed 42**. Five older directories contain only a checkpoint: their original seed, complete configuration, training history and evaluation are not recorded in this archive. Missing fields remain empty and missing curve points are not synthesized.

## Weight storage by hyperparameter

A saved best checkpoint is **2,842,846,488 bytes = 2.843 GB = 2.648 GiB**. Each row below counts distinct saved parameter settings on the inventoried server, not copies from multiple servers.

| Hyperparameter | Saved best checkpoints | Values with a checkpoint | Best-checkpoint total GB |
| --- | --- | --- | --- |
| lambda_fl | 10 | 0.00, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 3.0, 6.0 | 28.43 |
| lambda_ff | 10 | 0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 5.0, 8.0 | 28.43 |
| lambda_trans_align | 6 | 0.2, 0.4, 0.5, 0.6, 0.8, 1 | 17.06 |
| lambda_align | 2 | 3.0, 5.0 | 5.69 |
| fft_num_bands | 4 | 2, 6, 8, 10 | 11.37 |
| **Total** | **32** | | **90.97** |

The original directory label `lambda_a` corresponds to the configuration field `lambda_align`. The FFT baseline K=4 reuses the existing main-run checkpoint and is not counted as an additional saved sweep checkpoint.

There are also three `last_model.pth` files for lambda_trans_align = 0.4, 0.6 and 1, totaling **8.53 GB**. Best plus last files occupy **99.50 GB**. Weights remain outside Git; this update does not remove, convert or download them.

## Curves and metrics

The left panel of each figure uses the original CSV's total training loss and recorded epoch number. Different curves can stop at different epochs. The right panel shows the available validation **Mean = (R@5 + GS@1) / 2**. The original result field `acc` is a different statistic and remains unchanged.

- [lambda_fl](figures/lambda_fl.svg)
- [lambda_ff](figures/lambda_ff.svg)
- [lambda_trans_align](figures/lambda_trans_align.svg)
- [lambda_align](figures/lambda_align.svg)
- [FFT band count](figures/fft_num_bands.svg)

These are single-seed records; no uncertainty intervals or multi-seed averages are implied. The FFT figure marks K=4 as an existing baseline rather than a newly trained sweep run.

## Code and data provenance

The root repository's main-model code remains unchanged by this archive. Its reference commit is `03d61c460436275b08fbd885a79e9b72634fc7f0`.

`code_snapshot/current/` captures the first-party source and sweep scripts available when archived, including FFT-band support. `before_fft_patch/` contains four recovered pre-FFT backup files. The source manifest and `VERSION.json` identify these files by content. Historical experiment records did not contain immutable Git commit IDs, so the exact training-time source commit for each run is **not established**. A snapshot taken later must not be presented as a proven historical commit.

`DATA_VERSION.json` fingerprints the current normalized annotations, image-to-index mappings and train/val/test feature tensors. It records the **2026-10-21 data snapshot**; the archived experiments did not independently record training-time data hashes. Matching the historical training-time data version therefore remains unverified. Dataset contents and feature tensors are not uploaded.

The archived sweep scripts retain original machine paths and are provenance records. Use the root README for the packaged main-model training/evaluation workflow. The first-party source snapshots rely on the root `src/lavis/` dependency and applicable third-party licenses; they do not install a second runtime.

## Missing original records

The following settings have a saved best checkpoint but no original args/history/evaluation files in the inventoried directory:

- lambda_fl = 0.00
- lambda_ff = 0
- lambda_ff = 0.5
- lambda_trans_align = 0.2
- lambda_trans_align = 0.5

Their existence and storage cost are included in the checkpoint inventory. No seed or metric has been guessed for them.
