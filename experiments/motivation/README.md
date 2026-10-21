# Motivation experiments / 动机实验

This archive preserves the available **CustomCIRR / PatternCIRRS test** motivation records from the v9 `runA` checkpoint. The complete test set has **6,684 queries** and **1,947 gallery images**. These are recorded evaluations, not new runs. PatternCIR motivation results were not found on the inventoried server.

此前 GitHub 仅在敏感性实验的代码快照中保留了动机实验脚本；本目录补充独立的原始结果、执行日志、汇总及复现说明。权重不包含在仓库中。

## Contents

| Path | Contents |
| --- | --- |
| `records/CustomCIRR_test/shortcut_CustomCIRR_test.json` | Full, text-only and reference-only retrieval; per-action metrics |
| `records/CustomCIRR_test/action_perturb_CustomCIRR_test.json` | Graph-only swap, text+graph swap and graph-action shuffle; full set and changed subset |
| `records/CustomCIRR_test/size_buckets_CustomCIRR_test.json` | Two size buckets at area ratio 0.4; per-query object ratios |
| `records/CustomCIRR_test/size_buckets_CustomCIRR_test_0.3.json` | Separate two-bucket result at threshold 0.3 |
| `records/CustomCIRR_test/motivation_experiments.log` | Original append-only execution log covering multiple settings |
| `summary.json` | Compact metrics extracted from the four result JSON files |
| `SOURCE_MANIFEST.json` | Source paths, SHA-256 hashes and packaging changes |
| [`tools/run_motivation.py`](../../tools/run_motivation.py) | Launcher with explicit asset and dataset paths |
| [`src/SSN_v9/motivation_experiments/`](../../src/SSN_v9/motivation_experiments/) | Evaluation implementation |

## Recorded results

All metrics below are percentages, rounded only for display. JSON files retain the original precision. No result is selected or recomputed from the append-only log.

### Reference dependence / shortcut

| Query | Queries | R@1 | R@5 | GS@1 |
| --- | ---: | ---: | ---: | ---: |
| Full composed query | 6,684 | 29.74 | 61.62 | 66.23 |
| Text only | 6,684 | 8.26 | 21.60 | 52.45 |
| Reference image only | 6,684 | 7.39 | 23.56 | 20.42 |

Text-only and reference-only use the corresponding encoder embeddings. They are inference checks on the existing checkpoint, not independently trained baselines.

### Action perturbation: original Add/Remove subset

| Condition | Queries | R@1 | R@5 | GS@1 |
| --- | ---: | ---: | ---: | ---: |
| Full | 5,456 | 30.90 | 62.26 | 67.72 |
| Swap graph action only | 5,456 | 31.05 | 61.73 | 67.63 |
| Swap text and graph action | 5,456 | 16.70 | 40.05 | 46.32 |
| Shuffle graph action | 5,456 | 30.98 | 62.26 | 67.72 |

The subset contains samples originally categorized as Add or Remove. Move samples are excluded from this table. The swap exchanges Add with Remove (including actions mapped to that category), preserving reference images, gallery and original retrieval targets. The text+graph condition also rewrites the action phrase in the query; the graph-only condition preserves the text. The shuffle condition permutes graph actions and is distinct from the deterministic swap.

The large drop is observed when **both text and graph action change**. Graph-only swap and graph-action shuffle have much smaller effects. These results should not be described as evidence that graph-action changes alone cause a large retrieval drop. The text+graph comparison does not isolate the graph branch.

### Object-size buckets / FOE

Size is estimated by the ObjectLocalizer mask on reference-image patches, not ground-truth object boxes. In hard-area mode, the ratio is the fraction of patches with mask response **>= 0.5**. Add samples are excluded because the added object is absent from the reference image, leaving **5,274** eligible queries. `w/o FOE` disables the Ff frequency branch at inference (`ablate_no_ff`); it is not retraining without FOE.

| Area threshold | Bucket | Queries | Full R@1 | w/o FOE R@1 | Full GS@1 | w/o FOE GS@1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.4 | small: ratio < 0.4 | 4,741 | 28.94 | 24.85 | 65.30 | 59.40 |
| 0.4 | non-small: ratio >= 0.4 | 533 | 35.08 | 28.52 | 74.30 | 64.17 |
| 0.3 | small: ratio < 0.3 | 4,413 | 27.94 | 24.20 | 64.56 | 58.78 |
| 0.3 | non-small: ratio >= 0.3 | 861 | 37.86 | 30.43 | 74.68 | 65.51 |

Both buckets benefit from the frequency branch in these records. The absolute GS@1 improvement is larger for the non-small bucket at both thresholds; these records do not support a claim that the gain is largest for small objects.

## Reproduce with local assets

Install the repository dependencies and prepare the assets described in the root [README](../../README.md). The recorded checkpoint is `CustomCIRR/ssn_v9/runA/best_arithmetic.pth`, distributed locally as `weights/FIVE_PatternCIRRS.pth`. The launcher additionally checks the selector and BLIP2 pretrained files used by the original motivation script. No separate motivation-training checkpoint is required.

Run from the repository root:

```bash
python tools/run_motivation.py --dataset PatternCIRRS \
  --assets-root /path/to/code \
  --data-root /path/to/CustomCIRR_graph \
  --feature-dir /path/to/features/CustomCIRR \
  --experiments shortcut,size,action --size-threshold 0.4 \
  --output-dir outputs/motivation/PatternCIRRS/threshold_0.4 --gpu 0

python tools/run_motivation.py --dataset PatternCIRRS \
  --assets-root /path/to/code \
  --data-root /path/to/CustomCIRR_graph \
  --feature-dir /path/to/features/CustomCIRR \
  --experiments size --size-threshold 0.3 \
  --output-dir outputs/motivation/PatternCIRRS/threshold_0.3 --gpu 0
```

Use separate output directories: the raw script uses the same size-result filename for different thresholds. Pass `--checkpoint` to override the checkpoint path. `--dry-run` prints the command without importing model dependencies, while `--check` verifies resource existence without loading the model. The launcher sets BERT, EVA, BLIP2, selector and graph paths through the same environment variables as the released training/evaluation launcher.

The default seed is 42 in the available source and in the launcher. The result JSON files do not record the seed or complete invocation, so this is a reconstruction of the recorded settings, not independent proof of the original invocation. The source defaults to three size buckets; the launcher explicitly selects **two** buckets to match the archived JSON results.

PatternCIR can use `--dataset PatternCIR --experiments shortcut,action` with its graph, image and feature resources. This is a supported code path, not an archived PatternCIR result or a verified reproduction in this upload.

## Provenance and limitations

- Source and records were collected on 2026-10-21. The append-only log records multiple executions on 2026-07-09, including earlier three-bucket settings and earlier action variants. Later runs overwrote some JSON filenames. Preserve both the log and final JSON, and do not mix their rows into one experiment table.
- `SOURCE_MANIFEST.json` preserves the original hashes. The published runner changes only the BLIP2 asset path override and the module description; the root model, training objectives and metric implementation are unchanged by this archive.
- The original motivation script already loads checkpoint state with `strict=False` and logs missing/unexpected keys. Review those messages during a rerun; they were not independently revalidated in this upload.
- The collected server source had no Git metadata. Source hashes identify available files, not a verified historical training commit. Exact historical data/feature versions and checkpoint equality on this server were not newly established by hashing the large assets.
- This upload validates record integrity, summary consistency, Python syntax and launcher commands. It does not rerun GPU inference, establish confidence intervals, or guarantee identical results on another software/hardware stack.
