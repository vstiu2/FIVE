# Release validation

The release contains the saved SSN v9 implementation, checkpoint-associated training arguments and recorded evaluation metrics for PatternCIRRS and PatternCIR.

## Checks performed

- Python source files were parsed for syntax errors.
- Both dataset launch commands were generated and checked for the intended dataset, labeled split and checkpoint path.
- The packaged inference entry point imported successfully and printed `--help` in both source servers' existing `five` environments.
- Both recorded training configurations were parsed through the packaged original training parser on both servers. PatternCIRRS retained 35 epochs, learning rate 1e-5 and stage switch 25; PatternCIR retained 35 epochs, learning rate 1.3e-5 and stage switch 35.
- The resource checker detects missing data, feature and model assets before model construction.
- The existing server checkpoint SHA-256 values are recorded in the README and the verification utility. Local downloads are checked against these values separately.

## Evaluation scope

Neither source server currently exposes a usable CUDA device (`torch.cuda.is_available()` is false). Full retrieval evaluation and training were not rerun during this release. The result table therefore identifies historical checkpoint-associated metrics, rather than claiming newly reproduced scores.

The existing environments also emit a SciPy/NumPy compatibility warning because their current NumPy version is 2.5.1. The installation instructions retain NumPy 1.26.4 from the historical export. A fresh installation of the documented environment has not been validated in this release.

## PatternCIR metric reconciliation

The saved full-model result has GS@3 = 93.98865784499056. It rounds to 93.99, whereas the supplied manuscript row says 93.89. Mean = (74.21550094517958 + 72.4007561436673) / 2 = 73.30812854442344. The full precision source JSON is preserved in `results/PatternCIR_recorded.json`.
