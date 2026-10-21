#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/data/lj/FIVE"
PYTHON_BIN="/root/autodl-tmp/envs/five/bin/python"
DATA_ROOT="/data/lj/CIR/datasets/merged_query_text_with_graph_allclass_normalized"
FEATURE_DIR="/data/lj/TME_v1/features/CustomCIRR"
EXP_PREFIX="ssn_v9_sensitivity"
LOG_ROOT="${PROJECT_ROOT}/logs/SSN_BLIP2/CustomCIRR/${EXP_PREFIX}"
RUNNER_LOG_ROOT="${LOG_ROOT}/_test_runner_logs_round2"
SUMMARY_CSV="${LOG_ROOT}/sensitivity_round2_val_results.csv"
FORCE_RETEST="${FORCE_RETEST:-0}"

export HF_HOME="/root/autodl-tmp/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUSTOM_CIRR_ROOT="${DATA_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi

for required_path in "${FEATURE_DIR}" "${DATA_ROOT}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "ERROR: Required path not found: ${required_path}" >&2
    exit 1
  fi
done

if pgrep -af '[r]un_ssn_v9_sensitivity_round2.sh|[t]rain_ssn_v9.py' >/dev/null; then
  echo "ERROR: Sensitivity training is still running. Wait before testing." >&2
  exit 1
fi

if pgrep -af '[t]est_ssn_v9.py' >/dev/null; then
  echo "ERROR: Another test_ssn_v9.py process is already running." >&2
  exit 1
fi

EXPERIMENTS=(
  "lambda_fl 0.7"
  "lambda_fl 0.8"
  "lambda_fl 0.9"
  "lambda_ff 0.3"
  "lambda_ff 0.4"
  "lambda_ff 0.7"
  "lambda_ff 0.8"
  "lambda_trans_align 0.8"
)

for spec in "${EXPERIMENTS[@]}"; do
  read -r parameter value <<< "${spec}"
  best_ckpt="${LOG_ROOT}/${parameter}/${value}/best_arithmetic.pth"
  if [[ ! -s "${best_ckpt}" ]]; then
    echo "ERROR: Missing best checkpoint: ${best_ckpt}" >&2
    exit 1
  fi
done

mkdir -p "${RUNNER_LOG_ROOT}"

test_one() {
  local parameter="$1"
  local value="$2"
  local output_dir="${LOG_ROOT}/${parameter}/${value}"
  local best_ckpt="${output_dir}/best_arithmetic.pth"
  local result_json="${output_dir}/test_results_val.json"
  local runner_log="${RUNNER_LOG_ROOT}/${parameter}_${value}.log"

  if [[ "${FORCE_RETEST}" != "1" && -s "${result_json}" && "${result_json}" -nt "${best_ckpt}" ]]; then
    echo "[$(date '+%F %T')] SKIP ${parameter}=${value}: current val result already exists."
    return 0
  fi

  echo "[$(date '+%F %T')] START TEST ${parameter}=${value}"

  "${PYTHON_BIN}" src/SSN_v9/test_ssn_v9.py \
    --dataset CustomCIRR \
    --split val \
    --feature_dir "${FEATURE_DIR}" \
    --model_path "${best_ckpt}" \
    --gpu 0 \
    2>&1 | tee -a "${runner_log}"

  if [[ ! -s "${result_json}" ]]; then
    echo "ERROR: Test exited without producing ${result_json}" >&2
    exit 1
  fi

  echo "[$(date '+%F %T')] DONE TEST ${parameter}=${value}"
}

for spec in "${EXPERIMENTS[@]}"; do
  read -r parameter value <<< "${spec}"
  test_one "${parameter}" "${value}"
done

"${PYTHON_BIN}" - "${LOG_ROOT}" "${SUMMARY_CSV}" <<'PY'
import csv
import json
import sys
from pathlib import Path

log_root = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
experiments = [
    ("lambda_fl", "0.7"),
    ("lambda_fl", "0.8"),
    ("lambda_fl", "0.9"),
    ("lambda_ff", "0.3"),
    ("lambda_ff", "0.4"),
    ("lambda_ff", "0.7"),
    ("lambda_ff", "0.8"),
    ("lambda_trans_align", "0.8"),
]
metrics = ["R@1", "R@5", "R@10", "R@50", "acc", "GS@1", "GS@2", "GS@3"]
rows = []
for parameter, value in experiments:
    result_path = log_root / parameter / value / "test_results_val.json"
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    row = {"parameter": parameter, "value": value}
    row.update({metric: result.get(metric) for metric in metrics})
    rows.append(row)

with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["parameter", "value", *metrics])
    writer.writeheader()
    writer.writerows(rows)

print(f"Summary written to: {summary_csv}")
PY

echo "[$(date '+%F %T')] All round-2 sensitivity tests are complete."
echo "Summary: ${SUMMARY_CSV}"
