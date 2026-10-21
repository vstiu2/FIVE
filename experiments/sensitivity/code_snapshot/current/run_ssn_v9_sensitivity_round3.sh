#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/data/lj/FIVE"
PYTHON_BIN="/root/autodl-tmp/envs/five/bin/python"
TRAIN_ENTRY="src/SSN_v9/train_ssn_v9_best_only.py"
DATA_ROOT="/data/lj/CIR/datasets/merged_query_text_with_graph_allclass_normalized"
FEATURE_DIR="/data/lj/TME_v1/features/CustomCIRR"
SELECTOR_CKPT="/data/lj/SSN_v1/weight/selector_eva.pt"
EXP_PREFIX="ssn_v9_sensitivity"
LOG_ROOT="${PROJECT_ROOT}/logs/SSN_BLIP2/CustomCIRR/${EXP_PREFIX}"
RUNNER_LOG_ROOT="${LOG_ROOT}/_runner_logs_round3"
SUMMARY_CSV="${LOG_ROOT}/sensitivity_round3_val_results.csv"
SYSTEM_OVERFLOW_ROOT="/root/ssn_v9_sensitivity_overflow"
MIN_FREE_KB=4194304
FORCE_RERUN="${FORCE_RERUN:-0}"
FORCE_RETEST="${FORCE_RETEST:-0}"

export HF_HOME="/root/autodl-tmp/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUSTOM_CIRR_ROOT="${DATA_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"

for required_path in "${PYTHON_BIN}" "${TRAIN_ENTRY}" "${FEATURE_DIR}" "${SELECTOR_CKPT}" "${DATA_ROOT}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "ERROR: Required path not found: ${required_path}" >&2
    exit 1
  fi
done

if pgrep -af '[t]rain_ssn_v9' >/dev/null; then
  echo "ERROR: Another SSN_v9 training process is already running." >&2
  pgrep -af '[t]rain_ssn_v9' >&2 || true
  exit 1
fi

mkdir -p "${RUNNER_LOG_ROOT}"

COMMON_ARGS=(
  "${TRAIN_ENTRY}"
  --dataset CustomCIRR
  --feature_dir "${FEATURE_DIR}"
  --selector_ckpt "${SELECTOR_CKPT}"
  --gpu 0
  --batch_size 64
  --num_epochs 35
  --lambda_align 0.4
  --gamma_init 0.05
  --beta_init 0.05
  --patch_dropout 0.15
  --lambda_trans_align 0.1
  --lambda_fl 0.1
  --lambda_ff 0.1
  --save_training
)

test_one() {
  local directory_parameter="$1"
  local value="$2"
  local best_ckpt="$3"
  local output_dir
  local result_json
  local test_log

  output_dir="$(dirname "${best_ckpt}")"
  result_json="${output_dir}/test_results_val.json"
  test_log="${RUNNER_LOG_ROOT}/${directory_parameter}_${value}_test.log"

  if [[ "${FORCE_RETEST}" != "1" && -s "${result_json}" && "${result_json}" -nt "${best_ckpt}" ]]; then
    echo "[$(date '+%F %T')] SKIP TEST ${directory_parameter}=${value}: current result exists."
    return 0
  fi

  echo "[$(date '+%F %T')] START TEST ${directory_parameter}=${value}"
  "${PYTHON_BIN}" src/SSN_v9/test_ssn_v9.py \
    --dataset CustomCIRR \
    --split val \
    --feature_dir "${FEATURE_DIR}" \
    --model_path "${best_ckpt}" \
    --gpu 0 \
    2>&1 | tee -a "${test_log}"

  if [[ ! -s "${result_json}" ]]; then
    echo "ERROR: Test did not produce ${result_json}" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] DONE TEST ${directory_parameter}=${value}"
}

run_one() {
  local directory_parameter="$1"
  local cli_parameter="$2"
  local value="$3"
  local exp_name="${EXP_PREFIX}/${directory_parameter}/${value}"
  local output_dir="${LOG_ROOT}/${directory_parameter}/${value}"
  local best_ckpt="${output_dir}/best_arithmetic.pth"
  local runner_log="${RUNNER_LOG_ROOT}/${directory_parameter}_${value}.log"
  local available_kb
  local system_available_kb
  local overflow_dir

  if [[ -s "${best_ckpt}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[$(date '+%F %T')] SKIP ${directory_parameter}=${value}: best checkpoint already exists."
    test_one "${directory_parameter}" "${value}" "${best_ckpt}"
    return 0
  fi

  if [[ -e "${output_dir}" && ! -s "${best_ckpt}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "ERROR: Partial output exists without a best checkpoint: ${output_dir}" >&2
    exit 1
  fi

  available_kb="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')"
  if (( available_kb < MIN_FREE_KB )); then
    system_available_kb="$(df -Pk / | awk 'NR==2 {print $4}')"
    if (( system_available_kb < MIN_FREE_KB )); then
      echo "ERROR: Neither disk has the 4 GiB minimum required for another best checkpoint." >&2
      df -h /root/autodl-tmp / >&2
      exit 1
    fi

    overflow_dir="${SYSTEM_OVERFLOW_ROOT}/${directory_parameter}/${value}"
    mkdir -p "$(dirname "${output_dir}")" "${overflow_dir}"
    ln -s "${overflow_dir}" "${output_dir}"
    echo "[$(date '+%F %T')] Data disk is low; redirecting to system disk: ${overflow_dir}"
  fi

  echo "[$(date '+%F %T')] START ${directory_parameter}=${value} via --${cli_parameter}"

  "${PYTHON_BIN}" "${COMMON_ARGS[@]}" \
    --exp_name "${exp_name}" \
    "--${cli_parameter}" "${value}" \
    2>&1 | tee -a "${runner_log}"

  if [[ ! -s "${best_ckpt}" ]]; then
    echo "ERROR: Training exited without producing ${best_ckpt}" >&2
    exit 1
  fi

  if [[ -e "${output_dir}/last_model.pth" ]]; then
    echo "ERROR: Best-only training unexpectedly produced last_model.pth." >&2
    exit 1
  fi

  echo "[$(date '+%F %T')] DONE ${directory_parameter}=${value}"
  test_one "${directory_parameter}" "${value}" "${best_ckpt}"
  df -h /root/autodl-tmp /
}

run_one lambda_fl lambda_fl 3.0
run_one lambda_fl lambda_fl 6.0
run_one lambda_ff lambda_ff 8.0
run_one lambda_ff lambda_ff 5.0
run_one lambda_a lambda_align 3.0
run_one lambda_a lambda_align 5.0

"${PYTHON_BIN}" - "${LOG_ROOT}" "${SUMMARY_CSV}" <<'PY'
import csv
import json
import sys
from pathlib import Path

log_root = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
experiments = [
    ("lambda_fl", "3.0"),
    ("lambda_fl", "6.0"),
    ("lambda_ff", "8.0"),
    ("lambda_ff", "5.0"),
    ("lambda_a", "3.0"),
    ("lambda_a", "5.0"),
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

echo "[$(date '+%F %T')] All requested round-3 sensitivity experiments are complete."
echo "Summary: ${SUMMARY_CSV}"
