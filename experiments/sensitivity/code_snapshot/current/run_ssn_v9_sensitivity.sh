#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/data/lj/FIVE"
PYTHON_BIN="/root/autodl-tmp/envs/five/bin/python"
DATA_ROOT="/data/lj/CIR/datasets/merged_query_text_with_graph_allclass_normalized"
FEATURE_DIR="/data/lj/TME_v1/features/CustomCIRR"
SELECTOR_CKPT="/data/lj/SSN_v1/weight/selector_eva.pt"
EXP_PREFIX="ssn_v9_sensitivity"
LOG_ROOT="${PROJECT_ROOT}/logs/SSN_BLIP2/CustomCIRR/${EXP_PREFIX}"
RUNNER_LOG_ROOT="${LOG_ROOT}/_runner_logs"
SYSTEM_OVERFLOW_ROOT="/root/ssn_v9_sensitivity_overflow"
MIN_FREE_KB=7340032

# Defaults are safe for resuming a sweep:
# - completed experiments are skipped;
# - newly generated last_model.pth files are removed after best_arithmetic.pth
#   has been verified, because sensitivity experiments only need the best model.
# Set FORCE_RERUN=1 to rerun completed experiments.
# Set KEEP_LAST_MODEL=1 to retain last_model.pth for newly run experiments.
FORCE_RERUN="${FORCE_RERUN:-0}"
KEEP_LAST_MODEL="${KEEP_LAST_MODEL:-0}"

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

for required_path in "${FEATURE_DIR}" "${SELECTOR_CKPT}" "${DATA_ROOT}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "ERROR: Required path not found: ${required_path}" >&2
    exit 1
  fi
done

if pgrep -af '[t]rain_ssn_v9.py' >/dev/null; then
  echo "ERROR: Another train_ssn_v9.py process is already running." >&2
  pgrep -af '[t]rain_ssn_v9.py' >&2 || true
  exit 1
fi

mkdir -p "${RUNNER_LOG_ROOT}"

COMMON_ARGS=(
  src/SSN_v9/train_ssn_v9.py
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

run_one() {
  local parameter="$1"
  local value="$2"
  local exp_name="${EXP_PREFIX}/${parameter}/${value}"
  local output_dir="${LOG_ROOT}/${parameter}/${value}"
  local best_ckpt="${output_dir}/best_arithmetic.pth"
  local last_ckpt="${output_dir}/last_model.pth"
  local runner_log="${RUNNER_LOG_ROOT}/${parameter}_${value}.log"
  local available_kb
  local system_available_kb
  local overflow_dir

  if [[ -s "${best_ckpt}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[$(date '+%F %T')] SKIP ${parameter}=${value}: best checkpoint already exists."
    return 0
  fi

  if [[ -d "${output_dir}" && ! -s "${best_ckpt}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "ERROR: Partial experiment directory exists without a best checkpoint:" >&2
    echo "       ${output_dir}" >&2
    echo "Inspect it first, or rerun with FORCE_RERUN=1." >&2
    exit 1
  fi

  available_kb="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')"
  if (( available_kb < MIN_FREE_KB )); then
    system_available_kb="$(df -Pk / | awk 'NR==2 {print $4}')"
    if (( system_available_kb < MIN_FREE_KB )); then
      echo "ERROR: Both the data disk and system disk have less than 7 GiB free." >&2
      df -h /root/autodl-tmp / >&2
      exit 1
    fi

    overflow_dir="${SYSTEM_OVERFLOW_ROOT}/${parameter}/${value}"
    mkdir -p "$(dirname "${output_dir}")" "${overflow_dir}"
    ln -s "${overflow_dir}" "${output_dir}"
    echo "[$(date '+%F %T')] Data disk is low; redirecting this experiment to the system disk."
    echo "Physical output: ${overflow_dir}"
    echo "Stable access path: ${output_dir}"
  fi

  echo "[$(date '+%F %T')] START ${parameter}=${value}"
  echo "Output: ${output_dir}"
  echo "Runner log: ${runner_log}"

  "${PYTHON_BIN}" "${COMMON_ARGS[@]}" \
    --exp_name "${exp_name}" \
    "--${parameter}" "${value}" \
    2>&1 | tee -a "${runner_log}"

  if [[ ! -s "${best_ckpt}" ]]; then
    echo "ERROR: Training exited without producing ${best_ckpt}" >&2
    exit 1
  fi

  if [[ "${KEEP_LAST_MODEL}" != "1" && -f "${last_ckpt}" ]]; then
    rm -f -- "${last_ckpt}"
    echo "[$(date '+%F %T')] Removed unneeded last checkpoint: ${last_ckpt}"
  fi

  echo "[$(date '+%F %T')] DONE ${parameter}=${value}"
  df -h /root/autodl-tmp
}

# Requested execution order.
run_one lambda_fl 0.5
run_one lambda_fl 1.0
run_one lambda_ff 1.0
run_one lambda_ff 0.6
run_one lambda_fl 0.6
run_one lambda_fl 0.4
run_one lambda_trans_align 0.6

echo "[$(date '+%F %T')] All requested sensitivity experiments are complete."
