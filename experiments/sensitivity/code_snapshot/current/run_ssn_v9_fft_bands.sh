#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/data/lj/FIVE"
PYTHON_BIN="/root/autodl-tmp/envs/five/bin/python"
TRAIN_ENTRY="src/SSN_v9/train_ssn_v9_best_only.py"
TEST_ENTRY="src/SSN_v9/test_ssn_v9.py"
DATA_ROOT="/data/lj/CIR/datasets/merged_query_text_with_graph_allclass_normalized"
FEATURE_DIR="/data/lj/TME_v1/features/CustomCIRR"
SELECTOR_CKPT="/data/lj/SSN_v1/weight/selector_eva.pt"
EXP_PREFIX="ssn_v9_sensitivity/fft_num_bands"
LOG_ROOT="${PROJECT_ROOT}/logs/SSN_BLIP2/CustomCIRR/${EXP_PREFIX}"
RUNNER_LOG_ROOT="${LOG_ROOT}/_runner_logs"
SUMMARY_CSV="${LOG_ROOT}/fft_band_sensitivity_val_results.csv"
BASELINE_RESULT="${PROJECT_ROOT}/logs/SSN_BLIP2/CustomCIRR/ssn_v9/runA/test_results_val.json"
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

for required_path in \
  "${PYTHON_BIN}" "${TRAIN_ENTRY}" "${TEST_ENTRY}" "${FEATURE_DIR}" \
  "${SELECTOR_CKPT}" "${DATA_ROOT}"; do
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
  --seed 42
  --num_workers 8
  --batch_size 64
  --num_epochs 35
  --warmup_epoch 3
  --lr 1e-5
  --weight_decay 0.05
  --lambda_align 0.4
  --gamma_init 0.05
  --gamma_max 0.5
  --aux_ramp_epochs 5
  --lambda_global 0
  --lambda_colbert 0
  --label_smoothing 0
  --alpha_init 0.01
  --beta_init 0.05
  --alpha_max 0.5
  --beta_max 0.5
  --max_txt_len 32
  --swa_last_n 5
  --resume ""
  --patch_dropout 0.15
  --early_stop_patience 7
  --lambda_kl_action 0.1
  --mask_hidden_dim 512
  --mask_dropout 0.1
  --mask_bias_init 0
  --inject_hidden_dim 1024
  --inject_dropout 0.1
  --alpha_inject_init 0.2
  --alpha_inject_max 0.5
  --lambda_counterfact 0
  --counterfact_margin 0.05
  --blend_dropout 0
  --mask_target_lo 0.4
  --mask_target_hi 0.7
  --mask_target_w 1
  --mask_div_w 0.1
  --lambda_trans_align 0.1
  --f_trans_via_fusion true
  --f_trans_via_gamma true
  --stage_switch_epoch 25
  --stage2_mask_target_w 2
  --stage2_lambda_kl_action 0.05
  --top_k_fl 4
  --lambda_fl 0.1
  --lambda_ff 0.1
  --fl_multi true
  --ff_multi true
  --save_training
)

test_one() {
  local num_bands="$1"
  local output_dir="$2"
  local best_ckpt="${output_dir}/best_arithmetic.pth"
  local result_json="${output_dir}/test_results_val.json"
  local test_log="${RUNNER_LOG_ROOT}/bands_${num_bands}_test.log"

  if [[ "${FORCE_RETEST}" != "1" && -s "${result_json}" && -e "${output_dir}/.complete" ]]; then
    echo "[$(date '+%F %T')] SKIP TEST bands=${num_bands}: completed result exists."
    return 0
  fi

  echo "[$(date '+%F %T')] START TEST bands=${num_bands}"
  "${PYTHON_BIN}" "${TEST_ENTRY}" \
    --dataset CustomCIRR \
    --split val \
    --feature_dir "${FEATURE_DIR}" \
    --model_path "${best_ckpt}" \
    --fft_num_bands "${num_bands}" \
    --gpu 0 \
    2>&1 | tee -a "${test_log}"

  if [[ ! -s "${result_json}" ]]; then
    echo "ERROR: Test did not produce ${result_json}" >&2
    exit 1
  fi
  touch "${output_dir}/.complete"
  echo "[$(date '+%F %T')] DONE TEST bands=${num_bands}"
}

run_one() {
  local num_bands="$1"
  local exp_name="${EXP_PREFIX}/${num_bands}"
  local output_dir="${LOG_ROOT}/${num_bands}"
  local best_ckpt="${output_dir}/best_arithmetic.pth"
  local runner_log="${RUNNER_LOG_ROOT}/bands_${num_bands}_train.log"
  local available_kb

  if [[ -e "${output_dir}/.training_complete" && -s "${best_ckpt}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[$(date '+%F %T')] SKIP TRAIN bands=${num_bands}: completed checkpoint exists."
    test_one "${num_bands}" "${output_dir}"
    return 0
  fi

  if [[ -e "${output_dir}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "ERROR: Partial output exists; refusing to mix runs: ${output_dir}" >&2
    echo "Inspect it, then move it aside or rerun with FORCE_RERUN=1." >&2
    exit 1
  fi

  available_kb="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')"
  if (( available_kb < MIN_FREE_KB )); then
    echo "ERROR: Less than 4 GiB remains on /root/autodl-tmp." >&2
    df -h /root/autodl-tmp >&2
    exit 1
  fi

  echo "[$(date '+%F %T')] START TRAIN bands=${num_bands}"
  "${PYTHON_BIN}" "${COMMON_ARGS[@]}" \
    --exp_name "${exp_name}" \
    --fft_num_bands "${num_bands}" \
    2>&1 | tee -a "${runner_log}"

  if [[ ! -s "${best_ckpt}" ]]; then
    echo "ERROR: Training exited without producing ${best_ckpt}" >&2
    exit 1
  fi
  if [[ -e "${output_dir}/last_model.pth" ]]; then
    echo "ERROR: Best-only training unexpectedly produced last_model.pth." >&2
    exit 1
  fi

  touch "${output_dir}/.training_complete"
  echo "[$(date '+%F %T')] DONE TRAIN bands=${num_bands}"
  test_one "${num_bands}" "${output_dir}"
  df -h /root/autodl-tmp
}

for num_bands in 2 6 8 10; do
  run_one "${num_bands}"
done

"${PYTHON_BIN}" - "${LOG_ROOT}" "${BASELINE_RESULT}" "${SUMMARY_CSV}" <<'PY'
import csv
import json
import sys
from pathlib import Path

log_root = Path(sys.argv[1])
baseline_result = Path(sys.argv[2])
summary_csv = Path(sys.argv[3])
metrics = ["R@1", "R@5", "R@10", "R@50", "acc", "GS@1", "GS@2", "GS@3"]
rows = []
for num_bands in (2, 4, 6, 8, 10):
    result_path = (
        baseline_result
        if num_bands == 4
        else log_root / str(num_bands) / "test_results_val.json"
    )
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    row = {"num_bands": num_bands, "result_path": str(result_path)}
    row.update({metric: result.get(metric) for metric in metrics})
    rows.append(row)

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle, fieldnames=["num_bands", *metrics, "result_path"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Summary written to: {summary_csv}")
PY

echo "[$(date '+%F %T')] FFT-band sensitivity sweep complete."
echo "Summary: ${SUMMARY_CSV}"
