#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

GPUS=""
SEEDS="42,777"
STAGE="all"  # aug|loss|batch|all
NUM_EPOCHS=20
DECODE_AVG=9
MAX_DURATION_DEFAULT=1000
DECODE_MAX_DURATION=1000
BPE_MODEL="data/lang_bpe_500/bpe.model"
EXP_PREFIX="zipformer/exp_m_ablation"
DECODE_METHODS="modified_beam_search"  # comma-separated; e.g. greedy_search,modified_beam_search

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --asr-dir PATH                 ASR recipe dir (default: ${ASR_DIR})
  --gpus IDS                     GPU ids, e.g. 0,1 (default: empty -> CPU/auto)
  --seeds CSV                    Seeds, e.g. 42,777 (default: ${SEEDS})
  --stage NAME                   aug|loss|batch|all (default: ${STAGE})
  --num-epochs N                 Training epochs per run (default: ${NUM_EPOCHS})
  --decode-avg N                 Decode avg checkpoints (default: ${DECODE_AVG})
  --decode-max-duration N        Decode max-duration (default: ${DECODE_MAX_DURATION})
  --max-duration-default N       Default train max-duration (default: ${MAX_DURATION_DEFAULT})
  --bpe-model PATH               BPE model relative to ASR dir (default: ${BPE_MODEL})
  --exp-prefix RELPATH           Relative exp prefix (default: ${EXP_PREFIX})
  --decode-methods CSV           Decoding methods list (default: ${DECODE_METHODS})
  -h, --help                     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asr-dir) ASR_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --num-epochs) NUM_EPOCHS="$2"; shift 2 ;;
    --decode-avg) DECODE_AVG="$2"; shift 2 ;;
    --decode-max-duration) DECODE_MAX_DURATION="$2"; shift 2 ;;
    --max-duration-default) MAX_DURATION_DEFAULT="$2"; shift 2 ;;
    --bpe-model) BPE_MODEL="$2"; shift 2 ;;
    --exp-prefix) EXP_PREFIX="$2"; shift 2 ;;
    --decode-methods) DECODE_METHODS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -n "${GPUS}" ]]; then
  WORLD_SIZE="$(echo "${GPUS}" | awk -F',' '{print NF}')"
  DECODE_GPU="${GPUS%%,*}"
else
  WORLD_SIZE=1
  DECODE_GPU=""
fi

CSV_FILE="${RESULTS_DIR}/ablation_runs.csv"
if [[ ! -f "${CSV_FILE}" ]]; then
  echo "phase,run_id,seed,exp_dir,decoding_method,epoch,avg,use_averaged_model,beam_size,enable_musan,enable_spec_aug,prune_range,simple_loss_scale,lm_scale,am_scale,max_duration,elapsed_sec,status,notes" > "${CSV_FILE}"
fi

run_timed() {
  local -a cmd=("$@")
  local start_ts end_ts elapsed status
  start_ts="$(date +%s)"
  set +e
  "${cmd[@]}"
  status=$?
  set -e
  end_ts="$(date +%s)"
  elapsed=$(( end_ts - start_ts ))
  echo "${status},${elapsed}"
}

train_once() {
  local run_id="$1"
  local seed="$2"
  local enable_musan="$3"
  local enable_spec_aug="$4"
  local prune_range="$5"
  local simple_loss_scale="$6"
  local lm_scale="$7"
  local am_scale="$8"
  local max_duration="$9"
  local exp_dir_rel="${EXP_PREFIX}_${run_id}_seed${seed}"

  local -a cmd=(
    python3 zipformer/train.py
    --world-size "${WORLD_SIZE}"
    --num-epochs "${NUM_EPOCHS}"
    --start-epoch 1
    --use-fp16 1
    --exp-dir "${exp_dir_rel}"
    --subset M
    --bpe-model "${BPE_MODEL}"
    --seed "${seed}"
    --enable-musan "${enable_musan}"
    --enable-spec-aug "${enable_spec_aug}"
    --prune-range "${prune_range}"
    --simple-loss-scale "${simple_loss_scale}"
    --lm-scale "${lm_scale}"
    --am-scale "${am_scale}"
    --max-duration "${max_duration}"
  )

  local pair status elapsed
  if [[ -n "${GPUS}" ]]; then
    pair="$(run_timed env CUDA_VISIBLE_DEVICES="${GPUS}" "${cmd[@]}")"
  else
    pair="$(run_timed "${cmd[@]}")"
  fi
  status="${pair%%,*}"
  elapsed="${pair##*,}"

  if [[ "${status}" == "0" ]]; then
    echo "train,${run_id},${seed},${exp_dir_rel},NA,${NUM_EPOCHS},NA,NA,NA,${enable_musan},${enable_spec_aug},${prune_range},${simple_loss_scale},${lm_scale},${am_scale},${max_duration},${elapsed},success,ablation_train" >> "${CSV_FILE}"
  else
    echo "train,${run_id},${seed},${exp_dir_rel},NA,${NUM_EPOCHS},NA,NA,NA,${enable_musan},${enable_spec_aug},${prune_range},${simple_loss_scale},${lm_scale},${am_scale},${max_duration},${elapsed},fail,ablation_train" >> "${CSV_FILE}"
  fi
  echo "${status}"
}

decode_once() {
  local run_id="$1"
  local seed="$2"
  local method="$3"
  local beam_size="$4"
  local enable_musan="$5"
  local enable_spec_aug="$6"
  local prune_range="$7"
  local simple_loss_scale="$8"
  local lm_scale="$9"
  local am_scale="${10}"
  local max_duration="${11}"
  local exp_dir_rel="${EXP_PREFIX}_${run_id}_seed${seed}"

  local -a cmd=(
    python3 zipformer/decode.py
    --epoch "${NUM_EPOCHS}"
    --avg "${DECODE_AVG}"
    --exp-dir "${exp_dir_rel}"
    --max-duration "${DECODE_MAX_DURATION}"
    --decoding-method "${method}"
    --use-averaged-model 1
    --bpe-model "${BPE_MODEL}"
  )
  if [[ "${method}" == "modified_beam_search" ]]; then
    cmd+=(--beam-size "${beam_size}")
  fi

  local pair status elapsed
  if [[ -n "${DECODE_GPU}" ]]; then
    pair="$(run_timed env CUDA_VISIBLE_DEVICES="${DECODE_GPU}" "${cmd[@]}")"
  else
    pair="$(run_timed "${cmd[@]}")"
  fi
  status="${pair%%,*}"
  elapsed="${pair##*,}"

  if [[ "${status}" == "0" ]]; then
    echo "decode,${run_id},${seed},${exp_dir_rel},${method},${NUM_EPOCHS},${DECODE_AVG},1,${beam_size},${enable_musan},${enable_spec_aug},${prune_range},${simple_loss_scale},${lm_scale},${am_scale},${max_duration},${elapsed},success,ablation_decode" >> "${CSV_FILE}"
  else
    echo "decode,${run_id},${seed},${exp_dir_rel},${method},${NUM_EPOCHS},${DECODE_AVG},1,${beam_size},${enable_musan},${enable_spec_aug},${prune_range},${simple_loss_scale},${lm_scale},${am_scale},${max_duration},${elapsed},fail,ablation_decode" >> "${CSV_FILE}"
  fi
}

build_experiments() {
  local stage="$1"
  case "${stage}" in
    aug)
      cat <<EOF
aug_base|1|1|5|0.5|0.25|0.0|${MAX_DURATION_DEFAULT}
aug_no_musan|0|1|5|0.5|0.25|0.0|${MAX_DURATION_DEFAULT}
aug_no_specaug|1|0|5|0.5|0.25|0.0|${MAX_DURATION_DEFAULT}
aug_no_aug|0|0|5|0.5|0.25|0.0|${MAX_DURATION_DEFAULT}
EOF
      ;;
    loss)
      cat <<EOF
loss_base|1|1|5|0.5|0.25|0.0|${MAX_DURATION_DEFAULT}
loss_pr3|1|1|3|0.5|0.25|0.0|${MAX_DURATION_DEFAULT}
loss_pr7|1|1|7|0.5|0.25|0.0|${MAX_DURATION_DEFAULT}
loss_simple03|1|1|5|0.3|0.25|0.0|${MAX_DURATION_DEFAULT}
loss_simple07|1|1|5|0.7|0.25|0.0|${MAX_DURATION_DEFAULT}
loss_lm020_am005|1|1|5|0.5|0.20|0.05|${MAX_DURATION_DEFAULT}
loss_lm030_am000|1|1|5|0.5|0.30|0.0|${MAX_DURATION_DEFAULT}
EOF
      ;;
    batch)
      cat <<EOF
batch_800|1|1|5|0.5|0.25|0.0|800
batch_1000|1|1|5|0.5|0.25|0.0|1000
batch_1200|1|1|5|0.5|0.25|0.0|1200
EOF
      ;;
    all)
      build_experiments aug
      build_experiments loss
      build_experiments batch
      ;;
    *)
      echo "Unknown stage: ${stage}" >&2
      exit 1
      ;;
  esac
}

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"
IFS=',' read -r -a DECODE_METHOD_ARR <<< "${DECODE_METHODS}"

EXPERIMENTS="$(build_experiments "${STAGE}")"

pushd "${ASR_DIR}" >/dev/null
while IFS='|' read -r run_id enable_musan enable_spec_aug prune_range simple_loss_scale lm_scale am_scale max_duration; do
  [[ -z "${run_id}" ]] && continue
  for seed in "${SEED_ARR[@]}"; do
    status="$(train_once "${run_id}" "${seed}" "${enable_musan}" "${enable_spec_aug}" "${prune_range}" "${simple_loss_scale}" "${lm_scale}" "${am_scale}" "${max_duration}")"
    exp_dir_rel="${EXP_PREFIX}_${run_id}_seed${seed}"
    if [[ "${status}" != "0" && ! -f "${exp_dir_rel}/epoch-${NUM_EPOCHS}.pt" ]]; then
      echo "decode,${run_id},${seed},${exp_dir_rel},NA,${NUM_EPOCHS},${DECODE_AVG},1,NA,${enable_musan},${enable_spec_aug},${prune_range},${simple_loss_scale},${lm_scale},${am_scale},${max_duration},0,skip,no_checkpoint" >> "${CSV_FILE}"
      continue
    fi
    for method in "${DECODE_METHOD_ARR[@]}"; do
      if [[ "${method}" == "modified_beam_search" ]]; then
        decode_once "${run_id}" "${seed}" "${method}" 4 "${enable_musan}" "${enable_spec_aug}" "${prune_range}" "${simple_loss_scale}" "${lm_scale}" "${am_scale}" "${max_duration}" || true
      else
        decode_once "${run_id}" "${seed}" "${method}" 0 "${enable_musan}" "${enable_spec_aug}" "${prune_range}" "${simple_loss_scale}" "${lm_scale}" "${am_scale}" "${max_duration}" || true
      fi
    done
  done
done <<< "${EXPERIMENTS}"
popd >/dev/null

echo "Ablation stage (${STAGE}) finished. CSV: ${CSV_FILE}"

