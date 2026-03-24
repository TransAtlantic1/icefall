#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

GPUS=""
SEEDS="42,777"
NUM_EPOCHS=40
MAX_DURATION=1000
DECODE_MAX_DURATION=1000
DECODE_AVG=9
BPE_MODEL="data/lang_bpe_500/bpe.model"
TRAIN=1
DECODE=1
EXP_PREFIX="zipformer/exp_m_baseline"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --asr-dir PATH                 ASR recipe dir (default: ${ASR_DIR})
  --gpus IDS                     GPU ids, e.g. 0,1 (default: empty -> CPU/auto)
  --seeds CSV                    Seeds, e.g. 42,777 (default: ${SEEDS})
  --num-epochs N                 Training epochs (default: ${NUM_EPOCHS})
  --max-duration N               Train max-duration (default: ${MAX_DURATION})
  --decode-max-duration N        Decode max-duration (default: ${DECODE_MAX_DURATION})
  --decode-avg N                 Decode avg checkpoints (default: ${DECODE_AVG})
  --bpe-model PATH               BPE model relative to ASR dir (default: ${BPE_MODEL})
  --train 0|1                    Run training phase (default: ${TRAIN})
  --decode 0|1                   Run decode phase (default: ${DECODE})
  --exp-prefix RELPATH           Relative exp prefix (default: ${EXP_PREFIX})
  -h, --help                     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asr-dir) ASR_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --num-epochs) NUM_EPOCHS="$2"; shift 2 ;;
    --max-duration) MAX_DURATION="$2"; shift 2 ;;
    --decode-max-duration) DECODE_MAX_DURATION="$2"; shift 2 ;;
    --decode-avg) DECODE_AVG="$2"; shift 2 ;;
    --bpe-model) BPE_MODEL="$2"; shift 2 ;;
    --train) TRAIN="$2"; shift 2 ;;
    --decode) DECODE="$2"; shift 2 ;;
    --exp-prefix) EXP_PREFIX="$2"; shift 2 ;;
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

CSV_FILE="${RESULTS_DIR}/baseline_runs.csv"
if [[ ! -f "${CSV_FILE}" ]]; then
  echo "phase,seed,exp_dir,decoding_method,epoch,avg,use_averaged_model,beam_size,elapsed_sec,status,notes" > "${CSV_FILE}"
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

run_train() {
  local seed="$1"
  local exp_dir_rel="$2"
  local -a cmd=(
    python3 zipformer/train.py
    --world-size "${WORLD_SIZE}"
    --num-epochs "${NUM_EPOCHS}"
    --start-epoch 1
    --use-fp16 1
    --exp-dir "${exp_dir_rel}"
    --max-duration "${MAX_DURATION}"
    --subset M
    --bpe-model "${BPE_MODEL}"
    --seed "${seed}"
  )

  local status elapsed pair
  if [[ -n "${GPUS}" ]]; then
    pair="$(run_timed env CUDA_VISIBLE_DEVICES="${GPUS}" "${cmd[@]}")"
  else
    pair="$(run_timed "${cmd[@]}")"
  fi
  status="${pair%%,*}"
  elapsed="${pair##*,}"

  if [[ "${status}" == "0" ]]; then
    echo "train,${seed},${exp_dir_rel},NA,${NUM_EPOCHS},NA,NA,NA,${elapsed},success,baseline_train" >> "${CSV_FILE}"
  else
    echo "train,${seed},${exp_dir_rel},NA,${NUM_EPOCHS},NA,NA,NA,${elapsed},fail,baseline_train" >> "${CSV_FILE}"
  fi
  return "${status}"
}

run_decode() {
  local seed="$1"
  local exp_dir_rel="$2"
  local method="$3"
  local use_averaged_model="$4"
  local beam_size="$5"

  local -a cmd=(
    python3 zipformer/decode.py
    --epoch "${NUM_EPOCHS}"
    --avg "${DECODE_AVG}"
    --exp-dir "${exp_dir_rel}"
    --max-duration "${DECODE_MAX_DURATION}"
    --decoding-method "${method}"
    --use-averaged-model "${use_averaged_model}"
    --bpe-model "${BPE_MODEL}"
  )
  if [[ "${method}" == "modified_beam_search" ]]; then
    cmd+=(--beam-size "${beam_size}")
  fi

  local status elapsed pair
  if [[ -n "${DECODE_GPU}" ]]; then
    pair="$(run_timed env CUDA_VISIBLE_DEVICES="${DECODE_GPU}" "${cmd[@]}")"
  else
    pair="$(run_timed "${cmd[@]}")"
  fi
  status="${pair%%,*}"
  elapsed="${pair##*,}"

  if [[ "${status}" == "0" ]]; then
    echo "decode,${seed},${exp_dir_rel},${method},${NUM_EPOCHS},${DECODE_AVG},${use_averaged_model},${beam_size},${elapsed},success,baseline_decode" >> "${CSV_FILE}"
  else
    echo "decode,${seed},${exp_dir_rel},${method},${NUM_EPOCHS},${DECODE_AVG},${use_averaged_model},${beam_size},${elapsed},fail,baseline_decode" >> "${CSV_FILE}"
  fi
  return "${status}"
}

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"

pushd "${ASR_DIR}" >/dev/null
for seed in "${SEED_ARR[@]}"; do
  exp_dir_rel="${EXP_PREFIX}_seed${seed}"
  mkdir -p "${exp_dir_rel}"

  train_ok=1
  if [[ "${TRAIN}" == "1" ]]; then
    if ! run_train "${seed}" "${exp_dir_rel}"; then
      train_ok=0
    fi
  fi

  if [[ "${DECODE}" == "1" ]]; then
    ckpt_path="${exp_dir_rel}/epoch-${NUM_EPOCHS}.pt"
    if [[ "${train_ok}" == "0" && ! -f "${ckpt_path}" ]]; then
      echo "decode,${seed},${exp_dir_rel},NA,${NUM_EPOCHS},${DECODE_AVG},NA,NA,0,skip,no_checkpoint" >> "${CSV_FILE}"
      continue
    fi

    run_decode "${seed}" "${exp_dir_rel}" "greedy_search" 1 0 || true
    run_decode "${seed}" "${exp_dir_rel}" "modified_beam_search" 1 4 || true
  fi
done
popd >/dev/null

echo "Baseline stage finished. CSV: ${CSV_FILE}"

