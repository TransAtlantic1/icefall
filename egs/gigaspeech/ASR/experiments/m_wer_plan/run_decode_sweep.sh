#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

GPUS=""
EXP_DIR=""  # relative to ASR dir
EPOCH=40
AVG_LIST="5,9,15"
USE_AVERAGED_MODEL_LIST="0,1"
BEAM_SIZE_LIST="2,4,6,8"
DECODE_MAX_DURATION=1000
BPE_MODEL="data/lang_bpe_500/bpe.model"

usage() {
  cat <<EOF
Usage: $(basename "$0") --exp-dir RELPATH [options]

Required:
  --exp-dir RELPATH              Experiment directory relative to ASR dir, e.g. zipformer/exp_m_baseline_seed42

Options:
  --asr-dir PATH                 ASR recipe dir (default: ${ASR_DIR})
  --gpus IDS                     GPU ids, e.g. 0 (default: empty -> CPU/auto)
  --epoch N                      Decode epoch (default: ${EPOCH})
  --avg-list CSV                 Avg list (default: ${AVG_LIST})
  --use-averaged-model-list CSV  0/1 list (default: ${USE_AVERAGED_MODEL_LIST})
  --beam-size-list CSV           Beam-size list for modified beam search (default: ${BEAM_SIZE_LIST})
  --decode-max-duration N        Decode max-duration (default: ${DECODE_MAX_DURATION})
  --bpe-model PATH               BPE model relative to ASR dir (default: ${BPE_MODEL})
  -h, --help                     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asr-dir) ASR_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --exp-dir) EXP_DIR="$2"; shift 2 ;;
    --epoch) EPOCH="$2"; shift 2 ;;
    --avg-list) AVG_LIST="$2"; shift 2 ;;
    --use-averaged-model-list) USE_AVERAGED_MODEL_LIST="$2"; shift 2 ;;
    --beam-size-list) BEAM_SIZE_LIST="$2"; shift 2 ;;
    --decode-max-duration) DECODE_MAX_DURATION="$2"; shift 2 ;;
    --bpe-model) BPE_MODEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${EXP_DIR}" ]]; then
  echo "--exp-dir is required"
  usage
  exit 1
fi

if [[ -n "${GPUS}" ]]; then
  DECODE_GPU="${GPUS%%,*}"
else
  DECODE_GPU=""
fi

CSV_FILE="${RESULTS_DIR}/decode_sweep_runs.csv"
if [[ ! -f "${CSV_FILE}" ]]; then
  echo "phase,exp_dir,decoding_method,epoch,avg,use_averaged_model,beam_size,elapsed_sec,status,notes" > "${CSV_FILE}"
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

run_decode() {
  local method="$1"
  local avg="$2"
  local use_avg_model="$3"
  local beam_size="$4"

  local -a cmd=(
    python3 zipformer/decode.py
    --epoch "${EPOCH}"
    --avg "${avg}"
    --exp-dir "${EXP_DIR}"
    --max-duration "${DECODE_MAX_DURATION}"
    --decoding-method "${method}"
    --use-averaged-model "${use_avg_model}"
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
    echo "decode,${EXP_DIR},${method},${EPOCH},${avg},${use_avg_model},${beam_size},${elapsed},success,decode_sweep" >> "${CSV_FILE}"
  else
    echo "decode,${EXP_DIR},${method},${EPOCH},${avg},${use_avg_model},${beam_size},${elapsed},fail,decode_sweep" >> "${CSV_FILE}"
  fi
}

IFS=',' read -r -a AVG_ARR <<< "${AVG_LIST}"
IFS=',' read -r -a USE_AVG_ARR <<< "${USE_AVERAGED_MODEL_LIST}"
IFS=',' read -r -a BEAM_ARR <<< "${BEAM_SIZE_LIST}"

pushd "${ASR_DIR}" >/dev/null
for avg in "${AVG_ARR[@]}"; do
  for use_avg in "${USE_AVG_ARR[@]}"; do
    run_decode "greedy_search" "${avg}" "${use_avg}" 0 || true
    for beam in "${BEAM_ARR[@]}"; do
      run_decode "modified_beam_search" "${avg}" "${use_avg}" "${beam}" || true
    done
  done
done
popd >/dev/null

echo "Decode sweep finished. CSV: ${CSV_FILE}"

