#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ICEFALL_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
RECIPE_DIR="${ICEFALL_ROOT}/egs/gigaspeech_16k/ASR"
VALIDATION_ROOT="$(cd -- "${ICEFALL_ROOT}/.." && pwd)/experiments/main_flow_validation/giga16k"

resolve_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf "%s\n" "$path"
  else
    printf "%s\n" "${SCRIPT_DIR}/${path}"
  fi
}

find_exp_dir() {
  local base_dir="$1"
  if [[ -f "${base_dir}/epoch-${EPOCH}.pt" ]]; then
    printf "%s\n" "${base_dir}"
    return 0
  fi

  mapfile -t matches < <(find "${base_dir}" -maxdepth 1 -mindepth 1 -type d -name 'zipformer_m_g*' | sort)
  if [[ "${#matches[@]}" -eq 0 ]]; then
    echo "No GigaSpeech exp dir found under ${base_dir}" >&2
    exit 1
  fi

  printf "%s\n" "${matches[-1]}"
}

DATA_ROOT="${VALIDATION_ROOT}/workspace/data"
EXP_ROOT="${VALIDATION_ROOT}/exp/smoke"
EXP_DIR=""
EXPORT_ROOT="${VALIDATION_ROOT}/exports"
EPOCH="${EPOCH:-1}"
AVG="${AVG:-1}"
USE_AVERAGED_MODEL="${USE_AVERAGED_MODEL:-true}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --exp-root) EXP_ROOT="$2"; shift 2 ;;
    --exp-dir) EXP_DIR="$2"; shift 2 ;;
    --export-root) EXPORT_ROOT="$2"; shift 2 ;;
    --epoch) EPOCH="$2"; shift 2 ;;
    --avg) AVG="$2"; shift 2 ;;
    --help)
      cat <<'EOF'
Usage: test/recipes/giga16k/run_decode_export.sh [options]

Options:
  --data-root ../experiments/main_flow_validation/giga16k/workspace/data
  --exp-root ../experiments/main_flow_validation/giga16k/exp/smoke
  --exp-dir  ../experiments/main_flow_validation/giga16k/exp/smoke/zipformer_m_g0
  --export-root ../experiments/main_flow_validation/giga16k/exports
  --epoch 1
  --avg 1
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if (( EPOCH <= AVG )); then
  USE_AVERAGED_MODEL=false
fi

DATA_ROOT="$(resolve_path "${DATA_ROOT}")"
EXP_ROOT="$(resolve_path "${EXP_ROOT}")"
EXPORT_ROOT="$(resolve_path "${EXPORT_ROOT}")"
if [[ -n "${EXP_DIR}" ]]; then
  EXP_DIR="$(resolve_path "${EXP_DIR}")"
else
  EXP_DIR="$(find_exp_dir "${EXP_ROOT}")"
fi

mkdir -p "${EXPORT_ROOT}"

(cd "${RECIPE_DIR}" && python zipformer/decode.py \
  --epoch "${EPOCH}" \
  --avg "${AVG}" \
  --use-averaged-model "${USE_AVERAGED_MODEL}" \
  --exp-dir "${EXP_DIR}" \
  --manifest-dir "${DATA_ROOT}/fbank" \
  --bpe-model "${DATA_ROOT}/lang_bpe_500/bpe.model" \
  --lang-dir "${DATA_ROOT}/lang_bpe_500" \
  --max-duration 100 \
  --num-workers 0 \
  --decoding-method greedy_search)

(cd "${RECIPE_DIR}" && python zipformer/export.py \
  --epoch "${EPOCH}" \
  --avg "${AVG}" \
  --use-averaged-model "${USE_AVERAGED_MODEL}" \
  --exp-dir "${EXP_DIR}" \
  --tokens "${DATA_ROOT}/lang_bpe_500/tokens.txt")

cp -f "${EXP_DIR}/pretrained.pt" "${EXPORT_ROOT}/pretrained.pt"
echo "Decoded and exported 16k artifacts into ${EXP_DIR} and ${EXPORT_ROOT}"
