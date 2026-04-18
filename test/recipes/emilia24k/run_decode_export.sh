#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ICEFALL_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
RECIPE_DIR="${ICEFALL_ROOT}/egs/emilia_24k_multilang/emilia_24k_ZH/ASR"
VALIDATION_ROOT="$(cd -- "${ICEFALL_ROOT}/.." && pwd)/experiments/main_flow_validation/emilia24k"

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

  mapfile -t matches < <(find "${base_dir}" -maxdepth 1 -mindepth 1 -type d | sort)
  if [[ "${#matches[@]}" -eq 0 ]]; then
    echo "No Emilia exp dir found under ${base_dir}" >&2
    exit 1
  fi

  printf "%s\n" "${matches[-1]}"
}

ARTIFACT_ROOT="${VALIDATION_ROOT}/workspace/artifacts"
EXP_ROOT="${VALIDATION_ROOT}/exp/smoke"
EXP_DIR=""
EXPORT_ROOT="${VALIDATION_ROOT}/exports"
LANGUAGE="${LANGUAGE:-zh}"
EPOCH="${EPOCH:-1}"
AVG="${AVG:-1}"
USE_AVERAGED_MODEL="${USE_AVERAGED_MODEL:-true}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-root) ARTIFACT_ROOT="$2"; shift 2 ;;
    --exp-root) EXP_ROOT="$2"; shift 2 ;;
    --exp-dir) EXP_DIR="$2"; shift 2 ;;
    --export-root) EXPORT_ROOT="$2"; shift 2 ;;
    --language) LANGUAGE="$2"; shift 2 ;;
    --epoch) EPOCH="$2"; shift 2 ;;
    --avg) AVG="$2"; shift 2 ;;
    --help)
      cat <<'EOF'
Usage: test/recipes/emilia24k/run_decode_export.sh [options]

Options:
  --artifact-root ../experiments/main_flow_validation/emilia24k/workspace/artifacts
  --exp-root ../experiments/main_flow_validation/emilia24k/exp/smoke
  --exp-dir ../experiments/main_flow_validation/emilia24k/exp/smoke/<run_id>
  --export-root ../experiments/main_flow_validation/emilia24k/exports
  --language zh
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

ARTIFACT_ROOT="$(resolve_path "${ARTIFACT_ROOT}")"
EXP_ROOT="$(resolve_path "${EXP_ROOT}")"
EXPORT_ROOT="$(resolve_path "${EXPORT_ROOT}")"
if [[ -n "${EXP_DIR}" ]]; then
  EXP_DIR="$(resolve_path "${EXP_DIR}")"
else
  EXP_DIR="$(find_exp_dir "${EXP_ROOT}")"
fi

mkdir -p "${EXPORT_ROOT}"

EXPORT_TOKENS="${EXPORT_ROOT}/tokens.export.txt"
awk '
  $1 == "#0" {$1 = "<unused_disambig_0>"}
  {print $1, $2}
' "${ARTIFACT_ROOT}/data/lang_hybrid_zh/tokens.txt" >"${EXPORT_TOKENS}"

(cd "${RECIPE_DIR}" && python zipformer/decode.py \
  --language "${LANGUAGE}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --epoch "${EPOCH}" \
  --avg "${AVG}" \
  --use-averaged-model "${USE_AVERAGED_MODEL}" \
  --exp-dir "${EXP_DIR}" \
  --bpe-model "${ARTIFACT_ROOT}/data/lang_hybrid_zh/bpe.model" \
  --lang-dir "${ARTIFACT_ROOT}/data/lang_hybrid_zh" \
  --max-duration 100 \
  --num-workers 0 \
  --decoding-method greedy_search)

(cd "${RECIPE_DIR}" && python zipformer/export.py \
  --language "${LANGUAGE}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --epoch "${EPOCH}" \
  --avg "${AVG}" \
  --use-averaged-model "${USE_AVERAGED_MODEL}" \
  --exp-dir "${EXP_DIR}" \
  --tokens "${EXPORT_TOKENS}")

cp -f "${EXP_DIR}/pretrained.pt" "${EXPORT_ROOT}/pretrained.pt"
echo "Decoded and exported Emilia artifacts into ${EXP_DIR} and ${EXPORT_ROOT}"
