#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
icefall_root="$(cd -- "${script_dir}/../../.." && pwd)"

resolve_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf "%s\n" "$path"
  else
    printf "%s\n" "${script_dir}/${path}"
  fi
}

is_truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -f /opt/conda/etc/profile.d/conda.sh ]] && ! command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
fi

if command -v conda >/dev/null 2>&1 && [[ "${CONDA_DEFAULT_ENV:-}" != "icefall" ]]; then
  conda activate icefall
fi

export PYTHONPATH="${icefall_root}:${PYTHONPATH:-}"

exp_root_default="$(cd -- "${icefall_root}/.." && pwd)/experiments/gigaspeech_24k_train"
export EXP_ROOT="${EXP_ROOT:-${exp_root_default}}"
export WANDB_PROJECT="${WANDB_PROJECT:-gigaspeech-24k}"
export WANDB_GROUP="${WANDB_GROUP:-zipformer-m}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WORLD_SIZE="${WORLD_SIZE:-4}"
export MASTER_PORT="${MASTER_PORT:-12364}"
export DATA_ROOT="$(resolve_path "${DATA_ROOT:-data}")"
export FBANK_DIR="$(resolve_path "${FBANK_DIR:-${DATA_ROOT}/fbank}")"
export BPE_MODEL="$(resolve_path "${BPE_MODEL:-${DATA_ROOT}/lang_bpe_500/bpe.model}")"
export SMOKE_NUM_BATCHES="${SMOKE_NUM_BATCHES:-0}"
export SMOKE_SKIP_VALIDATION="${SMOKE_SKIP_VALIDATION:-True}"

gpu_tag="$(printf "%s" "${CUDA_VISIBLE_DEVICES}" | tr ',' '-')"
export EXP_DIR="${EXP_DIR:-${EXP_ROOT}/zipformer_m_g${gpu_tag}}"
export WANDB_DIR="${WANDB_DIR:-${EXP_ROOT}/wandb_offline}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-gsm-m-24k-zipformer-g${gpu_tag}}"
export WANDB_TAGS="${WANDB_TAGS:-gigaspeech,zipformer,subset-m,no-musan,24k,f5tts-mel}"

mkdir -p "${EXP_DIR}" "${WANDB_DIR}"

IFS=',' read -r -a gpu_ids <<<"${CUDA_VISIBLE_DEVICES}"
if [[ "${#gpu_ids[@]}" -ne "${WORLD_SIZE}" ]]; then
  echo "WORLD_SIZE=${WORLD_SIZE} does not match CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

for required_path in \
  "${FBANK_DIR}/gigaspeech_cuts_M.jsonl.gz" \
  "${FBANK_DIR}/gigaspeech_cuts_DEV.jsonl.gz" \
  "${BPE_MODEL}"
do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required input: ${required_path}" >&2
    echo "Finish 24k data preparation before starting training." >&2
    exit 1
  fi
done

FBANK_DIR="${FBANK_DIR}" EXPECTED_FEATURE_DIM=100 python - <<'PY'
from lhotse import load_manifest_lazy
import os

cuts = load_manifest_lazy(os.path.join(os.environ["FBANK_DIR"], "gigaspeech_cuts_DEV.jsonl.gz"))
first = next(iter(cuts))
expected_dim = int(os.environ["EXPECTED_FEATURE_DIM"])
if first.num_features != expected_dim:
    raise SystemExit(f"Expected {expected_dim}-dim features for 24k, got {first.num_features}")
print(f"Validated 24k feature dim: {first.num_features}")
PY

if is_truthy "${USE_WANDB:-True}"; then
  python - <<'PY'
import wandb

print(f"wandb version: {wandb.__version__}")
PY
fi

cd "${script_dir}"

exec python zipformer/train.py \
  --world-size "${WORLD_SIZE}" \
  --master-port "${MASTER_PORT}" \
  --num-epochs "${NUM_EPOCHS:-40}" \
  --use-fp16 "${USE_FP16:-1}" \
  --subset "${SUBSET:-M}" \
  --enable-musan "${ENABLE_MUSAN:-False}" \
  --max-duration "${MAX_DURATION:-1000}" \
  --log-interval "${LOG_INTERVAL:-50}" \
  --valid-interval "${VALID_INTERVAL:-500}" \
  --tensorboard "${TENSORBOARD:-True}" \
  --use-wandb "${USE_WANDB:-True}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-tags "${WANDB_TAGS}" \
  --wandb-resume "${WANDB_RESUME:-allow}" \
  --manifest-dir "${FBANK_DIR}" \
  --bpe-model "${BPE_MODEL}" \
  --smoke-num-batches "${SMOKE_NUM_BATCHES}" \
  --smoke-skip-validation "${SMOKE_SKIP_VALIDATION}" \
  --exp-dir "${EXP_DIR}" \
  "$@"
