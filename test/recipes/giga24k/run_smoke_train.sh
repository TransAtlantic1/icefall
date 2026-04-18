#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
recipe_dir="${repo_root}/egs/gigaspeech_24k/ASR"
validation_root_default="$(cd -- "${repo_root}/.." && pwd)/experiments/main_flow_validation/giga24k"

resolve_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf "%s\n" "$path"
  else
    printf "%s\n" "${script_dir}/${path}"
  fi
}

gpus="0,1"
exp_root="${validation_root_default}/exp/smoke"
data_root="${validation_root_default}/workspace/data"
master_port=12384
max_duration=200
smoke_num_batches=0
num_workers=0
use_fp16=1
skip_validation=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) gpus="$2"; shift 2 ;;
    --exp-root) exp_root="$2"; shift 2 ;;
    --data-root) data_root="$2"; shift 2 ;;
    --master-port) master_port="$2"; shift 2 ;;
    --max-duration) max_duration="$2"; shift 2 ;;
    --smoke-num-batches) smoke_num_batches="$2"; shift 2 ;;
    --num-workers) num_workers="$2"; shift 2 ;;
    --use-fp16) use_fp16="$2"; shift 2 ;;
    --skip-validation) skip_validation="$2"; shift 2 ;;
    --help)
      cat <<'EOF'
Usage: test/recipes/giga24k/run_smoke_train.sh [options]

Options:
  --gpus 0,1
  --exp-root ../experiments/main_flow_validation/giga24k/exp/smoke
  --data-root ../experiments/main_flow_validation/giga24k/workspace/data
  --master-port 12384
  --max-duration 200
  --smoke-num-batches 0
  --num-workers 0
  --use-fp16 1
  --skip-validation true
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

exp_root="$(resolve_path "${exp_root}")"
data_root="$(resolve_path "${data_root}")"

IFS=',' read -r -a gpu_ids <<<"${gpus}"
world_size="${#gpu_ids[@]}"

EXP_ROOT="${exp_root}" \
DATA_ROOT="${data_root}" \
CUDA_VISIBLE_DEVICES="${gpus}" \
WORLD_SIZE="${world_size}" \
MASTER_PORT="${master_port}" \
USE_WANDB=False \
TENSORBOARD=False \
NUM_EPOCHS=1 \
USE_FP16="${use_fp16}" \
SMOKE_NUM_BATCHES="${smoke_num_batches}" \
SMOKE_SKIP_VALIDATION="${skip_validation}" \
bash "${recipe_dir}/run_train_offline.sh" \
  --small-dev true \
  --num-workers "${num_workers}" \
  --max-duration "${max_duration}"
