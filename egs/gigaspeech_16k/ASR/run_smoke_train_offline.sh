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

gpus="0,1"
exp_root_default="$(cd -- "${icefall_root}/.." && pwd)/experiments/gigaspeech_16k_smoke"
exp_root="${exp_root_default}"
data_root="data"
master_port=12374
max_duration=300
smoke_num_batches=8
num_workers=0
use_fp16=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      gpus="$2"
      shift 2
      ;;
    --exp-root)
      exp_root="$2"
      shift 2
      ;;
    --data-root)
      data_root="$2"
      shift 2
      ;;
    --master-port)
      master_port="$2"
      shift 2
      ;;
    --max-duration)
      max_duration="$2"
      shift 2
      ;;
    --smoke-num-batches)
      smoke_num_batches="$2"
      shift 2
      ;;
    --num-workers)
      num_workers="$2"
      shift 2
      ;;
    --use-fp16)
      use_fp16="$2"
      shift 2
      ;;
    --help)
      cat <<'EOF'
Usage: run_smoke_train_offline.sh [options]

Options:
  --gpus 0,1
  --exp-root /abs/path
  --data-root /abs/path
  --master-port 12374
  --max-duration 300
  --smoke-num-batches 8
  --num-workers 0
  --use-fp16 1
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
SMOKE_SKIP_VALIDATION=True \
bash "${script_dir}/run_train_offline.sh" \
  --small-dev true \
  --num-workers "${num_workers}" \
  --max-duration "${max_duration}"
