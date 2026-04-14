#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ICEFALL_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)
PARSE_OPTIONS_SH="${ICEFALL_ROOT}/icefall/shared/parse_options.sh"

mode=prepare-subset
language=zh
artifact_root="/inspire/hdd/project/embodied-multimodality/public/emilia/fc71e07/icefall_emilia_zh_24k"
config_path="${SCRIPT_DIR}/configs/train_zh.yaml"
train_split_name="train_split_1000"
train_shard_ids="0000"
subset_name=""
subset_root=""
run_base=""
run_id=""
exp_dir=""
world_size=8
master_port=12460
cuda_visible_devices="0,1,2,3,4,5,6,7"
max_duration=-1
num_workers=4
num_buckets=30
use_wandb=false
tensorboard=false
profiler_timeout=90
gpu_monitor_interval=1

. "${PARSE_OPTIONS_SH}" || exit 1

if [[ "$mode" != "prepare-subset" && "$mode" != "profiler" && "$mode" != "smoke" && "$mode" != "full-smoke" ]]; then
  echo "$0: --mode must be one of prepare-subset, profiler, smoke, full-smoke"
  exit 1
fi

if [[ "$language" != "zh" ]]; then
  echo "$0: this helper currently supports only --language zh"
  exit 1
fi

if [ ! -f "$config_path" ]; then
  echo "$0: missing config_path=$config_path"
  exit 1
fi

manifest_source_dir="${artifact_root}/data/fbank/${language}"
train_split_dir="${manifest_source_dir}/${train_split_name}"
if [ "$language" = "zh" ]; then
  lang_dir="${artifact_root}/data/lang_hybrid_zh"
else
  lang_dir="${artifact_root}/data/lang_bpe_${language}_2000"
fi

if [ ! -d "$manifest_source_dir" ]; then
  echo "$0: missing manifest source dir $manifest_source_dir"
  exit 1
fi

if [ ! -d "$train_split_dir" ]; then
  echo "$0: missing processed train split dir $train_split_dir"
  exit 1
fi

if [ ! -d "$lang_dir" ]; then
  echo "$0: missing lang dir $lang_dir"
  exit 1
fi

if [ -z "$subset_name" ]; then
  subset_name=$(echo "$train_shard_ids" | tr ',' '_')
fi

if [ -z "$subset_root" ]; then
  subset_root="/tmp/emilia_${language}_subset_${subset_name}"
fi

if [ -z "$run_base" ]; then
  run_base="${artifact_root}/exp/zipformer/manual-smoke"
fi

if [ "$max_duration" -lt 0 ]; then
  if [ "$mode" = "full-smoke" ]; then
    max_duration=1000
  else
    max_duration=800
  fi
fi

mkdir -p "$subset_root" "$run_base"

prepare_subset() {
  local split_dir info_file shard_id shard_src shard_dst
  split_dir="${subset_root}/train_split_subset"
  mkdir -p "$split_dir"

  IFS=',' read -r -a shard_ids <<<"$train_shard_ids"
  if [ "${#shard_ids[@]}" -eq 0 ]; then
    echo "$0: no shard ids provided in --train-shard-ids"
    exit 1
  fi

  for shard_id in "${shard_ids[@]}"; do
    shard_id=$(printf "%04d" "$((10#$shard_id))")
    shard_src="${train_split_dir}/emilia_${language}_cuts_train.${shard_id}.jsonl.gz"
    shard_dst="${split_dir}/emilia_${language}_cuts_train.${shard_id}.jsonl.gz"
    if [ ! -f "$shard_src" ]; then
      echo "$0: missing processed shard $shard_src"
      exit 1
    fi
    ln -sfn "$shard_src" "$shard_dst"
  done

  ln -sfn "${manifest_source_dir}/emilia_${language}_cuts_dev.jsonl.gz" \
    "${subset_root}/emilia_${language}_cuts_dev.jsonl.gz"
  ln -sfn "${manifest_source_dir}/emilia_${language}_cuts_test.jsonl.gz" \
    "${subset_root}/emilia_${language}_cuts_test.jsonl.gz"

  info_file="${subset_root}/subset_info.txt"
  {
    printf 'created_utc=%s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    printf 'language=%s\n' "$language"
    printf 'artifact_root=%s\n' "$artifact_root"
    printf 'manifest_source_dir=%s\n' "$manifest_source_dir"
    printf 'train_split_dir=%s\n' "$train_split_dir"
    printf 'train_shard_ids=%s\n' "$train_shard_ids"
    printf 'subset_root=%s\n' "$subset_root"
  } >"$info_file"

  echo "$0: prepared subset_root=$subset_root"
  echo "$0: subset_info=$info_file"
}

prepare_subset

if [ "$mode" = "prepare-subset" ]; then
  exit 0
fi

if [ -z "$run_id" ]; then
  run_id="${mode}.${language}.subset-${subset_name}.md${max_duration}.$(date +%Y%m%d_%H%M%S)"
fi

if [ -z "$exp_dir" ]; then
  exp_dir="${run_base}/${run_id}"
fi
mkdir -p "$exp_dir"

export PYTHONPATH="${ICEFALL_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$cuda_visible_devices"

manifest_dir="$subset_root"
if [ "$mode" = "full-smoke" ]; then
  manifest_dir="$manifest_source_dir"
fi

cmd=(
  python zipformer/train.py
  --config "$config_path"
  --manifest-dir "$manifest_dir"
  --exp-dir "$exp_dir"
  --world-size "$world_size"
  --master-port "$master_port"
  --use-wandb "$use_wandb"
  --tensorboard "$tensorboard"
  --max-duration "$max_duration"
  --num-workers "$num_workers"
  --num-buckets "$num_buckets"
)

printf '%q ' "${cmd[@]}" >"${exp_dir}/launch_cmd.txt"
printf '\n' >>"${exp_dir}/launch_cmd.txt"

echo "$0: mode=$mode"
echo "$0: exp_dir=$exp_dir"
echo "$0: manifest_dir=$manifest_dir"
echo "$0: lang_dir=$lang_dir"
echo "$0: train_shard_ids=$train_shard_ids"
echo "$0: max_duration=$max_duration"

gpu_monitor_pid=""
cleanup() {
  if [ -n "$gpu_monitor_pid" ] && kill -0 "$gpu_monitor_pid" 2>/dev/null; then
    kill "$gpu_monitor_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [ "$mode" = "profiler" ]; then
  if ! command -v nvprof >/dev/null 2>&1; then
    echo "$0: nvprof is not available in PATH"
    exit 1
  fi
  profiler_cmd=(
    nvprof
    --profile-from-start on
    --timeout "$profiler_timeout"
    --csv
    --print-gpu-trace
    --log-file "${exp_dir}/nvprof.trace.%p.csv"
  )
  if [ "$world_size" -gt 1 ]; then
    profiler_cmd+=(--profile-child-processes)
  fi
  profiler_cmd+=("${cmd[@]}")
  printf '%q ' "${profiler_cmd[@]}" >"${exp_dir}/launch_cmd.txt"
  printf '\n' >>"${exp_dir}/launch_cmd.txt"
  "${profiler_cmd[@]}"
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "$0: nvidia-smi is not available in PATH"
  exit 1
fi

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm \
  --format=csv -l "$gpu_monitor_interval" >"${exp_dir}/gpu_1s.csv" &
gpu_monitor_pid=$!

stdbuf -oL -eL "${cmd[@]}" 2>&1 | tee "${exp_dir}/launcher.stdout.log"
