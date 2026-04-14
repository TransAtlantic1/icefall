#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_SELF="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
ICEFALL_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PARSE_OPTIONS_SH="${ICEFALL_ROOT}/icefall/shared/parse_options.sh"

language=zh
public_root=/inspire/qb-ilm/project/embodied-multimodality/chenxie-25019/public
dataset_root=""
artifact_root=""
instance_index=0
num_instances=4
recording_num_splits=1000
resample_num_workers=24
target_sample_rate=24000
log_root=""
detach=false
detach_log=""
pid_file=""

. "${PARSE_OPTIONS_SH}" || exit 1

if [[ "$language" != "zh" && "$language" != "en" ]]; then
  echo "$0: --language must be one of zh or en, got: $language"
  exit 1
fi

if [ -z "$dataset_root" ]; then
  dataset_root="/inspire/dataset/emilia/fc71e07"
fi

if [ -z "$artifact_root" ]; then
  artifact_root="${public_root%/}/emilia/fc71e07/icefall_emilia_${language}_24k"
fi

if [ -z "$log_root" ]; then
  log_root="${artifact_root}/logs"
fi

if [ "$num_instances" -le 0 ]; then
  echo "$0: --num-instances must be > 0"
  exit 1
fi

if [ "$instance_index" -lt 0 ] || [ "$instance_index" -ge "$num_instances" ]; then
  echo "$0: --instance-index must be in [0, ${num_instances}), got: $instance_index"
  exit 1
fi

if [ "$recording_num_splits" -le 0 ]; then
  echo "$0: --recording-num-splits must be > 0"
  exit 1
fi

shards_per_instance=$(( (recording_num_splits + num_instances - 1) / num_instances ))
resample_start=$(( instance_index * shards_per_instance ))
resample_stop=$(( resample_start + shards_per_instance ))
if [ "$resample_stop" -gt "$recording_num_splits" ]; then
  resample_stop="$recording_num_splits"
fi

if [ "$resample_start" -ge "$recording_num_splits" ]; then
  echo "$0: computed empty shard range for instance ${instance_index}"
  exit 1
fi

mkdir -p "$log_root"

log_file="${log_root}/resample.${language}.${resample_start}-$((resample_stop - 1)).log"
if [ -z "$detach_log" ]; then
  detach_log="${log_root}/launcher.resample.${language}.${instance_index}of${num_instances}.nohup.log"
fi
if [ -z "$pid_file" ]; then
  pid_file="${log_root}/launcher.resample.${language}.${instance_index}of${num_instances}.pid"
fi

if [ "$detach" = true ]; then
  cmd=(
    "${SCRIPT_SELF}"
    --language "$language"
    --public-root "$public_root"
    --dataset-root "$dataset_root"
    --artifact-root "$artifact_root"
    --instance-index "$instance_index"
    --num-instances "$num_instances"
    --recording-num-splits "$recording_num_splits"
    --resample-num-workers "$resample_num_workers"
    --target-sample-rate "$target_sample_rate"
    --log-root "$log_root"
    --detach false
    --detach-log "$detach_log"
    --pid-file "$pid_file"
  )
  nohup "${cmd[@]}" >>"$detach_log" 2>&1 &
  pid=$!
  echo "$0: detached pid=${pid}"
  echo "$0: launcher_log=${detach_log}"
  echo "$0: pid_file=${pid_file}"
  echo "$0: worker_log=${log_file}"
  exit 0
fi

cleanup_pid_file() {
  if [ -n "$pid_file" ]; then
    rm -f "$pid_file"
  fi
}

echo "$$" >"$pid_file"
trap cleanup_pid_file EXIT

echo "$0: language=${language}"
echo "$0: dataset_root=${dataset_root}"
echo "$0: artifact_root=${artifact_root}"
echo "$0: instance_index=${instance_index}/${num_instances}"
echo "$0: shard_range=[${resample_start}, ${resample_stop})"
echo "$0: detach_log=${detach_log}"
echo "$0: pid_file=${pid_file}"
echo "$0: log_file=${log_file}"

exec bash "${SCRIPT_DIR}/prepare.sh" \
  --language "$language" \
  --dataset-root "$dataset_root" \
  --artifact-root "$artifact_root" \
  --recording-num-splits "$recording_num_splits" \
  --target-sample-rate "$target_sample_rate" \
  --resample-num-workers "$resample_num_workers" \
  --stage 3 \
  --stop-stage 3 \
  --resample-start "$resample_start" \
  --resample-stop "$resample_stop" \
  >>"$log_file" 2>&1
