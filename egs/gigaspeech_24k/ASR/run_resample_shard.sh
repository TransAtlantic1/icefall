#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ICEFALL_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PARSE_OPTIONS_SH="${ICEFALL_ROOT}/icefall/shared/parse_options.sh"

instance_index=0
num_instances=4
recording_num_splits=1000
resample_num_workers=24
target_sample_rate=24000
log_root=""

. "${PARSE_OPTIONS_SH}" || exit 1

if [ -z "$log_root" ]; then
  log_root="${SCRIPT_DIR}/data/logs"
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

log_file="${log_root}/resample.${resample_start}-$((resample_stop - 1)).log"

echo "$0: instance_index=${instance_index}/${num_instances}"
echo "$0: shard_range=[${resample_start}, ${resample_stop})"
echo "$0: log_file=${log_file}"

exec bash "${SCRIPT_DIR}/prepare.sh" \
  --recording-num-splits "$recording_num_splits" \
  --target-sample-rate "$target_sample_rate" \
  --resample-num-workers "$resample_num_workers" \
  --stage 4 \
  --stop-stage 4 \
  --resample-start "$resample_start" \
  --resample-stop "$resample_stop" \
  >>"$log_file" 2>&1
