#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ICEFALL_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PARSE_OPTIONS_SH="${ICEFALL_ROOT}/icefall/shared/parse_options.sh"

language=zh
public_root=/inspire/qb-ilm/project/embodied-multimodality/chenxie-25019/public
artifact_root=""
instance_index=-1
num_instances=4
recording_num_splits=1000
resample_start=-1
resample_stop=-1
log_root=""
pid_file=""
signal=TERM
grace_seconds=15
force_kill=true

. "${PARSE_OPTIONS_SH}" || exit 1

if [[ "$language" != "zh" && "$language" != "en" ]]; then
  echo "$0: --language must be one of zh or en, got: $language"
  exit 1
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

if [ "$recording_num_splits" -le 0 ]; then
  echo "$0: --recording-num-splits must be > 0"
  exit 1
fi

if [ "$resample_start" -lt 0 ] || [ "$resample_stop" -lt 0 ]; then
  if [ "$instance_index" -lt 0 ] || [ "$instance_index" -ge "$num_instances" ]; then
    echo "$0: pass either --resample-start/--resample-stop or a valid --instance-index in [0, ${num_instances}), got: $instance_index"
    exit 1
  fi

  shards_per_instance=$(( (recording_num_splits + num_instances - 1) / num_instances ))
  resample_start=$(( instance_index * shards_per_instance ))
  resample_stop=$(( resample_start + shards_per_instance ))
  if [ "$resample_stop" -gt "$recording_num_splits" ]; then
    resample_stop="$recording_num_splits"
  fi
fi

if [ "$resample_start" -lt 0 ] || [ "$resample_stop" -le "$resample_start" ] || [ "$resample_stop" -gt "$recording_num_splits" ]; then
  echo "$0: invalid shard range [${resample_start}, ${resample_stop})"
  exit 1
fi

if [ -z "$pid_file" ]; then
  if [ "$instance_index" -ge 0 ] && [ "$resample_start" -eq $(( instance_index * ((recording_num_splits + num_instances - 1) / num_instances) )) ]; then
    pid_file="${log_root}/launcher.resample.${language}.${instance_index}of${num_instances}.pid"
  else
    pid_file="${log_root}/launcher.resample.${language}.${resample_start}-${resample_stop}.pid"
  fi
fi

echo "$0: artifact_root=${artifact_root}"
if [ "$instance_index" -ge 0 ]; then
  echo "$0: instance_index=${instance_index}/${num_instances}"
fi
echo "$0: shard_range=[${resample_start}, ${resample_stop})"
echo "$0: pid_file=${pid_file}"

stopped=false
declare -A target_pids=()

add_target_pid() {
  local pid="$1"
  if [[ -z "$pid" || ! "$pid" =~ ^[0-9]+$ ]]; then
    return 0
  fi
  target_pids["$pid"]=1
}

collect_descendants() {
  local parent_pid="$1"
  local child_pid
  while IFS= read -r child_pid; do
    if [[ -z "$child_pid" ]]; then
      continue
    fi
    add_target_pid "$child_pid"
    collect_descendants "$child_pid"
  done < <(pgrep -P "$parent_pid" || true)
}

collect_prepare_range_pids() {
  local pattern="${SCRIPT_DIR}/prepare.sh.*--stage 3.*--stop-stage 3.*--resample-start ${resample_start}.*--resample-stop ${resample_stop}"
  local pid
  while IFS= read -r pid; do
    if [[ -z "$pid" ]]; then
      continue
    fi
    add_target_pid "$pid"
    collect_descendants "$pid"
  done < <(pgrep -f -- "$pattern" || true)
}

collect_resample_range_pids() {
  local line
  local pid
  local cmd
  local shard_id
  local shard_num

  while IFS= read -r line; do
    if [[ -z "$line" ]]; then
      continue
    fi
    pid="${line%% *}"
    cmd="${line#* }"

    if [[ "$resample_start" -eq 0 ]] && [[ "$cmd" == *"_recordings_dev.jsonl.gz"* || "$cmd" == *"_recordings_test.jsonl.gz"* ]]; then
      add_target_pid "$pid"
      collect_descendants "$pid"
      continue
    fi

    if [[ "$cmd" =~ emilia_[a-z]+_recordings_train\.([0-9]{4})\.jsonl\.gz ]]; then
      shard_id="${BASH_REMATCH[1]}"
      shard_num=$((10#$shard_id))
      if [ "$shard_num" -ge "$resample_start" ] && [ "$shard_num" -lt "$resample_stop" ]; then
        add_target_pid "$pid"
        collect_descendants "$pid"
      fi
    fi
  done < <(pgrep -af 'local/resample_recordings_to_flac.py' || true)
}

stop_pid() {
  local pid="$1"
  if ! kill -0 "$pid" 2>/dev/null; then
    return 1
  fi

  echo "$0: sending SIG${signal} to pid=${pid}"
  kill "-${signal}" "$pid"
  pkill "-${signal}" -P "$pid" 2>/dev/null || true
  stopped=true

  for ((i=0; i<grace_seconds; ++i)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done

  if [ "$force_kill" = true ] && kill -0 "$pid" 2>/dev/null; then
    echo "$0: pid=${pid} still alive after ${grace_seconds}s, sending SIGKILL"
    pkill -KILL -P "$pid" 2>/dev/null || true
    kill -KILL "$pid"
  fi
}

collect_prepare_range_pids
collect_resample_range_pids

if [ -f "$pid_file" ]; then
  pid=$(cat "$pid_file")
  add_target_pid "$pid"
  collect_descendants "$pid"
fi

while IFS= read -r pid; do
  if [[ -z "$pid" ]]; then
    continue
  fi
  stop_pid "$pid" || true
done < <(printf '%s\n' "${!target_pids[@]}" | sort -nr)

rm -f "$pid_file"

if [ "$stopped" = true ]; then
  echo "$0: stop signal sent for shard_range=[${resample_start}, ${resample_stop})"
else
  echo "$0: no matching local stage-3 process found for shard_range=[${resample_start}, ${resample_stop})"
fi
