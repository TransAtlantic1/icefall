#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ICEFALL_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PARSE_OPTIONS_SH="${ICEFALL_ROOT}/icefall/shared/parse_options.sh"

language=zh
public_root=/inspire/qb-ilm/project/embodied-multimodality/chenxie-25019/public
dataset_root=""
artifact_root=""
recording_num_splits=1000
resample_start=-1
resample_stop=-1
resample_num_workers=24
target_sample_rate=24000
log_root=""
log_file=""
detach=true
detach_log=""
pid_file=""
startup_lock_dir=""

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

if [ "$recording_num_splits" -le 0 ]; then
  echo "$0: --recording-num-splits must be > 0"
  exit 1
fi

if [ "$resample_start" -lt 0 ] || [ "$resample_stop" -le "$resample_start" ] || [ "$resample_stop" -gt "$recording_num_splits" ]; then
  echo "$0: invalid shard range [${resample_start}, ${resample_stop})"
  exit 1
fi

mkdir -p "$log_root"

range_label="${resample_start}-$((resample_stop - 1))"
if [ -z "$log_file" ]; then
  log_file="${log_root}/resample.${language}.${range_label}.rebalance.log"
fi
if [ -z "$detach_log" ]; then
  detach_log="${log_root}/launcher.resample.range.${language}.${range_label}.nohup.log"
fi
if [ -z "$pid_file" ]; then
  pid_file="${log_root}/launcher.resample.range.${language}.${range_label}.pid"
fi
if [ -z "$startup_lock_dir" ]; then
  startup_lock_dir="${log_root}/launcher.resample.range.${language}.${range_label}.lock"
fi

prepare_pattern="${SCRIPT_DIR}/prepare.sh.*--stage 3.*--stop-stage 3.*--resample-start ${resample_start}.*--resample-stop ${resample_stop}"

find_active_prepare_pids() {
  pgrep -f -- "$prepare_pattern" || true
}

find_active_python_pids() {
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
      echo "$pid"
      continue
    fi

    if [[ "$cmd" =~ emilia_[a-z]+_recordings_train\.([0-9]{4})\.jsonl\.gz ]]; then
      shard_id="${BASH_REMATCH[1]}"
      shard_num=$((10#$shard_id))
      if [ "$shard_num" -ge "$resample_start" ] && [ "$shard_num" -lt "$resample_stop" ]; then
        echo "$pid"
      fi
    fi
  done < <(pgrep -af 'local/resample_recordings_to_flac.py' || true)
}

check_existing_run() {
  local pid
  if [ -f "$pid_file" ]; then
    pid=$(cat "$pid_file")
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "$0: range [${resample_start}, ${resample_stop}) already running with pid=${pid}"
      return 0
    fi
    rm -f "$pid_file"
  fi

  mapfile -t existing_prepare_pids < <(find_active_prepare_pids)
  if [ "${#existing_prepare_pids[@]}" -gt 0 ]; then
    printf '%s\n' "${existing_prepare_pids[0]}" >"$pid_file"
    echo "$0: range [${resample_start}, ${resample_stop}) already has prepare.sh pid=${existing_prepare_pids[0]}"
    return 0
  fi

  mapfile -t existing_python_pids < <(find_active_python_pids)
  if [ "${#existing_python_pids[@]}" -gt 0 ]; then
    echo "$0: range [${resample_start}, ${resample_stop}) already has active resample python pid=${existing_python_pids[0]}"
    return 0
  fi

  return 1
}

cleanup_foreground_pid_file() {
  rm -f "$pid_file"
}

cleanup_startup_lock() {
  rmdir "$startup_lock_dir" 2>/dev/null || true
}

if ! mkdir "$startup_lock_dir" 2>/dev/null; then
  echo "$0: startup lock is busy for range [${resample_start}, ${resample_stop}), refusing duplicate launch"
  exit 1
fi
trap cleanup_startup_lock EXIT

if check_existing_run; then
  exit 0
fi

echo "$0: language=${language}"
echo "$0: dataset_root=${dataset_root}"
echo "$0: artifact_root=${artifact_root}"
echo "$0: shard_range=[${resample_start}, ${resample_stop})"
echo "$0: log_file=${log_file}"
echo "$0: detach_log=${detach_log}"
echo "$0: pid_file=${pid_file}"

if [ "$detach" = true ]; then
  nohup bash "${SCRIPT_DIR}/prepare.sh" \
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
    >>"$log_file" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$pid_file"
  printf '%s\n' "$0: detached pid=${pid}" >>"$detach_log"
  printf '%s\n' "$0: worker_log=${log_file}" >>"$detach_log"
  echo "$0: detached pid=${pid}"
  echo "$0: detach_log=${detach_log}"
  echo "$0: worker_log=${log_file}"
  exit 0
fi

printf '%s\n' "$$" >"$pid_file"
trap cleanup_foreground_pid_file INT TERM EXIT

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
