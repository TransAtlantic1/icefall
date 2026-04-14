#!/usr/bin/env bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ICEFALL_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)
PARSE_OPTIONS_SH="${ICEFALL_ROOT}/icefall/shared/parse_options.sh"

role=""
worker_index=-1
num_stage7_workers=8
run_id=auto
run_stage10=true

language=zh
dataset_root="/inspire/dataset/emilia/fc71e07"
artifact_root=""

feature_num_splits=100
feature_num_workers=20
feature_batch_duration=1000
feature_device=auto
enable_musan=false

poll_seconds=30
heartbeat_seconds=60
stale_seconds=900

max_attempts=3
retry_backoff_seconds=120
retry_backoff_multiplier=2
retry_backoff_max_seconds=900
retry_jitter_seconds=30

. "${PARSE_OPTIONS_SH}" || exit 1

if [[ "$role" != "host" && "$role" != "worker" ]]; then
  echo "$0: --role must be one of host or worker"
  exit 1
fi

if [[ "$language" != "zh" && "$language" != "en" ]]; then
  echo "$0: --language must be one of zh or en, got: $language"
  exit 1
fi

if [ -z "$artifact_root" ]; then
  artifact_root="/inspire/qb-ilm/project/embodied-multimodality/chenxie-25019/icefall_emilia_${language}_24k"
fi

if [ "$num_stage7_workers" -le 0 ]; then
  echo "$0: --num-stage7-workers must be > 0"
  exit 1
fi

if [ "$feature_num_splits" -le 0 ]; then
  echo "$0: --feature-num-splits must be > 0"
  exit 1
fi

if [ "$max_attempts" -le 0 ]; then
  echo "$0: --max-attempts must be > 0"
  exit 1
fi

if [ "$retry_backoff_seconds" -lt 0 ] || [ "$retry_backoff_multiplier" -lt 1 ] || [ "$retry_backoff_max_seconds" -lt 0 ] || [ "$retry_jitter_seconds" -lt 0 ]; then
  echo "$0: invalid retry backoff configuration"
  exit 1
fi

if [ "$poll_seconds" -le 0 ] || [ "$heartbeat_seconds" -le 0 ] || [ "$stale_seconds" -le 0 ]; then
  echo "$0: poll/heartbeat/stale seconds must be > 0"
  exit 1
fi

if [ "$role" = "worker" ] && { [ "$worker_index" -lt 0 ] || [ "$worker_index" -ge "$num_stage7_workers" ]; }; then
  echo "$0: --worker-index must be in [0, ${num_stage7_workers}) for worker mode"
  exit 1
fi

prefix="emilia_${language}"
data_root="${artifact_root}/data"
fbank_dir="${data_root}/fbank/${language}"
train_feature_split_dir="${fbank_dir}/train_split_${feature_num_splits}"
state_root="${artifact_root}/orchestration/stage4_9/${language}"
current_run_id_file="${state_root}/current_run_id"

role_log_file=""
run_dir=""
logs_root=""
attempts_root=""
stage7_dir=""
stage7_generations_dir=""
stage7_assignment_lock_dir=""
stage7_current_generation_file=""
stage7_done_marker=""
worker_runtime_dir=""

held_stage7_assignment_lock=false
worker_heartbeat_pid=""
worker_heartbeat_state_file=""
worker_heartbeat_attempt_file=""
worker_heartbeat_generation_file=""
worker_heartbeat_target_file=""

declare -a ALL_RAW_SHARD_PATHS=()
declare -A RAW_SHARD_PATH_BY_IDX=()
declare -A RAW_SHARD_PATH_BY_NUM=()

log() {
  local line
  line="[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
  echo "$line"
  if [ -n "$role_log_file" ]; then
    mkdir -p "$(dirname "$role_log_file")"
    printf '%s\n' "$line" >>"$role_log_file"
  fi
}

write_text_atomic() {
  local path="$1"
  local content="$2"
  local tmp
  tmp="${path}.tmp.$$.$RANDOM"
  mkdir -p "$(dirname "$path")"
  printf '%s' "$content" >"$tmp"
  mv "$tmp" "$path"
}

write_marker() {
  local path="$1"
  local extra="${2:-}"
  local tmp
  tmp="${path}.tmp.$$.$RANDOM"
  mkdir -p "$(dirname "$path")"
  {
    printf 'time=%s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    printf 'host=%s\n' "$(hostname)"
    printf 'pid=%s\n' "$$"
    if [ -n "$extra" ]; then
      printf '%s\n' "$extra"
    fi
  } >"$tmp"
  mv "$tmp" "$path"
}

remove_path_if_exists() {
  local path
  for path in "$@"; do
    if [ -e "$path" ] || [ -L "$path" ]; then
      rm -rf "$path"
    fi
  done
}

file_mtime_epoch() {
  local path="$1"
  stat -c '%Y' "$path"
}

file_age_seconds() {
  local path="$1"
  local now
  now=$(date +%s)
  echo $((now - $(file_mtime_epoch "$path")))
}

sleep_with_backoff() {
  local attempt="$1"
  local delay="$retry_backoff_seconds"
  local i
  local jitter=0
  for ((i=1; i<attempt; ++i)); do
    delay=$((delay * retry_backoff_multiplier))
    if [ "$delay" -ge "$retry_backoff_max_seconds" ]; then
      delay="$retry_backoff_max_seconds"
      break
    fi
  done
  if [ "$delay" -gt "$retry_backoff_max_seconds" ]; then
    delay="$retry_backoff_max_seconds"
  fi
  if [ "$retry_jitter_seconds" -gt 0 ]; then
    jitter=$((RANDOM % (retry_jitter_seconds + 1)))
  fi
  log "Sleeping $((delay + jitter))s before retry"
  sleep $((delay + jitter))
}

assert_cut_manifest_readable() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "Missing manifest: $path" >&2
    return 1
  fi
  python3 - "$path" <<'PY'
import sys
from pathlib import Path
from lhotse import CutSet

path = Path(sys.argv[1])
cuts = CutSet.from_file(path)
it = iter(cuts)
try:
    next(it)
except StopIteration:
    pass
PY
}

discover_raw_shards() {
  local raw_path
  local file_name
  local idx

  ALL_RAW_SHARD_PATHS=()
  RAW_SHARD_PATH_BY_IDX=()
  RAW_SHARD_PATH_BY_NUM=()

  if [ ! -d "$train_feature_split_dir" ]; then
    return 0
  fi

  mapfile -t ALL_RAW_SHARD_PATHS < <(
    find "$train_feature_split_dir" -maxdepth 1 -name "${prefix}_cuts_train_raw.*.jsonl.gz" | sort
  )
  for raw_path in "${ALL_RAW_SHARD_PATHS[@]}"; do
    file_name=$(basename "$raw_path")
    idx="${file_name#${prefix}_cuts_train_raw.}"
    idx="${idx%.jsonl.gz}"
    RAW_SHARD_PATH_BY_IDX["$idx"]="$raw_path"
    RAW_SHARD_PATH_BY_NUM["$((10#$idx))"]="$raw_path"
  done
}

resolve_raw_shard_path() {
  local shard_id="$1"
  local normalized
  normalized=$(printf '%d' "$((10#$shard_id))")
  printf '%s\n' "${RAW_SHARD_PATH_BY_IDX[$shard_id]:-${RAW_SHARD_PATH_BY_NUM[$normalized]:-}}"
}

load_shard_ids_from_list() {
  local list_path="$1"
  local -n out_ref="$2"
  out_ref=()
  if [ ! -f "$list_path" ]; then
    return 0
  fi
  mapfile -t out_ref < <(
    sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$list_path"
  )
}

worker_list_file_for_generation() {
  local generation="$1"
  printf '%s/%s/worker-%02d.shards.txt\n' "$stage7_generations_dir" "$generation" "$worker_index"
}

worker_done_marker_for_generation() {
  local generation="$1"
  printf '%s/%s/worker-%02d.done\n' "$stage7_generations_dir" "$generation" "$worker_index"
}

worker_failed_marker_for_generation() {
  local generation="$1"
  printf '%s/%s/worker-%02d.failed\n' "$stage7_generations_dir" "$generation" "$worker_index"
}

worker_heartbeat_file_for_generation() {
  local generation="$1"
  printf '%s/%s/worker-%02d.heartbeat\n' "$stage7_generations_dir" "$generation" "$worker_index"
}

output_cuts_path_for_idx() {
  local idx="$1"
  printf '%s/%s_cuts_train.%s.jsonl.gz\n' "$train_feature_split_dir" "$prefix" "$idx"
}

output_storage_path_for_idx() {
  local idx="$1"
  printf '%s/%s_feats_train_%s\n' "$train_feature_split_dir" "$prefix" "$idx"
}

cleanup_stage4_outputs() {
  local split
  for split in train dev test; do
    remove_path_if_exists \
      "${fbank_dir}/${prefix}_supervisions_${split}_norm.jsonl.gz" \
      "${fbank_dir}/${prefix}_supervisions_${split}_norm_fixed.jsonl.gz" \
      "${fbank_dir}/${prefix}_cuts_${split}_raw.jsonl.gz"
  done
}

cleanup_stage5_outputs() {
  local split
  for split in dev test; do
    remove_path_if_exists \
      "${fbank_dir}/${prefix}_cuts_${split}.jsonl.gz" \
      "${fbank_dir}/${prefix}_feats_${split}" \
      "${fbank_dir}/${prefix}_feats_${split}.lca"
  done
}

cleanup_stage6_outputs() {
  remove_path_if_exists "$train_feature_split_dir"
}

cleanup_stage8_outputs() {
  remove_path_if_exists \
    "${fbank_dir}/musan_feats" \
    "${fbank_dir}/musan_cuts.jsonl.gz" \
    "${fbank_dir}/.musan.done"
}

cleanup_stage9_outputs() {
  remove_path_if_exists "${fbank_dir}/${prefix}_cuts_train.jsonl.gz"
}

cleanup_stage7_outputs_for_list() {
  local list_path="$1"
  local shard_ids=()
  local shard_id
  local raw_path
  local idx
  discover_raw_shards
  load_shard_ids_from_list "$list_path" shard_ids
  for shard_id in "${shard_ids[@]}"; do
    raw_path=$(resolve_raw_shard_path "$shard_id")
    if [ -z "$raw_path" ]; then
      continue
    fi
    idx=$(basename "$raw_path")
    idx="${idx#${prefix}_cuts_train_raw.}"
    idx="${idx%.jsonl.gz}"
    remove_path_if_exists \
      "$(output_cuts_path_for_idx "$idx")" \
      "$(output_storage_path_for_idx "$idx")" \
      "$(output_storage_path_for_idx "$idx").lca"
  done
}

prepare_common_args=(
  --language "$language"
  --dataset-root "$dataset_root"
  --artifact-root "$artifact_root"
  --feature-num-splits "$feature_num_splits"
  --feature-device "$feature_device"
  --feature-num-workers "$feature_num_workers"
  --feature-batch-duration "$feature_batch_duration"
  --enable-musan "$enable_musan"
)

run_prepare_stage4() {
  bash "${SCRIPT_DIR}/prepare.sh" "${prepare_common_args[@]}" --stage 4 --stop-stage 4
}

run_prepare_stage5() {
  bash "${SCRIPT_DIR}/prepare.sh" "${prepare_common_args[@]}" --stage 5 --stop-stage 5
}

run_prepare_stage6() {
  bash "${SCRIPT_DIR}/prepare.sh" "${prepare_common_args[@]}" --stage 6 --stop-stage 6
}

run_prepare_stage8() {
  bash "${SCRIPT_DIR}/prepare.sh" "${prepare_common_args[@]}" --stage 8 --stop-stage 8
}

run_prepare_stage9() {
  bash "${SCRIPT_DIR}/prepare.sh" "${prepare_common_args[@]}" --stage 9 --stop-stage 9
}

run_prepare_stage10() {
  bash "${SCRIPT_DIR}/prepare.sh" "${prepare_common_args[@]}" --stage 10 --stop-stage 10
}

verify_stage4() {
  local split
  for split in train dev test; do
    assert_cut_manifest_readable "${fbank_dir}/${prefix}_cuts_${split}_raw.jsonl.gz"
  done
}

verify_stage5() {
  local split
  for split in dev test; do
    assert_cut_manifest_readable "${fbank_dir}/${prefix}_cuts_${split}.jsonl.gz"
  done
}

verify_stage6() {
  discover_raw_shards
  if [ "${#ALL_RAW_SHARD_PATHS[@]}" -ne "$feature_num_splits" ]; then
    echo "Expected ${feature_num_splits} raw train split manifests, got ${#ALL_RAW_SHARD_PATHS[@]}" >&2
    return 1
  fi
}

verify_stage8() {
  if [ "$enable_musan" = false ]; then
    return 0
  fi
  [ -e "${fbank_dir}/.musan.done" ]
}

verify_stage9() {
  assert_cut_manifest_readable "${fbank_dir}/${prefix}_cuts_train.jsonl.gz"
}

run_host_stage_with_retries() {
  local label="$1"
  local cleanup_fn="$2"
  local run_fn="$3"
  local verify_fn="$4"
  local done_marker="${run_dir}/${label}.done"
  local failed_marker="${run_dir}/${label}.failed"
  local attempt
  local attempt_dir="${attempts_root}/${label}"
  local attempt_log
  local status

  if [ -f "$done_marker" ]; then
    log "${label}: already done"
    return 0
  fi

  mkdir -p "$attempt_dir"
  rm -f "$failed_marker"

  for ((attempt=1; attempt<=max_attempts; ++attempt)); do
    attempt_log="${attempt_dir}/attempt-${attempt}.log"
    log "${label}: attempt ${attempt}/${max_attempts}"
    "$cleanup_fn"
    if "$run_fn" >"$attempt_log" 2>&1 && "$verify_fn" >>"$attempt_log" 2>&1; then
      write_marker "$done_marker" "attempt=${attempt}"
      log "${label}: success"
      return 0
    fi

    status=$?
    log "${label}: attempt ${attempt} failed with status=${status}, see ${attempt_log}"
    if [ "$attempt" -eq "$max_attempts" ]; then
      write_marker "$failed_marker" "attempt=${attempt}"$'\n'"status=${status}"
      return "$status"
    fi
    sleep_with_backoff "$attempt"
  done
}

start_worker_heartbeat_loop() {
  if [ -n "$worker_heartbeat_pid" ] && kill -0 "$worker_heartbeat_pid" 2>/dev/null; then
    return 0
  fi

  (
    set +e
    while true; do
      target_file=""
      state="idle"
      attempt="0"
      generation=""
      tmp=""

      if [ -f "$worker_heartbeat_target_file" ]; then
        target_file=$(cat "$worker_heartbeat_target_file")
      fi
      if [ -f "$worker_heartbeat_state_file" ]; then
        state=$(cat "$worker_heartbeat_state_file")
      fi
      if [ -f "$worker_heartbeat_attempt_file" ]; then
        attempt=$(cat "$worker_heartbeat_attempt_file")
      fi
      if [ -f "$worker_heartbeat_generation_file" ]; then
        generation=$(cat "$worker_heartbeat_generation_file")
      fi

      if [ -n "$target_file" ]; then
        tmp="${target_file}.tmp.$$.$RANDOM"
        mkdir -p "$(dirname "$target_file")"
        {
          printf 'time=%s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
          printf 'host=%s\n' "$(hostname)"
          printf 'pid=%s\n' "$$"
          printf 'worker_index=%s\n' "$worker_index"
          printf 'state=%s\n' "$state"
          printf 'attempt=%s\n' "$attempt"
          printf 'generation=%s\n' "$generation"
        } >"$tmp"
        mv "$tmp" "$target_file"
      fi

      sleep "$heartbeat_seconds"
    done
  ) &
  worker_heartbeat_pid=$!
}

update_worker_heartbeat() {
  local target_file="$1"
  local generation="$2"
  local state="$3"
  local attempt="$4"
  write_text_atomic "$worker_heartbeat_target_file" "$target_file"
  write_text_atomic "$worker_heartbeat_generation_file" "$generation"
  write_text_atomic "$worker_heartbeat_state_file" "$state"
  write_text_atomic "$worker_heartbeat_attempt_file" "$attempt"
}

stop_worker_heartbeat_loop() {
  if [ -n "$worker_heartbeat_pid" ] && kill -0 "$worker_heartbeat_pid" 2>/dev/null; then
    kill "$worker_heartbeat_pid" 2>/dev/null || true
    wait "$worker_heartbeat_pid" 2>/dev/null || true
  fi
  worker_heartbeat_pid=""
}

run_prepare_stage7_for_list() {
  local list_path="$1"
  bash "${SCRIPT_DIR}/prepare.sh" \
    "${prepare_common_args[@]}" \
    --stage 7 \
    --stop-stage 7 \
    --feature-shard-list "$list_path"
}

verify_stage7_outputs_for_list() {
  local list_path="$1"
  local shard_ids=()
  local shard_id
  local raw_path
  local idx

  discover_raw_shards
  load_shard_ids_from_list "$list_path" shard_ids
  for shard_id in "${shard_ids[@]}"; do
    raw_path=$(resolve_raw_shard_path "$shard_id")
    if [ -z "$raw_path" ]; then
      echo "Missing raw shard for ${shard_id}" >&2
      return 1
    fi
    idx=$(basename "$raw_path")
    idx="${idx#${prefix}_cuts_train_raw.}"
    idx="${idx%.jsonl.gz}"
    assert_cut_manifest_readable "$(output_cuts_path_for_idx "$idx")"
  done
}

run_worker_generation_with_retries() {
  local generation="$1"
  local list_path="$2"
  local heartbeat_file
  local done_marker
  local failed_marker
  local attempt
  local attempt_dir
  local attempt_log
  local status

  done_marker=$(worker_done_marker_for_generation "$generation")
  failed_marker=$(worker_failed_marker_for_generation "$generation")
  heartbeat_file=$(worker_heartbeat_file_for_generation "$generation")
  attempt_dir="${attempts_root}/stage7/worker-${worker_index}/gen-${generation}"

  if [ -f "$done_marker" ]; then
    log "stage7 ${generation}: worker ${worker_index} already done"
    return 0
  fi

  mkdir -p "$attempt_dir"
  rm -f "$failed_marker"
  start_worker_heartbeat_loop

  for ((attempt=1; attempt<=max_attempts; ++attempt)); do
    attempt_log="${attempt_dir}/attempt-${attempt}.log"
    update_worker_heartbeat "$heartbeat_file" "$generation" "running" "$attempt"
    log "stage7 ${generation}: worker ${worker_index} attempt ${attempt}/${max_attempts}"
    cleanup_stage7_outputs_for_list "$list_path"
    if run_prepare_stage7_for_list "$list_path" >"$attempt_log" 2>&1 && verify_stage7_outputs_for_list "$list_path" >>"$attempt_log" 2>&1; then
      write_marker "$done_marker" "attempt=${attempt}"
      update_worker_heartbeat "$heartbeat_file" "$generation" "done" "$attempt"
      log "stage7 ${generation}: worker ${worker_index} success"
      return 0
    fi

    status=$?
    log "stage7 ${generation}: worker ${worker_index} attempt ${attempt} failed with status=${status}, see ${attempt_log}"
    if [ "$attempt" -eq "$max_attempts" ]; then
      write_marker "$failed_marker" "attempt=${attempt}"$'\n'"status=${status}"
      update_worker_heartbeat "$heartbeat_file" "$generation" "failed" "$attempt"
      return "$status"
    fi

    update_worker_heartbeat "$heartbeat_file" "$generation" "backoff" "$attempt"
    sleep_with_backoff "$attempt"
  done
}

ensure_run_id_for_host() {
  mkdir -p "$state_root"
  if [ "$run_id" = auto ]; then
    run_id="run-$(date -u '+%Y%m%dT%H%M%SZ')-$(hostname -s)-$$"
  fi
  write_text_atomic "$current_run_id_file" "$run_id"
}

ensure_run_id_for_worker() {
  mkdir -p "$state_root"
  if [ "$run_id" != auto ]; then
    return 0
  fi
  while [ ! -f "$current_run_id_file" ]; do
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] waiting for ${current_run_id_file}"
    sleep "$poll_seconds"
  done
  run_id=$(cat "$current_run_id_file")
}

initialize_run_layout() {
  run_dir="${state_root}/runs/${run_id}"
  logs_root="${artifact_root}/logs/stage4_9/${run_id}"
  attempts_root="${logs_root}/attempts"
  stage7_dir="${run_dir}/stage7"
  stage7_generations_dir="${stage7_dir}/generations"
  stage7_assignment_lock_dir="${stage7_dir}/assignment.lock"
  stage7_current_generation_file="${stage7_dir}/current_generation"
  stage7_done_marker="${stage7_dir}/stage7.done"

  mkdir -p "$run_dir" "$logs_root" "$attempts_root" "$stage7_generations_dir"
  if [ "$role" = "host" ]; then
    role_log_file="${logs_root}/host.log"
  else
    role_log_file="${logs_root}/worker.$(printf '%02d' "$worker_index").log"
    worker_runtime_dir="${run_dir}/worker-$(printf '%02d' "$worker_index").runtime"
    mkdir -p "$worker_runtime_dir"
    worker_heartbeat_state_file="${worker_runtime_dir}/heartbeat.state"
    worker_heartbeat_attempt_file="${worker_runtime_dir}/heartbeat.attempt"
    worker_heartbeat_generation_file="${worker_runtime_dir}/heartbeat.generation"
    worker_heartbeat_target_file="${worker_runtime_dir}/heartbeat.target"
  fi
}

cleanup_on_exit() {
  stop_worker_heartbeat_loop
  if [ -n "$stage7_assignment_lock_dir" ] && [ "$held_stage7_assignment_lock" = true ]; then
    rm -rf "$stage7_assignment_lock_dir"
  fi
}

trap cleanup_on_exit EXIT INT TERM

acquire_stage7_assignment_lock() {
  local owner_file
  local lock_age
  local wait_logged=false

  while true; do
    if mkdir "$stage7_assignment_lock_dir" 2>/dev/null; then
      held_stage7_assignment_lock=true
      owner_file="${stage7_assignment_lock_dir}/owner"
      write_marker "$owner_file" "role=host"$'\n'"run_id=${run_id}"
      return 0
    fi

    if [ -d "$stage7_assignment_lock_dir" ]; then
      lock_age=$(file_age_seconds "$stage7_assignment_lock_dir")
      if [ "$lock_age" -gt "$stale_seconds" ]; then
        log "Removing stale stage7 assignment lock ${stage7_assignment_lock_dir}"
        rm -rf "$stage7_assignment_lock_dir"
        continue
      fi
    fi

    if [ "$wait_logged" = false ]; then
      log "Waiting for stage7 assignment lock ${stage7_assignment_lock_dir}"
      wait_logged=true
    fi
    sleep "$poll_seconds"
  done
}

release_stage7_assignment_lock() {
  if [ "$held_stage7_assignment_lock" = true ]; then
    rm -rf "$stage7_assignment_lock_dir"
    held_stage7_assignment_lock=false
  fi
}

read_current_generation() {
  if [ ! -f "$stage7_current_generation_file" ]; then
    return 1
  fi
  cat "$stage7_current_generation_file"
}

next_stage7_generation_name() {
  local generation_dir
  local generation_base
  local generation_num
  local next_num=1

  for generation_dir in "$stage7_generations_dir"/gen-*; do
    generation_base="${generation_dir##*/gen-}"
    generation_num=$((10#$generation_base))
    if [ "$generation_num" -ge "$next_num" ]; then
      next_num=$((generation_num + 1))
    fi
  done
  printf 'gen-%05d\n' "$next_num"
}

collect_remaining_stage7_entries() {
  local -n out_entries="$1"
  local raw_path
  local idx
  local idx_num
  local output_path
  local weight

  out_entries=()
  discover_raw_shards
  for raw_path in "${ALL_RAW_SHARD_PATHS[@]}"; do
    idx=$(basename "$raw_path")
    idx="${idx#${prefix}_cuts_train_raw.}"
    idx="${idx%.jsonl.gz}"
    idx_num=$(printf '%d' "$((10#$idx))")
    output_path=$(output_cuts_path_for_idx "$idx")
    if [ -f "$output_path" ] && assert_cut_manifest_readable "$output_path" >/dev/null 2>&1; then
      continue
    fi
    weight=$(stat -c '%s' "$raw_path")
    out_entries+=("${weight}"$'\t'"${idx_num}"$'\t'"${idx}")
  done
}

create_stage7_generation() {
  local remaining_entries=()
  local generation
  local generation_dir
  local entry
  local sorted_entries=()
  local size
  local idx_num
  local idx
  local best_worker=0
  local best_load=0
  local worker
  local list_file
  local -a worker_loads=()
  local -a worker_lists=()
  local assignment_summary=""

  acquire_stage7_assignment_lock
  collect_remaining_stage7_entries remaining_entries
  if [ "${#remaining_entries[@]}" -eq 0 ]; then
    touch "$stage7_done_marker"
    release_stage7_assignment_lock
    echo ""
    return 0
  fi

  generation=$(next_stage7_generation_name)
  generation_dir="${stage7_generations_dir}/${generation}"
  mkdir -p "$generation_dir"

  for ((worker=0; worker<num_stage7_workers; ++worker)); do
    worker_loads[$worker]=0
    worker_lists[$worker]=""
  done

  mapfile -t sorted_entries < <(
    printf '%s\n' "${remaining_entries[@]}" | sort -t $'\t' -k1,1nr -k2,2n
  )

  for entry in "${sorted_entries[@]}"; do
    IFS=$'\t' read -r size idx_num idx <<<"$entry"
    best_worker=0
    best_load=${worker_loads[0]}
    for ((worker=1; worker<num_stage7_workers; ++worker)); do
      if [ "${worker_loads[$worker]}" -lt "$best_load" ]; then
        best_worker="$worker"
        best_load=${worker_loads[$worker]}
      fi
    done
    worker_loads[$best_worker]=$((worker_loads[$best_worker] + size))
    worker_lists[$best_worker]+="${idx}"$'\n'
  done

  for ((worker=0; worker<num_stage7_workers; ++worker)); do
    list_file="${generation_dir}/worker-$(printf '%02d' "$worker").shards.txt"
    write_text_atomic "$list_file" "${worker_lists[$worker]}"
    assignment_summary+="worker-$(printf '%02d' "$worker") shards=$(tr '\n' ' ' <"$list_file" | sed 's/[[:space:]]*$//') load=${worker_loads[$worker]}"$'\n'
  done

  write_marker "${generation_dir}/created" "remaining_shards=${#sorted_entries[@]}"
  write_text_atomic "${generation_dir}/assignment_summary.txt" "$assignment_summary"
  write_text_atomic "$stage7_current_generation_file" "$generation"
  release_stage7_assignment_lock
  log "Created ${generation} with ${#sorted_entries[@]} remaining stage7 shards"
  printf '%s\n' "$generation"
}

generation_all_done() {
  local generation="$1"
  local generation_dir="${stage7_generations_dir}/${generation}"
  local worker
  for ((worker=0; worker<num_stage7_workers; ++worker)); do
    if [ ! -f "${generation_dir}/worker-$(printf '%02d' "$worker").done" ]; then
      return 1
    fi
  done
  return 0
}

generation_has_failed_workers() {
  local generation="$1"
  local generation_dir="${stage7_generations_dir}/${generation}"
  local worker
  for ((worker=0; worker<num_stage7_workers; ++worker)); do
    if [ -f "${generation_dir}/worker-$(printf '%02d' "$worker").failed" ]; then
      return 0
    fi
  done
  return 1
}

generation_has_fresh_activity() {
  local generation="$1"
  local generation_dir="${stage7_generations_dir}/${generation}"
  local worker
  local heartbeat_file
  for ((worker=0; worker<num_stage7_workers; ++worker)); do
    if [ -f "${generation_dir}/worker-$(printf '%02d' "$worker").done" ]; then
      continue
    fi
    heartbeat_file="${generation_dir}/worker-$(printf '%02d' "$worker").heartbeat"
    if [ -f "$heartbeat_file" ] && [ "$(file_age_seconds "$heartbeat_file")" -le "$stale_seconds" ]; then
      return 0
    fi
  done
  return 1
}

generation_has_stale_workers() {
  local generation="$1"
  local generation_dir="${stage7_generations_dir}/${generation}"
  local generation_created="${generation_dir}/created"
  local generation_age
  local worker
  local heartbeat_file

  if [ ! -f "$generation_created" ]; then
    return 1
  fi

  generation_age=$(file_age_seconds "$generation_created")
  if [ "$generation_age" -le "$stale_seconds" ]; then
    return 1
  fi

  for ((worker=0; worker<num_stage7_workers; ++worker)); do
    if [ -f "${generation_dir}/worker-$(printf '%02d' "$worker").done" ]; then
      continue
    fi
    if [ -f "${generation_dir}/worker-$(printf '%02d' "$worker").failed" ]; then
      continue
    fi
    heartbeat_file="${generation_dir}/worker-$(printf '%02d' "$worker").heartbeat"
    if [ ! -f "$heartbeat_file" ] || [ "$(file_age_seconds "$heartbeat_file")" -gt "$stale_seconds" ]; then
      return 0
    fi
  done
  return 1
}

wait_for_generation_terminal() {
  local generation="$1"
  while true; do
    if generation_all_done "$generation"; then
      log "${generation}: all workers done"
      return 0
    fi
    if generation_has_failed_workers "$generation"; then
      log "${generation}: found failed worker marker"
      return 1
    fi
    if generation_has_stale_workers "$generation"; then
      log "${generation}: found stale worker"
      return 1
    fi
    sleep "$poll_seconds"
  done
}

run_host_stage7() {
  local generation=""
  local remaining_entries=()

  if [ -f "$stage7_done_marker" ]; then
    log "stage7: already done"
    return 0
  fi

  while true; do
    collect_remaining_stage7_entries remaining_entries
    if [ "${#remaining_entries[@]}" -eq 0 ]; then
      touch "$stage7_done_marker"
      log "stage7: no remaining shards"
      return 0
    fi

    generation=$(read_current_generation || true)
    if [ -n "$generation" ]; then
      if generation_all_done "$generation"; then
        log "stage7: current generation ${generation} already finished, rescanning"
      elif generation_has_fresh_activity "$generation"; then
        log "stage7: attaching to active generation ${generation}"
        wait_for_generation_terminal "$generation" || return 1
      else
        log "stage7: generation ${generation} has no fresh activity, reallocating remaining shards"
      fi
      collect_remaining_stage7_entries remaining_entries
      if [ "${#remaining_entries[@]}" -eq 0 ]; then
        touch "$stage7_done_marker"
        log "stage7: all shards are done after generation ${generation}"
        return 0
      fi
    fi

    generation=$(create_stage7_generation)
    if [ -z "$generation" ]; then
      log "stage7: nothing left after allocation scan"
      return 0
    fi
    wait_for_generation_terminal "$generation" || return 1
  done
}

wait_for_stage6_done() {
  local marker="${run_dir}/stage6.done"
  while [ ! -f "$marker" ]; do
    if [ -f "${run_dir}/stage6.failed" ]; then
      log "stage6 failed marker exists, worker cannot continue"
      return 1
    fi
    if [ -f "$stage7_done_marker" ]; then
      return 0
    fi
    log "worker ${worker_index}: waiting for ${marker}"
    sleep "$poll_seconds"
  done
}

wait_for_stage7_generation_ready() {
  while true; do
    if [ -f "$stage7_done_marker" ]; then
      echo ""
      return 0
    fi
    if [ -d "$stage7_assignment_lock_dir" ]; then
      sleep "$poll_seconds"
      continue
    fi
    if [ -f "$stage7_current_generation_file" ]; then
      cat "$stage7_current_generation_file"
      return 0
    fi
    sleep "$poll_seconds"
  done
}

run_worker() {
  local generation=""
  local last_generation=""
  local list_file=""
  local shard_ids=()

  wait_for_stage6_done

  while true; do
    if [ -f "$stage7_done_marker" ]; then
      log "worker ${worker_index}: stage7 is complete"
      return 0
    fi

    generation=$(wait_for_stage7_generation_ready)
    if [ -z "$generation" ]; then
      log "worker ${worker_index}: stage7 is complete"
      return 0
    fi

    list_file="${stage7_generations_dir}/${generation}/worker-$(printf '%02d' "$worker_index").shards.txt"
    if [ ! -f "$list_file" ]; then
      log "worker ${worker_index}: waiting for shard list in ${generation}"
      sleep "$poll_seconds"
      continue
    fi

    if [ "$generation" = "$last_generation" ] && [ -f "$(worker_done_marker_for_generation "$generation")" ]; then
      sleep "$poll_seconds"
      continue
    fi
    last_generation="$generation"

    load_shard_ids_from_list "$list_file" shard_ids
    if [ "${#shard_ids[@]}" -eq 0 ]; then
      write_marker "$(worker_done_marker_for_generation "$generation")" "attempt=0"$'\n'"empty_assignment=true"
      log "worker ${worker_index}: ${generation} has empty shard list"
      continue
    fi

    rm -f "$(worker_failed_marker_for_generation "$generation")"
    run_worker_generation_with_retries "$generation" "$list_file"
  done
}

setup_role_and_run() {
  if [ "$role" = "host" ]; then
    ensure_run_id_for_host
  else
    ensure_run_id_for_worker
  fi
  initialize_run_layout
}

run_host() {
  log "role=host run_id=${run_id}"
  log "dataset_root=${dataset_root}"
  log "artifact_root=${artifact_root}"
  log "feature_num_splits=${feature_num_splits}"
  log "num_stage7_workers=${num_stage7_workers}"
  log "feature_device=${feature_device}"

  run_host_stage_with_retries stage4 cleanup_stage4_outputs run_prepare_stage4 verify_stage4
  run_host_stage_with_retries stage5 cleanup_stage5_outputs run_prepare_stage5 verify_stage5
  run_host_stage_with_retries stage6 cleanup_stage6_outputs run_prepare_stage6 verify_stage6
  run_host_stage7
  run_host_stage_with_retries stage8 cleanup_stage8_outputs run_prepare_stage8 verify_stage8
  run_host_stage_with_retries stage9 cleanup_stage9_outputs run_prepare_stage9 verify_stage9

  if [ "$run_stage10" = true ]; then
    run_host_stage_with_retries stage10 true run_prepare_stage10 true
  fi
}

main() {
  setup_role_and_run
  if [ "$role" = "host" ]; then
    run_host
  else
    log "role=worker run_id=${run_id} worker_index=${worker_index}"
    log "dataset_root=${dataset_root}"
    log "artifact_root=${artifact_root}"
    log "feature_num_splits=${feature_num_splits}"
    log "feature_device=${feature_device}"
    run_worker
  fi
}

main "$@"
