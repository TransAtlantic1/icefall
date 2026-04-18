#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_SELF="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
ICEFALL_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PARSE_OPTIONS_SH="${ICEFALL_ROOT}/icefall/shared/parse_options.sh"
RECIPE_DIR="${ICEFALL_ROOT}/egs/emilia_24k_multilang/emilia_24k_ZH/ASR"
VALIDATION_ROOT="$(cd -- "${ICEFALL_ROOT}/.." && pwd)/experiments/main_flow_validation/emilia24k"

language=zh
source_artifact_root=""
probe_artifact_root=""
recording_num_splits=4
feature_num_splits=4
feature_start=0
feature_stop=-1
feature_num_workers=0
feature_batch_duration=120
feature_device=cpu
target_sample_rate=24000
log_root=""
log_file=""
detach=false
detach_log=""
pid_file=""
refresh_probe=true

. "${PARSE_OPTIONS_SH}" || exit 1

if [[ "$language" != "zh" && "$language" != "en" ]]; then
  echo "$0: --language must be one of zh or en, got: $language"
  exit 1
fi

if [ -z "$source_artifact_root" ]; then
  source_artifact_root="${VALIDATION_ROOT}/workspace/artifacts"
fi

if [ -z "$probe_artifact_root" ]; then
  probe_artifact_root="${VALIDATION_ROOT}/workspace/feature_probe"
fi

if [ -z "$log_root" ]; then
  log_root="${probe_artifact_root}/logs"
fi

if [ "$recording_num_splits" -le 0 ]; then
  echo "$0: --recording-num-splits must be > 0"
  exit 1
fi

if [ "$feature_num_splits" -le 0 ]; then
  echo "$0: --feature-num-splits must be > 0"
  exit 1
fi

if [ "$feature_start" -lt 0 ]; then
  echo "$0: --feature-start must be >= 0"
  exit 1
fi

if [ "$feature_stop" -lt 0 ]; then
  feature_stop="$feature_num_splits"
fi

if [ "$feature_stop" -le "$feature_start" ] || [ "$feature_stop" -gt "$feature_num_splits" ]; then
  echo "$0: invalid feature range [${feature_start}, ${feature_stop})"
  exit 1
fi

mkdir -p "$log_root"

range_label="${feature_start}-$((feature_stop - 1))"
if [ -z "$log_file" ]; then
  log_file="${log_root}/partial-train-feature.${language}.${range_label}.log"
fi
if [ -z "$detach_log" ]; then
  detach_log="${log_root}/launcher.partial-train-feature.${language}.${range_label}.nohup.log"
fi
if [ -z "$pid_file" ]; then
  pid_file="${log_root}/launcher.partial-train-feature.${language}.${range_label}.pid"
fi

cleanup_pid_file() {
  rm -f "$pid_file"
}

if [ "$detach" = true ]; then
  if [ -f "$pid_file" ]; then
    pid=$(cat "$pid_file")
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "$0: partial train probe already running with pid=${pid}"
      exit 0
    fi
    rm -f "$pid_file"
  fi

  cmd=(
    "${SCRIPT_SELF}"
    --language "$language"
    --source-artifact-root "$source_artifact_root"
    --probe-artifact-root "$probe_artifact_root"
    --recording-num-splits "$recording_num_splits"
    --feature-num-splits "$feature_num_splits"
    --feature-start "$feature_start"
    --feature-stop "$feature_stop"
    --feature-num-workers "$feature_num_workers"
    --feature-batch-duration "$feature_batch_duration"
    --feature-device "$feature_device"
    --target-sample-rate "$target_sample_rate"
    --log-root "$log_root"
    --log-file "$log_file"
    --detach false
    --detach-log "$detach_log"
    --pid-file "$pid_file"
    --refresh-probe "$refresh_probe"
  )
  nohup "${cmd[@]}" >>"$detach_log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$pid_file"
  echo "$0: detached pid=${pid}"
  echo "$0: detach_log=${detach_log}"
  echo "$0: worker_log=${log_file}"
  exit 0
fi

printf '%s\n' "$$" >"$pid_file"
trap cleanup_pid_file EXIT

source_manifest_root="${source_artifact_root}/data/manifests"
source_manifest_dir="${source_manifest_root}/${language}"
source_resampled_manifest_dir="${source_artifact_root}/data/manifests_resampled/${language}/${target_sample_rate}"
source_split_dir="${source_resampled_manifest_dir}/recordings_train_split_${recording_num_splits}"

probe_data_root="${probe_artifact_root}/data"
probe_manifest_root="${probe_data_root}/manifests"
probe_resampled_manifest_dir="${probe_data_root}/manifests_resampled/${language}/${target_sample_rate}"
probe_split_dir="${probe_resampled_manifest_dir}/recordings_train_split_${recording_num_splits}"
probe_fbank_dir="${probe_data_root}/fbank/${language}"
snapshot_summary="${probe_artifact_root}/snapshot_summary.txt"

if [ ! -d "$source_manifest_dir" ]; then
  echo "$0: missing source manifest dir ${source_manifest_dir}"
  exit 1
fi

if [ ! -d "$source_split_dir" ]; then
  echo "$0: missing source resampled split dir ${source_split_dir}"
  exit 1
fi

echo "$0: source_artifact_root=${source_artifact_root}"
echo "$0: probe_artifact_root=${probe_artifact_root}"
echo "$0: feature_range=[${feature_start}, ${feature_stop})"
echo "$0: feature_device=${feature_device}"
echo "$0: log_file=${log_file}"

mkdir -p "$probe_data_root" "$probe_resampled_manifest_dir" "$log_root"

if [ "$refresh_probe" = true ]; then
  rm -rf "$probe_fbank_dir" "$probe_resampled_manifest_dir"
  rm -f "$snapshot_summary"
  mkdir -p "$probe_resampled_manifest_dir"
fi

rm -rf "$probe_manifest_root"
ln -snf "$source_manifest_root" "$probe_manifest_root"

mkdir -p "$probe_split_dir"

if [ -f "${source_resampled_manifest_dir}/emilia_${language}_recordings_dev.jsonl.gz" ]; then
  ln -snf \
    "${source_resampled_manifest_dir}/emilia_${language}_recordings_dev.jsonl.gz" \
    "${probe_resampled_manifest_dir}/emilia_${language}_recordings_dev.jsonl.gz"
fi
if [ -f "${source_resampled_manifest_dir}/emilia_${language}_recordings_test.jsonl.gz" ]; then
  ln -snf \
    "${source_resampled_manifest_dir}/emilia_${language}_recordings_test.jsonl.gz" \
    "${probe_resampled_manifest_dir}/emilia_${language}_recordings_test.jsonl.gz"
fi

linked_shards=0
for src in "${source_split_dir}"/emilia_"${language}"_recordings_train.*.jsonl.gz; do
  if [ ! -e "$src" ]; then
    continue
  fi
  ln -snf "$src" "${probe_split_dir}/$(basename "$src")"
  linked_shards=$((linked_shards + 1))
done

printf 'snapshot_time=%s\nlinked_train_shards=%s\nsource_split_dir=%s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" \
  "$linked_shards" \
  "$source_split_dir" >"$snapshot_summary"

echo "$0: linked_train_shards=${linked_shards}"
if [ "$linked_shards" -eq 0 ]; then
  echo "$0: no completed train resampled shard manifests found in ${source_split_dir}"
  exit 1
fi

bash "${RECIPE_DIR}/prepare.sh" \
  --language "$language" \
  --artifact-root "$probe_artifact_root" \
  --recording-num-splits "$recording_num_splits" \
  --target-sample-rate "$target_sample_rate" \
  --stage 4 \
  --stop-stage 4 \
  >>"$log_file" 2>&1

bash "${RECIPE_DIR}/prepare.sh" \
  --language "$language" \
  --artifact-root "$probe_artifact_root" \
  --recording-num-splits "$recording_num_splits" \
  --feature-num-splits "$feature_num_splits" \
  --stage 6 \
  --stop-stage 6 \
  >>"$log_file" 2>&1

env CUDA_VISIBLE_DEVICES="" bash "${RECIPE_DIR}/prepare.sh" \
  --language "$language" \
  --artifact-root "$probe_artifact_root" \
  --recording-num-splits "$recording_num_splits" \
  --feature-num-splits "$feature_num_splits" \
  --feature-start "$feature_start" \
  --feature-stop "$feature_stop" \
  --feature-device "$feature_device" \
  --feature-num-workers "$feature_num_workers" \
  --feature-batch-duration "$feature_batch_duration" \
  --stage 7 \
  --stop-stage 7 \
  >>"$log_file" 2>&1

echo "$0: completed partial train feature probe at ${probe_artifact_root}"
