#!/usr/bin/env bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ICEFALL_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PARSE_OPTIONS_SH="${ICEFALL_ROOT}/icefall/shared/parse_options.sh"

stage=0
stop_stage=10

language=zh
public_root="/inspire/qb-ilm/project/embodied-multimodality/chenxie-25019/public"
default_dataset_root="/inspire/dataset/emilia/fc71e07"

download_root=""
dataset_root=""
audio_cache_root=""
data_root=""
artifact_root=""

dev_ratio=0.001
test_ratio=0.001

recording_num_splits=1000
resample_start=0
resample_stop=-1
resample_num_workers=32

# Backward-compatible feature split options.
num_splits=100
start=0
stop=-1
num_workers=20
batch_duration=1000

feature_num_splits=""
feature_start=""
feature_stop=""
feature_num_workers=""
feature_batch_duration=""
feature_device="auto"

target_sample_rate=24000
use_resampled_audio=true
# For the local fc71e07 Emilia copy, source audio is not uniformly 32 kHz.
# Keep this enabled unless you have separately verified the exact source
# sample-rate distribution for the subset you are processing.
speed_perturb=false
enable_musan=false

max_jsonl_files=-1
max_utterances=-1

. "${PARSE_OPTIONS_SH}" || exit 1

if [ -z "$feature_num_splits" ]; then
  feature_num_splits=$num_splits
fi
if [ -z "$feature_start" ]; then
  feature_start=$start
fi
if [ -z "$feature_stop" ]; then
  feature_stop=$stop
fi
if [ -z "$feature_num_workers" ]; then
  feature_num_workers=$num_workers
fi
if [ -z "$feature_batch_duration" ]; then
  feature_batch_duration=$batch_duration
fi

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [[ "$language" != "zh" && "$language" != "en" ]]; then
  echo "$0: --language must be one of zh or en, got: $language"
  exit 1
fi

if [ -z "$artifact_root" ]; then
  artifact_root="${public_root%/}/emilia/fc71e07/icefall_emilia_${language}_24k"
fi
if [ -z "$download_root" ]; then
  download_root="${artifact_root}/download"
fi
if [ -z "$dataset_root" ]; then
  dataset_root="${default_dataset_root}"
fi
if [ -z "$audio_cache_root" ]; then
  audio_cache_root="${artifact_root}/audio_cache"
fi
if [ -z "$data_root" ]; then
  data_root="${artifact_root}/data"
fi

prefix="emilia_${language}"
manifests_root="${data_root}/manifests"
manifest_dir="${manifests_root}/${language}"
resampled_manifest_dir="${data_root}/manifests_resampled/${language}/${target_sample_rate}"
fbank_dir="${data_root}/fbank/${language}"
recording_split_dir="${manifest_dir}/recordings_train_split_${recording_num_splits}"
resampled_recording_split_dir="${resampled_manifest_dir}/recordings_train_split_${recording_num_splits}"
resample_lock_dir="${artifact_root}/locks/resample/${language}/${target_sample_rate}/recordings_train_split_${recording_num_splits}"
train_feature_split_dir="${fbank_dir}/train_split_${feature_num_splits}"
cache_dir="${audio_cache_root}/emilia/${language}"
input_audio_sampling_rate=$target_sample_rate
if [ "$use_resampled_audio" = false ]; then
  # Warning: this fallback assumes a uniform 32 kHz source distribution.
  # That assumption is false for the local fc71e07 copy, which contains
  # mixed source sample rates (at least 24 kHz / 32 kHz / 44.1 kHz across
  # inspected subsets). Using this branch may cause declared-sample-count
  # mismatches unless the processed subset is known to be uniformly 32 kHz.
  input_audio_sampling_rate=32000
fi

if [[ "$language" == "zh" ]]; then
  vocab_size=2000
  lang_dir="${data_root}/lang_bpe_zh_${vocab_size}"
  transcript_file="${lang_dir}/transcript_chars.txt"
else
  vocab_size=500
  lang_dir="${data_root}/lang_bpe_en_${vocab_size}"
  transcript_file="${lang_dir}/transcript_words.txt"
fi

mkdir -p "$data_root" "$manifest_dir" "$fbank_dir" "$resampled_manifest_dir"

run_resample_with_lock() {
  local input_manifest="$1"
  local output_manifest="$2"
  local lock_dir="${resample_lock_dir}/$(basename "${output_manifest}").lock"

  if [ -f "$output_manifest" ]; then
    log "Stage 3: Reusing existing resampled manifest ${output_manifest}"
    return 0
  fi

  mkdir -p "$resample_lock_dir"
  if ! mkdir "$lock_dir" 2>/dev/null; then
    log "Stage 3: Lock busy for ${output_manifest}, skipping on this worker"
    return 0
  fi

  printf 'host=%s\npid=%s\ntime=%s\n' \
    "$(hostname)" "$$" "$(date '+%Y-%m-%d %H:%M:%S')" >"${lock_dir}/owner"

  (
    trap 'rm -rf "$lock_dir"' EXIT INT TERM

    if [ -f "$output_manifest" ]; then
      log "Stage 3: Reusing existing resampled manifest ${output_manifest} after locking"
      exit 0
    fi

    python3 "${SCRIPT_DIR}/local/resample_recordings_to_flac.py" \
      --input-manifest "$input_manifest" \
      --output-manifest "$output_manifest" \
      --source-root "$dataset_root" \
      --cache-root "$cache_dir" \
      --target-sample-rate "$target_sample_rate" \
      --num-workers "$resample_num_workers"
  )
}

log "language: $language"
log "artifact_root: $artifact_root"
log "data_root: $data_root"
log "download_root: $download_root"
log "dataset_root: $dataset_root"
log "audio_cache_root: $audio_cache_root"
log "target_sample_rate: $target_sample_rate"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Prepare Emilia ${language} manifests"
  python3 "${SCRIPT_DIR}/local/prepare_emilia_manifests.py" \
    --dataset-root "$dataset_root" \
    --language "$language" \
    --output-dir "$manifest_dir" \
    --dev-ratio "$dev_ratio" \
    --test-ratio "$test_ratio" \
    --max-jsonl-files "$max_jsonl_files" \
    --max-utterances "$max_utterances"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Split train recordings into ${recording_num_splits} shards"
  if [ ! -f "${recording_split_dir}/.split_completed" ]; then
    mkdir -p "$recording_split_dir"
    lhotse split \
      "$recording_num_splits" \
      "${manifest_dir}/${prefix}_recordings_train.jsonl.gz" \
      "$recording_split_dir"
    touch "${recording_split_dir}/.split_completed"
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  if [ "$enable_musan" = false ]; then
    log "Stage 2: Skipping MUSAN manifest prep because enable_musan=false"
  elif [ -e "${SCRIPT_DIR}/../../librispeech/ASR/data/fbank/.musan.done" ]; then
    log "Stage 2: Shared Librispeech MUSAN features are available; no local MUSAN manifest prep needed"
  else
    log "Stage 2: Prepare MUSAN manifests"
    musan_source_dir="${download_root}/musan"
    if [ ! -d "$musan_source_dir" ] && [ -d "${dataset_root%/}/../musan" ]; then
      musan_source_dir="${dataset_root%/}/../musan"
    fi
    if [ ! -d "$musan_source_dir" ]; then
      log "Downloading MUSAN"
      lhotse download musan "$download_root"
      musan_source_dir="${download_root}/musan"
    fi
    if [ ! -f "${manifests_root}/.musan.done" ]; then
      lhotse prepare musan "$musan_source_dir" "$manifests_root"
      touch "${manifests_root}/.musan.done"
    fi
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  if [ "$use_resampled_audio" = false ]; then
    log "Stage 3: Skipping offline resampling because use_resampled_audio=false"
    log "Stage 3: Warning: downstream stages will assume 32000 Hz input audio; this is unsafe for mixed-rate fc71e07 subsets unless you have verified the subset is uniformly 32 kHz"
  else
    log "Stage 3: Offline resample recordings to ${target_sample_rate} Hz"
    mkdir -p "$resampled_manifest_dir" "$resampled_recording_split_dir" "$cache_dir" "$resample_lock_dir"
    if [ ! -d "$recording_split_dir" ]; then
      echo "$0: Missing recording split dir ${recording_split_dir}. Run stage 1 first."
      exit 1
    fi

    if [ "$resample_start" -eq 0 ]; then
      for split in dev test; do
        run_resample_with_lock \
          "${manifest_dir}/${prefix}_recordings_${split}.jsonl.gz" \
          "${resampled_manifest_dir}/${prefix}_recordings_${split}.jsonl.gz"
      done
    else
      log "Stage 3: resample_start=${resample_start}, skipping dev/test resampling on this worker"
    fi

    mapfile -t recording_shards < <(
      find "$recording_split_dir" -maxdepth 1 -name "${prefix}_recordings_train.*.jsonl.gz" | sort
    )
    if [ ${#recording_shards[@]} -eq 0 ]; then
      echo "$0: No train recording shards found in ${recording_split_dir}"
      exit 1
    fi

    total_recording_shards=${#recording_shards[@]}
    if [ "$resample_stop" -lt "$resample_start" ]; then
      resample_stop="$total_recording_shards"
    fi
    if [ "$resample_stop" -gt "$total_recording_shards" ]; then
      resample_stop="$total_recording_shards"
    fi

    for ((i=resample_start; i<resample_stop; ++i)); do
      shard_path="${recording_shards[$i]}"
      shard_name=$(basename "$shard_path")
      run_resample_with_lock \
        "$shard_path" \
        "${resampled_recording_split_dir}/${shard_name}"
    done
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Normalize transcripts and build raw cuts"
  speed_perturb_flag=()
  if [ "$speed_perturb" = true ]; then
    speed_perturb_flag+=(--speed-perturb)
  fi

  recordings_manifest_args=()
  if [ "$use_resampled_audio" = true ]; then
    recordings_manifest_args+=(--recordings-manifest-dir "$resampled_manifest_dir")
  fi

  python3 "${SCRIPT_DIR}/local/prepare_emilia_raw_cuts.py" \
    --language "$language" \
    --manifest-dir "$manifest_dir" \
    --output-dir "$fbank_dir" \
    "${recordings_manifest_args[@]}" \
    "${speed_perturb_flag[@]}"
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Compute features for dev/test"
  for split in dev test; do
    python3 "${SCRIPT_DIR}/local/compute_emilia_features.py" \
      --raw-cuts-path "${fbank_dir}/${prefix}_cuts_${split}_raw.jsonl.gz" \
      --output-cuts-path "${fbank_dir}/${prefix}_cuts_${split}.jsonl.gz" \
      --storage-path "${fbank_dir}/${prefix}_feats_${split}" \
      --num-workers "$feature_num_workers" \
      --batch-duration "$feature_batch_duration" \
      --device "$feature_device" \
      --sampling-rate "$input_audio_sampling_rate"
  done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Split train raw cuts into ${feature_num_splits} shards"
  if [ ! -f "${train_feature_split_dir}/.split_completed" ]; then
    mkdir -p "$train_feature_split_dir"
    lhotse split \
      "$feature_num_splits" \
      "${fbank_dir}/${prefix}_cuts_train_raw.jsonl.gz" \
      "$train_feature_split_dir"
    touch "${train_feature_split_dir}/.split_completed"
  fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Compute features for train splits"
  if [ ! -d "$train_feature_split_dir" ]; then
    echo "$0: Missing split dir ${train_feature_split_dir}. Run stage 6 first."
    exit 1
  fi

  mapfile -t raw_paths < <(
    find "$train_feature_split_dir" -maxdepth 1 -name "${prefix}_cuts_train_raw.*.jsonl.gz" | sort
  )
  if [ ${#raw_paths[@]} -eq 0 ]; then
    echo "$0: No split manifests found in ${train_feature_split_dir}"
    exit 1
  fi

  total_feature_splits=${#raw_paths[@]}
  if [ "$feature_stop" -lt "$feature_start" ]; then
    feature_stop="$total_feature_splits"
  fi
  if [ "$feature_stop" -gt "$total_feature_splits" ]; then
    feature_stop="$total_feature_splits"
  fi

  for ((i=feature_start; i<feature_stop; ++i)); do
    raw_path="${raw_paths[$i]}"
    file_name=$(basename "$raw_path")
    idx="${file_name#${prefix}_cuts_train_raw.}"
    idx="${idx%.jsonl.gz}"
    out_path="${train_feature_split_dir}/${prefix}_cuts_train.${idx}.jsonl.gz"
    storage_path="${train_feature_split_dir}/${prefix}_feats_train_${idx}"
    python3 "${SCRIPT_DIR}/local/compute_emilia_features.py" \
      --raw-cuts-path "$raw_path" \
      --output-cuts-path "$out_path" \
      --storage-path "$storage_path" \
      --num-workers "$feature_num_workers" \
      --batch-duration "$feature_batch_duration" \
      --device "$feature_device" \
      --sampling-rate "$input_audio_sampling_rate"
  done
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  if [ "$enable_musan" = false ]; then
    log "Stage 8: Skipping MUSAN features because enable_musan=false"
  elif [ -e "${SCRIPT_DIR}/../../librispeech/ASR/data/fbank/.musan.done" ]; then
    log "Stage 8: Link shared Librispeech MUSAN features"
    mkdir -p "$fbank_dir"
    ln -snf \
      "$(realpath "${SCRIPT_DIR}/../../librispeech/ASR/data/fbank/musan_feats")" \
      "${fbank_dir}/musan_feats"
    ln -snf \
      "$(realpath "${SCRIPT_DIR}/../../librispeech/ASR/data/fbank/musan_cuts.jsonl.gz")" \
      "${fbank_dir}/musan_cuts.jsonl.gz"
    touch "${fbank_dir}/.musan.done"
  else
    log "Stage 8: Compute MUSAN features"
    if [ ! -f "${manifests_root}/.musan.done" ]; then
      echo "$0: Missing MUSAN manifests. Run stage 2 with --enable-musan true first."
      exit 1
    fi
    python3 "${SCRIPT_DIR}/local/compute_fbank_musan.py" \
      --manifest-dir "${manifests_root}" \
      --output-dir "$fbank_dir"
    touch "${fbank_dir}/.musan.done"
  fi
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Combine train split cut manifests"
  if [ ! -f "${fbank_dir}/${prefix}_cuts_train.jsonl.gz" ]; then
    pieces=$(find "$train_feature_split_dir" -name "${prefix}_cuts_train.[0-9]*.jsonl.gz" | sort)
    if [ -z "$pieces" ]; then
      echo "$0: No processed split manifests found in ${train_feature_split_dir}"
      exit 1
    fi
    lhotse combine $pieces "${fbank_dir}/${prefix}_cuts_train.jsonl.gz"
  fi
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
  log "Stage 10: Prepare BPE based language dir"
  mkdir -p "$lang_dir"

  cuts_source="${fbank_dir}/${prefix}_cuts_train.jsonl.gz"
  if [ ! -f "$cuts_source" ]; then
    cuts_source="$fbank_dir"
  fi

  python3 "${SCRIPT_DIR}/local/prepare_emilia_bpe_data.py" \
    --cuts-path "$cuts_source" \
    --language "$language" \
    --lang-dir "$lang_dir"

  if [ ! -f "${lang_dir}/bpe.model" ]; then
    python3 "${SCRIPT_DIR}/local/train_bpe_model.py" \
      --lang-dir "$lang_dir" \
      --transcript "$transcript_file" \
      --vocab-size "$vocab_size"
  fi

  if [ ! -f "${lang_dir}/tokens.txt" ]; then
    python3 "${SCRIPT_DIR}/local/bpe_model_to_tokens.py" "${lang_dir}/bpe.model" > "${lang_dir}/tokens.txt"
  fi

  if [ ! -f "${lang_dir}/L_disambig.pt" ]; then
    python3 "${SCRIPT_DIR}/local/prepare_lang_bpe.py" --lang-dir "$lang_dir"
    python3 "${SCRIPT_DIR}/local/validate_bpe_lexicon.py" \
      --lexicon "${lang_dir}/lexicon.txt" \
      --bpe-model "${lang_dir}/bpe.model"
  fi
fi

log "prepare.sh: DONE"
