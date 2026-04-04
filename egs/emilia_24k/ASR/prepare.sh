#!/usr/bin/env bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -euo pipefail

stage=0
stop_stage=10

language=zh
dataset_root=$PWD/download/Emilia
audio_cache_root=$PWD/download/audio_cache

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
batch_duration=600

feature_num_splits=""
feature_start=""
feature_stop=""
feature_num_workers=""
feature_batch_duration=""

target_sample_rate=24000
use_resampled_audio=true
speed_perturb=true
enable_musan=false

max_jsonl_files=-1
max_utterances=-1

. shared/parse_options.sh || exit 1

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

prefix="emilia_${language}"
manifest_dir="data/manifests/${language}"
resampled_manifest_dir="data/manifests_resampled/${language}/${target_sample_rate}"
fbank_dir="data/fbank/${language}"
recording_split_dir="${manifest_dir}/recordings_train_split_${recording_num_splits}"
resampled_recording_split_dir="${resampled_manifest_dir}/recordings_train_split_${recording_num_splits}"
train_feature_split_dir="${fbank_dir}/train_split_${feature_num_splits}"
cache_dir="${audio_cache_root}/emilia/${language}"
input_audio_sampling_rate=$target_sample_rate
if [ "$use_resampled_audio" = false ]; then
  input_audio_sampling_rate=32000
fi

if [[ "$language" == "zh" ]]; then
  vocab_size=2000
  lang_dir="data/lang_bpe_zh_${vocab_size}"
  transcript_file="${lang_dir}/transcript_chars.txt"
else
  vocab_size=500
  lang_dir="data/lang_bpe_en_${vocab_size}"
  transcript_file="${lang_dir}/transcript_words.txt"
fi

mkdir -p data "$manifest_dir" "$fbank_dir" "$resampled_manifest_dir"

log "language: $language"
log "dataset_root: $dataset_root"
log "audio_cache_root: $audio_cache_root"
log "target_sample_rate: $target_sample_rate"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Prepare Emilia ${language} manifests"
  python3 ./local/prepare_emilia_manifests.py \
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
  elif [ -e ../../librispeech/ASR/data/fbank/.musan.done ]; then
    log "Stage 2: Shared Librispeech MUSAN features are available; no local MUSAN manifest prep needed"
  else
    log "Stage 2: Prepare MUSAN manifests"
    musan_source_dir="$PWD/download/musan"
    if [ ! -d "$musan_source_dir" ] && [ -d "${dataset_root%/}/../musan" ]; then
      musan_source_dir="${dataset_root%/}/../musan"
    fi
    if [ ! -d "$musan_source_dir" ]; then
      log "Downloading MUSAN"
      lhotse download musan "$PWD/download"
      musan_source_dir="$PWD/download/musan"
    fi
    if [ ! -f data/manifests/.musan.done ]; then
      lhotse prepare musan "$musan_source_dir" data/manifests
      touch data/manifests/.musan.done
    fi
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  if [ "$use_resampled_audio" = false ]; then
    log "Stage 3: Skipping offline resampling because use_resampled_audio=false"
  else
    log "Stage 3: Offline resample recordings to ${target_sample_rate} Hz"
    mkdir -p "$resampled_manifest_dir" "$resampled_recording_split_dir" "$cache_dir"
    if [ ! -d "$recording_split_dir" ]; then
      echo "$0: Missing recording split dir ${recording_split_dir}. Run stage 1 first."
      exit 1
    fi

    if [ "$resample_start" -eq 0 ]; then
      for split in dev test; do
        python3 ./local/resample_recordings_to_flac.py \
          --input-manifest "${manifest_dir}/${prefix}_recordings_${split}.jsonl.gz" \
          --output-manifest "${resampled_manifest_dir}/${prefix}_recordings_${split}.jsonl.gz" \
          --source-root "$dataset_root" \
          --cache-root "$cache_dir" \
          --target-sample-rate "$target_sample_rate" \
          --num-workers "$resample_num_workers"
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
      python3 ./local/resample_recordings_to_flac.py \
        --input-manifest "$shard_path" \
        --output-manifest "${resampled_recording_split_dir}/${shard_name}" \
        --source-root "$dataset_root" \
        --cache-root "$cache_dir" \
        --target-sample-rate "$target_sample_rate" \
        --num-workers "$resample_num_workers"
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

  python3 ./local/prepare_emilia_raw_cuts.py \
    --language "$language" \
    --manifest-dir "$manifest_dir" \
    --output-dir "$fbank_dir" \
    "${recordings_manifest_args[@]}" \
    "${speed_perturb_flag[@]}"
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Compute features for dev/test"
  for split in dev test; do
    python3 ./local/compute_emilia_features.py \
      --raw-cuts-path "${fbank_dir}/${prefix}_cuts_${split}_raw.jsonl.gz" \
      --output-cuts-path "${fbank_dir}/${prefix}_cuts_${split}.jsonl.gz" \
      --storage-path "${fbank_dir}/${prefix}_feats_${split}" \
      --num-workers "$feature_num_workers" \
      --batch-duration "$feature_batch_duration" \
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
    python3 ./local/compute_emilia_features.py \
      --raw-cuts-path "$raw_path" \
      --output-cuts-path "$out_path" \
      --storage-path "$storage_path" \
      --num-workers "$feature_num_workers" \
      --batch-duration "$feature_batch_duration" \
      --sampling-rate "$input_audio_sampling_rate"
  done
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  if [ "$enable_musan" = false ]; then
    log "Stage 8: Skipping MUSAN features because enable_musan=false"
  elif [ -e ../../librispeech/ASR/data/fbank/.musan.done ]; then
    log "Stage 8: Link shared Librispeech MUSAN features"
    mkdir -p "$fbank_dir"
    (
      cd "$fbank_dir"
      ln -snf "$(realpath ../../../../librispeech/ASR/data/fbank/musan_feats)" .
      ln -snf "$(realpath ../../../../librispeech/ASR/data/fbank/musan_cuts.jsonl.gz)" .
    )
    touch "${fbank_dir}/.musan.done"
  else
    log "Stage 8: Compute MUSAN features"
    if [ ! -f data/manifests/.musan.done ]; then
      echo "$0: Missing MUSAN manifests. Run stage 2 with --enable-musan true first."
      exit 1
    fi
    python3 ./local/compute_fbank_musan.py --output-dir "$fbank_dir"
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

  python3 ./local/prepare_emilia_bpe_data.py \
    --cuts-path "$cuts_source" \
    --language "$language" \
    --lang-dir "$lang_dir"

  if [ ! -f "${lang_dir}/bpe.model" ]; then
    python3 ./local/train_bpe_model.py \
      --lang-dir "$lang_dir" \
      --transcript "$transcript_file" \
      --vocab-size "$vocab_size"
  fi

  if [ ! -f "${lang_dir}/tokens.txt" ]; then
    python3 ./local/bpe_model_to_tokens.py "${lang_dir}/bpe.model" > "${lang_dir}/tokens.txt"
  fi

  if [ ! -f "${lang_dir}/L_disambig.pt" ]; then
    python3 ./local/prepare_lang_bpe.py --lang-dir "$lang_dir"
    python3 ./local/validate_bpe_lexicon.py \
      --lexicon "${lang_dir}/lexicon.txt" \
      --bpe-model "${lang_dir}/bpe.model"
  fi
fi

log "prepare.sh: DONE"
