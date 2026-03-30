#!/usr/bin/env bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -euo pipefail

stage=0
stop_stage=7

language=zh
dataset_root=$PWD/download/Emilia

dev_ratio=0.001
test_ratio=0.001

num_splits=100
start=0
stop=-1

max_jsonl_files=-1
max_utterances=-1

num_workers=20
batch_duration=600
speed_perturb=true

. shared/parse_options.sh || exit 1

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
fbank_dir="data/fbank/${language}"

if [[ "$language" == "zh" ]]; then
  vocab_size=2000
  lang_dir="data/lang_bpe_zh_${vocab_size}"
  transcript_file="${lang_dir}/transcript_chars.txt"
else
  vocab_size=500
  lang_dir="data/lang_bpe_en_${vocab_size}"
  transcript_file="${lang_dir}/transcript_words.txt"
fi

mkdir -p data "$manifest_dir" "$fbank_dir"

log "language: $language"
log "dataset_root: $dataset_root"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Prepare Emilia ${language} manifests"
  python3 ./local/prepare_emilia.py \
    --dataset-root "$dataset_root" \
    --language "$language" \
    --output-dir "$manifest_dir" \
    --dev-ratio "$dev_ratio" \
    --test-ratio "$test_ratio" \
    --max-jsonl-files "$max_jsonl_files" \
    --max-utterances "$max_utterances"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Normalize transcripts and build raw cuts"
  speed_perturb_flag=()
  if [ "$speed_perturb" = true ]; then
    speed_perturb_flag+=(--speed-perturb)
  fi
  python3 ./local/preprocess_emilia.py \
    --language "$language" \
    --manifest-dir "$manifest_dir" \
    --output-dir "$fbank_dir" \
    "${speed_perturb_flag[@]}"
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Compute features for dev/test"
  for split in dev test; do
    python3 ./local/compute_fbank_emilia.py \
      --raw-cuts-path "${fbank_dir}/${prefix}_cuts_${split}_raw.jsonl.gz" \
      --output-cuts-path "${fbank_dir}/${prefix}_cuts_${split}.jsonl.gz" \
      --storage-path "${fbank_dir}/${prefix}_feats_${split}" \
      --num-workers "$num_workers" \
      --batch-duration "$batch_duration"
  done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Split train raw cuts"
  split_dir="${fbank_dir}/train_split_${num_splits}"
  if [ ! -f "${split_dir}/.split_completed" ]; then
    mkdir -p "$split_dir"
    lhotse split "$num_splits" "${fbank_dir}/${prefix}_cuts_train_raw.jsonl.gz" "$split_dir"
    touch "${split_dir}/.split_completed"
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute features for train splits"
  split_dir="${fbank_dir}/train_split_${num_splits}"
  if [ ! -d "$split_dir" ]; then
    echo "$0: Missing split dir ${split_dir}. Run stage 3 first."
    exit 1
  fi

  mapfile -t raw_paths < <(find "$split_dir" -maxdepth 1 -name "${prefix}_cuts_train_raw.*.jsonl.gz" | sort)
  if [ ${#raw_paths[@]} -eq 0 ]; then
    echo "$0: No split manifests found in ${split_dir}"
    exit 1
  fi

  total_splits=${#raw_paths[@]}
  if [ "$stop" -lt "$start" ]; then
    stop="$total_splits"
  fi
  if [ "$stop" -gt "$total_splits" ]; then
    stop="$total_splits"
  fi

  for ((i=start; i<stop; ++i)); do
    raw_path="${raw_paths[$i]}"
    file_name=$(basename "$raw_path")
    idx="${file_name#${prefix}_cuts_train_raw.}"
    idx="${idx%.jsonl.gz}"
    out_path="${split_dir}/${prefix}_cuts_train.${idx}.jsonl.gz"
    storage_path="${split_dir}/${prefix}_feats_train_${idx}"
    python3 ./local/compute_fbank_emilia.py \
      --raw-cuts-path "$raw_path" \
      --output-cuts-path "$out_path" \
      --storage-path "$storage_path" \
      --num-workers "$num_workers" \
      --batch-duration "$batch_duration"
  done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Combine train split cut manifests"
  split_dir="${fbank_dir}/train_split_${num_splits}"
  if [ ! -f "${fbank_dir}/${prefix}_cuts_train.jsonl.gz" ]; then
    pieces=$(find "$split_dir" -name "${prefix}_cuts_train.[0-9]*.jsonl.gz" | sort)
    if [ -z "$pieces" ]; then
      echo "$0: No processed split manifests found in ${split_dir}"
      exit 1
    fi
    lhotse combine $pieces "${fbank_dir}/${prefix}_cuts_train.jsonl.gz"
  fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare Musan features"
  if [ -e ../../librispeech/ASR/data/fbank/.musan.done ]; then
    mkdir -p "$fbank_dir"
    (
      cd "$fbank_dir"
      ln -snf "$(realpath ../../../../librispeech/ASR/data/fbank/musan_feats)" .
      ln -snf "$(realpath ../../../../librispeech/ASR/data/fbank/musan_cuts.jsonl.gz)" .
    )
  else
    if [ ! -d "${dataset_root%/}/../musan" ] && [ ! -d "$PWD/download/musan" ]; then
      log "Downloading Musan"
      lhotse download musan "$PWD/download"
    fi

    if [ ! -f data/manifests/.musan.done ]; then
      lhotse prepare musan "$PWD/download/musan" data/manifests
      touch data/manifests/.musan.done
    fi

    python3 ./local/compute_fbank_musan.py --output-dir "$fbank_dir"
    touch "${fbank_dir}/.musan.done"
  fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Prepare BPE based language dir"
  mkdir -p "$lang_dir"

  python3 ./local/prepare_emilia_bpe_data.py \
    --cuts-path "${fbank_dir}/${prefix}_cuts_train.jsonl.gz" \
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
