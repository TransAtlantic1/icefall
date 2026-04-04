#!/usr/bin/env bash
# prepare_data.sh — 一条命令跑完 Emilia 单语言数据准备全流程
#
# 核心优化：stage 4（train fbank 提取）自动启动 N 个后台进程，
#           每个进程绑定不同 GPU，并行处理不同的 split 分段。
#
# 用法示例:
#
#   # 中文全量数据准备（8*H200 并行提特征）
#   bash prepare_data.sh --language zh --num-gpus 8
#
#   # 英文全量
#   bash prepare_data.sh --language en --num-gpus 8
#
#   # Smoke test（在 4090 上快速验证链路）
#   bash prepare_data.sh --language zh --num-gpus 2 \
#     --max-jsonl-files 1 --max-utterances 500 --num-splits 4
#
#   # 只跑某些 stage
#   bash prepare_data.sh --language zh --stage 4 --stop-stage 5 --num-gpus 8

set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# icefall 根目录加入 PYTHONPATH（供 prepare_lang_bpe.py 等脚本 import icefall）
ICEFALL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${ICEFALL_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# 让 kaldifeat/k2 能找到 pip 安装的 CUDA 运行时库（libnvrtc 等）
_nvidia_lib="$(python3 -c 'import nvidia.cuda_nvrtc; import os; print(os.path.dirname(nvidia.cuda_nvrtc.__file__))' 2>/dev/null)/lib"
if [[ -d "$_nvidia_lib" ]]; then
  export LD_LIBRARY_PATH="${_nvidia_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# -----------------------------------------------------------------------
# 默认参数
# -----------------------------------------------------------------------
stage=0
stop_stage=7

language=zh
dataset_root=/inspire/dataset/emilia/fc71e07

num_splits=200
num_gpus=8             # stage 4 并行进程数，每个进程占一张 GPU
num_workers=0  # Docker 默认 /dev/shm 只有 64MB，多进程 DataLoader 会 OOM；用 0 走单进程模式
sampling_rate=32000  # Emilia mp3 音频采样率
batch_duration=600
speed_perturb=true
dev_ratio=0.001
test_ratio=0.001
max_jsonl_files=-1
max_utterances=-1
vocab_size=-1          # -1 = 使用语言默认值（zh:2000, en:500）；smoke test 时可设为小值如 500

# -----------------------------------------------------------------------
# 参数解析
# -----------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)           stage="$2";           shift 2 ;;
    --stop-stage)      stop_stage="$2";      shift 2 ;;
    --language)        language="$2";        shift 2 ;;
    --dataset-root)    dataset_root="$2";    shift 2 ;;
    --num-splits)      num_splits="$2";      shift 2 ;;
    --num-gpus)        num_gpus="$2";        shift 2 ;;
    --num-workers)     num_workers="$2";     shift 2 ;;
    --batch-duration)  batch_duration="$2";  shift 2 ;;
    --speed-perturb)   speed_perturb="$2";   shift 2 ;;
    --dev-ratio)       dev_ratio="$2";       shift 2 ;;
    --test-ratio)      test_ratio="$2";      shift 2 ;;
    --max-jsonl-files) max_jsonl_files="$2"; shift 2 ;;
    --max-utterances)  max_utterances="$2";  shift 2 ;;
    --vocab-size)      vocab_size="$2";      shift 2 ;;
    --sampling-rate)   sampling_rate="$2";  shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# -----------------------------------------------------------------------
# 校验
# -----------------------------------------------------------------------
if [[ "$language" != "zh" && "$language" != "en" ]]; then
  echo "ERROR: --language must be zh or en, got: $language"
  exit 1
fi

# -----------------------------------------------------------------------
# 路径派生
# -----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

prefix="emilia_${language}"
manifest_dir="data/manifests/${language}"
fbank_dir="data/fbank/${language}"

if [[ "$language" == "zh" ]]; then
  [[ "$vocab_size" -le 0 ]] && vocab_size=2000
  lang_dir="data/lang_bpe_zh_${vocab_size}"
  transcript_file="${lang_dir}/transcript_chars.txt"
else
  [[ "$vocab_size" -le 0 ]] && vocab_size=500
  lang_dir="data/lang_bpe_en_${vocab_size}"
  transcript_file="${lang_dir}/transcript_words.txt"
fi

mkdir -p data "$manifest_dir" "$fbank_dir"

log() {
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') [prepare_data.sh] $*"
}

log "=== Emilia data preparation ==="
log "language=$language  stage=$stage  stop_stage=$stop_stage"
log "dataset_root=$dataset_root"
log "num_splits=$num_splits  num_gpus=$num_gpus"

# -----------------------------------------------------------------------
# Stage 0: JSONL -> lhotse manifests
# -----------------------------------------------------------------------
if [[ $stage -le 0 && $stop_stage -ge 0 ]]; then
  log "Stage 0: Prepare Emilia ${language} manifests"
  python3 ./local/prepare_emilia.py \
    --dataset-root    "$dataset_root" \
    --language        "$language" \
    --output-dir      "$manifest_dir" \
    --dev-ratio       "$dev_ratio" \
    --test-ratio      "$test_ratio" \
    --max-jsonl-files "$max_jsonl_files" \
    --max-utterances  "$max_utterances"
fi

# -----------------------------------------------------------------------
# Stage 1: 文本归一化 + 构建 raw CutSet
# -----------------------------------------------------------------------
if [[ $stage -le 1 && $stop_stage -ge 1 ]]; then
  log "Stage 1: Normalize transcripts and build raw cuts"
  speed_perturb_flag=()
  if [[ "$speed_perturb" == "true" ]]; then
    speed_perturb_flag+=(--speed-perturb)
  fi
  python3 ./local/preprocess_emilia.py \
    --language     "$language" \
    --manifest-dir "$manifest_dir" \
    --output-dir   "$fbank_dir" \
    "${speed_perturb_flag[@]}"
fi

# -----------------------------------------------------------------------
# Stage 2: dev/test fbank
# -----------------------------------------------------------------------
if [[ $stage -le 2 && $stop_stage -ge 2 ]]; then
  log "Stage 2: Compute fbank for dev/test"
  for split in dev test; do
    raw_cuts="${fbank_dir}/${prefix}_cuts_${split}_raw.jsonl.gz"
    if [[ ! -f "$raw_cuts" ]]; then
      log "Stage 2: $raw_cuts not found, skipping $split split"
      continue
    fi
    python3 ./local/compute_fbank_emilia.py \
      --raw-cuts-path    "$raw_cuts" \
      --output-cuts-path "${fbank_dir}/${prefix}_cuts_${split}.jsonl.gz" \
      --storage-path     "${fbank_dir}/${prefix}_feats_${split}" \
      --num-workers      "$num_workers" \
      --batch-duration   "$batch_duration" \
      --sampling-rate    "$sampling_rate"
  done
fi

# -----------------------------------------------------------------------
# Stage 3: 切分 train raw cuts
# -----------------------------------------------------------------------
if [[ $stage -le 3 && $stop_stage -ge 3 ]]; then
  log "Stage 3: Split train raw cuts into $num_splits pieces"
  split_dir="${fbank_dir}/train_split_${num_splits}"
  if [[ ! -f "${split_dir}/.split_completed" ]]; then
    mkdir -p "$split_dir"
    lhotse split "$num_splits" \
      "${fbank_dir}/${prefix}_cuts_train_raw.jsonl.gz" \
      "$split_dir"
    touch "${split_dir}/.split_completed"
  else
    log "Stage 3: split already done, skipping"
  fi
fi

# -----------------------------------------------------------------------
# Stage 4: train fbank — 多 GPU 并行
#
#   原理：compute_fbank_emilia.py 内部写死 torch.device("cuda", 0)。
#   我们启动 num_gpus 个后台子进程，每个子进程通过
#   CUDA_VISIBLE_DEVICES=<物理GPU编号> 让它只看到一张卡，
#   各自处理 num_splits/num_gpus 份 split，互不冲突。
#
#   每个子进程的日志写到 logs/stage4_gpu<N>.log，可随时 tail 查看。
# -----------------------------------------------------------------------
if [[ $stage -le 4 && $stop_stage -ge 4 ]]; then
  log "Stage 4: Compute fbank for train splits (parallel across $num_gpus GPUs)"
  split_dir="${fbank_dir}/train_split_${num_splits}"

  mapfile -t raw_paths < <(
    find "$split_dir" -maxdepth 1 \
      -name "${prefix}_cuts_train_raw.*.jsonl.gz" | sort
  )
  total_splits=${#raw_paths[@]}
  if [[ $total_splits -eq 0 ]]; then
    echo "ERROR: No split manifests in ${split_dir}. Run stage 3 first."
    exit 1
  fi
  log "Stage 4: found $total_splits split files, dispatching to $num_gpus GPUs"

  mkdir -p logs
  pids=()
  per_gpu=$(( (total_splits + num_gpus - 1) / num_gpus ))  # 向上取整

  for ((gpu=0; gpu<num_gpus; gpu++)); do
    range_start=$((gpu * per_gpu))
    range_stop=$(( (gpu + 1) * per_gpu ))
    if [[ $range_start -ge $total_splits ]]; then
      break  # GPU 数多于 split 数，剩余 GPU 不用启动
    fi
    if [[ $range_stop -gt $total_splits ]]; then
      range_stop=$total_splits
    fi

    log_file="logs/stage4_gpu${gpu}.log"
    log "  GPU $gpu: splits [$range_start, $range_stop) -> $log_file"

    (
      for ((i=range_start; i<range_stop; i++)); do
        raw_path="${raw_paths[$i]}"
        file_name=$(basename "$raw_path")
        idx="${file_name#${prefix}_cuts_train_raw.}"
        idx="${idx%.jsonl.gz}"
        out_path="${split_dir}/${prefix}_cuts_train.${idx}.jsonl.gz"
        storage_path="${split_dir}/${prefix}_feats_train_${idx}"
        CUDA_VISIBLE_DEVICES=$gpu python3 ./local/compute_fbank_emilia.py \
          --raw-cuts-path    "$raw_path" \
          --output-cuts-path "$out_path" \
          --storage-path     "$storage_path" \
          --num-workers      "$num_workers" \
          --batch-duration   "$batch_duration" \
          --sampling-rate    "$sampling_rate"
      done
    ) > "$log_file" 2>&1 &

    pids+=($!)
  done

  # 等待所有后台进程完成
  log "Stage 4: waiting for ${#pids[@]} GPU workers to finish..."
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      log "ERROR: worker PID $pid failed"
      failed=$((failed + 1))
    fi
  done

  if [[ $failed -gt 0 ]]; then
    log "ERROR: $failed GPU worker(s) failed. Check logs/stage4_gpu*.log"
    exit 1
  fi
  log "Stage 4: all GPU workers finished successfully"
fi

# -----------------------------------------------------------------------
# Stage 5: 合并 train splits
# -----------------------------------------------------------------------
if [[ $stage -le 5 && $stop_stage -ge 5 ]]; then
  log "Stage 5: Combine train split manifests"
  split_dir="${fbank_dir}/train_split_${num_splits}"
  if [[ ! -f "${fbank_dir}/${prefix}_cuts_train.jsonl.gz" ]]; then
    pieces=$(find "$split_dir" -name "${prefix}_cuts_train.[0-9]*.jsonl.gz" | sort)
    if [[ -z "$pieces" ]]; then
      echo "ERROR: No processed split manifests in ${split_dir}. Run stage 4 first."
      exit 1
    fi
    # shellcheck disable=SC2086
    lhotse combine $pieces "${fbank_dir}/${prefix}_cuts_train.jsonl.gz"
  else
    log "Stage 5: combined manifest already exists, skipping"
  fi
fi

# -----------------------------------------------------------------------
# Stage 6: MUSAN
# -----------------------------------------------------------------------
if [[ $stage -le 6 && $stop_stage -ge 6 ]]; then
  log "Stage 6: Prepare MUSAN features"
  if [[ -e ../../librispeech/ASR/data/fbank/.musan.done ]]; then
    (
      cd "$fbank_dir"
      ln -snf "$(realpath ../../../../librispeech/ASR/data/fbank/musan_feats)" .
      ln -snf "$(realpath ../../../../librispeech/ASR/data/fbank/musan_cuts.jsonl.gz)" .
    )
  else
    musan_dir="$PWD/download/musan"
    if [[ ! -d "$musan_dir" ]]; then
      log "Downloading MUSAN..."
      lhotse download musan "$PWD/download"
    fi
    if [[ ! -f data/manifests/.musan.done ]]; then
      lhotse prepare musan "$musan_dir" data/manifests
      touch data/manifests/.musan.done
    fi
    python3 ./local/compute_fbank_musan.py --output-dir "$fbank_dir"
    touch "${fbank_dir}/.musan.done"
  fi
fi

# -----------------------------------------------------------------------
# Stage 7: BPE 语言目录
# -----------------------------------------------------------------------
if [[ $stage -le 7 && $stop_stage -ge 7 ]]; then
  log "Stage 7: Prepare BPE language dir ($lang_dir, vocab=$vocab_size)"
  mkdir -p "$lang_dir"

  python3 ./local/prepare_emilia_bpe_data.py \
    --cuts-path  "${fbank_dir}/${prefix}_cuts_train.jsonl.gz" \
    --language   "$language" \
    --lang-dir   "$lang_dir"

  if [[ ! -f "${lang_dir}/bpe.model" ]]; then
    python3 ./local/train_bpe_model.py \
      --lang-dir   "$lang_dir" \
      --transcript "$transcript_file" \
      --vocab-size "$vocab_size"
  fi

  if [[ ! -f "${lang_dir}/tokens.txt" ]]; then
    python3 ./local/bpe_model_to_tokens.py "${lang_dir}/bpe.model" \
      > "${lang_dir}/tokens.txt"
  fi

  if [[ ! -f "${lang_dir}/L_disambig.pt" ]]; then
    python3 ./local/prepare_lang_bpe.py --lang-dir "$lang_dir"
    python3 ./local/validate_bpe_lexicon.py \
      --lexicon   "${lang_dir}/lexicon.txt" \
      --bpe-model "${lang_dir}/bpe.model"
  fi
fi

log "=== prepare_data.sh DONE ==="
