#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
fi

if command -v conda >/dev/null 2>&1 && [[ "${CONDA_DEFAULT_ENV:-}" != "icefall" ]]; then
  conda activate icefall
fi

DECODE_CUDA_VISIBLE_DEVICES="${DECODE_CUDA_VISIBLE_DEVICES:-0}"
AVG="${AVG:-9}"
MAX_DURATION="${MAX_DURATION:-1000}"
BEAM_SIZE="${BEAM_SIZE:-4}"
DECODE_NUM_WORKERS="${DECODE_NUM_WORKERS:-0}"
POLL_SECONDS="${POLL_SECONDS:-300}"
TARGET_EPOCHS="${TARGET_EPOCHS:-10 20 30}"
RUN_ONCE="${RUN_ONCE:-0}"
EXIT_WHEN_DONE="${EXIT_WHEN_DONE:-1}"
TARGET_JOBS="${TARGET_JOBS:-16k 24k}"

JOB16_NAME="${JOB16_NAME:-16k}"
JOB16_RECIPE_DIR="${JOB16_RECIPE_DIR:-${repo_root}/egs/gigaspeech_16k/ASR}"
JOB16_EXP_DIR="${JOB16_EXP_DIR:-/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_h200/16k_train_g0-3/zipformer_m_g0-1-2-3}"
JOB16_MANIFEST_DIR="${JOB16_MANIFEST_DIR:-/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_16k/ASR/data/fbank}"
JOB16_BPE_MODEL="${JOB16_BPE_MODEL:-/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_16k/ASR/data/lang_bpe_500/bpe.model}"
JOB16_LANG_DIR="${JOB16_LANG_DIR:-/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/icefall/egs/gigaspeech_16k/ASR/data/lang_bpe_500}"

JOB24_NAME="${JOB24_NAME:-24k}"
JOB24_RECIPE_DIR="${JOB24_RECIPE_DIR:-${repo_root}/egs/gigaspeech_24k/ASR}"
JOB24_EXP_DIR="${JOB24_EXP_DIR:-/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_h200/24k_train_g4-7/zipformer_m_g4-5-6-7}"
JOB24_MANIFEST_DIR="${JOB24_MANIFEST_DIR:-/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_h200/24k_data_ready/fbank}"
JOB24_BPE_MODEL="${JOB24_BPE_MODEL:-/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_h200/24k_data_ready/lang_bpe_500/bpe.model}"
JOB24_LANG_DIR="${JOB24_LANG_DIR:-/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_h200/24k_data_ready/lang_bpe_500}"

JOB_NAMES=("${JOB16_NAME}" "${JOB24_NAME}")

declare -A JOB_RECIPE_DIRS=(
  ["${JOB16_NAME}"]="${JOB16_RECIPE_DIR}"
  ["${JOB24_NAME}"]="${JOB24_RECIPE_DIR}"
)

declare -A JOB_EXP_DIRS=(
  ["${JOB16_NAME}"]="${JOB16_EXP_DIR}"
  ["${JOB24_NAME}"]="${JOB24_EXP_DIR}"
)

declare -A JOB_MANIFEST_DIRS=(
  ["${JOB16_NAME}"]="${JOB16_MANIFEST_DIR}"
  ["${JOB24_NAME}"]="${JOB24_MANIFEST_DIR}"
)

declare -A JOB_BPE_MODELS=(
  ["${JOB16_NAME}"]="${JOB16_BPE_MODEL}"
  ["${JOB24_NAME}"]="${JOB24_BPE_MODEL}"
)

declare -A JOB_LANG_DIRS=(
  ["${JOB16_NAME}"]="${JOB16_LANG_DIR}"
  ["${JOB24_NAME}"]="${JOB24_LANG_DIR}"
)

log() {
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

decode_epoch() {
  local job_name="$1"
  local epoch="$2"
  local method="$3"

  local recipe_dir="${JOB_RECIPE_DIRS[${job_name}]}"
  local exp_dir="${JOB_EXP_DIRS[${job_name}]}"
  local manifest_dir="${JOB_MANIFEST_DIRS[${job_name}]}"
  local bpe_model="${JOB_BPE_MODELS[${job_name}]}"
  local lang_dir="${JOB_LANG_DIRS[${job_name}]}"
  local state_dir="${exp_dir}/auto_decode_state"
  local log_dir="${exp_dir}/auto_decode_logs"
  local done_marker="${state_dir}/epoch-${epoch}.${method}.done"
  local log_file="${log_dir}/epoch-${epoch}.${method}.log"

  mkdir -p "${state_dir}" "${log_dir}"

  if [[ -f "${done_marker}" ]]; then
    return 0
  fi

  local cmd=(
    python zipformer/decode.py
    --epoch "${epoch}"
    --avg "${AVG}"
    --exp-dir "${exp_dir}"
    --manifest-dir "${manifest_dir}"
    --bpe-model "${bpe_model}"
    --lang-dir "${lang_dir}"
    --max-duration "${MAX_DURATION}"
    --num-workers "${DECODE_NUM_WORKERS}"
    --decoding-method "${method}"
  )

  if [[ "${method}" == "modified_beam_search" ]]; then
    cmd+=(--beam-size "${BEAM_SIZE}")
  fi

  log "${job_name}: start decode epoch=${epoch} method=${method}"

  if (
    cd "${recipe_dir}"
    CUDA_VISIBLE_DEVICES="${DECODE_CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
  ) >"${log_file}" 2>&1; then
    touch "${done_marker}"
    log "${job_name}: done decode epoch=${epoch} method=${method}"
  else
    log "${job_name}: failed decode epoch=${epoch} method=${method}, see ${log_file}"
    return 1
  fi
}

epoch_ready() {
  local exp_dir="$1"
  local epoch="$2"
  local avg_start_epoch=$((epoch - AVG))

  if (( avg_start_epoch < 1 )); then
    return 1
  fi

  [[ -f "${exp_dir}/epoch-${epoch}.pt" && -f "${exp_dir}/epoch-${avg_start_epoch}.pt" ]]
}

job_epoch_done() {
  local exp_dir="$1"
  local epoch="$2"
  [[ -f "${exp_dir}/auto_decode_state/epoch-${epoch}.greedy_search.done" ]] \
    && [[ -f "${exp_dir}/auto_decode_state/epoch-${epoch}.modified_beam_search.done" ]]
}

print_config() {
  log "decode watcher config:"
  log "  DECODE_CUDA_VISIBLE_DEVICES=${DECODE_CUDA_VISIBLE_DEVICES}"
  log "  AVG=${AVG}"
  log "  MAX_DURATION=${MAX_DURATION}"
  log "  BEAM_SIZE=${BEAM_SIZE}"
  log "  DECODE_NUM_WORKERS=${DECODE_NUM_WORKERS}"
  log "  POLL_SECONDS=${POLL_SECONDS}"
  log "  TARGET_EPOCHS=${TARGET_EPOCHS}"
  log "  TARGET_JOBS=${TARGET_JOBS}"
  for job_name in "${JOB_NAMES[@]}"; do
    log "  ${job_name}: recipe_dir=${JOB_RECIPE_DIRS[${job_name}]}"
    log "  ${job_name}: exp_dir=${JOB_EXP_DIRS[${job_name}]}"
    log "  ${job_name}: manifest_dir=${JOB_MANIFEST_DIRS[${job_name}]}"
    log "  ${job_name}: bpe_model=${JOB_BPE_MODELS[${job_name}]}"
    log "  ${job_name}: lang_dir=${JOB_LANG_DIRS[${job_name}]}"
  done
}

main() {
  print_config

  while true; do
    local all_done=1

    for job_name in ${TARGET_JOBS}; do
      local exp_dir="${JOB_EXP_DIRS[${job_name}]}"

      for epoch in ${TARGET_EPOCHS}; do
        if job_epoch_done "${exp_dir}" "${epoch}"; then
          continue
        fi

        all_done=0

        if ! epoch_ready "${exp_dir}" "${epoch}"; then
          continue
        fi

        decode_epoch "${job_name}" "${epoch}" "greedy_search" || true
        decode_epoch "${job_name}" "${epoch}" "modified_beam_search" || true
      done
    done

    if [[ "${RUN_ONCE}" == "1" ]]; then
      break
    fi

    if [[ "${EXIT_WHEN_DONE}" == "1" && "${all_done}" == "1" ]]; then
      log "all target epochs are already decoded"
      break
    fi

    sleep "${POLL_SECONDS}"
  done
}

main "$@"
