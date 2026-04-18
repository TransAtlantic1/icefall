#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SCAN_SECONDS="${SCAN_SECONDS:-120}"
LOG_PATH="${LOG_PATH:-/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_h200/decode_progress_monitor.log}"

declare -A EXP_DIRS=(
  ["16k"]="/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_h200/16k_train_g0-3/zipformer_m_g0-1-2-3"
  ["24k"]="/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_h200/24k_train_g4-7/zipformer_m_g4-5-6-7"
)

log() {
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" >> "${LOG_PATH}"
}

latest_test_wer() {
  local exp_dir="$1"
  local method_dir="$2"
  local epoch="$3"
  local pattern="${exp_dir}/${method_dir}/wer-summary-test-*epoch-${epoch}-*.txt"
  local file
  file="$(compgen -G "${pattern}" | head -n 1 || true)"
  if [[ -z "${file}" ]]; then
    echo "-"
    return
  fi
  awk 'NR==2 {print $2}' "${file}" 2>/dev/null || echo "-"
}

while true; do
  for job in 16k 24k; do
    exp_dir="${EXP_DIRS[${job}]}"
    summary="job=${job}"
    for epoch in 10 20 30; do
      g="$(latest_test_wer "${exp_dir}" greedy_search "${epoch}")"
      m="$(latest_test_wer "${exp_dir}" modified_beam_search "${epoch}")"
      summary="${summary} epoch${epoch}[greedy=${g},mbs=${m}]"
    done
    log "${summary}"
  done
  sleep "${SCAN_SECONDS}"
done
