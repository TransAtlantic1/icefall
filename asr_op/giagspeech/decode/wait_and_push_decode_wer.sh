#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
summary_script="${repo_root}/summarize_decode_wer.py"

SENDKEY="${SENDKEY:-}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_PATH="${LOG_PATH:-/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_h200/decode_push_notify.log}"
MARKER_PATH="${MARKER_PATH:-/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_h200/.decode_push_sent}"

if [[ -z "${SENDKEY}" ]]; then
  echo "SENDKEY is required" >&2
  exit 1
fi

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
fi

if command -v conda >/dev/null 2>&1 && [[ "${CONDA_DEFAULT_ENV:-}" != "icefall" ]]; then
  conda activate icefall
fi

SUMMARY_EPOCHS="${SUMMARY_EPOCHS:-}"
SUMMARY_ARGS=()
if [[ -n "${SUMMARY_EPOCHS}" ]]; then
  SUMMARY_ARGS+=(--epochs)
  for epoch in ${SUMMARY_EPOCHS}; do
    SUMMARY_ARGS+=("${epoch}")
  done
fi

chmod +x "${summary_script}"

log() {
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" >> "${LOG_PATH}"
}

if [[ -f "${MARKER_PATH}" ]]; then
  log "marker exists, nothing to do"
  exit 0
fi

while true; do
  if python "${summary_script}" "${SUMMARY_ARGS[@]}" --check-complete >/dev/null; then
    summary="$(python "${summary_script}" "${SUMMARY_ARGS[@]}")"
    title="ASR decode WER done"
    desp=$'最终 WER 汇总如下：\n\n'"${summary}"

    response="$(
      curl -sS --fail \
        --data-urlencode "title=${title}" \
        --data-urlencode "desp=${desp}" \
        "https://sctapi.ftqq.com/${SENDKEY}.send"
    )"

    printf '%s\n' "${response}" > "${MARKER_PATH}"
    log "push sent successfully"
    exit 0
  fi

  log "WER not complete yet"
  sleep "${POLL_SECONDS}"
done
