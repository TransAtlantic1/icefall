#!/usr/bin/env bash

set -euo pipefail

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
fi

if command -v conda >/dev/null 2>&1 && [[ "${CONDA_DEFAULT_ENV:-}" != "icefall" ]]; then
  conda activate icefall
fi

SYNC_INTERVAL_SEC="${SYNC_INTERVAL_SEC:-600}"
WANDB_ENTITY="${WANDB_ENTITY:-maine-004-sjtu-hpc-center}"

RUN16_ID="${RUN16_ID:-rkzkclvq}"
RUN24_ID="${RUN24_ID:-s4rx55ce}"

DIR16="${DIR16:-/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_h200/16k_train_g0-3/wandb_offline/wandb/offline-run-20260408_020736-glt62m94}"
DIR24="${DIR24:-/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_h200/24k_train_g4-7/wandb_offline/wandb/offline-run-20260408_022222-sy372le8}"

log() {
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

sync_one() {
  local project="$1"
  local run_id="$2"
  local run_dir="$3"

  if [[ ! -d "${run_dir}" ]]; then
    log "skip ${project}: missing dir ${run_dir}"
    return 0
  fi

  log "sync start ${project} -> ${run_id}"
  if wandb sync \
    --include-offline \
    --include-synced \
    --append \
    --entity "${WANDB_ENTITY}" \
    --project "${project}" \
    --id "${run_id}" \
    "${run_dir}"; then
    log "sync done ${project} -> ${run_id}"
  else
    log "sync failed ${project} -> ${run_id}"
  fi
}

while true; do
  sync_one "gigaspeech-16k" "${RUN16_ID}" "${DIR16}"
  sync_one "gigaspeech-24k" "${RUN24_ID}" "${DIR24}"
  log "sleep ${SYNC_INTERVAL_SEC}s"
  sleep "${SYNC_INTERVAL_SEC}"
done
