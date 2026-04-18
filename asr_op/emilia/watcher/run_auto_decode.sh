#!/usr/bin/env bash
#
# Watch an Emilia Zipformer training exp dir from another machine and decode
# newly saved iteration checkpoints periodically.
#
# The training recipe saves:
#   - checkpoint-${global_batch_idx}.pt every --save-every-n steps
#   - epoch-${epoch}.pt at the end of each epoch
#
# For Emilia full-data training, iteration checkpoints are the stable trigger
# because the run may only cover a few very large epochs.
#
# Example:
#   ./run_auto_decode.sh \
#     --exp-dir /path/to/zipformer/exp-zh-24k \
#     --artifact-root /path/to/icefall_emilia_zh_24k \
#     --decode-every-n 16000 \
#     --avg 1 \
#     --decode-cuda-visible-devices 0
#
# If train.py created a timestamped run subdir under --exp-dir, this watcher
# will automatically lock onto the newest run-* child directory.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ICEFALL_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PARSE_OPTIONS_SH="${ICEFALL_ROOT}/icefall/shared/parse_options.sh"

exp_dir=""
language=zh
artifact_root=""
manifest_dir=""
lang_dir=""
bpe_model=""
avg=3
beam_size=4
decode_every_n=5000
poll_seconds=120
decode_max_duration=1000
decode_num_workers=0
decode_cuda_visible_devices=""
use_averaged_model=true
decoding_methods="greedy_search,modified_beam_search"
train_done_marker=""
state_dir=""
log_path=""
once=false
dry_run=false
start_iter=0
auto_resolve_run_dir=true

. "${PARSE_OPTIONS_SH}" || exit 1

if [ -z "${exp_dir}" ]; then
  echo "$0: --exp-dir is required"
  exit 1
fi

if [ "${avg}" -lt 1 ]; then
  echo "$0: --avg must be >= 1"
  exit 1
fi

if [ "${beam_size}" -lt 1 ]; then
  echo "$0: --beam-size must be >= 1"
  exit 1
fi

if [ "${decode_every_n}" -lt 0 ]; then
  echo "$0: --decode-every-n must be >= 0"
  exit 1
fi

if [ "${poll_seconds}" -lt 1 ]; then
  echo "$0: --poll-seconds must be >= 1"
  exit 1
fi

if [ "${decode_num_workers}" -lt 0 ]; then
  echo "$0: --decode-num-workers must be >= 0"
  exit 1
fi

if [ "${decode_max_duration}" -le 0 ]; then
  echo "$0: --decode-max-duration must be > 0"
  exit 1
fi

if [ "${start_iter}" -lt 0 ]; then
  echo "$0: --start-iter must be >= 0"
  exit 1
fi

if [ -z "${state_dir}" ]; then
  state_hash="$(printf '%s' "${exp_dir}" | cksum | awk '{print $1}')"
  state_dir="/tmp/icefall-auto-decode/${language}-${state_hash}"
fi
mkdir -p "${state_dir}"

if [ -z "${log_path}" ]; then
  log_path="${state_dir}/watcher.log"
fi

lock_dir="${state_dir}/lock"
if ! mkdir "${lock_dir}" 2>/dev/null; then
  echo "$0: another watcher is already using state_dir=${state_dir}"
  exit 1
fi

cleanup() {
  rmdir "${lock_dir}" 2>/dev/null || true
}
trap cleanup EXIT

decoded_iters_file="${state_dir}/decoded_iters.txt"
resolved_run_dir_file="${state_dir}/resolved_run_dir.txt"
touch "${decoded_iters_file}"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*" >>"${log_path}"
}

trim_spaces() {
  printf '%s' "$1" | tr -d '[:space:]'
}

checkpoint_iter_from_path() {
  local checkpoint_path="$1"
  local checkpoint_name
  checkpoint_name="$(basename -- "${checkpoint_path}")"
  if [[ "${checkpoint_name}" =~ ^checkpoint-([0-9]+)\.pt$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

resolve_watch_dir() {
  local base_dir="$1"
  local latest_run_dir=""
  local candidate=""
  local base_name=""

  base_name="$(basename -- "${base_dir}")"

  if compgen -G "${base_dir}/checkpoint-*.pt" >/dev/null; then
    printf '%s\n' "${base_dir}"
    return 0
  fi

  if [ -d "${base_dir}" ] && [[ "${base_name}" == run-* ]]; then
    printf '%s\n' "${base_dir}"
    return 0
  fi

  if [ "${auto_resolve_run_dir}" = true ] && [ -d "${base_dir}" ]; then
    while IFS= read -r candidate; do
      latest_run_dir="${candidate}"
    done < <(find "${base_dir}" -maxdepth 1 -mindepth 1 -type d -name 'run-*' | sort)
  fi

  if [ -n "${latest_run_dir}" ]; then
    printf '%s\n' "${latest_run_dir}"
  else
    printf '%s\n' "${base_dir}"
  fi
}

iter_already_decoded() {
  local iter="$1"
  grep -Fxq "${iter}" "${decoded_iters_file}"
}

required_checkpoint_count="${avg}"
if [ "${use_averaged_model}" = true ]; then
  required_checkpoint_count=$((avg + 1))
fi

cd "${SCRIPT_DIR}"
export PYTHONPATH="${ICEFALL_ROOT}:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

raw_methods=()
IFS=',' read -r -a raw_methods <<<"${decoding_methods}"
methods=()
for raw_method in "${raw_methods[@]}"; do
  method="$(trim_spaces "${raw_method}")"
  if [ -n "${method}" ]; then
    methods+=("${method}")
  fi
done

if [ "${#methods[@]}" -eq 0 ]; then
  echo "$0: no decoding methods configured"
  exit 1
fi

run_decode_for_iter() {
  local actual_exp_dir="$1"
  local iter="$2"
  local method=""
  local method_log=""
  local -a cmd

  log "starting decode for iter=${iter} exp_dir=${actual_exp_dir}"

  for method in "${methods[@]}"; do
    method_log="${state_dir}/decode-iter-${iter}-${method}.log"
    cmd=(
      python zipformer/decode.py
      --language "${language}"
      --exp-dir "${actual_exp_dir}"
      --iter "${iter}"
      --avg "${avg}"
      --use-averaged-model "${use_averaged_model}"
      --max-duration "${decode_max_duration}"
      --num-workers "${decode_num_workers}"
      --decoding-method "${method}"
    )

    if [ -n "${artifact_root}" ]; then
      cmd+=(--artifact-root "${artifact_root}")
    fi
    if [ -n "${manifest_dir}" ]; then
      cmd+=(--manifest-dir "${manifest_dir}")
    fi
    if [ -n "${lang_dir}" ]; then
      cmd+=(--lang-dir "${lang_dir}")
    fi
    if [ -n "${bpe_model}" ]; then
      cmd+=(--bpe-model "${bpe_model}")
    fi
    if [ "${method}" = "modified_beam_search" ]; then
      cmd+=(--beam-size "${beam_size}")
    fi

    log "decode command: CUDA_VISIBLE_DEVICES=${decode_cuda_visible_devices:-<empty-for-cpu>} $(printf '%q ' "${cmd[@]}")"
    if [ "${dry_run}" = true ]; then
      continue
    fi

    if env CUDA_VISIBLE_DEVICES="${decode_cuda_visible_devices}" "${cmd[@]}" >"${method_log}" 2>&1; then
      log "finished decode for iter=${iter} method=${method} log=${method_log}"
    else
      log "decode failed for iter=${iter} method=${method} log=${method_log}"
      return 1
    fi
  done

  return 0
}

log "watcher started base_exp_dir=${exp_dir} state_dir=${state_dir}"
log "settings: avg=${avg} required_checkpoint_count=${required_checkpoint_count} decode_every_n=${decode_every_n} poll_seconds=${poll_seconds} start_iter=${start_iter} methods=${decoding_methods}"

resolved_exp_dir=""
idle_notice_printed=false

while true; do
  if [ -z "${resolved_exp_dir}" ]; then
    candidate_exp_dir="$(resolve_watch_dir "${exp_dir}")"
    candidate_name="$(basename -- "${candidate_exp_dir}")"
    if [[ "${candidate_name}" == run-* ]] || [ "${candidate_exp_dir}" != "${exp_dir}" ] || compgen -G "${candidate_exp_dir}/checkpoint-*.pt" >/dev/null; then
      resolved_exp_dir="${candidate_exp_dir}"
      printf '%s\n' "${resolved_exp_dir}" >"${resolved_run_dir_file}"
      log "resolved run dir: ${resolved_exp_dir}"
    fi
  fi

  if [ -z "${resolved_exp_dir}" ]; then
    if [ "${idle_notice_printed}" = false ]; then
      log "waiting for exp_dir to appear: ${exp_dir}"
      idle_notice_printed=true
    fi
    if [ "${once}" = true ]; then
      break
    fi
    sleep "${poll_seconds}"
    continue
  fi

  idle_notice_printed=false

  checkpoint_paths=()
  while IFS= read -r checkpoint_path; do
    checkpoint_paths+=("${checkpoint_path}")
  done < <(find "${resolved_exp_dir}" -maxdepth 1 -type f -name 'checkpoint-*.pt' | sort -V)

  pending_found=false
  checkpoint_count="${#checkpoint_paths[@]}"

  for ((i = 0; i < checkpoint_count; ++i)); do
    iter="$(checkpoint_iter_from_path "${checkpoint_paths[$i]}")"

    if [ "${iter}" -lt "${start_iter}" ]; then
      continue
    fi

    if [ "${decode_every_n}" -gt 0 ] && (( iter % decode_every_n != 0 )); then
      continue
    fi

    if iter_already_decoded "${iter}"; then
      continue
    fi

    if (( i + 1 < required_checkpoint_count )); then
      continue
    fi

    pending_found=true
    if run_decode_for_iter "${resolved_exp_dir}" "${iter}"; then
      if [ "${dry_run}" != true ]; then
        printf '%s\n' "${iter}" >>"${decoded_iters_file}"
      fi
    else
      log "will retry iter=${iter} on the next poll"
      break
    fi
  done

  if [ "${once}" = true ]; then
    break
  fi

  if [ -n "${train_done_marker}" ] && [ -e "${train_done_marker}" ] && [ "${pending_found}" = false ]; then
    log "train_done_marker detected and no pending decode remains; exiting"
    break
  fi

  sleep "${poll_seconds}"
done

log "watcher stopped"
