#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_SELF="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
ICEFALL_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PARSE_OPTIONS_SH="${ICEFALL_ROOT}/icefall/shared/parse_options.sh"

language=zh
public_root=/inspire/qb-ilm/project/embodied-multimodality/chenxie-25019/public
dataset_root=""
artifact_root=""
feature_num_workers=20
feature_batch_duration=1000
feature_device=cpu
log_root=""
detach=false
detach_log=""

. "${PARSE_OPTIONS_SH}" || exit 1

if [[ "$language" != "zh" && "$language" != "en" ]]; then
  echo "$0: --language must be one of zh or en, got: $language"
  exit 1
fi

if [ -z "$dataset_root" ]; then
  dataset_root="/inspire/dataset/emilia/fc71e07"
fi

if [ -z "$artifact_root" ]; then
  artifact_root="${public_root%/}/emilia/fc71e07/icefall_emilia_${language}_24k"
fi

if [ -z "$log_root" ]; then
  log_root="${artifact_root}/logs"
fi

mkdir -p "$log_root"

log_file="${log_root}/feature.cpu.stage5.${language}.log"
if [ -z "$detach_log" ]; then
  detach_log="${log_root}/launcher.feature.cpu.stage5.${language}.nohup.log"
fi

if [ "$detach" = true ]; then
  cmd=(
    "${SCRIPT_SELF}"
    --language "$language"
    --public-root "$public_root"
    --dataset-root "$dataset_root"
    --artifact-root "$artifact_root"
    --feature-num-workers "$feature_num_workers"
    --feature-batch-duration "$feature_batch_duration"
    --feature-device "$feature_device"
    --log-root "$log_root"
    --detach false
    --detach-log "$detach_log"
  )
  nohup "${cmd[@]}" >>"$detach_log" 2>&1 &
  pid=$!
  echo "$0: detached pid=${pid}"
  echo "$0: launcher_log=${detach_log}"
  echo "$0: worker_log=${log_file}"
  exit 0
fi

echo "$0: language=${language}"
echo "$0: dataset_root=${dataset_root}"
echo "$0: artifact_root=${artifact_root}"
echo "$0: feature_device=${feature_device}"
echo "$0: detach_log=${detach_log}"
echo "$0: log_file=${log_file}"

exec env CUDA_VISIBLE_DEVICES="" bash "${SCRIPT_DIR}/prepare.sh" \
  --language "$language" \
  --dataset-root "$dataset_root" \
  --artifact-root "$artifact_root" \
  --feature-device "$feature_device" \
  --feature-num-workers "$feature_num_workers" \
  --feature-batch-duration "$feature_batch_duration" \
  --stage 5 \
  --stop-stage 5 \
  >>"$log_file" 2>&1
