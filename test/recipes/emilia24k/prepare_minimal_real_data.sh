#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ICEFALL_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
RECIPE_DIR="${ICEFALL_ROOT}/egs/emilia_24k_multilang/emilia_24k_ZH/ASR"
VALIDATION_ROOT="$(cd -- "${ICEFALL_ROOT}/.." && pwd)/experiments/main_flow_validation/emilia24k"

ARTIFACT_ROOT="${VALIDATION_ROOT}/workspace/artifacts"
DATASET_ROOT="${DATASET_ROOT:-/inspire/dataset/emilia/fc71e07}"
LANGUAGE="${LANGUAGE:-zh}"
MAX_JSONL_FILES="${MAX_JSONL_FILES:-4}"
MAX_UTTERANCES="${MAX_UTTERANCES:-256}"
DEV_RATIO="${DEV_RATIO:-0.2}"
TEST_RATIO="${TEST_RATIO:-0.2}"
RECORDING_NUM_SPLITS="${RECORDING_NUM_SPLITS:-4}"
FEATURE_NUM_SPLITS="${FEATURE_NUM_SPLITS:-4}"
FEATURE_NUM_WORKERS="${FEATURE_NUM_WORKERS:-0}"
FEATURE_BATCH_DURATION="${FEATURE_BATCH_DURATION:-80}"

mkdir -p "${ARTIFACT_ROOT}"

bash "${RECIPE_DIR}/prepare.sh" \
  --language "${LANGUAGE}" \
  --dataset-root "${DATASET_ROOT}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --dev-ratio "${DEV_RATIO}" \
  --test-ratio "${TEST_RATIO}" \
  --max-jsonl-files "${MAX_JSONL_FILES}" \
  --max-utterances "${MAX_UTTERANCES}" \
  --recording-num-splits "${RECORDING_NUM_SPLITS}" \
  --feature-num-splits "${FEATURE_NUM_SPLITS}" \
  --feature-num-workers "${FEATURE_NUM_WORKERS}" \
  --feature-batch-duration "${FEATURE_BATCH_DURATION}" \
  --feature-device cpu \
  --stage 0 \
  --stop-stage 10

printf '{\n  "status": "prepared",\n  "artifact_root": "%s"\n}\n' "${ARTIFACT_ROOT}" \
  >"${VALIDATION_ROOT}/validation_summary.json"
echo "Prepared minimal Emilia validation data at ${ARTIFACT_ROOT}"
