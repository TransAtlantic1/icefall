#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ICEFALL_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
RECIPE_DIR="${ICEFALL_ROOT}/egs/gigaspeech_16k/ASR"
VALIDATION_ROOT="$(cd -- "${ICEFALL_ROOT}/.." && pwd)/experiments/main_flow_validation/giga16k"

DATA_ROOT="${VALIDATION_ROOT}/workspace/data"
SOURCE_MANIFEST_DIR="${RECIPE_DIR}/data/manifests"
SOURCE_LANG_DIR="${RECIPE_DIR}/data/lang_bpe_500"
TRAIN_RECORDINGS="${TRAIN_RECORDINGS:-32}"
DEV_RECORDINGS="${DEV_RECORDINGS:-4}"
TEST_RECORDINGS="${TEST_RECORDINGS:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_DURATION="${BATCH_DURATION:-80}"

mkdir -p "${DATA_ROOT}/manifests" "${DATA_ROOT}/fbank"
rm -rf "${DATA_ROOT}/lang_bpe_500"
cp -a "${SOURCE_LANG_DIR}" "${DATA_ROOT}/lang_bpe_500"

SOURCE_MANIFEST_DIR="${SOURCE_MANIFEST_DIR}" \
TARGET_MANIFEST_DIR="${DATA_ROOT}/manifests" \
TRAIN_RECORDINGS="${TRAIN_RECORDINGS}" \
DEV_RECORDINGS="${DEV_RECORDINGS}" \
TEST_RECORDINGS="${TEST_RECORDINGS}" \
python - <<'PY'
import os
from pathlib import Path

from lhotse import RecordingSet, SupervisionSet
from lhotse.serialization import load_manifest_lazy_or_eager

src = Path(os.environ["SOURCE_MANIFEST_DIR"])
dst = Path(os.environ["TARGET_MANIFEST_DIR"])
counts = {
    "M": int(os.environ["TRAIN_RECORDINGS"]),
    "DEV": int(os.environ["DEV_RECORDINGS"]),
    "TEST": int(os.environ["TEST_RECORDINGS"]),
}

for split, limit in counts.items():
    recordings = load_manifest_lazy_or_eager(src / f"gigaspeech_recordings_{split}.jsonl.gz")
    supervisions = load_manifest_lazy_or_eager(src / f"gigaspeech_supervisions_{split}.jsonl.gz")

    keep_ids = []
    with RecordingSet.open_writer(dst / f"gigaspeech_recordings_{split}.jsonl.gz") as writer:
        for idx, recording in enumerate(recordings):
            if idx >= limit:
                break
            writer.write(recording)
            keep_ids.append(recording.id)

    keep_ids = set(keep_ids)
    kept = 0
    with SupervisionSet.open_writer(dst / f"gigaspeech_supervisions_{split}.jsonl.gz") as writer:
        for supervision in supervisions:
            if supervision.recording_id in keep_ids:
                writer.write(supervision)
                kept += 1

    if not keep_ids or kept == 0:
        raise SystemExit(f"Failed to build minimal subset for {split}")
PY

python "${RECIPE_DIR}/local/preprocess_gigaspeech.py" \
  --manifest-dir "${DATA_ROOT}/manifests" \
  --output-dir "${DATA_ROOT}/fbank" \
  --cpu-only true

python "${RECIPE_DIR}/local/compute_fbank_gigaspeech.py" \
  --fbank-dir "${DATA_ROOT}/fbank" \
  --num-workers "${NUM_WORKERS}" \
  --batch-duration "${BATCH_DURATION}"

printf 'data_root=%s\n' "${DATA_ROOT}" >"${VALIDATION_ROOT}/validation_summary.txt"
echo "Prepared minimal 16k validation data at ${DATA_ROOT}"
