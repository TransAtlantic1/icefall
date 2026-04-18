#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ICEFALL_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
RECIPE_DIR="${ICEFALL_ROOT}/egs/gigaspeech_24k/ASR"
VALIDATION_ROOT="$(cd -- "${ICEFALL_ROOT}/.." && pwd)/experiments/main_flow_validation/giga24k"

DATA_ROOT="${VALIDATION_ROOT}/workspace/data"
DOWNLOAD_ROOT="${VALIDATION_ROOT}/workspace/download"
SOURCE_MANIFEST_DIR="${RECIPE_DIR}/data/manifests"
SOURCE_LANG_DIR="${RECIPE_DIR}/data/lang_bpe_500"
SOURCE_GIGASPEECH_DIR="${RECIPE_DIR}/download/GigaSpeech"
TRAIN_RECORDINGS="${TRAIN_RECORDINGS:-24}"
DEV_RECORDINGS="${DEV_RECORDINGS:-4}"
TEST_RECORDINGS="${TEST_RECORDINGS:-4}"
RECORDING_NUM_SPLITS="${RECORDING_NUM_SPLITS:-4}"
RESAMPLE_NUM_WORKERS="${RESAMPLE_NUM_WORKERS:-1}"
FEATURE_NUM_WORKERS="${FEATURE_NUM_WORKERS:-0}"
FEATURE_BATCH_DURATION="${FEATURE_BATCH_DURATION:-80}"

mkdir -p "${DATA_ROOT}/manifests" "${DATA_ROOT}/fbank" "${DOWNLOAD_ROOT}"
rm -rf "${DATA_ROOT}/lang_bpe_500"
cp -a "${SOURCE_LANG_DIR}" "${DATA_ROOT}/lang_bpe_500"

SOURCE_MANIFEST_DIR="${SOURCE_MANIFEST_DIR}" \
SOURCE_GIGASPEECH_DIR="${SOURCE_GIGASPEECH_DIR}" \
DOWNLOAD_ROOT="${DOWNLOAD_ROOT}" \
TARGET_MANIFEST_DIR="${DATA_ROOT}/manifests" \
TRAIN_RECORDINGS="${TRAIN_RECORDINGS}" \
DEV_RECORDINGS="${DEV_RECORDINGS}" \
TEST_RECORDINGS="${TEST_RECORDINGS}" \
python - <<'PY'
import copy
import os
import shutil
from pathlib import Path

from lhotse import RecordingSet, SupervisionSet
from lhotse.serialization import load_manifest_lazy_or_eager

src = Path(os.environ["SOURCE_MANIFEST_DIR"])
source_audio_root = Path(os.environ["SOURCE_GIGASPEECH_DIR"])
target_audio_root = Path(os.environ["DOWNLOAD_ROOT"]) / "GigaSpeech"
dst = Path(os.environ["TARGET_MANIFEST_DIR"])
counts = {
    "M": int(os.environ["TRAIN_RECORDINGS"]),
    "DEV": int(os.environ["DEV_RECORDINGS"]),
    "TEST": int(os.environ["TEST_RECORDINGS"]),
}


def clone_recording_to_download_root(recording):
    cloned = copy.deepcopy(recording)
    for source in cloned.sources:
        if source.type != "file":
            continue
        source_path = Path(source.source)
        rel_path = source_path.relative_to(source_audio_root)
        target_path = target_audio_root / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if not target_path.exists():
            shutil.copy2(source_path, target_path)
        source.source = str(target_path)
    return cloned

for split, limit in counts.items():
    recordings = load_manifest_lazy_or_eager(src / f"gigaspeech_recordings_{split}.jsonl.gz")
    supervisions = load_manifest_lazy_or_eager(src / f"gigaspeech_supervisions_{split}.jsonl.gz")

    keep_ids = []
    with RecordingSet.open_writer(dst / f"gigaspeech_recordings_{split}.jsonl.gz") as writer:
        for idx, recording in enumerate(recordings):
            if idx >= limit:
                break
            cloned = clone_recording_to_download_root(recording)
            writer.write(cloned)
            keep_ids.append(cloned.id)

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

bash "${RECIPE_DIR}/prepare.sh" \
  --data-root "${DATA_ROOT}" \
  --download-dir "${DOWNLOAD_ROOT}" \
  --recording-num-splits "${RECORDING_NUM_SPLITS}" \
  --resample-num-workers "${RESAMPLE_NUM_WORKERS}" \
  --feature-num-workers "${FEATURE_NUM_WORKERS}" \
  --feature-batch-duration "${FEATURE_BATCH_DURATION}" \
  --feature-subsets DEV,TEST,M \
  --stage 3 \
  --stop-stage 6 \
  --cpu-only true

printf 'data_root=%s\n' "${DATA_ROOT}" >"${VALIDATION_ROOT}/validation_summary.txt"
echo "Prepared minimal 24k validation data at ${DATA_ROOT}"
