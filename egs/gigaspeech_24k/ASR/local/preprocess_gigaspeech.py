#!/usr/bin/env python3
# Copyright    2021  Johns Hopkins University (Piotr Żelasko)
# Copyright    2021  Xiaomi Corp.             (Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import lhotse
from lhotse import CutSet, RecordingSet, SupervisionSegment
from lhotse.recipes.utils import read_manifests_if_cached
from lhotse.serialization import load_manifest_lazy_or_eager

# Similar text filtering and normalization procedure as in:
# https://github.com/SpeechColab/GigaSpeech/blob/main/toolkits/kaldi/gigaspeech_data_prep.sh


def normalize_text(
    utt: str,
    punct_pattern=re.compile(r"<(COMMA|PERIOD|QUESTIONMARK|EXCLAMATIONPOINT)>"),
    whitespace_pattern=re.compile(r"\s\s+"),
) -> str:
    return whitespace_pattern.sub(" ", punct_pattern.sub("", utt))


def has_no_oov(
    sup: SupervisionSegment,
    oov_pattern=re.compile(r"<(SIL|MUSIC|NOISE|OTHER)>"),
) -> bool:
    return oov_pattern.search(sup.text) is None


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Input directory containing the original GigaSpeech manifests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/fbank"),
        help="Output directory for raw cut manifests.",
    )
    parser.add_argument(
        "--recordings-manifest-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing recording manifests used to build the raw cuts. "
            "If omitted, use --manifest-dir."
        ),
    )
    parser.add_argument(
        "--cpu-only",
        type=str2bool,
        default=False,
        help="Skip importing the full icefall package chain. Use this on CPU-only data prep instances.",
    )
    return parser.parse_args()


def load_recordings(
    partition: str, preferred_dir: Path, fallback_dir: Path
) -> Optional[RecordingSet]:
    candidate_dirs = [preferred_dir]
    if preferred_dir.resolve() != fallback_dir.resolve():
        candidate_dirs.append(fallback_dir)

    for manifest_dir in candidate_dirs:
        if partition == "M":
            split_dirs = sorted(manifest_dir.glob("recordings_M_split_*"))
            for split_dir in split_dirs:
                pieces = sorted(split_dir.glob("gigaspeech_recordings_M.*.jsonl.gz"))
                if pieces:
                    logging.info(
                        "Loading %s M recording shards from %s", len(pieces), split_dir
                    )
                    return lhotse.combine(
                        load_manifest_lazy_or_eager(piece) for piece in pieces
                    )

        recordings_path = manifest_dir / f"gigaspeech_recordings_{partition}.jsonl.gz"
        recordings = load_manifest_lazy_or_eager(recordings_path)
        if recordings is not None:
            logging.info("Using %s recordings from %s", partition, recordings_path)
            return recordings

    return None


def preprocess_gigaspeech(
    manifest_dir: Path, output_dir: Path, recordings_manifest_dir: Optional[Path] = None
):
    output_dir.mkdir(parents=True, exist_ok=True)
    recordings_dir = recordings_manifest_dir or manifest_dir

    dataset_parts = ("DEV", "TEST", "M")

    logging.info("Loading manifest (may take 4 minutes)")
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=manifest_dir,
        prefix="gigaspeech",
        suffix="jsonl.gz",
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    for partition, manifests_for_partition in manifests.items():
        logging.info("Processing %s", partition)
        raw_cuts_path = output_dir / f"gigaspeech_cuts_{partition}_raw.jsonl.gz"
        if raw_cuts_path.exists():
            raw_cuts_path.unlink()

        logging.info("Filtering OOV utterances from supervisions")
        supervisions = manifests_for_partition["supervisions"].filter(has_no_oov)
        logging.info("Normalizing text in %s", partition)
        for supervision in supervisions:
            supervision.text = normalize_text(supervision.text)

        recordings = load_recordings(
            partition=partition,
            preferred_dir=recordings_dir,
            fallback_dir=manifest_dir,
        )
        if recordings is None:
            raise ValueError(
                f"Unable to find recordings for {partition} in {recordings_dir} or {manifest_dir}"
            )

        cut_set = CutSet.from_manifests(
            recordings=recordings,
            supervisions=supervisions,
        )

        logging.info("Saving to %s", raw_cuts_path)
        cut_set.to_file(raw_cuts_path)


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    if not args.cpu_only:
        from icefall.utils import str2bool as _icefall_str2bool  # noqa: F401

    preprocess_gigaspeech(
        manifest_dir=args.manifest_dir,
        output_dir=args.output_dir,
        recordings_manifest_dir=args.recordings_manifest_dir,
    )


if __name__ == "__main__":
    main()
