#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import lhotse
from lhotse import CutSet, SupervisionSet, load_manifest_lazy
from lhotse.serialization import load_manifest_lazy_or_eager

from split_utils import manifest_prefix, validate_language
from text_normalization import normalize_text


def trim_supervisions_to_recordings_sequentially(
    recordings,
    supervisions,
    output_path: Path,
) -> Dict[str, int]:
    """
    Streamingly align supervisions to recordings and trim any supervision whose
    end exceeds the corresponding recording duration.

    In this recipe, the recordings manifest order is preserved across stages,
    while transcript normalization may drop some supervisions. That means the
    supervisions are a subsequence of the recordings and can be aligned with a
    single forward pass without materializing a full recording-id index.
    """

    if output_path.exists():
        output_path.unlink()

    stats = {
        "trim_input_supervisions": 0,
        "trim_written_supervisions": 0,
        "trimmed_supervisions": 0,
        "removed_supervisions": 0,
        "skipped_recordings_without_supervision": 0,
    }

    recording_iter = iter(recordings)
    current_recording = next(recording_iter, None)
    previous_supervision_recording_id = None

    with SupervisionSet.open_writer(output_path) as writer:
        for sup in supervisions:
            stats["trim_input_supervisions"] += 1

            if previous_supervision_recording_id == sup.recording_id:
                raise ValueError(
                    "Expected at most one supervision per recording in Emilia "
                    f"stage 4, but saw multiple supervisions for recording_id="
                    f"{sup.recording_id}. "
                    "The sequential trimming logic relies on a 1:1 "
                    "recording-to-supervision mapping."
                )

            while (
                current_recording is not None
                and current_recording.id != sup.recording_id
            ):
                stats["skipped_recordings_without_supervision"] += 1
                current_recording = next(recording_iter, None)

            if current_recording is None:
                raise ValueError(
                    "Unable to align supervision "
                    f"{sup.id} with the recordings manifest. "
                    "This recipe expects normalized supervisions to remain "
                    "an ordered subsequence of recordings."
                )

            fixed_sup = sup
            if sup.start >= current_recording.duration:
                stats["removed_supervisions"] += 1
            else:
                if sup.end > current_recording.duration:
                    fixed_sup = sup.trim(end=current_recording.duration)
                    stats["trimmed_supervisions"] += 1

                if fixed_sup.duration <= 0:
                    stats["removed_supervisions"] += 1
                else:
                    writer.write(fixed_sup)
                    stats["trim_written_supervisions"] += 1

            previous_supervision_recording_id = sup.recording_id
            current_recording = next(recording_iter, None)

    return stats


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["zh", "en"],
        help="Subset language to preprocess.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Input directory containing supervision manifests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for normalized supervisions and raw cuts.",
    )
    parser.add_argument(
        "--recordings-manifest-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing recordings manifests to use when building raw cuts. "
            "If omitted, use --manifest-dir."
        ),
    )
    parser.add_argument(
        "--speed-perturb",
        action="store_true",
        help="Apply 0.9x/1.1x speed perturbation to the train split.",
    )
    return parser.parse_args()


def load_recordings(
    prefix: str, split: str, preferred_dir: Path, fallback_dir: Path
):
    candidate_dirs = [preferred_dir]
    if preferred_dir.resolve() != fallback_dir.resolve():
        candidate_dirs.append(fallback_dir)

    for manifest_dir in candidate_dirs:
        if split == "train":
            split_dirs = sorted(manifest_dir.glob("recordings_train_split_*"))
            for split_dir in split_dirs:
                pieces = sorted(
                    split_dir.glob(f"{prefix}_recordings_train.*.jsonl.gz")
                )
                if pieces:
                    logging.info(
                        "Loading %s train recording shards from %s",
                        len(pieces),
                        split_dir,
                    )
                    return lhotse.combine(load_manifest_lazy(p) for p in pieces)

        recordings_path = manifest_dir / f"{prefix}_recordings_{split}.jsonl.gz"
        if not recordings_path.is_file():
            continue
        recordings = load_manifest_lazy_or_eager(recordings_path)
        if recordings is not None:
            logging.info("Using %s recordings from %s", split, recordings_path)
            return recordings

    return None


def main():
    args = get_args()
    language = validate_language(args.language)
    prefix = manifest_prefix(language)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recordings_manifest_dir = args.recordings_manifest_dir or args.manifest_dir

    summary = {}
    for split in ("train", "dev", "test"):
        supervisions_path = args.manifest_dir / f"{prefix}_supervisions_{split}.jsonl.gz"
        normalized_supervisions_path = (
            args.output_dir / f"{prefix}_supervisions_{split}_norm.jsonl.gz"
        )
        fixed_supervisions_path = (
            args.output_dir / f"{prefix}_supervisions_{split}_norm_fixed.jsonl.gz"
        )
        raw_cuts_path = args.output_dir / f"{prefix}_cuts_{split}_raw.jsonl.gz"

        if normalized_supervisions_path.exists():
            normalized_supervisions_path.unlink()
        if fixed_supervisions_path.exists():
            fixed_supervisions_path.unlink()
        if raw_cuts_path.exists():
            raw_cuts_path.unlink()

        if not supervisions_path.exists():
            logging.warning(
                "Skipping %s split: %s does not exist", split, supervisions_path
            )
            summary[split] = {
                "total_supervisions": 0,
                "kept_supervisions": 0,
                "raw_cuts_path": str(raw_cuts_path),
            }
            continue

        total = 0
        kept = 0
        raw_sups = load_manifest_lazy_or_eager(supervisions_path)
        if raw_sups is None:
            logging.warning("Skipping %s split: manifest is empty", split)
            summary[split] = {
                "total_supervisions": 0,
                "kept_supervisions": 0,
                "raw_cuts_path": str(raw_cuts_path),
            }
            continue

        with SupervisionSet.open_writer(normalized_supervisions_path) as writer:
            for sup in raw_sups:
                total += 1
                normalized = normalize_text(sup.text, language)
                if not normalized:
                    continue
                sup.text = normalized
                writer.write(sup)
                kept += 1

        recordings = load_recordings(
            prefix=prefix,
            split=split,
            preferred_dir=recordings_manifest_dir,
            fallback_dir=args.manifest_dir,
        )
        supervisions = load_manifest_lazy_or_eager(normalized_supervisions_path)
        if recordings is None or supervisions is None:
            logging.warning(
                "Skipping %s split: recordings or supervisions empty after normalization",
                split,
            )
            summary[split] = {
                "total_supervisions": total,
                "kept_supervisions": kept,
                "raw_cuts_path": str(raw_cuts_path),
                "recordings_manifest_dir": str(recordings_manifest_dir),
            }
            continue

        trim_stats = trim_supervisions_to_recordings_sequentially(
            recordings=recordings,
            supervisions=supervisions,
            output_path=fixed_supervisions_path,
        )
        recordings = load_recordings(
            prefix=prefix,
            split=split,
            preferred_dir=recordings_manifest_dir,
            fallback_dir=args.manifest_dir,
        )
        supervisions = load_manifest_lazy_or_eager(fixed_supervisions_path)
        if recordings is None or supervisions is None:
            raise ValueError(
                f"Failed to reload fixed manifests for split {split}: "
                f"recordings={recordings is not None}, "
                f"supervisions={supervisions is not None}"
            )

        cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
        if split == "train" and args.speed_perturb:
            cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)

        cuts.to_file(raw_cuts_path)
        summary[split] = {
            "total_supervisions": total,
            "kept_supervisions": kept,
            "raw_cuts_path": str(raw_cuts_path),
            "recordings_manifest_dir": str(recordings_manifest_dir),
            "fixed_supervisions_path": str(fixed_supervisions_path),
            **trim_stats,
        }
        logging.info(
            "Prepared %s split: kept %s/%s supervisions, trimmed=%s removed=%s",
            split,
            kept,
            total,
            trim_stats["trimmed_supervisions"],
            trim_stats["removed_supervisions"],
        )

    summary_path = args.output_dir / f"{prefix}_preprocess_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info("Wrote preprocess summary to %s", summary_path)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
