#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path

import lhotse
from lhotse import CutSet, SupervisionSet, load_manifest_lazy
from lhotse.serialization import load_manifest_lazy_or_eager

from split_utils import manifest_prefix, validate_language
from text_normalization import normalize_text


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
        raw_cuts_path = args.output_dir / f"{prefix}_cuts_{split}_raw.jsonl.gz"

        if normalized_supervisions_path.exists():
            normalized_supervisions_path.unlink()
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

        cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
        if split == "train" and args.speed_perturb:
            cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)

        cuts.to_file(raw_cuts_path)
        summary[split] = {
            "total_supervisions": total,
            "kept_supervisions": kept,
            "raw_cuts_path": str(raw_cuts_path),
            "recordings_manifest_dir": str(recordings_manifest_dir),
        }
        logging.info(
            "Prepared %s split: kept %s/%s supervisions",
            split,
            kept,
            total,
        )

    summary_path = args.output_dir / f"{prefix}_preprocess_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info("Wrote preprocess summary to %s", summary_path)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
