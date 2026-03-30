#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path

from lhotse import CutSet, load_manifest_lazy
from lhotse.serialization import load_manifest_lazy_or_eager
from lhotse import SupervisionSet

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
        help="Input directory containing split manifests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for normalized supervisions and raw cuts.",
    )
    parser.add_argument(
        "--speed-perturb",
        action="store_true",
        help="Apply 0.9x/1.1x speed perturbation to the train split.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    language = validate_language(args.language)
    prefix = manifest_prefix(language)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in ("train", "dev", "test"):
        recordings_path = args.manifest_dir / f"{prefix}_recordings_{split}.jsonl.gz"
        supervisions_path = args.manifest_dir / f"{prefix}_supervisions_{split}.jsonl.gz"
        normalized_supervisions_path = (
            args.output_dir / f"{prefix}_supervisions_{split}_norm.jsonl.gz"
        )
        raw_cuts_path = args.output_dir / f"{prefix}_cuts_{split}_raw.jsonl.gz"

        if normalized_supervisions_path.exists():
            normalized_supervisions_path.unlink()
        if raw_cuts_path.exists():
            raw_cuts_path.unlink()

        total = 0
        kept = 0
        with SupervisionSet.open_writer(normalized_supervisions_path) as writer:
            for sup in load_manifest_lazy_or_eager(supervisions_path):
                total += 1
                normalized = normalize_text(sup.text, language)
                if not normalized:
                    continue
                sup.text = normalized
                writer.write(sup)
                kept += 1

        recordings = load_manifest_lazy_or_eager(recordings_path)
        supervisions = load_manifest_lazy_or_eager(normalized_supervisions_path)
        cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
        if split == "train" and args.speed_perturb:
            cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)

        cuts.to_file(raw_cuts_path)
        summary[split] = {
            "total_supervisions": total,
            "kept_supervisions": kept,
            "raw_cuts_path": str(raw_cuts_path),
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

