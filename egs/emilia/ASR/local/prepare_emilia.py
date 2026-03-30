#!/usr/bin/env python3

import argparse
import json
import logging
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet

from split_utils import dataset_subdir_name, manifest_prefix, speaker_to_split, validate_language


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory of the Emilia dataset.",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["zh", "en"],
        help="Subset language to prepare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for split manifests.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.001,
        help="Speaker-disjoint development split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.001,
        help="Speaker-disjoint test split ratio.",
    )
    parser.add_argument(
        "--max-jsonl-files",
        type=int,
        default=-1,
        help="Optional cap on the number of top-level JSONL files to process.",
    )
    parser.add_argument(
        "--max-utterances",
        type=int,
        default=-1,
        help="Optional cap on the number of utterances to write.",
    )
    return parser.parse_args()


def unlink_if_exists(path: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()


def main():
    args = get_args()
    language = validate_language(args.language)
    prefix = manifest_prefix(language)
    language_dir = args.dataset_root / dataset_subdir_name(language)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(language_dir.glob("*.jsonl"))
    if args.max_jsonl_files > 0:
        jsonl_files = jsonl_files[: args.max_jsonl_files]

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {language_dir}")

    split_paths = {
        split: {
            "recordings": args.output_dir / f"{prefix}_recordings_{split}.jsonl.gz",
            "supervisions": args.output_dir / f"{prefix}_supervisions_{split}.jsonl.gz",
        }
        for split in ("train", "dev", "test")
    }
    for paths in split_paths.values():
        unlink_if_exists(paths["recordings"])
        unlink_if_exists(paths["supervisions"])

    stats = Counter()
    split_counts = Counter()

    with ExitStack() as stack:
        recording_writers = {
            split: stack.enter_context(RecordingSet.open_writer(paths["recordings"]))
            for split, paths in split_paths.items()
        }
        supervision_writers = {
            split: stack.enter_context(SupervisionSet.open_writer(paths["supervisions"]))
            for split, paths in split_paths.items()
        }

        stop = False
        for jsonl_path in jsonl_files:
            logging.info("Processing %s", jsonl_path)
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if args.max_utterances > 0 and stats["written"] >= args.max_utterances:
                        stop = True
                        break

                    stats["seen"] += 1
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        stats["json_error"] += 1
                        continue

                    entry_language = str(entry.get("language", language)).lower()
                    if entry_language != language:
                        stats["language_mismatch"] += 1
                        continue

                    utterance_id = str(entry["id"])
                    speaker = str(entry.get("speaker") or utterance_id)
                    audio_rel_path = Path(entry["wav"])
                    audio_path = language_dir / audio_rel_path
                    if not audio_path.is_file():
                        stats["missing_audio"] += 1
                        continue

                    duration = float(entry.get("duration", 0.0) or 0.0)
                    if duration <= 0:
                        stats["invalid_duration"] += 1
                        continue

                    try:
                        recording = Recording.from_file(audio_path, recording_id=utterance_id)
                    except Exception as ex:  # noqa: BLE001
                        logging.warning("Skip unreadable audio %s: %s", audio_path, ex)
                        stats["audio_error"] += 1
                        continue

                    raw_text = str(entry.get("text", ""))
                    split = speaker_to_split(
                        speaker=speaker,
                        dev_ratio=args.dev_ratio,
                        test_ratio=args.test_ratio,
                    )
                    supervision = SupervisionSegment(
                        id=utterance_id,
                        recording_id=recording.id,
                        start=0.0,
                        duration=recording.duration,
                        channel=0,
                        text=raw_text,
                        language=language,
                        speaker=speaker,
                        custom={
                            "raw_text": raw_text,
                            "dnsmos": entry.get("dnsmos"),
                            "source_jsonl": jsonl_path.name,
                        },
                    )

                    recording_writers[split].write(recording)
                    supervision_writers[split].write(supervision)
                    split_counts[split] += 1
                    stats["written"] += 1

            if stop:
                break

    summary_path = args.output_dir / f"{prefix}_summary.json"
    summary = {
        "language": language,
        "dataset_root": str(args.dataset_root),
        "processed_jsonl_files": [p.name for p in jsonl_files],
        "stats": dict(stats),
        "split_counts": dict(split_counts),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info("Finished preparing Emilia %s manifests", language)
    logging.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()

