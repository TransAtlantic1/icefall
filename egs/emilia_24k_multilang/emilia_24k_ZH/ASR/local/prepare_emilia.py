#!/usr/bin/env python3

import argparse
import json
import logging
import multiprocessing as mp
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Tuple

from lhotse import AudioSource, Recording, RecordingSet, SupervisionSegment, SupervisionSet

from split_utils import dataset_subdir_name, manifest_prefix, speaker_to_split, validate_language

# Emilia mp3 固定参数，避免每条都 open 音频文件读取头信息
EMILIA_SAMPLING_RATE = 32000
EMILIA_NUM_CHANNELS = 1


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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel worker processes for JSONL parsing.",
    )
    return parser.parse_args()


def unlink_if_exists(path: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()


def process_jsonl_file(args_tuple):
    """
    单个 JSONL 文件的处理函数，在 worker 进程中运行。
    核心优化：直接用 JSONL 的 duration/wav 字段构建 Recording，
    不调用 Recording.from_file()，跳过所有音频文件 IO。
    返回 (recordings_dict, supervisions_dict, stats, split_counts)
    """
    (
        jsonl_path,
        language_dir,
        language,
        dev_ratio,
        test_ratio,
        max_utterances,
    ) = args_tuple

    recordings: Dict[str, List[Recording]] = {"train": [], "dev": [], "test": []}
    supervisions: Dict[str, List[SupervisionSegment]] = {"train": [], "dev": [], "test": []}
    stats = Counter()
    split_counts = Counter()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_utterances > 0 and stats["written"] >= max_utterances:
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
            duration = float(entry.get("duration", 0.0) or 0.0)

            if duration <= 0:
                stats["invalid_duration"] += 1
                continue

            audio_rel_path = entry.get("wav", "")
            if not audio_rel_path:
                stats["missing_wav_field"] += 1
                continue

            audio_path = str(language_dir / audio_rel_path)

            # 直接构建 Recording，不打开音频文件，不做 is_file() 检查
            # 缺失音频会在 stage 4 fbank 提取时被发现并跳过
            recording = Recording(
                id=utterance_id,
                sources=[AudioSource(type="file", channels=[0], source=audio_path)],
                sampling_rate=EMILIA_SAMPLING_RATE,
                num_samples=int(duration * EMILIA_SAMPLING_RATE),
                duration=duration,
            )

            raw_text = str(entry.get("text", ""))
            split = speaker_to_split(
                speaker=speaker,
                dev_ratio=dev_ratio,
                test_ratio=test_ratio,
            )
            supervision = SupervisionSegment(
                id=utterance_id,
                recording_id=recording.id,
                start=0.0,
                duration=duration,
                channel=0,
                text=raw_text,
                language=language,
                speaker=speaker,
                custom={
                    "raw_text": raw_text,
                    "dnsmos": entry.get("dnsmos"),
                    "source_jsonl": Path(jsonl_path).name,
                },
            )

            recordings[split].append(recording)
            supervisions[split].append(supervision)
            split_counts[split] += 1
            stats["written"] += 1

    logging.info("Done %s: written=%d", Path(jsonl_path).name, stats["written"])
    return recordings, supervisions, stats, split_counts


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

    logging.info(
        "Found %d JSONL files, using %d worker processes", len(jsonl_files), args.num_workers
    )

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

    total_stats = Counter()
    total_split_counts = Counter()

    worker_args = [
        (
            str(jsonl_path),
            language_dir,
            language,
            args.dev_ratio,
            args.test_ratio,
            args.max_utterances,
        )
        for jsonl_path in jsonl_files
    ]

    num_workers = min(args.num_workers, len(jsonl_files))

    with ExitStack() as stack:
        recording_writers = {
            split: stack.enter_context(RecordingSet.open_writer(paths["recordings"]))
            for split, paths in split_paths.items()
        }
        supervision_writers = {
            split: stack.enter_context(SupervisionSet.open_writer(paths["supervisions"]))
            for split, paths in split_paths.items()
        }

        with mp.Pool(processes=num_workers) as pool:
            for i, (recordings, supervisions, stats, split_counts) in enumerate(
                pool.imap_unordered(process_jsonl_file, worker_args)
            ):
                total_stats += stats
                total_split_counts += split_counts

                for split in ("train", "dev", "test"):
                    for rec in recordings[split]:
                        recording_writers[split].write(rec)
                    for sup in supervisions[split]:
                        supervision_writers[split].write(sup)

                if (i + 1) % 10 == 0 or (i + 1) == len(jsonl_files):
                    logging.info(
                        "Progress: %d/%d files done, total written=%d",
                        i + 1,
                        len(jsonl_files),
                        total_stats["written"],
                    )

                # 全局 max_utterances 近似控制（以文件为粒度）
                if args.max_utterances > 0 and total_stats["written"] >= args.max_utterances:
                    pool.terminate()
                    break

    summary_path = args.output_dir / f"{prefix}_summary.json"
    summary = {
        "language": language,
        "dataset_root": str(args.dataset_root),
        "processed_jsonl_files": [p.name for p in jsonl_files],
        "stats": dict(total_stats),
        "split_counts": dict(total_split_counts),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info("Finished preparing Emilia %s manifests", language)
    logging.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
