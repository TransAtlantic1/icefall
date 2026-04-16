#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence


DEFAULT_DATASET_ROOT = Path("/inspire/dataset/emilia/fc71e07")
REPORT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_JSON = REPORT_ROOT / "emilia_24k_non_zh_original_sample_rate_scan.json"
DEFAULT_OUTPUT_MD = REPORT_ROOT / "emilia_24k_non_zh_original_sample_rate_report.md"
DEFAULT_EXCLUDE_LANGUAGES = {"ZH"}
MP3_SAMPLE_RATES = {
    3: (44100, 48000, 32000),
    2: (22050, 24000, 16000),
    0: (11025, 12000, 8000),
}


@dataclass
class ScanConfig:
    dataset_root: Path
    languages: Sequence[str]
    workers: int
    chunk_size: int
    initial_read_bytes: int
    fallback_read_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan original Emilia MP3 sample rates and write JSON/Markdown reports."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Dataset root. Default: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Languages to scan. Default: all dataset language directories except ZH.",
    )
    parser.add_argument(
        "--exclude-languages",
        nargs="*",
        default=sorted(DEFAULT_EXCLUDE_LANGUAGES),
        help="Languages to exclude from the discovered set.",
    )
    parser.add_argument("--workers", type=int, default=256, help="Thread pool size.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Number of audio paths to scan per executor.map() chunk.",
    )
    parser.add_argument(
        "--initial-read-bytes",
        type=int,
        default=4096,
        help="Bytes to read after seeking near the first MP3 frame.",
    )
    parser.add_argument(
        "--fallback-read-bytes",
        type=int,
        default=16384,
        help="Fallback bytes to read if the first frame is not found in the initial window.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Where to write the raw JSON scan result. Default: {DEFAULT_OUTPUT_JSON}",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help=f"Where to write the Markdown report. Default: {DEFAULT_OUTPUT_MD}",
    )
    return parser.parse_args()


def discover_languages(dataset_root: Path, exclude_languages: Sequence[str]) -> List[str]:
    exclude = {lang.upper() for lang in exclude_languages}
    languages = [
        path.name
        for path in sorted(dataset_root.iterdir())
        if path.is_dir()
        and len(path.name) == 2
        and path.name.isalpha()
        and path.name.upper() == path.name
        and path.name.upper() not in exclude
    ]
    if not languages:
        raise ValueError(f"No language directories found under {dataset_root}")
    return languages


def synchsafe_to_int(buf: bytes) -> int:
    return (
        ((buf[0] & 0x7F) << 21)
        | ((buf[1] & 0x7F) << 14)
        | ((buf[2] & 0x7F) << 7)
        | (buf[3] & 0x7F)
    )


def parse_mp3_sample_rate(data: bytes) -> Optional[int]:
    index = 0
    limit = len(data)
    while index + 4 <= limit:
        sync = data.find(b"\xff", index)
        if sync < 0 or sync + 4 > limit:
            return None
        if (data[sync + 1] & 0xE0) != 0xE0:
            index = sync + 1
            continue
        header = int.from_bytes(data[sync : sync + 4], "big")
        version = (header >> 19) & 0b11
        layer = (header >> 17) & 0b11
        sample_rate_index = (header >> 10) & 0b11
        if version != 1 and layer != 0 and sample_rate_index != 3:
            return MP3_SAMPLE_RATES[version][sample_rate_index]
        index = sync + 1
    return None


def get_original_sample_rate(
    path: Path, initial_read_bytes: int, fallback_read_bytes: int
) -> Optional[int]:
    with path.open("rb") as f:
        head = f.read(10)
        if len(head) < 10:
            return None
        if head[:3] == b"ID3":
            offset = 10 + synchsafe_to_int(head[6:10])
            f.seek(offset)
        else:
            f.seek(0)

        first_window = f.read(initial_read_bytes)
        sample_rate = parse_mp3_sample_rate(first_window)
        if sample_rate is not None:
            return sample_rate

        if fallback_read_bytes <= initial_read_bytes:
            return None

        remaining = fallback_read_bytes - len(first_window)
        if remaining <= 0:
            return None
        sample_rate = parse_mp3_sample_rate(first_window + f.read(remaining))
        return sample_rate


def iter_batch_jsonls(language_root: Path) -> Iterator[Path]:
    yield from sorted(language_root.glob(f"{language_root.name}_B*.jsonl"))


def iter_audio_paths(jsonl_path: Path, language_root: Path) -> Iterator[Path]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            yield language_root / entry["wav"]


def chunked(items: Iterable[Path], chunk_size: int) -> Iterator[List[Path]]:
    chunk: List[Path] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def format_int(value: int) -> str:
    return f"{value:,}"


def format_rate_counts(rate_counts: Dict[str, int]) -> str:
    if not rate_counts:
        return "-"
    parts = [f"{rate} Hz: {format_int(rate_counts[rate])}" for rate in sorted(rate_counts, key=int)]
    return ", ".join(parts)


def status_for_batch(rate_counts: Dict[str, int], bad_files: int) -> str:
    nonzero_rates = [rate for rate, count in rate_counts.items() if count > 0]
    if bad_files:
        return "bad-files"
    if len(nonzero_rates) <= 1:
        return "uniform"
    return "mixed"


def scan_language(config: ScanConfig, language: str) -> Dict[str, object]:
    language_root = config.dataset_root / language
    jsonls = list(iter_batch_jsonls(language_root))
    if not jsonls:
        raise ValueError(f"No batch jsonl files found for {language} under {language_root}")

    sample_rate_examples: Dict[str, str] = {}
    bad_examples: List[str] = []
    overall_counts: Counter[int] = Counter()
    batch_results: List[Dict[str, object]] = []
    files_done = 0
    started = time.time()

    print(
        json.dumps(
            {
                "event": "language_start",
                "language": language,
                "batches": len(jsonls),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        sample_rate_getter = partial(
            get_original_sample_rate,
            initial_read_bytes=config.initial_read_bytes,
            fallback_read_bytes=config.fallback_read_bytes,
        )
        for batch_index, jsonl_path in enumerate(jsonls, start=1):
            batch_name = jsonl_path.stem
            batch_counts: Counter[int] = Counter()
            batch_bad_files = 0
            batch_started = time.time()

            for paths in chunked(iter_audio_paths(jsonl_path, language_root), config.chunk_size):
                sample_rates = executor.map(sample_rate_getter, paths, chunksize=256)
                for path, sample_rate in zip(paths, sample_rates):
                    files_done += 1
                    if sample_rate is None:
                        batch_bad_files += 1
                        if len(bad_examples) < 20:
                            bad_examples.append(str(path))
                        continue
                    batch_counts[sample_rate] += 1
                    sample_rate_examples.setdefault(str(sample_rate), str(path))

            overall_counts.update(batch_counts)
            batch_elapsed = time.time() - batch_started
            batch_total_files = sum(batch_counts.values()) + batch_bad_files
            batch_results.append(
                {
                    "batch": batch_name,
                    "files": batch_total_files,
                    "sample_rate_counts": {
                        str(rate): count for rate, count in sorted(batch_counts.items())
                    },
                    "bad_files": batch_bad_files,
                    "status": status_for_batch(
                        {str(rate): count for rate, count in batch_counts.items()},
                        batch_bad_files,
                    ),
                    "elapsed_sec": round(batch_elapsed, 3),
                }
            )
            elapsed = time.time() - started
            print(
                json.dumps(
                    {
                        "event": "batch_done",
                        "language": language,
                        "batch_index": batch_index,
                        "batch": batch_name,
                        "files_done": files_done,
                        "batch_files": batch_total_files,
                        "batch_sample_rate_counts": {
                            str(rate): count for rate, count in sorted(batch_counts.items())
                        },
                        "batch_bad_files": batch_bad_files,
                        "elapsed_sec": round(elapsed, 1),
                        "files_per_sec": round(files_done / elapsed, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    elapsed = time.time() - started
    mixed_batches = sum(1 for batch in batch_results if batch["status"] == "mixed")
    bad_batches = sum(1 for batch in batch_results if batch["bad_files"] > 0)
    uniform_batches = len(batch_results) - mixed_batches - bad_batches
    bad_files_total = sum(batch["bad_files"] for batch in batch_results)

    return {
        "language": language,
        "dataset_root": str(language_root),
        "batch_count": len(batch_results),
        "files": files_done,
        "sample_rate_counts": {
            str(rate): count for rate, count in sorted(overall_counts.items())
        },
        "bad_files": bad_files_total,
        "uniform_batches": uniform_batches,
        "mixed_batches": mixed_batches,
        "bad_batches": bad_batches,
        "sample_rate_examples": sample_rate_examples,
        "bad_examples": bad_examples,
        "batches": batch_results,
        "elapsed_sec": round(elapsed, 3),
        "files_per_sec": round(files_done / elapsed, 1) if elapsed else 0.0,
    }


def build_markdown_report(results: Dict[str, object]) -> str:
    generated_at = results["generated_at_utc"]
    config = results["config"]
    languages = results["languages"]
    total = results["total"]
    excluded_languages = config["excluded_languages"]

    lines: List[str] = []
    lines.append("# Emilia 24k Non-ZH Original Sample Rate Report")
    lines.append("")
    lines.append(f"- Generated at: `{generated_at}`")
    lines.append(f"- Dataset root: `{config['dataset_root']}`")
    lines.append(f"- Languages scanned: `{', '.join(config['languages'])}`")
    lines.append(f"- Excluded languages: `{', '.join(excluded_languages) or '-'}`")
    lines.append(
        "- Method: read the original MP3 file header directly, skip any ID3 tag, "
        "then parse the first valid MPEG audio frame header to obtain the source sample rate."
    )
    lines.append(
        f"- Scanner config: `workers={config['workers']}`, `chunk_size={config['chunk_size']}`, "
        f"`initial_read_bytes={config['initial_read_bytes']}`, "
        f"`fallback_read_bytes={config['fallback_read_bytes']}`"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Language | Batches | Files | Sample-rate distribution | Mixed batches | Bad files | Scan speed |")
    lines.append("|---|---:|---:|---|---:|---:|---:|")
    for language_result in languages:
        lines.append(
            "| "
            f"{language_result['language']} | "
            f"{format_int(language_result['batch_count'])} | "
            f"{format_int(language_result['files'])} | "
            f"{format_rate_counts(language_result['sample_rate_counts'])} | "
            f"{format_int(language_result['mixed_batches'])} | "
            f"{format_int(language_result['bad_files'])} | "
            f"{language_result['files_per_sec']} files/s |"
        )
    lines.append(
        "| Total | "
        f"{format_int(total['batch_count'])} | "
        f"{format_int(total['files'])} | "
        f"{format_rate_counts(total['sample_rate_counts'])} | "
        f"{format_int(total['mixed_batches'])} | "
        f"{format_int(total['bad_files'])} | "
        f"{total['files_per_sec']} files/s |"
    )
    lines.append("")
    lines.append("## Per-Language Details")
    lines.append("")

    for language_result in languages:
        lines.append(f"### {language_result['language']}")
        lines.append("")
        lines.append(f"- Files scanned: `{format_int(language_result['files'])}`")
        lines.append(f"- Batch count: `{format_int(language_result['batch_count'])}`")
        lines.append(
            f"- Sample-rate distribution: `{format_rate_counts(language_result['sample_rate_counts'])}`"
        )
        lines.append(
            f"- Batch status: `uniform={format_int(language_result['uniform_batches'])}`, "
            f"`mixed={format_int(language_result['mixed_batches'])}`, "
            f"`bad={format_int(language_result['bad_batches'])}`"
        )
        if language_result["sample_rate_examples"]:
            sample_examples = ", ".join(
                f"{rate} Hz -> `{path}`"
                for rate, path in sorted(language_result["sample_rate_examples"].items(), key=lambda item: int(item[0]))
            )
            lines.append(f"- Example files: {sample_examples}")
        if language_result["bad_examples"]:
            bad_examples = ", ".join(f"`{path}`" for path in language_result["bad_examples"][:5])
            lines.append(f"- Bad-file examples: {bad_examples}")
        lines.append("")
        lines.append("| Batch | Files | Sample-rate distribution | Bad files | Status |")
        lines.append("|---|---:|---|---:|---|")
        for batch_result in language_result["batches"]:
            lines.append(
                "| "
                f"{batch_result['batch']} | "
                f"{format_int(batch_result['files'])} | "
                f"{format_rate_counts(batch_result['sample_rate_counts'])} | "
                f"{format_int(batch_result['bad_files'])} | "
                f"{batch_result['status']} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    if args.languages:
        languages = [lang.upper() for lang in args.languages]
    else:
        languages = discover_languages(dataset_root, args.exclude_languages)

    config = ScanConfig(
        dataset_root=dataset_root,
        languages=languages,
        workers=args.workers,
        chunk_size=args.chunk_size,
        initial_read_bytes=args.initial_read_bytes,
        fallback_read_bytes=args.fallback_read_bytes,
    )

    started = time.time()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    language_results = [scan_language(config, language) for language in config.languages]
    total_counts: Counter[int] = Counter()
    total_files = 0
    total_bad_files = 0
    total_batches = 0
    total_mixed_batches = 0
    total_bad_batches = 0

    for language_result in language_results:
        total_files += int(language_result["files"])
        total_bad_files += int(language_result["bad_files"])
        total_batches += int(language_result["batch_count"])
        total_mixed_batches += int(language_result["mixed_batches"])
        total_bad_batches += int(language_result["bad_batches"])
        for rate, count in language_result["sample_rate_counts"].items():
            total_counts[int(rate)] += int(count)

    elapsed = time.time() - started
    total_result = {
        "files": total_files,
        "batch_count": total_batches,
        "sample_rate_counts": {
            str(rate): count for rate, count in sorted(total_counts.items())
        },
        "mixed_batches": total_mixed_batches,
        "bad_batches": total_bad_batches,
        "bad_files": total_bad_files,
        "elapsed_sec": round(elapsed, 3),
        "files_per_sec": round(total_files / elapsed, 1) if elapsed else 0.0,
    }

    result = {
        "generated_at_utc": generated_at,
        "config": {
            "dataset_root": str(dataset_root),
            "languages": list(config.languages),
            "excluded_languages": sorted(args.exclude_languages),
            "workers": config.workers,
            "chunk_size": config.chunk_size,
            "initial_read_bytes": config.initial_read_bytes,
            "fallback_read_bytes": config.fallback_read_bytes,
        },
        "languages": language_results,
        "total": total_result,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(build_markdown_report(result), encoding="utf-8")

    print(
        json.dumps(
            {
                "event": "done",
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "total_files": total_result["files"],
                "elapsed_sec": total_result["elapsed_sec"],
                "files_per_sec": total_result["files_per_sec"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
