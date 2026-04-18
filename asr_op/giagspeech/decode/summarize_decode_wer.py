#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/inspire/hdd/project/embodied-multimodality/chenxie-25019/fj/experiments/gigaspeech_h200")

JOBS = {
    "16k": ROOT / "16k_train_g0-3/zipformer_m_g0-1-2-3",
    "24k": ROOT / "24k_train_g4-7/zipformer_m_g4-5-6-7",
}

DEFAULT_EPOCHS = (10, 20, 30)
METHODS = ("greedy", "mbs")


def find_summary(exp_dir: Path, epoch: int, method: str) -> Path | None:
    if method == "greedy":
        pattern = f"greedy_search/wer-summary-test-greedy_search-epoch-{epoch}-*.txt"
    else:
        pattern = (
            "modified_beam_search/"
            f"wer-summary-test-beam_size_4-epoch-{epoch}-*.txt"
        )

    matches = sorted(exp_dir.glob(pattern))
    return matches[0] if matches else None


def read_wer(path: Path | None) -> str:
    if path is None or not path.is_file():
        return "missing"

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return "missing"

    parts = lines[1].split()
    return parts[-1] if parts else "missing"


def collect(epochs: tuple[int, ...]) -> tuple[list[dict[str, str]], bool]:
    rows: list[dict[str, str]] = []
    complete = True

    for job, exp_dir in JOBS.items():
        for epoch in epochs:
            row = {"job": job, "epoch": str(epoch)}
            for method in METHODS:
                path = find_summary(exp_dir, epoch, method)
                wer = read_wer(path)
                row[method] = wer
                if wer == "missing":
                    complete = False
            rows.append(row)

    return rows, complete


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| job | epoch | greedy | mbs |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['job']} | {row['epoch']} | {row['greedy']} | {row['mbs']} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--check-complete", action="store_true")
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=list(DEFAULT_EPOCHS),
        help="Epochs to summarize, e.g. --epochs 40 50 60",
    )
    args = parser.parse_args()

    rows, complete = collect(tuple(args.epochs))

    if args.format == "json":
        print(json.dumps({"complete": complete, "rows": rows}, ensure_ascii=False))
    else:
        print(markdown_table(rows))

    if args.check_complete:
        return 0 if complete else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
