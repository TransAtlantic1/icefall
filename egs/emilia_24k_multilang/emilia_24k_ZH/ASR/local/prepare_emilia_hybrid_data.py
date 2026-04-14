#!/usr/bin/env python3

import argparse
from pathlib import Path

import lhotse
from lhotse import load_manifest_lazy
from lhotse.serialization import load_manifest_lazy_or_eager

from hybrid_text import collect_char_tokens, extract_english_word_runs, read_raw_text


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cuts-path",
        type=Path,
        required=True,
        help="Combined train cuts file or a directory containing train_split_* pieces.",
    )
    parser.add_argument("--lang-dir", type=Path, required=True)
    return parser.parse_args()


def load_cuts(path: Path):
    if path.is_file():
        cuts = load_manifest_lazy_or_eager(path)
        if cuts is None:
            raise ValueError(f"Unable to load cuts from {path}")
        return cuts

    if not path.is_dir():
        raise FileNotFoundError(f"Could not find cuts file or directory at {path}")

    split_dirs = sorted(path.glob("train_split_*"))
    for split_dir in split_dirs:
        pieces = sorted(split_dir.glob("emilia_*_cuts_train.*.jsonl.gz"))
        if pieces:
            return lhotse.combine(load_manifest_lazy(p) for p in pieces)

    raise FileNotFoundError(f"Could not find Emilia split train cuts under {path}")


def main():
    args = get_args()
    args.lang_dir.mkdir(parents=True, exist_ok=True)

    raw_text_path = args.lang_dir / "raw_text.txt"
    english_text_path = args.lang_dir / "english_text.txt"
    char_tokens_path = args.lang_dir / "char_tokens.txt"

    char_tokens = set()
    cuts = load_cuts(args.cuts_path)
    with open(raw_text_path, "w", encoding="utf-8") as raw_f, open(
        english_text_path, "w", encoding="utf-8"
    ) as english_f:
        for cut in cuts:
            if not cut.supervisions:
                continue
            raw_text = read_raw_text(cut.supervisions[0]).strip()
            if not raw_text:
                continue

            raw_f.write(raw_text + "\n")
            char_tokens.update(collect_char_tokens(raw_text))

            english_runs = extract_english_word_runs(raw_text)
            if english_runs:
                english_f.write(" ".join(english_runs) + "\n")

    with open(char_tokens_path, "w", encoding="utf-8") as f:
        for token in sorted(char_tokens):
            f.write(token + "\n")


if __name__ == "__main__":
    main()
