#!/usr/bin/env python3

import argparse
from pathlib import Path

import lhotse
from lhotse import load_manifest_lazy
from lhotse.serialization import load_manifest_lazy_or_eager

from split_utils import validate_language
from text_normalization import normalize_text, tokenize_zh_text


SPECIAL_WORDS = ["<eps>", "!SIL", "<SPOKEN_NOISE>", "<UNK>"]


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
    parser.add_argument("--language", type=str, choices=["zh", "en"], required=True)
    parser.add_argument("--lang-dir", type=Path, required=True)
    return parser.parse_args()


def write_words(words, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for idx, word in enumerate(SPECIAL_WORDS):
            f.write(f"{word} {idx}\n")
        for idx, word in enumerate(words, start=len(SPECIAL_WORDS)):
            f.write(f"{word} {idx}\n")


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
    language = validate_language(args.language)
    args.lang_dir.mkdir(parents=True, exist_ok=True)

    transcript_name = (
        "transcript_chars.txt" if language == "zh" else "transcript_words.txt"
    )
    transcript_path = args.lang_dir / transcript_name
    words_path = args.lang_dir / "words.txt"

    vocab = set()
    cuts = load_cuts(args.cuts_path)
    with open(transcript_path, "w", encoding="utf-8") as transcript_f:
        for cut in cuts:
            if not cut.supervisions:
                continue
            text = normalize_text(cut.supervisions[0].text, language)
            if not text:
                continue
            if language == "zh":
                tokenized = tokenize_zh_text(text)
                if not tokenized:
                    continue
                transcript_f.write(tokenized + "\n")
                vocab.update(tokenized.split())
            else:
                transcript_f.write(text + "\n")
                vocab.update(text.split())

    write_words(sorted(vocab), words_path)


if __name__ == "__main__":
    main()
