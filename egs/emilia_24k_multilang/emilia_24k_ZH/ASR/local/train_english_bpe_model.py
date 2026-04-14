#!/usr/bin/env python3

import argparse
from pathlib import Path

import sentencepiece as spm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang-dir", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=500)
    return parser.parse_args()


def ensure_non_empty_transcript(transcript: Path) -> Path:
    if transcript.read_text(encoding="utf-8").strip():
        return transcript

    fallback = transcript.with_name("english_text_fallback.txt")
    fallback.write_text(
        "a b c d e\nf g h i j\nk l m n o\np q r s t\nu v w x y z\n",
        encoding="utf-8",
    )
    return fallback


def main():
    args = get_args()
    args.lang_dir.mkdir(parents=True, exist_ok=True)
    transcript = ensure_non_empty_transcript(args.transcript)

    model_prefix = args.lang_dir / "english_bpe"
    model_file = model_prefix.with_suffix(".model")
    if model_file.is_file():
        print(f"{model_file} exists - skipping")
        return

    spm.SentencePieceTrainer.train(
        input=str(transcript),
        vocab_size=args.vocab_size,
        model_type="bpe",
        model_prefix=str(model_prefix),
        character_coverage=1.0,
        input_sentence_size=10000000,
        bos_id=-1,
        eos_id=-1,
        hard_vocab_limit=False,
        byte_fallback=False,
    )


if __name__ == "__main__":
    main()
