#!/usr/bin/env python3

import re
import sys
import unicodedata
from pathlib import Path
from typing import Callable, List, Optional


ASCII_ALPHA_PATTERN = re.compile(r"[A-Za-z]")
DIGIT_PATTERN = re.compile(r"\d")
CHINESE_NUMERIC_SPAN_PATTERN = re.compile(
    r"[零〇一二三四五六七八九十百千万亿点两幺壹贰叁肆伍陆柒捌玖拾佰仟萬億]+"
)

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SPEECHIO_LOCAL = _REPO_ROOT / "egs" / "speechio" / "ASR" / "local"
if str(_SPEECHIO_LOCAL) not in sys.path:
    sys.path.insert(0, str(_SPEECHIO_LOCAL))

from speechio_norm import chn2num  # noqa: E402


def read_raw_text(supervision) -> str:
    custom = getattr(supervision, "custom", None) or {}
    return str(custom.get("raw_text") or supervision.text or "")


def _consume_ascii_alpha_run(text: str, start: int) -> int:
    end = start
    while end < len(text) and ASCII_ALPHA_PATTERN.fullmatch(text[end]):
        end += 1
    return end


def extract_english_word_runs(text: str) -> List[str]:
    runs = []
    i = 0
    while i < len(text):
        if ASCII_ALPHA_PATTERN.fullmatch(text[i]):
            end = _consume_ascii_alpha_run(text, i)
            runs.append(text[i:end])
            i = end
            continue
        i += 1
    return runs


def collect_char_tokens(text: str) -> List[str]:
    tokens = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ASCII_ALPHA_PATTERN.fullmatch(ch):
            i = _consume_ascii_alpha_run(text, i)
            continue
        tokens.append(ch)
        i += 1
    return tokens


def hybrid_tokenize(
    text: str, english_encoder: Optional[Callable[[str], List[str]]] = None
) -> List[str]:
    tokens = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue

        if ASCII_ALPHA_PATTERN.fullmatch(ch):
            end = _consume_ascii_alpha_run(text, i)
            run = text[i:end]
            if english_encoder is None:
                tokens.extend(list(run))
            else:
                tokens.extend(english_encoder(run))
            i = end
            continue

        tokens.append(ch)
        i += 1
    return tokens


def _normalize_chinese_numeric_spans(text: str) -> str:
    def repl(match: re.Match) -> str:
        span = match.group(0)
        try:
            return chn2num(span)
        except Exception:
            return span

    return CHINESE_NUMERIC_SPAN_PATTERN.sub(repl, text)


def normalize_text_for_evaluation(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    return _normalize_chinese_numeric_spans(normalized)


def _should_keep_punctuation(text: str, index: int) -> bool:
    ch = text[index]
    if ch != ".":
        return False

    prev_is_digit = index > 0 and text[index - 1].isdigit()
    next_is_digit = index + 1 < len(text) and text[index + 1].isdigit()
    return prev_is_digit and next_is_digit


def strip_punctuation_for_evaluation(text: str) -> str:
    chars = []
    for i, ch in enumerate(text):
        if unicodedata.category(ch).startswith("P") and not _should_keep_punctuation(
            text, i
        ):
            continue
        chars.append(ch)
    return "".join(chars)


def literal_evaluation_tokens(text: str) -> List[str]:
    normalized = unicodedata.normalize("NFKC", str(text))
    return list("".join(normalized.split()))


def evaluation_tokens(text: str) -> List[str]:
    normalized = normalize_text_for_evaluation(text)
    normalized = strip_punctuation_for_evaluation(normalized)
    return list("".join(normalized.split()))


def contains_numeric_expression(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", str(text))
    return bool(
        DIGIT_PATTERN.search(normalized)
        or CHINESE_NUMERIC_SPAN_PATTERN.search(normalized)
    )
