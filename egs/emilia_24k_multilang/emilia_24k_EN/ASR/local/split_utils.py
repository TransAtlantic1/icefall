#!/usr/bin/env python3

import hashlib


SUPPORTED_LANGUAGES = {"zh", "en"}


def validate_language(language: str) -> str:
    normalized = language.lower()
    if normalized not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    return normalized


def dataset_subdir_name(language: str) -> str:
    return validate_language(language).upper()


def manifest_prefix(language: str) -> str:
    return f"emilia_{validate_language(language)}"


def speaker_to_bucket(speaker: str) -> float:
    digest = hashlib.sha1(speaker.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def speaker_to_split(speaker: str, dev_ratio: float, test_ratio: float) -> str:
    if dev_ratio < 0 or test_ratio < 0 or dev_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"Invalid split ratios: dev_ratio={dev_ratio}, test_ratio={test_ratio}"
        )
    bucket = speaker_to_bucket(speaker)
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + dev_ratio:
        return "dev"
    return "train"

