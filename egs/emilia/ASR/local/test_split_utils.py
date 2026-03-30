#!/usr/bin/env python3

from split_utils import (
    dataset_subdir_name,
    manifest_prefix,
    speaker_to_split,
    validate_language,
)


def test_validate_language():
    assert validate_language("ZH") == "zh"
    assert validate_language("en") == "en"


def test_helpers():
    assert dataset_subdir_name("zh") == "ZH"
    assert manifest_prefix("en") == "emilia_en"


def test_speaker_split_is_deterministic():
    split1 = speaker_to_split("speaker-1", dev_ratio=0.2, test_ratio=0.2)
    split2 = speaker_to_split("speaker-1", dev_ratio=0.2, test_ratio=0.2)
    assert split1 == split2
    assert split1 in {"train", "dev", "test"}


def test_zero_ratios_fall_back_to_train():
    assert speaker_to_split("speaker-2", dev_ratio=0.0, test_ratio=0.0) == "train"


if __name__ == "__main__":
    test_validate_language()
    test_helpers()
    test_speaker_split_is_deterministic()
    test_zero_ratios_fall_back_to_train()
    print("ok")
