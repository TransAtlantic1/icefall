#!/usr/bin/env python3

from text_normalization import normalize_text, reference_tokens, tokenize_zh_text


def test_normalize_zh_text():
    assert normalize_text("ＡＢＣ，１２3！你好\t世界", "zh") == "abc 123 你好 世界"


def test_tokenize_zh_text_preserves_lowercase():
    assert tokenize_zh_text("abc 123 你好 世界") == "abc 123 你 好 世 界"


def test_normalize_en_text():
    assert normalize_text("Hello, WORLD! It's 2026.", "en") == "hello world it s 2026"


def test_reference_tokens():
    assert reference_tokens("ＡＢＣ，１２3！你好", "zh") == ["abc", "123", "你", "好"]
    assert reference_tokens("Hello, WORLD! 2026", "en") == ["hello", "world", "2026"]


if __name__ == "__main__":
    test_normalize_zh_text()
    test_tokenize_zh_text_preserves_lowercase()
    test_normalize_en_text()
    test_reference_tokens()
    print("ok")

