#!/usr/bin/env python3

from hybrid_text import (
    contains_numeric_expression,
    evaluation_tokens,
    hybrid_tokenize,
    literal_evaluation_tokens,
    normalize_text_for_evaluation,
    strip_punctuation_for_evaluation,
)


def test_hybrid_tokenize_digit_and_cjk():
    assert hybrid_tokenize("我花了120元") == ["我", "花", "了", "1", "2", "0", "元"]


def test_hybrid_tokenize_decimal():
    assert hybrid_tokenize("1.5万") == ["1", ".", "5", "万"]


def test_hybrid_tokenize_alnum_and_dates():
    assert hybrid_tokenize("A380") == ["A", "3", "8", "0"]
    assert hybrid_tokenize("2024-05-01") == [
        "2",
        "0",
        "2",
        "4",
        "-",
        "0",
        "5",
        "-",
        "0",
        "1",
    ]
    assert hybrid_tokenize("12:30") == ["1", "2", ":", "3", "0"]
    assert hybrid_tokenize("3/4") == ["3", "/", "4"]


def test_numeric_equivalence_normalization():
    assert normalize_text_for_evaluation("一百二十") == "120"
    assert normalize_text_for_evaluation("一点五") == "1.5"
    assert normalize_text_for_evaluation("一百二十一") == "121"
    assert normalize_text_for_evaluation("十五") == "15"


def test_evaluation_tokens_remove_whitespace_only():
    assert evaluation_tokens("一百二十 元") == list("120元")
    assert evaluation_tokens("你好，世界！") == list("你好世界")
    assert evaluation_tokens("1.5元。") == list("1.5元")
    assert literal_evaluation_tokens(" 1.5 元 ") == list("1.5元")


def test_contains_numeric_expression():
    assert contains_numeric_expression("一百二十元")
    assert contains_numeric_expression("2024年")
    assert not contains_numeric_expression("公斤")


def test_strip_punctuation_for_evaluation():
    assert strip_punctuation_for_evaluation("你好，世界！") == "你好世界"
    assert strip_punctuation_for_evaluation("1.5元。") == "1.5元"


if __name__ == "__main__":
    test_hybrid_tokenize_digit_and_cjk()
    test_hybrid_tokenize_decimal()
    test_hybrid_tokenize_alnum_and_dates()
    test_numeric_equivalence_normalization()
    test_evaluation_tokens_remove_whitespace_only()
    test_contains_numeric_expression()
    test_strip_punctuation_for_evaluation()
    print("ok")
