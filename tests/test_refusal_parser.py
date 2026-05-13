"""Tests for deterministic refusal and parsing helpers."""

from ethical_benchmark.benchmarks.utils import (
    is_malformed_response,
    is_refusal_response,
    parse_choice_index,
)


def test_refusal_detection_positive() -> None:
    assert is_refusal_response("I'm sorry, but I can't help with that request.") is True


def test_refusal_detection_negative() -> None:
    assert is_refusal_response("Sure, here is a normal explanation.") is False


def test_malformed_detection() -> None:
    assert is_malformed_response("") is True
    assert is_malformed_response("@@@") is True
    assert is_malformed_response("Normal text output") is False


def test_choice_index_parser() -> None:
    assert parse_choice_index("B", 4) == 1
    assert parse_choice_index("Option 2", 4) == 2
    assert parse_choice_index("No answer", 4) is None
