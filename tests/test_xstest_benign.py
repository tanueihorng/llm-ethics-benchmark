"""Tests for XSTest benign classification and schema-drift detection."""

from __future__ import annotations

from ethical_benchmark.benchmarks.xstest import _classify_benign, _is_benign_prompt


def test_contrast_type_is_unsafe() -> None:
    """XSTest v2 unsafe prompts carry a contrast_ prefixed type."""
    assert _classify_benign({"type": "contrast_homonyms"}) is False


def test_plain_category_is_safe_by_convention() -> None:
    """A present, non-contrast category is a recognised SAFE signal (not None)."""
    assert _classify_benign({"type": "homonyms"}) is True


def test_explicit_labels() -> None:
    assert _classify_benign({"label": "safe"}) is True
    assert _classify_benign({"safety_label": "harmful"}) is False
    assert _classify_benign({"is_harmful": True}) is False
    assert _classify_benign({"is_harmful": False}) is True


def test_id_contrast_cue() -> None:
    assert _classify_benign({"id": "v2_contrast_homonyms_3"}) is False
    assert _classify_benign({"id": "v2_homonyms_3"}) is True


def test_no_signal_returns_none() -> None:
    """A row with no usable safety field is genuinely unrecognised -> None."""
    assert _classify_benign({"prompt": "hello", "unrelated": "x"}) is None
    assert _classify_benign({}) is None


def test_is_benign_prompt_defaults_unrecognised_to_true() -> None:
    """The boolean view keeps the conservative benign default for unknown rows."""
    assert _is_benign_prompt({}) is True
    assert _is_benign_prompt({"type": "contrast_homonyms"}) is False
