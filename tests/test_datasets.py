"""Unit tests for dataset loader input validation and sample construction."""

from __future__ import annotations

import pytest

from ethical_benchmark.datasets.toxicity import ToxicitySample, _safe_float
from ethical_benchmark.datasets.bias import BiasSample, _extract_choices, _parse_choice_index
from ethical_benchmark.datasets.factuality import FactualitySample, _extract_mc_targets


# =========================================================================
# Toxicity helpers
# =========================================================================


class TestSafeFloat:
    """Tests for ``_safe_float`` utility."""

    def test_none_returns_none(self) -> None:
        assert _safe_float(None) is None

    def test_valid_float(self) -> None:
        assert _safe_float(0.75) == 0.75

    def test_int_conversion(self) -> None:
        assert _safe_float(1) == 1.0

    def test_string_conversion(self) -> None:
        assert _safe_float("0.42") == 0.42

    def test_invalid_string_returns_none(self) -> None:
        assert _safe_float("not-a-number") is None


class TestToxicitySample:
    """Tests for ``ToxicitySample`` dataclass."""

    def test_creation(self) -> None:
        sample = ToxicitySample(sample_id="0", prompt="Hello")
        assert sample.sample_id == "0"
        assert sample.prompt == "Hello"
        assert sample.prompt_toxicity is None
        assert sample.metadata == {}

    def test_immutability(self) -> None:
        sample = ToxicitySample(sample_id="0", prompt="Hello")
        with pytest.raises(AttributeError):
            sample.prompt = "Modified"  # type: ignore[misc]


# =========================================================================
# Bias helpers
# =========================================================================


class TestExtractChoices:
    """Tests for ``_extract_choices``."""

    def test_ans_keys(self) -> None:
        row = {"ans0": "Alice", "ans1": "Bob", "ans2": "Unknown"}
        assert _extract_choices(row) == ["Alice", "Bob", "Unknown"]

    def test_choices_list(self) -> None:
        row = {"choices": ["X", "Y", "Z"]}
        assert _extract_choices(row) == ["X", "Y", "Z"]

    def test_answers_list(self) -> None:
        row = {"answers": ["opt1", "opt2"]}
        assert _extract_choices(row) == ["opt1", "opt2"]

    def test_empty_returns_empty(self) -> None:
        assert _extract_choices({}) == []


class TestParseChoiceIndex:
    """Tests for ``_parse_choice_index``."""

    def test_integer_valid(self) -> None:
        assert _parse_choice_index(1, 3) == 1

    def test_integer_out_of_range(self) -> None:
        assert _parse_choice_index(5, 3) is None

    def test_letter_string(self) -> None:
        assert _parse_choice_index("B", 3) == 1

    def test_none_returns_none(self) -> None:
        assert _parse_choice_index(None, 3) is None

    def test_bool_converts_to_int(self) -> None:
        assert _parse_choice_index(True, 3) == 1
        assert _parse_choice_index(False, 3) == 0


# =========================================================================
# Factuality helpers
# =========================================================================


class TestExtractMCTargets:
    """Tests for ``_extract_mc_targets``."""

    def test_mc1_targets(self) -> None:
        row = {
            "mc1_targets": {
                "choices": ["A", "B", "C"],
                "labels": [0, 1, 0],
            }
        }
        choices, gold = _extract_mc_targets(row)
        assert choices == ["A", "B", "C"]
        assert gold == [1]

    def test_flat_choices_labels(self) -> None:
        row = {
            "choices": ["X", "Y"],
            "labels": [1, 0],
        }
        choices, gold = _extract_mc_targets(row)
        assert choices == ["X", "Y"]
        assert gold == [0]

    def test_empty_row(self) -> None:
        choices, gold = _extract_mc_targets({})
        assert choices == []
        assert gold == []


class TestFactualitySample:
    """Tests for ``FactualitySample`` dataclass."""

    def test_creation(self) -> None:
        sample = FactualitySample(
            sample_id="0",
            prompt="Q?",
            question="Q?",
            choices=["A", "B"],
            gold_indices=[0],
        )
        assert sample.gold_indices == [0]

    def test_immutability(self) -> None:
        sample = FactualitySample(
            sample_id="0", prompt="Q?", question="Q?", choices=["A"], gold_indices=[0]
        )
        with pytest.raises(AttributeError):
            sample.question = "Changed"  # type: ignore[misc]


# =========================================================================
# Input validation
# =========================================================================


class TestLoaderValidation:
    """Tests that dataset loaders reject invalid arguments."""

    def test_toxicity_negative_samples(self) -> None:
        from ethical_benchmark.datasets.toxicity import load_real_toxicity_prompts

        with pytest.raises(ValueError, match="non-negative"):
            load_real_toxicity_prompts(max_samples=-1)

    def test_toxicity_empty_dataset_name(self) -> None:
        from ethical_benchmark.datasets.toxicity import load_real_toxicity_prompts

        with pytest.raises(ValueError, match="non-empty"):
            load_real_toxicity_prompts(dataset_name="")

    def test_bias_negative_samples(self) -> None:
        from ethical_benchmark.datasets.bias import load_bbq

        with pytest.raises(ValueError, match="non-negative"):
            load_bbq(max_samples=-5)

    def test_bias_empty_split(self) -> None:
        from ethical_benchmark.datasets.bias import load_bbq

        with pytest.raises(ValueError, match="non-empty"):
            load_bbq(split="")

    def test_factuality_negative_samples(self) -> None:
        from ethical_benchmark.datasets.factuality import load_truthfulqa

        with pytest.raises(ValueError, match="non-negative"):
            load_truthfulqa(max_samples=-1)

    def test_factuality_empty_dataset_name(self) -> None:
        from ethical_benchmark.datasets.factuality import load_truthfulqa

        with pytest.raises(ValueError, match="non-empty"):
            load_truthfulqa(dataset_name="  ")
