"""Unit tests for evaluator logic (answer parsing, scoring, summarization)."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from ethical_benchmark.evaluators.bias_eval import BiasEvalConfig, BiasEvaluator
from ethical_benchmark.evaluators.factuality_eval import (
    FactualityEvalConfig,
    FactualityEvaluator,
    _parse_answer_index as factuality_parse,
)
from ethical_benchmark.evaluators.toxicity_eval import (
    ToxicityEvalConfig,
    _extract_toxicity_probability,
)


# =========================================================================
# Answer parsing
# =========================================================================


class TestBiasAnswerParsing:
    """Tests for bias evaluator answer parsing."""

    def setup_method(self) -> None:
        self.evaluator = BiasEvaluator(config=BiasEvalConfig())

    def test_parse_single_letter(self) -> None:
        assert self.evaluator._parse_answer_index("B", num_choices=3) == 1

    def test_parse_letter_in_sentence(self) -> None:
        assert self.evaluator._parse_answer_index("The answer is C.", num_choices=3) == 2

    def test_parse_digit(self) -> None:
        assert self.evaluator._parse_answer_index("Option 2", num_choices=3) == 2

    def test_parse_returns_none_on_gibberish(self) -> None:
        assert self.evaluator._parse_answer_index("no answer here !!!", num_choices=3) is None

    def test_parse_out_of_range(self) -> None:
        # Letter D => index 3, but only 3 choices (max index 2)
        assert self.evaluator._parse_answer_index("D", num_choices=3) is None

    def test_parse_empty_string(self) -> None:
        assert self.evaluator._parse_answer_index("", num_choices=3) is None


class TestFactualityAnswerParsing:
    """Tests for factuality answer parsing."""

    def test_parse_single_letter(self) -> None:
        assert factuality_parse("A", 3) == 0

    def test_parse_letter_with_period(self) -> None:
        assert factuality_parse("B.", 4) == 1

    def test_parse_digit_fallback(self) -> None:
        assert factuality_parse("2", 5) == 2

    def test_parse_none_on_empty(self) -> None:
        assert factuality_parse("", 3) is None


# =========================================================================
# Toxicity probability extraction
# =========================================================================


class TestExtractToxicityProbability:
    """Tests for toxicity classifier output parsing."""

    def test_dict_toxic_label(self) -> None:
        output = {"label": "toxic", "score": 0.85}
        assert _extract_toxicity_probability(output) == 0.85

    def test_dict_non_toxic_label(self) -> None:
        output = {"label": "non-toxic", "score": 0.9}
        # non-toxic with 0.9 confidence => toxicity = 0.1
        assert abs(_extract_toxicity_probability(output) - 0.1) < 1e-6

    def test_list_output(self) -> None:
        output = [
            {"label": "toxic", "score": 0.7},
            {"label": "non-toxic", "score": 0.3},
        ]
        assert _extract_toxicity_probability(output) == 0.7

    def test_list_no_toxic_label(self) -> None:
        output = [
            {"label": "LABEL_0", "score": 0.6},
            {"label": "LABEL_1", "score": 0.4},
        ]
        result = _extract_toxicity_probability(output)
        assert isinstance(result, float)

    def test_fallback_on_unexpected_type(self) -> None:
        assert _extract_toxicity_probability(42) == 0.0


# =========================================================================
# Bias evaluator batch + summarize
# =========================================================================


class TestBiasEvaluator:
    """Tests for BiasEvaluator batch evaluation and summarization."""

    def test_evaluate_batch(self, bias_samples: list) -> None:
        evaluator = BiasEvaluator(config=BiasEvalConfig())
        responses = ["C", "B"]  # first correct, second stereotyped
        records = evaluator.evaluate_batch(bias_samples, responses)

        assert len(records) == 2
        assert records[0]["is_correct"] is True
        assert records[1]["is_correct"] is False
        assert records[1]["is_stereotyped_prediction"] is True

    def test_summarize_empty(self) -> None:
        evaluator = BiasEvaluator(config=BiasEvalConfig())
        summary = evaluator.summarize([])
        assert summary["num_samples"] == 0
        assert summary["accuracy"] is None

    def test_summarize_computes_accuracy(self, bias_samples: list) -> None:
        evaluator = BiasEvaluator(config=BiasEvalConfig())
        responses = ["C", "C"]  # first correct, second correct
        records = evaluator.evaluate_batch(bias_samples, responses)
        summary = evaluator.summarize(records)
        assert summary["accuracy"] == 1.0


# =========================================================================
# Factuality evaluator
# =========================================================================


class TestFactualityEvaluator:
    """Tests for FactualityEvaluator batch evaluation and summarization."""

    def test_evaluate_batch(self, factuality_samples: list) -> None:
        evaluator = FactualityEvaluator(config=FactualityEvalConfig())
        responses = ["B", "B"]  # both correct
        records = evaluator.evaluate_batch(factuality_samples, responses)

        assert len(records) == 2
        assert all(r["is_correct"] for r in records)

    def test_summarize_computes_accuracy(self, factuality_samples: list) -> None:
        evaluator = FactualityEvaluator(config=FactualityEvalConfig())
        responses = ["B", "A"]  # first correct, second wrong
        records = evaluator.evaluate_batch(factuality_samples, responses)
        summary = evaluator.summarize(records)
        assert summary["objective"]["mc_accuracy"] == 0.5

    def test_summarize_empty(self) -> None:
        evaluator = FactualityEvaluator(config=FactualityEvalConfig())
        summary = evaluator.summarize([])
        assert summary["num_samples"] == 0
