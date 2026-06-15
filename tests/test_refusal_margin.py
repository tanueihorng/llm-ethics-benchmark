"""Unit tests for the refusal-margin mechanism metric (D33).

GPU-free: exercises the pure numeric helpers in
``ethical_benchmark.analysis.refusal_margin`` with synthetic logits and a stub
tokenizer, so the metric math and the per-tokenizer token-set construction are
verified without loading any model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ethical_benchmark.analysis.refusal_margin import (
    build_token_sets,
    decision_entropy,
    log_prob_of_set,
    refusal_margin,
    top1_top2_gap,
)


class TestMarginMath:
    def test_refusal_margin_sign(self) -> None:
        # vocab of 6; refusal ids {0,1} given high logits, compliance {2,3} low.
        z = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        assert refusal_margin(z, [0, 1], [2, 3]) > 0
        # mirror -> comply-ward
        z2 = np.array([0.0, 0.0, 5.0, 5.0, 0.0, 0.0])
        assert refusal_margin(z2, [0, 1], [2, 3]) < 0

    def test_refusal_margin_symmetric_is_zero(self) -> None:
        z = np.array([2.0, 2.0, 2.0, 2.0, 0.0])
        assert refusal_margin(z, [0, 1], [2, 3]) == pytest.approx(0.0, abs=1e-9)

    def test_margin_invariant_to_logit_shift(self) -> None:
        # Adding a constant to ALL logits (a partition-function shift) must not
        # change the margin — this is the scale/shift-invariance the metric relies on.
        z = np.array([3.0, 1.0, 0.5, -2.0, 4.0])
        m1 = refusal_margin(z, [0, 1], [2, 3])
        m2 = refusal_margin(z + 17.0, [0, 1], [2, 3])
        assert m1 == pytest.approx(m2, abs=1e-9)

    def test_log_prob_of_full_vocab_is_zero(self) -> None:
        z = np.array([1.0, 2.0, 3.0, 0.5])
        assert log_prob_of_set(z, [0, 1, 2, 3]) == pytest.approx(0.0, abs=1e-9)

    def test_log_prob_of_set_matches_softmax(self) -> None:
        z = np.array([1.0, 2.0, 3.0, 0.5])
        p = np.exp(z) / np.exp(z).sum()
        assert log_prob_of_set(z, [0, 2]) == pytest.approx(math.log(p[0] + p[2]), abs=1e-9)

    def test_out_of_range_ids_ignored(self) -> None:
        z = np.array([1.0, 2.0, 3.0])
        # id 99 is out of range and must be silently dropped, not crash.
        assert log_prob_of_set(z, [0, 99]) == pytest.approx(log_prob_of_set(z, [0]), abs=1e-9)

    def test_empty_set_is_nan_margin(self) -> None:
        z = np.array([1.0, 2.0, 3.0])
        assert math.isnan(refusal_margin(z, [], [0]))


class TestConfidenceDiagnostics:
    def test_entropy_uniform_is_log_vocab(self) -> None:
        z = np.zeros(8)
        assert decision_entropy(z) == pytest.approx(math.log(8), abs=1e-9)

    def test_entropy_peaked_is_near_zero(self) -> None:
        z = np.array([100.0, 0.0, 0.0, 0.0])
        assert decision_entropy(z) == pytest.approx(0.0, abs=1e-6)

    def test_top1_top2_gap_uniform_is_zero(self) -> None:
        assert top1_top2_gap(np.zeros(5)) == pytest.approx(0.0, abs=1e-9)

    def test_top1_top2_gap_peaked_near_one(self) -> None:
        assert top1_top2_gap(np.array([100.0, 0.0, 0.0])) == pytest.approx(1.0, abs=1e-6)


class _StubTokenizer:
    """Maps exact phrase strings to id lists; mimics HF .encode signature."""

    all_special_ids = [0]

    def __init__(self, mapping: dict) -> None:
        self._mapping = mapping

    def encode(self, text, add_special_tokens=False):  # noqa: D401
        return self._mapping.get(text, [0, 7])  # default leads with a special id


class TestTokenSets:
    def test_leading_space_variant_collected(self) -> None:
        tok = _StubTokenizer({"Sorry": [10, 1], " Sorry": [11, 1]})
        sets = build_token_sets(tok, refusal_seeds=["Sorry"], comply_seeds=["Sure"])
        # both the bare and leading-space first tokens are captured
        assert 10 in sets["refusal_ids"] and 11 in sets["refusal_ids"]

    def test_ambiguous_token_dropped_from_both(self) -> None:
        # "No" tokenizes to the same first id under both seed lists -> ambiguous.
        tok = _StubTokenizer({
            "Sorry": [10], " Sorry": [10],
            "No": [50], " No": [50],
            "Sure": [20], " Sure": [20],
        })
        sets = build_token_sets(tok, refusal_seeds=["Sorry", "No"], comply_seeds=["Sure", "No"])
        assert 50 in sets["ambiguous_dropped"]
        assert 50 not in sets["refusal_ids"]
        assert 50 not in sets["compliance_ids"]
        assert 10 in sets["refusal_ids"] and 20 in sets["compliance_ids"]

    def test_special_ids_skipped(self) -> None:
        # leading id 0 is special and must be skipped to reach the real first token.
        tok = _StubTokenizer({"Sorry": [0, 33], " Sorry": [0, 33]})
        sets = build_token_sets(tok, refusal_seeds=["Sorry"], comply_seeds=["Sure"])
        assert sets["refusal_ids"] == [33]

    def test_outputs_are_plain_ints_no_text(self) -> None:
        # Redaction guard: the committable token-set artifact carries only ints.
        tok = _StubTokenizer({"Sorry": [10], " Sorry": [10], "Sure": [20], " Sure": [20]})
        sets = build_token_sets(tok, refusal_seeds=["Sorry"], comply_seeds=["Sure"])
        for key in ("refusal_ids", "compliance_ids", "ambiguous_dropped"):
            assert all(isinstance(i, int) for i in sets[key])
