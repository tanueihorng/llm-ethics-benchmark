"""Tests for the ARC-Challenge plugin (T23). Offline — load_dataset is faked."""

from __future__ import annotations

from typing import Any, Dict, List

import ethical_benchmark.benchmarks.arc as arc_mod
from ethical_benchmark.benchmarks.arc import ARCChallengePlugin, _gold_index_from_label
from ethical_benchmark.benchmarks.base import BenchmarkItem
from ethical_benchmark.benchmarks.registry import build_benchmark_plugin

# Mixed schema: letter labels, numeric labels, a 3-option item, a malformed row.
_FAKE_ROWS: List[Dict[str, Any]] = [
    {
        "id": "q_letter",
        "question": "What gas do plants absorb?",
        "choices": {"text": ["Oxygen", "Carbon dioxide", "Nitrogen", "Helium"],
                    "label": ["A", "B", "C", "D"]},
        "answerKey": "B",
    },
    {
        "id": "q_numeric",
        "question": "How many legs does a spider have?",
        "choices": {"text": ["Six", "Eight", "Ten"], "label": ["1", "2", "3"]},
        "answerKey": "2",
    },
    {
        "id": "q_bad",  # no answerKey -> skipped
        "question": "Unanswered?",
        "choices": {"text": ["a", "b"], "label": ["A", "B"]},
        "answerKey": None,
    },
]


def test_gold_index_letter_numeric_and_missing() -> None:
    assert _gold_index_from_label("B", ["A", "B", "C", "D"]) == 1
    assert _gold_index_from_label("2", ["1", "2", "3"]) == 1  # numeric 1-based -> idx 1
    assert _gold_index_from_label("C", ["A", "B", "C"]) == 2
    assert _gold_index_from_label(None, ["A", "B"]) is None
    assert _gold_index_from_label("Z", ["A", "B"]) is None


def test_load_items_parses_arc_schema(monkeypatch) -> None:
    monkeypatch.setattr(arc_mod, "load_dataset", lambda *a, **k: list(_FAKE_ROWS))
    plugin = ARCChallengePlugin({})
    items = plugin.load_items(max_samples=None, seed=42)
    assert len(items) == 2  # the answerKey=None row is dropped
    by_id = {it.prompt_id: it for it in items}
    assert by_id["q_letter"].metadata["gold_index"] == 1
    assert by_id["q_letter"].metadata["num_choices"] == 4
    assert by_id["q_numeric"].metadata["gold_index"] == 1
    assert by_id["q_numeric"].metadata["num_choices"] == 3
    # Options are re-labelled A,B,C,... regardless of the source label scheme.
    assert "A. Six" in by_id["q_numeric"].prompt_text
    assert by_id["q_letter"].prompt_text.rstrip().endswith("reply with only the option letter.")


def test_score_response_exact_match() -> None:
    plugin = ARCChallengePlugin({})
    item = BenchmarkItem("q", "prompt", {"gold_index": 1, "num_choices": 4, "subject": "arc_challenge"})
    correct = plugin.score_response(item, "B")
    assert correct["is_answered"] and correct["is_correct"]
    wrong = plugin.score_response(item, "A")
    assert wrong["is_answered"] and not wrong["is_correct"]
    blank = plugin.score_response(item, "I am not sure.")
    assert not blank["is_answered"] and not blank["is_correct"]


def test_aggregate_accuracy() -> None:
    plugin = ARCChallengePlugin({})
    records = [
        {"score_fields": {"is_answered": True, "is_correct": True, "subject": "arc_challenge"}},
        {"score_fields": {"is_answered": True, "is_correct": False, "subject": "arc_challenge"}},
    ]
    agg = plugin.aggregate(records)
    assert agg["num_samples"] == 2
    assert agg["accuracy"] == 0.5
    assert agg["answered_rate"] == 1.0


def test_registry_builds_arc() -> None:
    assert isinstance(build_benchmark_plugin("arc", {}), ARCChallengePlugin)
