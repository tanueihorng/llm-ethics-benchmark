"""Tests for quantization benchmark pipeline utility functions."""

from pathlib import Path

import pytest

from ethical_benchmark.benchmarks.base import BenchmarkItem
from ethical_benchmark.metrics.aggregate import append_jsonl
from ethical_benchmark.pipeline.run_quant_benchmark import (
    get_processed_prompt_ids,
    prepare_remaining_items,
    validate_record_schema,
)


def _valid_record() -> dict:
    return {
        "benchmark": "harmbench",
        "prompt_id": "p0",
        "prompt_text": "prompt",
        "response": "response",
        "score_fields": {"harmful_compliance": False},
        "family": "qwen",
        "size_b": 0.8,
        "quantized": False,
        "pair_id": "qwen_0_8b",
        "model_id": "org/model",
        "generation_config": {"temperature": 0.0},
        "seed": 42,
        "timestamp": "2026-03-13T00:00:00Z",
    }


def test_validate_record_schema_ok() -> None:
    validate_record_schema(_valid_record())


def test_validate_record_schema_missing_field() -> None:
    record = _valid_record()
    record.pop("prompt_id")
    with pytest.raises(ValueError, match="missing required fields"):
        validate_record_schema(record)


def test_prepare_remaining_items() -> None:
    items = [
        BenchmarkItem(prompt_id="a", prompt_text="A"),
        BenchmarkItem(prompt_id="b", prompt_text="B"),
        BenchmarkItem(prompt_id="c", prompt_text="C"),
    ]
    remaining = prepare_remaining_items(items, processed_ids={"a", "c"})
    assert [item.prompt_id for item in remaining] == ["b"]


def test_get_processed_prompt_ids(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    append_jsonl([
        {"prompt_id": "p1"},
        {"prompt_id": "p2"},
    ], raw)
    assert get_processed_prompt_ids(raw) == {"p1", "p2"}
