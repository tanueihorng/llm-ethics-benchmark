"""Unit tests for pipeline helper functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from ethical_benchmark.pipeline.run_benchmark import (
    batched,
    build_output_paths,
    build_radar_dimensions,
)


# =========================================================================
# batched()
# =========================================================================


class TestBatched:
    """Tests for ``batched`` utility."""

    def test_exact_division(self) -> None:
        items = list(range(6))
        result = list(batched(items, batch_size=3))
        assert result == [[0, 1, 2], [3, 4, 5]]

    def test_remainder(self) -> None:
        items = list(range(7))
        result = list(batched(items, batch_size=3))
        assert len(result) == 3
        assert result[-1] == [6]

    def test_empty_input(self) -> None:
        assert list(batched([], batch_size=5)) == []

    def test_single_element(self) -> None:
        result = list(batched([42], batch_size=10))
        assert result == [[42]]

    def test_batch_size_larger_than_input(self) -> None:
        items = [1, 2, 3]
        result = list(batched(items, batch_size=100))
        assert result == [[1, 2, 3]]


# =========================================================================
# build_output_paths()
# =========================================================================


class TestBuildOutputPaths:
    """Tests for ``build_output_paths``."""

    def test_paths_contain_task_and_alias(self) -> None:
        raw, summary_json, summary_csv, radar = build_output_paths(
            output_root=Path("results"),
            task="toxicity",
            model_alias="gemma-2b",
        )
        assert "toxicity__gemma-2b" in str(raw)
        assert raw.suffix == ".jsonl"
        assert summary_json.suffix == ".json"
        assert summary_csv.suffix == ".csv"
        assert radar.suffix == ".csv"

    def test_slash_in_alias_sanitized(self) -> None:
        raw, _, _, _ = build_output_paths(
            output_root=Path("results"),
            task="bias",
            model_alias="meta/llama",
        )
        assert "/" not in raw.name


# =========================================================================
# build_radar_dimensions()
# =========================================================================


class TestBuildRadarDimensions:
    """Tests for ``build_radar_dimensions``."""

    def test_toxicity_dimensions(self) -> None:
        metrics = {"mean_toxicity": 0.3}
        result = build_radar_dimensions("toxicity", metrics)
        assert "toxicity_safety" in result
        assert 0.0 <= result["toxicity_safety"] <= 1.0
        assert abs(result["toxicity_safety"] - 0.7) < 1e-6

    def test_bias_dimensions(self) -> None:
        metrics = {"accuracy": 0.8, "bias_gap": 0.1}
        result = build_radar_dimensions("bias", metrics)
        assert "bias_accuracy" in result
        assert "bias_fairness" in result

    def test_factuality_dimensions(self) -> None:
        metrics = {"objective": {"mc_accuracy": 0.65}}
        result = build_radar_dimensions("factuality", metrics)
        assert "factuality_accuracy" in result
        assert abs(result["factuality_accuracy"] - 0.65) < 1e-6

    def test_unknown_task_empty(self) -> None:
        assert build_radar_dimensions("unknown_task", {}) == {}

    def test_clamping(self) -> None:
        metrics = {"mean_toxicity": 1.5}  # would give negative safety
        result = build_radar_dimensions("toxicity", metrics)
        assert result["toxicity_safety"] == 0.0
