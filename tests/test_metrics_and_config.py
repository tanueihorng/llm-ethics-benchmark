"""Unit tests for metric aggregation, persistence utilities, and config validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from ethical_benchmark.metrics.aggregate import (
    append_jsonl,
    append_summary_csv,
    compute_bootstrap_ci,
    export_radar_csv,
    flatten_for_csv,
    read_jsonl,
    write_json,
)


# =========================================================================
# Bootstrap CI
# =========================================================================


class TestBootstrapCI:
    """Tests for ``compute_bootstrap_ci``."""

    def test_non_empty_array(self) -> None:
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = compute_bootstrap_ci(values, num_resamples=500)

        assert result is not None
        assert "mean" in result
        assert "lower" in result
        assert "upper" in result
        assert result["lower"] <= result["mean"] <= result["upper"]

    def test_empty_array_returns_none(self) -> None:
        assert compute_bootstrap_ci(np.array([])) is None

    def test_confidence_level(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_bootstrap_ci(values, confidence=0.90)
        assert result is not None
        assert result["confidence"] == 0.90

    def test_single_value(self) -> None:
        values = np.array([0.5])
        result = compute_bootstrap_ci(values, num_resamples=100)
        assert result is not None
        assert result["mean"] == 0.5


# =========================================================================
# Flatten for CSV
# =========================================================================


class TestFlattenForCSV:
    """Tests for ``flatten_for_csv``."""

    def test_flat_dict(self) -> None:
        data = {"a": 1, "b": 2}
        assert flatten_for_csv(data) == {"a": 1, "b": 2}

    def test_nested_dict(self) -> None:
        data = {"level1": {"level2": 42}}
        assert flatten_for_csv(data) == {"level1.level2": 42}

    def test_deeply_nested(self) -> None:
        data = {"a": {"b": {"c": "deep"}}}
        assert flatten_for_csv(data) == {"a.b.c": "deep"}

    def test_mixed_types(self) -> None:
        data = {"scalar": 1, "nested": {"key": "val"}}
        result = flatten_for_csv(data)
        assert result == {"scalar": 1, "nested.key": "val"}


# =========================================================================
# JSONL round-trip
# =========================================================================


class TestJSONLRoundTrip:
    """Tests for ``append_jsonl`` and ``read_jsonl``."""

    def test_write_and_read(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        records = [{"id": 1, "val": "a"}, {"id": 2, "val": "b"}]
        append_jsonl(records, path)

        loaded = read_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0]["id"] == 1

    def test_append_preserves_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        append_jsonl([{"x": 1}], path)
        append_jsonl([{"x": 2}], path)

        loaded = read_jsonl(path)
        assert len(loaded) == 2

    def test_read_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.jsonl"
        assert read_jsonl(path) == []


# =========================================================================
# JSON write
# =========================================================================


class TestWriteJSON:
    """Tests for ``write_json``."""

    def test_write_and_verify(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        data = {"key": "value", "num": 42}
        write_json(data, path)

        with path.open() as f:
            loaded = json.load(f)
        assert loaded == data


# =========================================================================
# CSV utilities
# =========================================================================


class TestSummaryCSV:
    """Tests for ``append_summary_csv``."""

    def test_creates_file_with_header(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.csv"
        row = {"task": "toxicity", "accuracy": 0.85}
        append_summary_csv(row, path)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 row

    def test_append_row(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.csv"
        append_summary_csv({"a": 1}, path)
        append_summary_csv({"a": 2}, path)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows


class TestRadarCSV:
    """Tests for ``export_radar_csv``."""

    def test_write_and_verify(self, tmp_path: Path) -> None:
        path = tmp_path / "radar.csv"
        export_radar_csv({"dim_a": 0.8, "dim_b": 0.6}, path)

        lines = path.read_text().strip().split("\n")
        assert lines[0] == "dimension,value"
        assert len(lines) == 3


# =========================================================================
# Config validation
# =========================================================================


class TestConfigValidation:
    """Tests for pydantic config schema validation."""

    def test_valid_config_passes(self, default_config: Dict[str, Any]) -> None:
        from ethical_benchmark.config_schema import validate_config

        result = validate_config(default_config)
        assert "test-model" in result.models

    def test_missing_models_fails(self) -> None:
        from pydantic import ValidationError

        from ethical_benchmark.config_schema import validate_config

        with pytest.raises(ValidationError):
            validate_config({"tasks": {"t": {"dataset_name": "x", "split": "y"}}})

    def test_invalid_dtype_fails(self, default_config: Dict[str, Any]) -> None:
        from pydantic import ValidationError

        from ethical_benchmark.config_schema import validate_config

        default_config["models"]["test-model"]["dtype"] = "invalid_dtype"
        with pytest.raises(ValidationError):
            validate_config(default_config)

    def test_negative_max_new_tokens_fails(self, default_config: Dict[str, Any]) -> None:
        from pydantic import ValidationError

        from ethical_benchmark.config_schema import validate_config

        default_config["decoding"]["max_new_tokens"] = -1
        with pytest.raises(ValidationError):
            validate_config(default_config)

    def test_temperature_out_of_range(self, default_config: Dict[str, Any]) -> None:
        from pydantic import ValidationError

        from ethical_benchmark.config_schema import validate_config

        default_config["decoding"]["temperature"] = -0.5
        with pytest.raises(ValidationError):
            validate_config(default_config)
