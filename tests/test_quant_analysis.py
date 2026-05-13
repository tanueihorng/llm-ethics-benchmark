"""Tests for quantization pairwise analysis logic."""

from __future__ import annotations

import json
from pathlib import Path

from ethical_benchmark.analysis.compare_quant_pairs import (
    build_pairwise_report,
    classify_pair_change,
    compute_cross_family_consistency,
    compute_relative_delta,
    compute_scale_sensitivity,
    summarize_pair_labels,
)
from ethical_benchmark.pipeline.run_quant_benchmark import build_run_paths
from ethical_benchmark.quant.config_schema import QuantizationConfig


def test_compute_relative_delta() -> None:
    assert compute_relative_delta(0.5, 0.1) == 0.2
    assert compute_relative_delta(0.0, 0.1) is None


def test_classify_pair_change_labels() -> None:
    assert classify_pair_change(0.05, 0.0, 0.0) == "alignment_degradation"
    assert (
        classify_pair_change(-0.08, 0.03, -0.10)
        == "capability_collapse_masquerading_as_safety"
    )
    assert classify_pair_change(0.0, 0.0, -0.01) == "robust_preservation"
    assert classify_pair_change(0.01, 0.05, -0.08) == "broad_degradation"


def test_classify_pair_change_incomplete_when_any_delta_missing() -> None:
    """Partial runs should label as 'incomplete', not silently 'broad_degradation'."""
    assert classify_pair_change(None, 0.0, 0.0) == "incomplete"
    assert classify_pair_change(0.0, None, 0.0) == "incomplete"
    assert classify_pair_change(0.0, 0.0, None) == "incomplete"
    assert classify_pair_change(None, None, None) == "incomplete"


def test_build_pairwise_and_reports(tmp_path: Path) -> None:
    raw_cfg = {
        "study_name": "test",
        "models": {
            "qwen_small_base": {
                "family": "qwen",
                "size_b": 0.8,
                "quantized": False,
                "pair_id": "qwen_small",
                "model_id": "org/qwen-small-base",
                "benchmarks": ["harmbench", "xstest", "mmlu"],
            },
            "qwen_small_4bit": {
                "family": "qwen",
                "size_b": 0.8,
                "quantized": True,
                "pair_id": "qwen_small",
                "model_id": "org/qwen-small-4bit",
                "benchmarks": ["harmbench", "xstest", "mmlu"],
            },
            "qwen_large_base": {
                "family": "qwen",
                "size_b": 4.0,
                "quantized": False,
                "pair_id": "qwen_large",
                "model_id": "org/qwen-large-base",
                "benchmarks": ["harmbench", "xstest", "mmlu"],
            },
            "qwen_large_4bit": {
                "family": "qwen",
                "size_b": 4.0,
                "quantized": True,
                "pair_id": "qwen_large",
                "model_id": "org/qwen-large-4bit",
                "benchmarks": ["harmbench", "xstest", "mmlu"],
            },
            "llama_base": {
                "family": "llama",
                "size_b": 3.0,
                "quantized": False,
                "pair_id": "llama_pair",
                "model_id": "org/llama-base",
                "benchmarks": ["harmbench", "xstest", "mmlu"],
            },
            "llama_4bit": {
                "family": "llama",
                "size_b": 3.0,
                "quantized": True,
                "pair_id": "llama_pair",
                "model_id": "org/llama-4bit",
                "benchmarks": ["harmbench", "xstest", "mmlu"],
            },
        },
        "benchmarks": {
            "harmbench": {"dataset_name": "dummy", "split": "test"},
            "xstest": {"dataset_name": "dummy", "split": "test"},
            "mmlu": {"dataset_name": "dummy", "split": "test"},
        },
    }
    config = QuantizationConfig.model_validate(raw_cfg)

    metric_values = {
        "qwen_small_base": {"harmbench": 0.30, "xstest": 0.10, "mmlu": 0.50},
        "qwen_small_4bit": {"harmbench": 0.38, "xstest": 0.18, "mmlu": 0.38},
        "qwen_large_base": {"harmbench": 0.25, "xstest": 0.09, "mmlu": 0.60},
        "qwen_large_4bit": {"harmbench": 0.27, "xstest": 0.10, "mmlu": 0.56},
        "llama_base": {"harmbench": 0.22, "xstest": 0.08, "mmlu": 0.58},
        "llama_4bit": {"harmbench": 0.24, "xstest": 0.11, "mmlu": 0.52},
    }

    metric_key = {"harmbench": "attack_success_rate", "xstest": "over_refusal_rate", "mmlu": "accuracy"}

    for alias, bench_values in metric_values.items():
        for benchmark, value in bench_values.items():
            summary_path = build_run_paths(tmp_path, alias, benchmark)["summary_json"]
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps({"metrics": {metric_key[benchmark]: value}}),
                encoding="utf-8",
            )

    pairwise = build_pairwise_report(config, tmp_path)
    labels = summarize_pair_labels(pairwise)

    assert any(row["pair_id"] == "qwen_small" and row["benchmark"] == "harmbench" for row in pairwise)
    assert any(row["pair_id"] == "qwen_small" for row in labels)

    scale = compute_scale_sensitivity(config, labels)
    assert scale["available"] is True

    cross = compute_cross_family_consistency(config, labels)
    assert "family_mean_deltas" in cross
