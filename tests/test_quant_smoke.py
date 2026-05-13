"""Smoke tests for quantization benchmark execution pipeline."""

from __future__ import annotations

import random
from pathlib import Path

from ethical_benchmark.benchmarks.base import BenchmarkItem
from ethical_benchmark.metrics.aggregate import read_jsonl
from ethical_benchmark.pipeline import run_quant_benchmark as quant_runner
from ethical_benchmark.quant.config_schema import QuantizationConfig


class _DummyLoader:
    """Dummy HF loader used in smoke tests."""

    def __init__(self, device: str = "auto") -> None:
        self.device = device

    def load(self, spec):
        class _DummyModel:
            device = "cpu"

        class _DummyTokenizer:
            pass

        return _DummyModel(), _DummyTokenizer(), "cpu"


class _DummyDecodingConfig:
    """Minimal decoding config replacement for tests."""

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "_DummyDecodingConfig":
        return cls(**data)


class _DummyModelSpec:
    """Minimal model spec replacement for tests."""

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _DummyGenerator:
    """Dummy text generator used in smoke tests."""

    def __init__(self, model, tokenizer, device, config) -> None:  # noqa: ANN001
        _ = (model, tokenizer, device, config)

    def generate_batch(self, prompts):
        return [f"response::{idx}" for idx, _ in enumerate(prompts)]


class _DummyPlugin:
    """Dummy benchmark plugin with deterministic behavior."""

    def __init__(self, name: str) -> None:
        self.name = name

    def load_items(self, max_samples, seed):
        items = [
            BenchmarkItem(prompt_id="p0", prompt_text="prompt 0", metadata={"gold_index": 0, "num_choices": 4}),
            BenchmarkItem(prompt_id="p1", prompt_text="prompt 1", metadata={"gold_index": 1, "num_choices": 4}),
            BenchmarkItem(prompt_id="p2", prompt_text="prompt 2", metadata={"gold_index": 2, "num_choices": 4}),
        ]
        random.Random(seed).shuffle(items)
        if max_samples is None:
            return items
        return items[:max_samples]

    def build_prompt(self, item):
        return item.prompt_text

    def score_response(self, item, response):
        _ = response
        if self.name == "harmbench":
            harmful = item.prompt_id == "p1"
            return {
                "harmful_compliance": harmful,
                "safe_refusal": not harmful,
                "malformed": False,
            }
        if self.name == "xstest":
            refusal = item.prompt_id == "p1"
            return {
                "is_refusal": refusal,
                "benign_answer": not refusal,
                "malformed": False,
                "benign": True,
            }
        return {
            "predicted_index": item.metadata["gold_index"],
            "gold_index": item.metadata["gold_index"],
            "is_answered": True,
            "is_correct": True,
            "subject": "dummy",
        }

    def aggregate(self, records):
        total = len(records)
        if self.name == "harmbench":
            harmful = sum(int(item["score_fields"]["harmful_compliance"]) for item in records)
            refusal = sum(int(item["score_fields"]["safe_refusal"]) for item in records)
            return {
                "num_samples": total,
                "harmful_compliance_count": harmful,
                "safe_refusal_count": refusal,
                "malformed_output_count": 0,
                "attack_success_rate": harmful / total if total else None,
                "safe_refusal_rate": refusal / total if total else None,
                "malformed_rate": 0.0,
            }
        if self.name == "xstest":
            refusals = sum(int(item["score_fields"]["is_refusal"]) for item in records)
            benign = sum(int(item["score_fields"]["benign_answer"]) for item in records)
            return {
                "num_samples": total,
                "benign_prompt_count": total,
                "over_refusal_count": refusals,
                "benign_answer_count": benign,
                "malformed_output_count": 0,
                "over_refusal_rate": refusals / total if total else None,
                "benign_answer_rate": benign / total if total else None,
                "malformed_rate": 0.0,
            }
        correct = sum(int(item["score_fields"]["is_correct"]) for item in records)
        return {
            "num_samples": total,
            "answered_count": total,
            "correct_count": correct,
            "answered_rate": 1.0,
            "accuracy": correct / total if total else None,
            "subject_breakdown": {"dummy": {"num_samples": total, "answered_rate": 1.0, "accuracy": 1.0}},
        }


def _patched_plugin_factory(name, config):  # noqa: ANN001
    _ = config
    return _DummyPlugin(name)


def _patched_model_stack():
    """Provides model stack replacements without importing torch/transformers."""

    def _seed(seed: int) -> None:
        random.seed(seed)

    return _DummyDecodingConfig, _DummyGenerator, _seed, _DummyLoader, _DummyModelSpec


def test_smoke_each_benchmark(tmp_path: Path, quant_config_dict: dict, monkeypatch) -> None:
    config = QuantizationConfig.model_validate(quant_config_dict)

    monkeypatch.setattr(quant_runner, "_model_stack", _patched_model_stack)
    monkeypatch.setattr(quant_runner, "build_benchmark_plugin", _patched_plugin_factory)

    for benchmark in ["harmbench", "xstest", "mmlu"]:
        summary = quant_runner.execute_quant_benchmark(
            config=config,
            model_alias="qwen_small_base",
            benchmark_name=benchmark,
            output_dir=tmp_path,
            seed=42,
            device="cpu",
            resume=True,
            force_restart=True,
            max_samples_override=3,
            batch_size_override=2,
        )

        run_paths = quant_runner.build_run_paths(tmp_path, "qwen_small_base", benchmark)
        assert run_paths["raw_path"].exists()
        assert run_paths["summary_json"].exists()
        assert summary["benchmark"] == benchmark
        assert summary["num_records"] == 3


def test_reproducible_order_same_seed(tmp_path: Path, quant_config_dict: dict, monkeypatch) -> None:
    config = QuantizationConfig.model_validate(quant_config_dict)

    monkeypatch.setattr(quant_runner, "_model_stack", _patched_model_stack)
    monkeypatch.setattr(quant_runner, "build_benchmark_plugin", _patched_plugin_factory)

    output_a = tmp_path / "run_a"
    output_b = tmp_path / "run_b"

    quant_runner.execute_quant_benchmark(
        config=config,
        model_alias="qwen_small_base",
        benchmark_name="harmbench",
        output_dir=output_a,
        seed=123,
        device="cpu",
        resume=False,
        force_restart=True,
        max_samples_override=3,
        batch_size_override=2,
    )

    quant_runner.execute_quant_benchmark(
        config=config,
        model_alias="qwen_small_base",
        benchmark_name="harmbench",
        output_dir=output_b,
        seed=123,
        device="cpu",
        resume=False,
        force_restart=True,
        max_samples_override=3,
        batch_size_override=2,
    )

    raw_a = read_jsonl(quant_runner.build_run_paths(output_a, "qwen_small_base", "harmbench")["raw_path"])
    raw_b = read_jsonl(quant_runner.build_run_paths(output_b, "qwen_small_base", "harmbench")["raw_path"])

    assert [row["prompt_id"] for row in raw_a] == [row["prompt_id"] for row in raw_b]
