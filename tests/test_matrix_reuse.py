"""Tests for matrix runner model reuse behavior."""

from __future__ import annotations

from pathlib import Path

from ethical_benchmark.pipeline import run_quant_matrix as matrix_module
from ethical_benchmark.quant.config_schema import QuantizationConfig


class _DummyDecodingConfig:
    """Minimal decoding config replacement for matrix tests."""

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "_DummyDecodingConfig":
        return cls(**data)


class _DummyModelSpec:
    """Minimal model spec replacement."""

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _DummyLoader:
    """Dummy loader with load counter."""

    load_count = 0

    def __init__(self, device: str = "auto") -> None:
        _ = device

    def load(self, spec):  # noqa: ANN001
        _ = spec
        _DummyLoader.load_count += 1

        class _M:
            device = "cpu"

        class _T:
            pass

        return _M(), _T(), "cpu"


class _DummyGenerator:
    """Dummy generator placeholder."""

    def __init__(self, model, tokenizer, device, config):  # noqa: ANN001
        _ = (model, tokenizer, device, config)



def _dummy_stack():
    def _seed(seed: int) -> None:
        _ = seed

    return _DummyDecodingConfig, _DummyGenerator, _seed, _DummyLoader, _DummyModelSpec


def test_matrix_reuse_loads_once_per_model(monkeypatch, quant_config_dict: dict, tmp_path: Path) -> None:
    config = QuantizationConfig.model_validate(quant_config_dict)
    calls = []

    def _fake_loaded(**kwargs):  # noqa: ANN001
        calls.append((kwargs["model_alias"], kwargs["benchmark_name"]))
        return {"ok": True}

    monkeypatch.setattr(matrix_module, "get_model_stack", _dummy_stack)
    monkeypatch.setattr(matrix_module, "execute_quant_benchmark_loaded", _fake_loaded)

    _DummyLoader.load_count = 0
    matrix_module.run_quant_matrix(
        config=config,
        output_dir=tmp_path,
        seed=42,
        device="cpu",
        max_samples=5,
        batch_size=2,
        selected_models=["qwen_small_base"],
        benchmark_filter=None,
        resume=True,
        force_restart=False,
        reuse_loaded_model=True,
    )

    assert _DummyLoader.load_count == 1
    assert len(calls) == len(config.models["qwen_small_base"].benchmarks)


def test_matrix_no_reuse_calls_single_runner(monkeypatch, quant_config_dict: dict, tmp_path: Path) -> None:
    config = QuantizationConfig.model_validate(quant_config_dict)
    calls = []

    def _fake_single(**kwargs):  # noqa: ANN001
        calls.append((kwargs["model_alias"], kwargs["benchmark_name"]))
        return {"ok": True}

    monkeypatch.setattr(matrix_module, "execute_quant_benchmark", _fake_single)

    matrix_module.run_quant_matrix(
        config=config,
        output_dir=tmp_path,
        seed=42,
        device="cpu",
        max_samples=5,
        batch_size=2,
        selected_models=["qwen_small_base"],
        benchmark_filter=None,
        resume=True,
        force_restart=False,
        reuse_loaded_model=False,
    )

    assert len(calls) == len(config.models["qwen_small_base"].benchmarks)
