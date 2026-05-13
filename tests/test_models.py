"""Unit tests for model loading utilities and generation config."""

from __future__ import annotations

from typing import Any, Dict

import pytest

import ethical_benchmark.models.generation as generation_module
import ethical_benchmark.models.loader as loader_module
from ethical_benchmark.models.generation import DecodingConfig, set_global_seed
from ethical_benchmark.models.loader import HFModelLoader, ModelSpec, build_model_spec


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        return None


class _FakeCudnn:
    deterministic = False
    benchmark = True


class _FakeBackends:
    cudnn = _FakeCudnn()


class _FakeTorch:
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"
    cuda = _FakeCuda()
    backends = _FakeBackends()

    @staticmethod
    def manual_seed(seed: int) -> None:
        return None


@pytest.fixture()
def fake_torch(monkeypatch):
    monkeypatch.setattr(loader_module, "_get_torch", lambda: _FakeTorch)
    monkeypatch.setattr(generation_module, "_get_torch", lambda: _FakeTorch)
    return _FakeTorch


# =========================================================================
# ModelSpec
# =========================================================================


class TestModelSpec:
    """Tests for ``ModelSpec`` dataclass."""

    def test_defaults(self) -> None:
        spec = ModelSpec(alias="test", hf_id="owner/model")
        assert spec.trust_remote_code is False
        assert spec.dtype == "auto"
        assert spec.revision is None
        assert spec.quantized is False

    def test_immutability(self) -> None:
        spec = ModelSpec(alias="test", hf_id="owner/model")
        with pytest.raises(AttributeError):
            spec.alias = "changed"  # type: ignore[misc]


# =========================================================================
# build_model_spec
# =========================================================================


class TestBuildModelSpec:
    """Tests for ``build_model_spec``."""

    def test_valid_alias(self) -> None:
        registry = {
            "my-model": {
                "hf_id": "org/my-model",
                "trust_remote_code": True,
                "dtype": "float16",
            }
        }
        spec = build_model_spec("my-model", registry)
        assert spec.hf_id == "org/my-model"
        assert spec.trust_remote_code is True
        assert spec.dtype == "float16"
        assert spec.quantized is False

    def test_quantized_flag_is_propagated(self) -> None:
        registry = {
            "my-4bit": {
                "hf_id": "org/my-4bit",
                "quantized": True,
            }
        }
        spec = build_model_spec("my-4bit", registry)
        assert spec.quantized is True

    def test_unknown_alias_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown model alias"):
            build_model_spec("missing", {"other": {"hf_id": "x"}})


# =========================================================================
# DecodingConfig
# =========================================================================


class TestDecodingConfig:
    """Tests for ``DecodingConfig``."""

    def test_from_dict_defaults(self) -> None:
        cfg = DecodingConfig.from_dict({})
        assert cfg.max_new_tokens == 128
        assert cfg.temperature == 0.0
        assert cfg.use_chat_template is True

    def test_from_dict_custom_values(self) -> None:
        cfg = DecodingConfig.from_dict(
            {"max_new_tokens": 256, "temperature": 0.7, "top_p": 0.9}
        )
        assert cfg.max_new_tokens == 256
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.9


# =========================================================================
# Device resolution
# =========================================================================


class TestDeviceResolution:
    """Tests for HFModelLoader device resolution."""

    def test_cpu_forced(self, fake_torch) -> None:
        loader = HFModelLoader(device="cpu")
        assert loader._resolve_runtime_device() == "cpu"

    def test_auto_resolves(self, fake_torch) -> None:
        loader = HFModelLoader(device="auto")
        device = loader._resolve_runtime_device()
        assert device in ("cpu", "cuda")


# =========================================================================
# Dtype resolution
# =========================================================================


class TestDtypeResolution:
    """Tests for ``HFModelLoader._resolve_dtype``."""

    def test_auto_cpu(self, fake_torch) -> None:
        result = HFModelLoader._resolve_dtype("auto", "cpu")
        assert result == fake_torch.float32

    def test_explicit_float16(self, fake_torch) -> None:
        result = HFModelLoader._resolve_dtype("float16", "cpu")
        assert result == fake_torch.float16

    def test_alias_fp16(self, fake_torch) -> None:
        result = HFModelLoader._resolve_dtype("fp16", "cpu")
        assert result == fake_torch.float16

    def test_bfloat16(self, fake_torch) -> None:
        result = HFModelLoader._resolve_dtype("bfloat16", "cpu")
        assert result == fake_torch.bfloat16

    def test_unsupported_raises(self, fake_torch) -> None:
        with pytest.raises(ValueError, match="Unsupported dtype"):
            HFModelLoader._resolve_dtype("int8", "cpu")


# =========================================================================
# Global seed
# =========================================================================


class TestSetGlobalSeed:
    """Tests for ``set_global_seed``."""

    def test_does_not_raise(self, fake_torch) -> None:
        set_global_seed(123)  # should complete without error

    def test_determinism(self, fake_torch) -> None:
        import random

        set_global_seed(42)
        a = random.random()
        set_global_seed(42)
        b = random.random()
        assert a == b
