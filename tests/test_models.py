"""Unit tests for model loading utilities and generation config."""

from __future__ import annotations

from typing import Any, Dict

import pytest

import ethical_benchmark.models.generation as generation_module
import ethical_benchmark.models.loader as loader_module
from ethical_benchmark.models.generation import DecodingConfig, TextGenerator, set_global_seed
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


class _FakeModel:
    device = "cpu"


class _ThinkingAwareTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "</s>"

    def __init__(self) -> None:
        self.eos_token_id = 1
        self.kwargs: Dict[str, Any] | None = None
        self.padding_side = "right"

    def apply_chat_template(self, messages: list, **kwargs: Any) -> str:
        self.kwargs = kwargs
        return messages[0]["content"]


class _LegacyTokenizer(_ThinkingAwareTokenizer):
    def apply_chat_template(self, messages: list, **kwargs: Any) -> str:
        if "enable_thinking" in kwargs:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        self.kwargs = kwargs
        return messages[0]["content"]


class TestPromptFormatting:
    def test_disables_thinking_when_template_supports_it(self) -> None:
        tokenizer = _ThinkingAwareTokenizer()
        generator = TextGenerator(
            model=_FakeModel(),
            tokenizer=tokenizer,
            device="cpu",
            config=DecodingConfig(),
        )

        assert generator._format_prompt("hello") == "hello"
        assert tokenizer.kwargs is not None
        assert tokenizer.kwargs["enable_thinking"] is False

    def test_falls_back_for_legacy_chat_templates(self) -> None:
        tokenizer = _LegacyTokenizer()
        generator = TextGenerator(
            model=_FakeModel(),
            tokenizer=tokenizer,
            device="cpu",
            config=DecodingConfig(),
        )

        assert generator._format_prompt("hello") == "hello"
        assert tokenizer.kwargs is not None
        assert "enable_thinking" not in tokenizer.kwargs


class _WrappingTokenizer(_ThinkingAwareTokenizer):
    """A tokenizer whose chat template actually transforms the prompt."""

    def apply_chat_template(self, messages: list, **kwargs: Any) -> str:
        self.kwargs = kwargs
        return f"<|user|>{messages[0]['content']}<|assistant|>"


class TestPromptProvenance:
    """prompt_was_templated records whether the model saw a templated prompt (audit M11)."""

    def test_flags_a_genuinely_templated_prompt(self) -> None:
        generator = TextGenerator(
            model=_FakeModel(), tokenizer=_WrappingTokenizer(), device="cpu", config=DecodingConfig(),
        )
        assert generator.prompt_was_templated("hello") is True

    def test_not_flagged_when_templating_disabled(self) -> None:
        generator = TextGenerator(
            model=_FakeModel(), tokenizer=_WrappingTokenizer(), device="cpu",
            config=DecodingConfig(use_chat_template=False),
        )
        assert generator.prompt_was_templated("hello") is False


# =========================================================================
# Quantization-active guard (refuse fp16-mislabelled-as-4bit)
# =========================================================================


class _FakeModel:
    """Minimal stand-in carrying whichever quantization signals a test sets."""

    def __init__(self, **attrs: Any) -> None:
        for key, value in attrs.items():
            setattr(self, key, value)


class TestQuantizationActive:
    """`_quantization_active` must detect any of the version-dependent signals."""

    def test_is_loaded_in_4bit(self) -> None:
        assert loader_module._quantization_active(_FakeModel(is_loaded_in_4bit=True)) is True

    def test_is_quantized(self) -> None:
        assert loader_module._quantization_active(_FakeModel(is_quantized=True)) is True

    def test_hf_quantizer_present(self) -> None:
        assert loader_module._quantization_active(_FakeModel(hf_quantizer=object())) is True

    def test_no_signal_is_false(self) -> None:
        # A plain fp16 model exposes none of the signals -> not quantized.
        assert loader_module._quantization_active(_FakeModel()) is False
        assert (
            loader_module._quantization_active(
                _FakeModel(is_loaded_in_4bit=False, is_quantized=False, hf_quantizer=None)
            )
            is False
        )


# =========================================================================
# attn_implementation threading (T26: Phi-4-mini needs eager on the V100)
# =========================================================================


class _LoadedFakeModel:
    """A loaded model with NO quantization signals (plain fp16/fp32 path)."""

    def to(self, device: str) -> "_LoadedFakeModel":
        return self

    def eval(self) -> "_LoadedFakeModel":
        return self


class _FakeTokenizerInstance:
    pad_token = "<pad>"
    eos_token = "</s>"


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(hf_id: str, **kwargs: Any) -> _FakeTokenizerInstance:
        return _FakeTokenizerInstance()


class _CapturingAutoModel:
    """Captures the kwargs HFModelLoader passes to from_pretrained."""

    captured_kwargs: Dict[str, Any] = {}
    captured_hf_id: str | None = None

    @classmethod
    def from_pretrained(cls, hf_id: str, **kwargs: Any) -> _LoadedFakeModel:
        cls.captured_hf_id = hf_id
        cls.captured_kwargs = dict(kwargs)
        return _LoadedFakeModel()


class TestModelSpecAttnImplementation:
    """`ModelSpec.attn_implementation` field behaviour."""

    def test_defaults_to_none(self) -> None:
        spec = ModelSpec(alias="t", hf_id="o/m")
        assert spec.attn_implementation is None

    def test_is_settable(self) -> None:
        spec = ModelSpec(alias="t", hf_id="o/m", attn_implementation="eager")
        assert spec.attn_implementation == "eager"

    def test_build_model_spec_propagates(self) -> None:
        registry = {"phi": {"hf_id": "microsoft/Phi-4-mini-instruct", "attn_implementation": "eager"}}
        assert build_model_spec("phi", registry).attn_implementation == "eager"

    def test_build_model_spec_defaults_none(self) -> None:
        assert build_model_spec("m", {"m": {"hf_id": "x"}}).attn_implementation is None


class TestAttnImplementationLoading:
    """The load path must inject attn_implementation only when set.

    This is the guard that keeps the *existing* models' from_pretrained call
    byte-identical to before the field existed (fairness invariant), while
    letting Phi-4-mini request the eager backend on the V100.
    """

    def _patch(self, monkeypatch) -> None:
        _CapturingAutoModel.captured_kwargs = {}
        _CapturingAutoModel.captured_hf_id = None
        monkeypatch.setattr(
            loader_module,
            "_get_transformer_classes",
            lambda: (_CapturingAutoModel, _FakeAutoTokenizer),
        )

    def test_injected_when_set(self, fake_torch, monkeypatch) -> None:
        self._patch(monkeypatch)
        loader = HFModelLoader(device="cpu")
        spec = ModelSpec(
            alias="phi4_mini_base",
            hf_id="microsoft/Phi-4-mini-instruct",
            attn_implementation="eager",
        )
        loader.load(spec)
        assert _CapturingAutoModel.captured_kwargs.get("attn_implementation") == "eager"

    def test_omitted_when_none(self, fake_torch, monkeypatch) -> None:
        self._patch(monkeypatch)
        loader = HFModelLoader(device="cpu")
        spec = ModelSpec(alias="qwen_2b_base", hf_id="Qwen/Qwen3-1.7B")  # no attn field
        loader.load(spec)
        # The key must be entirely ABSENT, not None — proves the existing
        # models' load call is unchanged by this feature.
        assert "attn_implementation" not in _CapturingAutoModel.captured_kwargs
