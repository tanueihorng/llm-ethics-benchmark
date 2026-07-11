"""Tests for the INT8 precision point (T29): quant_method field, loader branch,
8-bit detection, and the configs/tc1_int8.yaml sweep config.

GPU-free and bitsandbytes-free: the loader's int8/nf4 selection is tested by
monkeypatching the two BitsAndBytesConfig builders to sentinels, so neither a
CUDA device nor a bitsandbytes install is required. The backward-compatibility
invariant (quant_method=None still selects the NF4 path, byte-identical to
before this field existed) is asserted explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

import ethical_benchmark.models.loader as loader_module
from ethical_benchmark.models.loader import HFModelLoader, ModelSpec, build_model_spec
from ethical_benchmark.quant.config_schema import ModelEntry, load_quant_config

ROOT = Path(__file__).resolve().parent.parent
INT8_CONFIG = ROOT / "configs" / "tc1_int8.yaml"


# --------------------------------------------------------------------------
# Schema: quant_method validation + consistency
# --------------------------------------------------------------------------
class TestQuantMethodSchema:
    def _entry(self, **kw: Any) -> Dict[str, Any]:
        base = dict(family="qwen", size_b=1.7, quantized=True, pair_id="p",
                    model_id="x/y", benchmarks=["harmbench"])
        base.update(kw)
        return base

    def test_accepts_int8_and_nf4_and_none(self) -> None:
        assert ModelEntry(**self._entry(quant_method="int8")).quant_method == "int8"
        assert ModelEntry(**self._entry(quant_method="NF4")).quant_method == "nf4"  # normalised
        assert ModelEntry(**self._entry()).quant_method is None

    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(ValueError):
            ModelEntry(**self._entry(quant_method="fp8"))

    def test_rejects_quant_method_on_baseline(self) -> None:
        # quant_method only makes sense on a quantized member.
        with pytest.raises(ValueError):
            ModelEntry(**self._entry(quantized=False, quant_method="int8"))


# --------------------------------------------------------------------------
# ModelSpec + build_model_spec threading
# --------------------------------------------------------------------------
class TestModelSpecQuantMethod:
    def test_defaults_to_none(self) -> None:
        assert ModelSpec(alias="t", hf_id="o/m").quant_method is None

    def test_is_settable(self) -> None:
        assert ModelSpec(alias="t", hf_id="o/m", quantized=True,
                         quant_method="int8").quant_method == "int8"

    def test_build_model_spec_propagates(self) -> None:
        reg = {"m": {"hf_id": "x", "quantized": True, "quant_method": "int8"}}
        assert build_model_spec("m", reg).quant_method == "int8"

    def test_build_model_spec_defaults_none(self) -> None:
        assert build_model_spec("m", {"m": {"hf_id": "x"}}).quant_method is None


# --------------------------------------------------------------------------
# 8-bit detection in _quantization_active (the silent-fp16 guard)
# --------------------------------------------------------------------------
class TestEightBitDetection:
    def test_is_loaded_in_8bit_detected(self) -> None:
        model = type("M", (), {"is_loaded_in_8bit": True})()
        assert loader_module._quantization_active(model) is True

    def test_plain_model_not_active(self) -> None:
        assert loader_module._quantization_active(type("M", (), {})()) is False


# --------------------------------------------------------------------------
# The silent-fp16 RAISE path (_require_quantization_engaged) — the guard that
# refuses to emit fp16 results mislabelled as quantized.
# --------------------------------------------------------------------------
class TestQuantizationGuard:
    @staticmethod
    def _spec(quantized: bool, method=None):
        return type("S", (), {"quantized": quantized, "quant_method": method, "alias": "x"})()

    def test_raises_when_quantized_but_loaded_fp16(self) -> None:
        fp16_model = type("M", (), {})()  # no is_loaded_in_* / hf_quantizer attrs
        with pytest.raises(RuntimeError, match="no quantization is active"):
            loader_module._require_quantization_engaged(self._spec(True, "int8"), fp16_model)

    def test_raise_message_names_the_method(self) -> None:
        with pytest.raises(RuntimeError, match="int8 quantization"):
            loader_module._require_quantization_engaged(self._spec(True, "int8"), type("M", (), {})())

    def test_no_raise_when_quantization_active(self) -> None:
        active = type("M", (), {"is_loaded_in_8bit": True})()
        loader_module._require_quantization_engaged(self._spec(True, "int8"), active)  # must not raise

    def test_baseline_fp16_does_not_raise(self) -> None:
        # Baselines load in fp16 with quantized=False — the guard must stay silent.
        loader_module._require_quantization_engaged(self._spec(False), type("M", (), {})())


# --------------------------------------------------------------------------
# Loader branch: int8 -> 8-bit config; None/nf4 -> 4-bit config
# --------------------------------------------------------------------------
class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        return None


class _Torch:
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"
    cuda = _Cuda()
    backends = type("B", (), {"cudnn": type("C", (), {"deterministic": False, "benchmark": True})()})()

    @staticmethod
    def manual_seed(seed: int) -> None:
        return None


class _Loaded:
    def to(self, device: str) -> "_Loaded":
        return self

    def eval(self) -> "_Loaded":
        return self


class _Tok:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0


class _FakeTokClass:
    @staticmethod
    def from_pretrained(hf_id: str, **kwargs: Any) -> _Tok:
        return _Tok()


class _CapturingModel:
    captured: Dict[str, Any] = {}

    @classmethod
    def from_pretrained(cls, hf_id: str, **kwargs: Any) -> _Loaded:
        cls.captured = dict(kwargs)
        return _Loaded()


class TestLoaderQuantBranch:
    def _patch(self, monkeypatch) -> None:
        _CapturingModel.captured = {}
        monkeypatch.setattr(loader_module, "_get_torch", lambda: _Torch)
        monkeypatch.setattr(loader_module, "_get_transformer_classes",
                            lambda: (_CapturingModel, _FakeTokClass))
        monkeypatch.setattr(loader_module, "_build_bnb_8bit_config", lambda: "INT8_CFG")
        monkeypatch.setattr(loader_module, "_build_bnb_4bit_config", lambda dt: "NF4_CFG")
        # the silent-fp16 guard would otherwise reject our sentinel-loaded model
        monkeypatch.setattr(loader_module, "_quantization_active", lambda m: True)

    def test_int8_selects_8bit_config(self, monkeypatch) -> None:
        self._patch(monkeypatch)
        loader = HFModelLoader(device="cuda")
        loader.load(ModelSpec(alias="qwen_2b_8bit", hf_id="Qwen/Qwen3-1.7B",
                              quantized=True, quant_method="int8"))
        assert _CapturingModel.captured.get("quantization_config") == "INT8_CFG"

    def test_none_method_selects_nf4_config(self, monkeypatch) -> None:
        # Backward-compat: existing entries (quant_method unset) still get NF4.
        self._patch(monkeypatch)
        loader = HFModelLoader(device="cuda")
        loader.load(ModelSpec(alias="qwen_2b_4bit", hf_id="Qwen/Qwen3-1.7B", quantized=True))
        assert _CapturingModel.captured.get("quantization_config") == "NF4_CFG"

    def test_explicit_nf4_selects_nf4_config(self, monkeypatch) -> None:
        self._patch(monkeypatch)
        loader = HFModelLoader(device="cuda")
        loader.load(ModelSpec(alias="m", hf_id="x/y", quantized=True, quant_method="nf4"))
        assert _CapturingModel.captured.get("quantization_config") == "NF4_CFG"

    def test_unquantized_has_no_quant_config(self, monkeypatch) -> None:
        self._patch(monkeypatch)
        loader = HFModelLoader(device="cuda")
        loader.load(ModelSpec(alias="b", hf_id="x/y", quantized=False))
        assert "quantization_config" not in _CapturingModel.captured


# --------------------------------------------------------------------------
# The sweep config loads and is well-formed
# --------------------------------------------------------------------------
class TestInt8Config:
    def test_loads_five_pairs_with_int8_members(self) -> None:
        cfg = load_quant_config(str(INT8_CONFIG))
        assert len(cfg.models) == 10  # 5 fp16 baselines + 5 int8
        pairs = {e.pair_id for e in cfg.models.values()}
        assert pairs == {"qwen_2b", "qwen_4b", "llama_3_2_3b", "mistral_7b", "phi4_mini"}

        # Audit P1-3: the INT8 precision comparison must be reproducibly tied to
        # the same checkpoint snapshots as the NF4 study. Every INT8 config member
        # pins a revision, both members of a pair share model_id + revision, and
        # the 512 INT8 revisions equal the NF4 (tc1_512.yaml) revisions per model.
        nf4_rev = {
            e.model_id: e.revision
            for e in load_quant_config(str(ROOT / "configs" / "tc1_512.yaml")).models.values()
        }
        for name in ("tc1_int8.yaml", "tc1_int8_512.yaml"):
            icfg = load_quant_config(str(ROOT / "configs" / name))
            by_pair: dict = {}
            for entry in icfg.models.values():
                assert entry.revision, f"{name}:{entry.model_id} has no pinned revision"
                by_pair.setdefault(entry.pair_id, []).append((entry.model_id, entry.revision))
            for pair_id, members in by_pair.items():
                assert len({m[0] for m in members}) == 1, f"{name}:{pair_id} model_id mismatch"
                assert len({m[1] for m in members}) == 1, f"{name}:{pair_id} revision mismatch"
            if name == "tc1_int8_512.yaml":
                for entry in icfg.models.values():
                    assert entry.revision == nf4_rev[entry.model_id], (
                        f"{name}:{entry.model_id} revision differs from the NF4 study"
                    )

    def test_int8_members_have_method_and_baselines_do_not(self) -> None:
        cfg = load_quant_config(str(INT8_CONFIG))
        int8 = [a for a, e in cfg.models.items() if e.quantized]
        base = [a for a, e in cfg.models.items() if not e.quantized]
        assert len(int8) == 5 and len(base) == 5
        for a in int8:
            assert a.endswith("_8bit")
            assert cfg.models[a].quant_method == "int8"
        for a in base:
            assert cfg.models[a].quant_method is None

    def test_phi_int8_keeps_eager(self) -> None:
        cfg = load_quant_config(str(INT8_CONFIG))
        assert cfg.models["phi4_mini_8bit"].attn_implementation == "eager"
