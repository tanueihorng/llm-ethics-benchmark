"""Validate the live study configs and the attn_implementation schema field.

These tests guard the T26 invariant: the two new pairs (mistral_7b, phi4_mini)
must load the real configs cleanly and carry the exact per-family flags, while
the existing pairs stay untouched. They load configs/tc1.yaml and
configs/default.yaml directly (no other test does), so a hand-edit that drops a
field or a pair fails loudly here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ethical_benchmark.quant.config_schema import ModelEntry, load_quant_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = [ROOT / "configs" / "tc1.yaml", ROOT / "configs" / "default.yaml"]

EXPECTED_BENCHMARKS = ["harmbench", "xstest", "mmlu", "arc"]
NEW_ALIASES = ["mistral_7b_base", "mistral_7b_4bit", "phi4_mini_base", "phi4_mini_4bit"]


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda p: p.name)
class TestLiveConfigShape:
    """The shipped configs must describe exactly 5 pairs across 4 families."""

    def test_ten_models_five_pairs_four_families(self, config_path: Path) -> None:
        config = load_quant_config(str(config_path))
        assert len(config.models) == 10
        pairs = {entry.pair_id for entry in config.models.values()}
        families = {entry.family for entry in config.models.values()}
        assert pairs == {"qwen_2b", "qwen_4b", "llama_3_2_3b", "mistral_7b", "phi4_mini"}
        assert families == {"qwen", "llama", "mistral", "phi"}

    def test_phi_flags(self, config_path: Path) -> None:
        config = load_quant_config(str(config_path))
        for alias in ("phi4_mini_base", "phi4_mini_4bit"):
            entry = config.models[alias]
            assert entry.model_id == "microsoft/Phi-4-mini-instruct"
            assert entry.trust_remote_code is True
            assert entry.attn_implementation == "eager"

    def test_mistral_flags(self, config_path: Path) -> None:
        config = load_quant_config(str(config_path))
        for alias in ("mistral_7b_base", "mistral_7b_4bit"):
            entry = config.models[alias]
            assert entry.model_id == "mistralai/Mistral-7B-Instruct-v0.3"
            assert entry.trust_remote_code is False
            assert entry.attn_implementation is None

    def test_new_pairs_share_methodology(self, config_path: Path) -> None:
        config = load_quant_config(str(config_path))
        for alias in NEW_ALIASES:
            entry = config.models[alias]
            assert entry.dtype == "auto"
            assert entry.benchmarks == EXPECTED_BENCHMARKS

    def test_each_new_pair_has_one_baseline_one_quantized(self, config_path: Path) -> None:
        config = load_quant_config(str(config_path))
        for pair_id in ("mistral_7b", "phi4_mini"):
            members = [e for e in config.models.values() if e.pair_id == pair_id]
            assert sum(m.quantized for m in members) == 1
            assert sum(not m.quantized for m in members) == 1
            assert len({m.model_id for m in members}) == 1  # both share one model_id

    def test_existing_models_unchanged(self, config_path: Path) -> None:
        config = load_quant_config(str(config_path))
        for alias in ("qwen_2b_base", "qwen_4b_4bit", "llama_3_2_3b_base", "llama_3_2_3b_4bit"):
            entry = config.models[alias]
            assert entry.trust_remote_code is False
            assert entry.attn_implementation is None
            assert entry.dtype == "auto"


class TestAttnImplementationField:
    """The schema field accepts the known backends and rejects typos."""

    def _entry(self, **overrides):
        base = dict(
            family="phi",
            size_b=3.8,
            quantized=False,
            pair_id="phi4_mini",
            model_id="microsoft/Phi-4-mini-instruct",
            benchmarks=["harmbench"],
        )
        base.update(overrides)
        return ModelEntry(**base)

    def test_defaults_none(self) -> None:
        assert self._entry().attn_implementation is None

    def test_accepts_eager(self) -> None:
        assert self._entry(attn_implementation="eager").attn_implementation == "eager"

    def test_normalizes_case(self) -> None:
        assert self._entry(attn_implementation="EAGER").attn_implementation == "eager"

    def test_rejects_typo(self) -> None:
        with pytest.raises(ValueError, match="attn_implementation must be one of"):
            self._entry(attn_implementation="eagerr")
