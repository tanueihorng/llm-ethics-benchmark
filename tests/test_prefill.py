"""Tests for the T87/D105 prefill arm: generator prefill, config and analysis inputs."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Dict, List

import pytest

import ethical_benchmark.models.generation as generation_module
from ethical_benchmark.models.generation import DecodingConfig, TextGenerator
from ethical_benchmark.quant.config_schema import load_quant_config

ROOT = Path(__file__).resolve().parents[1]
PREFILL = "Sure, here is"


class _TemplateTokenizer:
    """Minimal chat-template tokenizer. ``style`` mimics three real templates."""

    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "</s>"
    eos_token_id = 1

    def __init__(self, style: str) -> None:
        self.style = style
        self.padding_side = "right"

    def apply_chat_template(self, messages: list, tokenize: bool = False,
                            add_generation_prompt: bool = False, **kwargs: Any) -> str:
        user = messages[0]["content"]
        reply = messages[1]["content"] if len(messages) > 1 else None
        if self.style == "phi":
            head, sep, end = f"<|user|>{user}<|end|><|assistant|>", "", "<|end|>"
        elif self.style == "mistral":
            head, sep, end = f"<s>[INST] {user}[/INST]", " ", "</s>"
        elif self.style == "qwen":
            head = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            sep, end = "", "<|im_end|>\n"
        elif self.style == "rewrites_context":
            head, sep, end = f"<|user|>{user}<|assistant|>", "", ""
            if reply is not None:
                return f"<|system|><|user|>{user}<|assistant|>{reply}"
        else:
            raise AssertionError(self.style)
        if reply is None:
            return head if add_generation_prompt else head.rsplit("<", 1)[0]
        return head + sep + reply + end


def _generator(style: str, **cfg: Any) -> TextGenerator:
    return TextGenerator(model=object(), tokenizer=_TemplateTokenizer(style), device="cpu",
                         config=DecodingConfig(**cfg))


class TestDecodingConfigPrefill:
    def test_default_is_none(self) -> None:
        assert DecodingConfig().assistant_prefill is None
        assert DecodingConfig.from_dict({}).assistant_prefill is None

    def test_from_dict_reads_value_and_blank_means_none(self) -> None:
        assert DecodingConfig.from_dict({"assistant_prefill": PREFILL}).assistant_prefill == PREFILL
        assert DecodingConfig.from_dict({"assistant_prefill": ""}).assistant_prefill is None


class TestPrefillPlacement:
    @pytest.mark.parametrize("style,expected_tail", [
        ("phi", "<|assistant|>Sure, here is"),
        ("mistral", "[/INST] Sure, here is"),
        ("qwen", "<think>\n\n</think>\n\nSure, here is"),
    ])
    def test_prefill_sits_where_the_reply_begins(self, style: str, expected_tail: str) -> None:
        plain = _generator(style)._format_prompt("Q")
        prefilled = _generator(style, assistant_prefill=PREFILL)._format_prompt("Q")
        assert prefilled.endswith(expected_tail)
        assert prefilled.startswith(plain)

    def test_no_prefill_leaves_prompt_unchanged(self) -> None:
        gen = _generator("phi")
        assert gen._format_prompt("Q") == "<|user|>Q<|end|><|assistant|>"

    def test_refuses_without_chat_template(self) -> None:
        gen = _generator("phi", use_chat_template=False, assistant_prefill=PREFILL)
        with pytest.raises(RuntimeError, match="requires an applied chat template"):
            gen._format_prompt("Q")

    def test_refuses_when_template_rewrites_earlier_context(self) -> None:
        gen = _generator("rewrites_context", assistant_prefill=PREFILL)
        with pytest.raises(RuntimeError, match="does not extend"):
            gen._format_prompt("Q")

    def test_templated_provenance_ignores_prefill(self) -> None:
        assert _generator("phi", assistant_prefill=PREFILL).prompt_was_templated("Q") is True
        assert _generator("phi", use_chat_template=False).prompt_was_templated("Q") is False


class _Tensor:
    def __init__(self, rows: List[List[int]]) -> None:
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def to(self, device: Any) -> "_Tensor":
        return self


class _BatchTokenizer(_TemplateTokenizer):
    def __call__(self, prompts: List[str], **kwargs: Any) -> Dict[str, _Tensor]:
        self.seen = prompts
        return {"input_ids": _Tensor([[5, 6, 7] for _ in prompts])}

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return " a harmless placeholder continuation.\n"


class _Model:
    device = "cpu"

    def generate(self, input_ids: _Tensor, **kwargs: Any) -> List[List[int]]:
        return [row + [9, 9] for row in input_ids.rows]


class _NoGradTorch:
    @staticmethod
    def no_grad() -> contextlib.AbstractContextManager:
        return contextlib.nullcontext()


def test_response_is_prefill_plus_continuation(monkeypatch) -> None:
    monkeypatch.setattr(generation_module, "_get_torch", lambda: _NoGradTorch)
    tok = _BatchTokenizer("mistral")
    gen = TextGenerator(model=_Model(), tokenizer=tok, device="cpu",
                        config=DecodingConfig(assistant_prefill=PREFILL))
    out = gen.generate_batch(["Q1", "Q2"])
    assert out == ["Sure, here is a harmless placeholder continuation."] * 2
    assert all(p.endswith("[/INST] Sure, here is") for p in tok.seen)


def test_response_without_prefill_is_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(generation_module, "_get_torch", lambda: _NoGradTorch)
    gen = TextGenerator(model=_Model(), tokenizer=_BatchTokenizer("phi"), device="cpu",
                        config=DecodingConfig())
    assert gen.generate_batch(["Q"]) == ["a harmless placeholder continuation."]


class TestPrefillConfig:
    """configs/tc1_prefill_512.yaml must match the primary study except the prefill."""

    def test_matches_primary_except_prefill_and_benchmarks(self) -> None:
        pre = load_quant_config(str(ROOT / "configs" / "tc1_prefill_512.yaml"))
        primary = load_quant_config(str(ROOT / "configs" / "tc1_512.yaml"))
        assert set(pre.models) == set(primary.models)
        for alias, entry in pre.models.items():
            a, b = entry.model_dump(), primary.models[alias].model_dump()
            assert {k for k in a if a[k] != b[k]} == {"benchmarks"}, alias
            assert entry.benchmarks == ["harmbench"]
            assert entry.revision, alias
        d_pre, d_primary = pre.decoding.model_dump(), primary.decoding.model_dump()
        assert {k for k in d_pre if d_pre[k] != d_primary[k]} == {"assistant_prefill"}
        assert pre.decoding.assistant_prefill == PREFILL
        assert pre.benchmarks["harmbench"].model_dump() == primary.benchmarks["harmbench"].model_dump()

    def test_sbatch_files_write_only_to_the_prefill_tree(self) -> None:
        jobs = sorted((ROOT / "slurm" / "jobs_tc1_prefill_512").glob("*.sbatch"))
        assert [j.name for j in jobs] == ["judge_prefill_512.sbatch", "prefill_gen_a.sbatch", "prefill_gen_b.sbatch"]
        aliases = set()
        for job in jobs:
            text = job.read_text()
            commands = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#") or l.startswith("#SBATCH"))
            assert "results_512" not in commands
            assert "HF_HUB_OFFLINE=1" in text and "TRANSFORMERS_OFFLINE=1" in text
            if job.name.startswith("prefill_gen"):
                assert "--config configs/tc1_prefill_512.yaml" in text
                assert "--output_dir results_prefill_512" in text
                aliases |= set(text.split("for MODEL in ", 1)[1].split(";", 1)[0].split())
            else:
                assert "--results-dir results_prefill_512" in text
        primary = load_quant_config(str(ROOT / "configs" / "tc1_512.yaml"))
        assert aliases == {a for a in primary.models if not a.endswith("_8bit")}
