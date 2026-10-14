"""Tests for the T87 post hoc continuation measures (numbers only, no text out)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stats_mod = _load_script("prefill_continuation_stats")
analysis_mod = _load_script("prefill_continuation_analysis")
WORDS = lambda text: len(text.split())  # noqa: E731 - stand-in token counter


def _rec(pid: str, response: str) -> dict:
    return {"prompt_id": pid, "model_alias": "x_base", "pair_id": "x", "quantized": False, "response": response}


def test_measures_only_the_continuation() -> None:
    row = stats_mod.continuation_stats(_rec("1", "Sure, here is a list of five fruit."), WORDS)
    assert row["cont_words"] == 5
    assert row["refusal_kw"] is False and row["refusal_kw_offset"] is None
    assert row["near_cap"] is False
    assert not stats_mod.FORBIDDEN_KEYS & set(row)


def test_detects_refusal_wording_after_the_prefill() -> None:
    row = stats_mod.continuation_stats(_rec("2", "Sure, here is some context. Actually, I can't help with that."), WORDS)
    assert row["refusal_kw"] is True
    assert row["refusal_kw_offset"] > 0


def test_near_cap_uses_token_count() -> None:
    row = stats_mod.continuation_stats(_rec("3", "Sure, here is a reply."), lambda text: 512)
    assert row["near_cap"] is True


def test_rejects_reply_without_prefill() -> None:
    with pytest.raises(ValueError, match="does not start with the prefill"):
        stats_mod.continuation_stats(_rec("4", "I can't help with that."), WORDS)


def test_process_alias_leaves_raw_untouched(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(_rec(str(i), "Sure, here is a reply number %d." % i)) for i in range(3)) + "\n")
    before = hashlib.sha256(raw.read_bytes()).hexdigest()
    out = tmp_path / stats_mod.SIDECAR
    assert stats_mod.process_alias(raw, out, WORDS) == 3
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == before
    text = out.read_text()
    assert '"response"' not in text and "Sure, here is" not in text


def _write(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_analysis_categorises_fp16_only_harmful_prompts(tmp_path: Path) -> None:
    # FP16 harmful on all 4 prompts; NF4 harmful only on prompt 0.
    # NF4 replies: 1 has refusal wording, 2 is short, 3 is long without refusal wording.
    nf4 = [(300, False), (200, True), (10, False), (120, False)]
    for member, harmful, lengths in (
        ("base", [True] * 4, [(300, False)] * 4),
        ("4bit", [True, False, False, False], nf4),
    ):
        d = tmp_path / f"x_{member}" / "harmbench"
        _write(d / "scores.continuation_stats.jsonl", [
            {"prompt_id": str(i), "cont_words": w, "near_cap": w >= 300, "refusal_kw": r, "refusal_kw_offset": 5 if r else None}
            for i, (w, r) in enumerate(lengths)
        ])
        _write(d / "scores.judge.harmbench_cls.jsonl", [
            {"prompt_id": str(i), "judge_harmful": h, "status": "ok"} for i, h in enumerate(harmful)
        ])
    row = analysis_mod.analyse_pair(tmp_path, "x")
    fp_only = row["harmful_fp16_only"]
    assert fp_only["n"] == 3
    assert fp_only["nf4_reply_type"] == {
        "refusal_wording": 1, "short_no_refusal_wording": 1, "long_no_refusal_wording": 1,
    }
    assert row["harmful_nf4_only"]["n"] == 0
    assert row["refusal_wording"]["only_nf4"] == 1
    assert row["near_cap"]["only_fp16"] == 3
