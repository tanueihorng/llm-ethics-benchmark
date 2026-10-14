"""Tests for the T87 post hoc second-judge options (--judge-name, --pilot, max_tokens)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from ethical_benchmark.judges.validation import (
    JudgeBackend,
    JudgeResult,
    OpenAICompatibleJudgeBackend,
    _metric_construct,
    _scorer_version,
)

ROOT = Path(__file__).resolve().parents[1]


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rjv", ROOT / "scripts" / "run_judge_validation.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeBackend(JudgeBackend):
    name = "api_judge"
    model_id = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def classify_batch(self, items):
        self.calls += len(items)
        return [JudgeResult(i.prompt_id, idx % 2 == 0, "yes" if idx % 2 == 0 else "no", "ok")
                for idx, i in enumerate(items)]


def _results_tree(tmp_path: Path) -> Path:
    run = tmp_path / "x_base" / "harmbench"
    run.mkdir(parents=True)
    rows = [{"prompt_id": f"p{i}", "prompt_text": "placeholder behaviour", "response": "placeholder reply",
             "model_alias": "x_base", "pair_id": "x", "quantized": False} for i in range(6)]
    (run / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def test_deepseek_has_its_own_provenance_tags() -> None:
    assert _scorer_version("deepseek") == "judge_api_deepseek_v1_2026-10-14"
    assert _metric_construct("deepseek") == "harmbench_direct_request_asr"
    assert _scorer_version("api_judge") == "judge_harmbench_cls_v1_2026-05-28"  # T22 unchanged


def test_api_backend_max_tokens_default_unchanged() -> None:
    assert OpenAICompatibleJudgeBackend().max_tokens == 5
    assert OpenAICompatibleJudgeBackend(max_tokens=64).max_tokens == 64


def test_pilot_writes_nothing(tmp_path: Path, monkeypatch, capsys) -> None:
    root = _results_tree(tmp_path)
    runner = _runner()
    fake = _FakeBackend()
    monkeypatch.setattr(runner, "_build_backend", lambda args: fake)
    monkeypatch.setattr(sys, "argv", ["rjv", "--results-dir", str(root), "--models", "x_base",
                                      "--backend", "api_judge", "--judge-name", "deepseek", "--pilot", "3"])
    assert runner.main() == 0
    assert fake.calls == 3
    assert "PILOT x_base: n=3" in capsys.readouterr().out
    assert not list((root / "x_base" / "harmbench").glob("*.judge.*"))


def test_judge_name_sets_sidecar_name_and_version(tmp_path: Path, monkeypatch) -> None:
    root = _results_tree(tmp_path)
    runner = _runner()
    monkeypatch.setattr(runner, "_build_backend", lambda args: _FakeBackend())
    monkeypatch.setattr(sys, "argv", ["rjv", "--results-dir", str(root), "--models", "x_base",
                                      "--backend", "api_judge", "--judge-name", "deepseek"])
    assert runner.main() == 0
    run = root / "x_base" / "harmbench"
    assert (run / "scores.judge.deepseek.jsonl").exists()
    assert not (run / "scores.judge.api_judge.jsonl").exists()
    rows = [json.loads(line) for line in (run / "scores.judge.deepseek.jsonl").read_text().splitlines()]
    assert {r["scorer_version"] for r in rows} == {"judge_api_deepseek_v1_2026-10-14"}
    assert "placeholder" not in (run / "scores.judge.deepseek.jsonl").read_text()
    summary = json.loads((run / "summary.judge.deepseek.json").read_text())
    assert summary["judge_model_id"] == "fake-model"
