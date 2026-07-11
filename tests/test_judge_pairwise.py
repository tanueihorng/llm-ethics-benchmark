"""Tests for the T22 API-judge backend mapping and judge-vs-judge agreement.

All deterministic and offline — the OpenAI client is replaced with a fake, so no
network, key, or `openai` package is required.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import List

from ethical_benchmark.judges.validation import (
    JudgeInput,
    OpenAICompatibleJudgeBackend,
)

ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- fake OpenAI client -----------------------------------------------------


class _FakeCompletions:
    def __init__(self, answers: List[str]) -> None:
        self._answers = answers
        self.i = 0

    def create(self, **_kwargs):
        ans = self._answers[self.i]
        self.i += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=ans))])


class _FakeClient:
    def __init__(self, answers: List[str]) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(answers))


def test_api_judge_maps_yes_no_and_unparseable() -> None:
    backend = OpenAICompatibleJudgeBackend(model="gpt-4o")
    backend._client = _FakeClient(["yes", "No.", "maybe?"])  # bypass _ensure_client
    items = [JudgeInput(f"p{i}", "behavior", "generation") for i in range(3)]
    res = backend.classify_batch(items)
    assert [r.harmful for r in res] == [True, False, None]
    assert res[2].status == "parse_error"


def test_api_judge_exposes_model_id_for_summary() -> None:
    # Regression: the summary writer records getattr(backend, "model_id", None);
    # the API backend must expose model_id (not just model) or every
    # summary.judge.api_judge.json records judge_model_id: null.
    backend = OpenAICompatibleJudgeBackend(model="gpt-4o")
    assert getattr(backend, "model_id", None) == "gpt-4o"


# --- agreement / kappa math -------------------------------------------------


def test_agreement_and_kappa_values() -> None:
    mod = _load_script("judge_pairwise_agreement")
    # Perfect agreement, balanced labels -> kappa 1.0
    n, agree, kappa = mod.agreement_and_kappa({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0})
    assert (n, agree, kappa) == (2, 1.0, 1.0)
    # All-same-label degenerate -> kappa undefined (None), not a spurious 1.0
    _, agree2, kappa2 = mod.agreement_and_kappa({"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 1.0})
    assert agree2 == 1.0 and kappa2 is None
    # Half agreement, balanced -> kappa 0.0
    a = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0}
    b = {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0}
    _, agree3, kappa3 = mod.agreement_and_kappa(a, b)
    assert agree3 == 0.5 and kappa3 == 0.0

    # Audit P2 regression: the two standalone script kappa helpers previously
    # returned a spurious 1.0 on the all-constant degenerate slice (pe == 1, so
    # kappa is 0/0 = undefined). They must now return NaN, matching
    # judge_pairwise_agreement / judge_agreement, while a normal balanced slice
    # still returns 0.0.
    import math

    hla = _load_script("human_label_audit")
    gl = _load_script("genlen_512_analysis")
    assert math.isnan(hla._kappa([1, 1, 1, 1], [1, 1, 1, 1]))
    assert math.isnan(gl._kappa([0, 0, 0, 0], [0, 0, 0, 0]))
    assert hla._kappa([1, 0, 1, 0], [1, 1, 0, 0]) == 0.0
    assert gl._kappa([1, 0, 1, 0], [1, 1, 0, 0]) == 0.0


def test_load_judge_outcomes_skips_null(tmp_path: Path) -> None:
    mod = _load_script("judge_pairwise_agreement")
    mdir = tmp_path / "qwen_2b_base" / "harmbench"
    mdir.mkdir(parents=True)
    (mdir / "scores.judge.api_judge.jsonl").write_text(
        '{"prompt_id": "p1", "judge_harmful": true}\n'
        '{"prompt_id": "p2", "judge_harmful": false}\n'
        '{"prompt_id": "p3", "judge_harmful": null}\n',  # error / parse failure
        encoding="utf-8",
    )
    out = mod.load_judge_outcomes(mdir, "api_judge")
    assert out == {"p1": 1.0, "p2": 0.0}  # null row skipped
