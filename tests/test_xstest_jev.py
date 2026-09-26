from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from ethical_benchmark.judges.validation import (
    JevXSTestRefusalJudgeBackend,
    JudgeInput,
    XSTEST_JEV_SCORER_VERSION,
    run_xstest_judge_validation,
)


class FakeJevClient:
    def __init__(self) -> None:
        self.calls = []

    def system_one(self, **kwargs):
        self.calls.append(kwargs)
        answer = SimpleNamespace(
            choice="partial_refusal",
            confidence=0.72,
            probabilities={
                "full_compliance": 0.18,
                "partial_refusal": 0.72,
                "full_refusal": 0.10,
            },
        )
        return SimpleNamespace(
            model="jev-1.13.0",
            answers={
                "refusal_category": answer,
                "substantive_compliance": SimpleNamespace(noul=0.68),
                "contains_refusal": SimpleNamespace(noul=0.77),
            },
        )


def test_jev_xstest_backend_preserves_probabilities_and_pin() -> None:
    client = FakeJevClient()
    backend = JevXSTestRefusalJudgeBackend(
        client=client, question_factory=lambda: {"questions": "stub"}
    )
    result = backend.classify_batch([JudgeInput("p1", "safe request", "mixed response")])[0]
    assert result.label == "partial_refusal"
    assert result.choice_confidence == 0.72
    assert result.choice_probabilities["partial_refusal"] == 0.72
    assert result.substantive_compliance_probability == 0.68
    assert result.contains_refusal_probability == 0.77
    assert client.calls[0]["model"] == "jev-1.13.0"
    assert set(client.calls[0]["state"]) == {"user_request", "assistant_response"}


def test_jev_xstest_sidecar_is_redacted(tmp_path: Path) -> None:
    alias = "model_base"
    run_dir = tmp_path / alias / "xstest"
    run_dir.mkdir(parents=True)
    raw = {
        "prompt_id": "7",
        "prompt_text": "SECRET PROMPT",
        "response": "SECRET RESPONSE",
        "model_alias": alias,
        "benchmark": "xstest",
        "pair_id": "model",
        "quantized": False,
    }
    (run_dir / "raw.jsonl").write_text(json.dumps(raw) + "\n", encoding="utf-8")
    backend = JevXSTestRefusalJudgeBackend(
        client=FakeJevClient(), question_factory=lambda: {"questions": "stub"}
    )
    summary = run_xstest_judge_validation(tmp_path, alias, backend, judge_name=backend.name)
    path = run_dir / "scores.judge.xstest_jev.jsonl"
    text = path.read_text(encoding="utf-8")
    assert "SECRET" not in text
    row = json.loads(text)
    assert row["judge_label"] == "partial_refusal"
    assert row["judge_refusal_strict"] is False
    assert row["judge_refusal_broad"] is True
    assert row["judge_choice_probabilities"]["partial_refusal"] == 0.72
    assert row["scorer_version"] == XSTEST_JEV_SCORER_VERSION
    assert summary["judge_observed_model_ids"] == ["jev-1.13.0"]
