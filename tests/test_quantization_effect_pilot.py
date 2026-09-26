import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "quantization_effect_pilot", ROOT / "scripts" / "quantization_effect_pilot.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_question_options_are_distinct_and_complete():
    assert set(MODULE.QUESTIONS) == {
        "overall_preference", "refusal_relation", "task_success", "semantic_relation"
    }
    for question in MODULE.QUESTIONS.values():
        assert 2 <= len(question["options"]) <= 4
        assert len(question["options"]) == len(set(question["options"]))


def test_apply_labels_rejects_incomplete_and_writes_redacted(tmp_path, monkeypatch):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    monkeypatch.setattr(MODULE, "ANALYSIS", analysis)
    MODULE.write_jsonl(analysis / "quant_effect_pilot_manifest.jsonl", [
        {"item_id": "qe-001", "pair_id": "pair", "prompt_id": "1", "stratum": "stable"}
    ])
    labels = {
        "qe-001": {
            "overall_preference": "tie",
            "refusal_relation": "same",
            "task_success": "both",
            "semantic_relation": "materially_same",
        }
    }
    label_path = tmp_path / "labels.json"
    label_path.write_text(json.dumps(labels), encoding="utf-8")
    MODULE.apply_labels(label_path)
    row = json.loads((analysis / "quant_effect_pilot_human_labels.jsonl").read_text())
    assert row["human_labels_blinded"] == labels["qe-001"]
    assert not ({"prompt", "response_a", "response_b"} & set(row))
