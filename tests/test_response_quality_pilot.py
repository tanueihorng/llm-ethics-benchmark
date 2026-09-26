from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _module():
    path = ROOT / "scripts" / "response_quality_pilot.py"
    spec = importlib.util.spec_from_file_location("response_quality_pilot", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_labels_writes_only_redacted_fields(tmp_path: Path) -> None:
    module = _module()
    module.ANALYSIS = tmp_path
    manifest = {
        "item_id": "rq-001", "pair_id": "pair", "prompt_id": "7",
        "stratum": "stable",
    }
    (tmp_path / "response_quality_pilot_manifest.jsonl").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    labels = {"rq-001": {axis: "tie" for axis in module.AXES}}
    label_path = tmp_path / "labels.json"
    label_path.write_text(json.dumps(labels), encoding="utf-8")
    module.apply_labels(label_path)
    row = json.loads((tmp_path / "response_quality_pilot_human_labels.jsonl").read_text())
    assert row["human_labels_blinded"] == labels["rq-001"]
    assert not ({"prompt", "response", "generation", "response_a", "response_b"} & set(row))


def test_apply_labels_rejects_incomplete_axes(tmp_path: Path) -> None:
    module = _module()
    module.ANALYSIS = tmp_path
    (tmp_path / "response_quality_pilot_manifest.jsonl").write_text(
        json.dumps({"item_id": "rq-001", "pair_id": "pair", "prompt_id": "7", "stratum": "stable"}) + "\n"
    )
    label_path = tmp_path / "labels.json"
    label_path.write_text(json.dumps({"rq-001": {"overall_quality": "A"}}))
    with pytest.raises(SystemExit):
        module.apply_labels(label_path)
