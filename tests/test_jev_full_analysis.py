"""Offline checks for the redacted Jev full-corpus analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _module() -> ModuleType:
    path = ROOT / "scripts" / "jev_full_analysis.py"
    spec = importlib.util.spec_from_file_location("jev_full_analysis", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_agreement_preserves_directional_disagreements() -> None:
    mod = _module()
    reference = {"a": True, "b": True, "c": False, "d": False}
    candidate = {"a": True, "b": False, "c": True, "d": False}
    result = mod._agreement(reference, candidate)
    assert result["n_shared"] == 4
    assert result["agreement_rate"] == 0.5
    assert result["cohens_kappa"] == 0.0
    assert result["reference_harmful_candidate_safe"] == 1
    assert result["reference_safe_candidate_harmful"] == 1


def test_bh_adjust_is_monotone_in_rank() -> None:
    mod = _module()
    contrasts = [{"p_value": value} for value in (0.001, 0.01, 0.04, 0.5)]
    mod._bh_adjust(contrasts)
    q_values = [row["bh_q_value"] for row in contrasts]
    assert q_values == sorted(q_values)
    assert contrasts[0]["bh_significant_q05"] is True
    assert contrasts[-1]["bh_significant_q05"] is False


def test_human_metrics_uses_ids_labels_and_probabilities_only() -> None:
    mod = _module()
    human = {("qwen_2b_base", "p1"): True, ("qwen_2b_base", "p2"): False}
    rows = {
        "qwen_2b_base": {
            "p1": {"label": True, "probability": 0.8},
            "p2": {"label": False, "probability": 0.2},
        }
    }
    result = mod._human_metrics(human, rows)
    assert result["n_shared"] == 2
    assert result["agreement_rate"] == 1.0
    assert result["brier_score"] == pytest.approx(0.04)
    assert "behavior" not in result and "response" not in result
