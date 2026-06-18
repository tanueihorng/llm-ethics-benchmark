"""INT8 full-parity scoping tests (T29).

The diagnostic stack (judge_agreement, judge_pairwise_agreement,
harmbench_category_breakdown, mmlu_subject_breakdown, rescore_harmbench) was
extended with backward-compatible ``--models`` + ``--out-suffix`` so the INT8
precision dimension can be analysed WITHOUT touching the committed base-vs-4bit
artifacts the report cites. These tests pin two invariants per script:

  * a scoped INT8 run writes ``<stem>_int8.*`` and the default ``<stem>.*`` is
    left absent (no committed-artifact contamination); and
  * for an INT8-only model set the per-pair (base-vs-4bit) section is empty by
    construction — INT8 has no base-vs-4bit pairing (that is the sweep's job).

All fixtures are synthetic IDs + booleans; no prompt/response text.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _judge_sidecar(results: Path, alias: str, name: str, harmful_by_pid) -> None:
    _write_jsonl(
        results / alias / "harmbench" / f"scores.judge.{name}.jsonl",
        [{"prompt_id": pid, "judge_harmful": h} for pid, h in harmful_by_pid.items()],
    )


def _v2_sidecar(results: Path, alias: str, harmful_by_pid) -> None:
    _write_jsonl(
        results / alias / "harmbench" / "scores.v2.jsonl",
        [{"prompt_id": pid, "score_fields": {"harmful_compliance": h}}
         for pid, h in harmful_by_pid.items()],
    )


def _hb_raw(results: Path, alias: str, cat_by_pid) -> None:
    _write_jsonl(
        results / alias / "harmbench" / "raw.jsonl",
        [{"prompt_id": pid, "score_fields": {"category": c}} for pid, c in cat_by_pid.items()],
    )


def _mmlu_raw(results: Path, alias: str, correct_by_pid) -> None:
    _write_jsonl(
        results / alias / "mmlu" / "raw.jsonl",
        [{"prompt_id": pid, "score_fields": {"subject": "law", "is_answered": True, "is_correct": c}}
         for pid, c in correct_by_pid.items()],
    )


def _run(module: ModuleType, argv) -> int:
    saved = sys.argv
    sys.argv = ["prog", *argv]
    try:
        return module.main()
    finally:
        sys.argv = saved


# --- judge_agreement -------------------------------------------------------

def test_judge_agreement_int8_scoping(tmp_path: Path) -> None:
    results = tmp_path / "results"
    pids = {f"h{i}": (i % 3 == 0) for i in range(12)}
    # An old pair so the DEFAULT run has data, plus the INT8 alias.
    for alias in ("qwen_2b_base", "qwen_2b_4bit", "qwen_2b_8bit"):
        _judge_sidecar(results, alias, "harmbench_cls", pids)
        _v2_sidecar(results, alias, {p: (h or i % 2 == 0) for i, (p, h) in enumerate(pids.items())})
    ja = _load_script("judge_agreement")

    rc = _run(ja, ["--results-dir", str(results),
                   "--models", "qwen_2b_8bit", "--out-suffix", "_int8"])
    assert rc == 0
    assert (results / "analysis" / "judge_agreement_int8.json").exists()
    # The committed (default) artifact must NOT be created by a scoped run.
    assert not (results / "analysis" / "judge_agreement.json").exists()
    report = json.loads((results / "analysis" / "judge_agreement_int8.json").read_text())
    assert list(report["per_model"].keys()) == ["qwen_2b_8bit"]
    assert report["per_pair"] == []  # no base-vs-4bit pair for an INT8-only set


# --- judge_pairwise_agreement ---------------------------------------------

def test_judge_pairwise_int8_scoping(tmp_path: Path) -> None:
    results = tmp_path / "results"
    pids = {f"h{i}": (i % 4 == 0) for i in range(12)}
    for alias in ("qwen_2b_base", "qwen_2b_4bit", "qwen_2b_8bit"):
        _judge_sidecar(results, alias, "harmbench_cls", pids)
        _judge_sidecar(results, alias, "api_judge", {p: (h or i % 5 == 0)
                                                     for i, (p, h) in enumerate(pids.items())})
    jp = _load_script("judge_pairwise_agreement")

    rc = _run(jp, ["--results-dir", str(results),
                   "--models", "qwen_2b_8bit", "--out-suffix", "_int8"])
    assert rc == 0
    assert (results / "analysis" / "judge_pairwise_agreement_int8.json").exists()
    assert not (results / "analysis" / "judge_pairwise_agreement.json").exists()
    report = json.loads((results / "analysis" / "judge_pairwise_agreement_int8.json").read_text())
    assert [m["model_alias"] for m in report["per_model"]] == ["qwen_2b_8bit"]
    assert report["per_pair"] == []


# --- harmbench_category_breakdown -----------------------------------------

def test_category_breakdown_int8_scoping(tmp_path: Path) -> None:
    results = tmp_path / "results"
    cats = {f"h{i}": ("illegal" if i % 2 else "cybercrime_intrusion") for i in range(10)}
    harmful = {f"h{i}": (i % 3 == 0) for i in range(10)}
    for alias in ("qwen_2b_base", "qwen_2b_4bit", "qwen_2b_8bit"):
        _hb_raw(results, alias, cats)
        _judge_sidecar(results, alias, "harmbench_cls", harmful)
    hc = _load_script("harmbench_category_breakdown")

    rc = _run(hc, ["--results-dir", str(results),
                   "--models", "qwen_2b_8bit", "--out-suffix", "_int8"])
    assert rc == 0
    assert (results / "analysis" / "harmbench_category_breakdown_int8.json").exists()
    assert not (results / "analysis" / "harmbench_category_breakdown.json").exists()
    report = json.loads((results / "analysis" / "harmbench_category_breakdown_int8.json").read_text())
    assert list(report["per_model"].keys()) == ["qwen_2b_8bit"]
    assert report["per_pair_category_deltas"] == []


# --- mmlu_subject_breakdown -----------------------------------------------

def test_subject_breakdown_int8_scoping(tmp_path: Path) -> None:
    results = tmp_path / "results"
    correct = {f"m{i}": (i % 2 == 0) for i in range(10)}
    for alias in ("qwen_2b_base", "qwen_2b_4bit", "qwen_2b_8bit"):
        _mmlu_raw(results, alias, correct)
    ms = _load_script("mmlu_subject_breakdown")

    rc = _run(ms, ["--results-dir", str(results),
                   "--models", "qwen_2b_8bit", "--out-suffix", "_int8"])
    assert rc == 0
    assert (results / "analysis" / "mmlu_subject_breakdown_int8.json").exists()
    assert not (results / "analysis" / "mmlu_subject_breakdown.json").exists()
    report = json.loads((results / "analysis" / "mmlu_subject_breakdown_int8.json").read_text())
    assert list(report["per_model"].keys()) == ["qwen_2b_8bit"]
    assert report["per_pair_subject_deltas"] == []


# --- default-run backward compatibility -----------------------------------

def test_default_run_uses_full_alias_list(tmp_path: Path) -> None:
    """No --models -> the hardcoded ten aliases + non-suffixed output (unchanged behaviour)."""
    results = tmp_path / "results"
    pids = {f"h{i}": (i % 3 == 0) for i in range(9)}
    for alias in ("qwen_2b_base", "qwen_2b_4bit"):
        _judge_sidecar(results, alias, "harmbench_cls", pids)
        _v2_sidecar(results, alias, pids)
    ja = _load_script("judge_agreement")

    rc = _run(ja, ["--results-dir", str(results)])
    assert rc == 0
    assert (results / "analysis" / "judge_agreement.json").exists()
    report = json.loads((results / "analysis" / "judge_agreement.json").read_text())
    # Default model set is the evaluated ten; the populated pair is present.
    assert set(report["per_model"].keys()) == set(ja.MODEL_ALIASES)
    assert any(p["pair_id"] == "qwen_2b" for p in report["per_pair"])
