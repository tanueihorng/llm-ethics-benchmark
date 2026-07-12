"""Tests for the strict MMLU/ARC parser and scripts/rescore_capability_strict.py
(T38 / D46).

Covers the strict parser's tier behaviour, the sync contract with the primary
``parse_choice_index`` (the strict rules are byte-copies of tiers 1-2), the
num_choices inference, the redacted sidecar, immutability of raw.jsonl, and the
aggregate + paired-delta plumbing.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from ethical_benchmark.benchmarks.utils import (
    parse_choice_index,
    parse_choice_index_strict,
    parse_choice_index_strict_with_tier,
    _STRICT_LEAD_RE,
    _STRICT_ANSWER_RE,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- strict parser ----------------------------------------------------------
def test_strict_tiers():
    # tier 1: leading option letter
    assert parse_choice_index_strict_with_tier("B. because ...", 4) == (1, "leading_letter")
    assert parse_choice_index_strict_with_tier("(C)", 4) == (2, "leading_letter")
    assert parse_choice_index_strict_with_tier("**D.**", 4) == (3, "leading_letter")
    # tier 2: explicit answer declaration, last mention wins
    assert parse_choice_index_strict_with_tier("I think A, no the answer is C", 4) == (2, "answer_declaration")
    # lenient-only case (a stray in-range capital mid-sentence) -> unanswered
    assert parse_choice_index_strict_with_tier("I really am not sure here", 4) == (None, None)
    # numeric-only -> unanswered under strict (tier-4 disabled)
    assert parse_choice_index_strict_with_tier("2", 4) == (None, None)
    # out-of-range leading letter -> not tier 1; no declaration -> unanswered
    assert parse_choice_index_strict_with_tier("Z is my guess", 4) == (None, None)


def test_strict_regexes_match_primary_cascade():
    """Tripwire: the named strict regexes must be byte-identical to the tier-1/2
    patterns actually inlined in parse_choice_index. Two prior weakenings, both
    caught by mutation testing: (a) the first version compared the strict copies
    against literals repeated in THIS test, so a primary edit could not trip it;
    (b) the second version used a SUBSTRING check on the primary's source, so an
    extension that keeps the old pattern as a substring — e.g. prefixing tier 2
    with (?:FINAL\\s+)? — still passed while strict silently stopped being a
    restriction of the primary. This version EXTRACTS the literal passed to the
    primary's re.match (tier 1) and re.findall (tier 2) and asserts exact
    equality, so any in-place edit, extension, or added second match/findall in
    parse_choice_index fails here — forcing a deliberate re-sync."""
    import inspect
    import re as _re

    # 1. the strict copies are still the claim-locked values (guards the copies) ...
    assert _STRICT_LEAD_RE.pattern == r"[*\s]*\(?\s*([A-Z])\b"
    assert _STRICT_ANSWER_RE.pattern == r"ANSWER\s*(?:IS|:|=)?\s*\**\s*\(?\s*([A-Z])\b"
    # 2. ... AND they EQUAL the literals the PRIMARY parser actually uses.
    primary_src = inspect.getsource(parse_choice_index)
    tier1_lits = _re.findall(r're\.match\(\s*r"([^"]*)"', primary_src)
    tier2_lits = _re.findall(r're\.findall\(\s*r"([^"]*)"', primary_src)
    assert tier1_lits == [_STRICT_LEAD_RE.pattern], \
        f"tier-1 literal(s) in parse_choice_index drifted: {tier1_lits!r}; re-sync _STRICT_LEAD_RE"
    assert tier2_lits == [_STRICT_ANSWER_RE.pattern], \
        f"tier-2 literal(s) in parse_choice_index drifted: {tier2_lits!r}; re-sync _STRICT_ANSWER_RE"


def test_strict_is_restriction_of_primary():
    """Behavioural sync contract: whenever the strict parser answers, the full
    cascade returns the identical index (strict is a restriction of full)."""
    battery = [
        "A. foo", "(B)", "**C.**", "the answer is D", "answer: A",
        "I think it is B", "2", "", "Z only", "C) yes", "D - correct",
        "no idea", "The correct answer (A) is best", "first B then answer is C",
    ]
    for resp in battery:
        for n in (2, 4, 5):
            strict = parse_choice_index_strict(resp, n)
            if strict is not None:
                assert strict == parse_choice_index(resp, n), (resp, n)


# --- num_choices inference ---------------------------------------------------
def test_infer_num_choices():
    mod = _load_script("rescore_capability_strict")
    prompt = ("Question: q?\nOptions:\nA. one\nB. two\nC. three\nD. four\n"
              "Choose the best answer and reply with only the option letter.")
    assert mod.infer_num_choices(prompt, 0) == 4
    prompt3 = ("Question: q?\nOptions:\nA. one\nB. two\nC. three\n"
               "Choose the best answer and reply with only the option letter.")
    assert mod.infer_num_choices(prompt3, 0) == 3
    # missing block -> falls back to a sane bound from gold, clamped
    assert mod.infer_num_choices("no options here", 2) == 4


# --- synthetic tree + rescore ------------------------------------------------
def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _mc_prompt(gold_letter: str) -> str:
    return ("Question: q?\nOptions:\nA. one\nB. two\nC. three\nD. four\n"
            "Choose the best answer and reply with only the option letter.")


def _build_capability_tree(root: Path, alias: str, records: list) -> None:
    """records: list of (prompt_id, response, gold_index, primary_pred, primary_correct)."""
    rows = []
    for pid, resp, gold, ppred, pcorr in records:
        rows.append({
            "benchmark": "mmlu", "prompt_id": pid, "prompt_text": _mc_prompt("A"),
            "response": resp,
            "score_fields": {"predicted_index": ppred, "gold_index": gold,
                             "is_answered": ppred is not None, "is_correct": pcorr,
                             "subject": "s"},
            "model_alias": alias, "pair_id": "qwen_2b", "quantized": alias.endswith("4bit"),
        })
    _write_jsonl(root / alias / "mmlu" / "raw.jsonl", rows)
    (root / alias / "mmlu" / "summary.json").write_text(
        json.dumps({"metrics": {"accuracy": 0.5}, "num_records": len(rows)}))


def test_rescore_immutability_and_sidecar(tmp_path: Path):
    mod = _load_script("rescore_capability_strict")
    root = tmp_path / "results_512"
    # 4 records: leading-letter correct, answer-declaration wrong, fallback (primary
    # answered via a stray capital), and unanswered.
    recs = [
        ("p1", "A. yes", 0, 0, True),          # strict leading, correct
        ("p2", "the answer is B", 2, 1, False),  # strict declaration, wrong
        ("p3", "hmm maybe C works", 2, 2, True),  # primary fallback only (stray C)
        ("p4", "no idea at all", 0, None, False),  # unanswered by both
    ]
    _build_capability_tree(root, "qwen_2b_4bit", recs)
    raw_path = root / "qwen_2b_4bit" / "mmlu" / "raw.jsonl"
    summary_path = root / "qwen_2b_4bit" / "mmlu" / "summary.json"
    raw_before = raw_path.read_bytes()
    summary_before = summary_path.read_bytes()

    res = mod._rescore_benchmark("qwen_2b_4bit", "mmlu", root, dry_run=False)

    # immutability: raw + summary byte-identical
    assert raw_path.read_bytes() == raw_before
    assert summary_path.read_bytes() == summary_before
    # sidecars written
    side = root / "qwen_2b_4bit" / "mmlu" / "scores.parser_strict.jsonl"
    assert side.exists()
    # redaction: no prompt/response text in the sidecar
    blob = side.read_text()
    assert "prompt_text" not in blob and '"response"' not in blob
    # tier split: 1 leading, 1 declaration, 1 fallback, 1 unanswered
    tu = res["summary"]["tier_usage"]
    assert tu["leading_letter"] == 1 and tu["answer_declaration"] == 1
    assert tu["lenient_fallback"] == 1 and tu["unanswered"] == 1
    # strict accuracy: only p1 correct under strict -> 1/4
    assert res["summary"]["strict_accuracy"] == pytest.approx(0.25)
    # leading-letter accuracy = primary-correct within tier1 = p1 correct -> 1.0
    assert res["summary"]["leading_letter_accuracy"] == pytest.approx(1.0)
    # fallback accuracy = primary-correct within fallback = p3 correct -> 1.0
    assert res["summary"]["fallback_accuracy"] == pytest.approx(1.0)


def test_rescore_pair_delta_plumbing(tmp_path: Path):
    mod = _load_script("rescore_capability_strict")
    root = tmp_path / "results_512"
    # base: all 4 strict-correct; 4bit: only 1 strict-correct -> strict ΔACC = -0.75
    base = [("p1", "A. x", 0, 0, True), ("p2", "B. x", 1, 1, True),
            ("p3", "C. x", 2, 2, True), ("p4", "D. x", 3, 3, True)]
    quant = [("p1", "A. x", 0, 0, True), ("p2", "stray b guess", 1, 1, True),
             ("p3", "stray c guess", 2, 2, True), ("p4", "nope", 3, None, False)]
    _build_capability_tree(root, "qwen_2b_base", base)
    _build_capability_tree(root, "qwen_2b_4bit", quant)

    import io
    import contextlib
    argv = ["--results-dir", str(root), "--models", "qwen_2b_base", "qwen_2b_4bit",
            "--benchmarks", "mmlu"]
    import sys
    old = sys.argv
    sys.argv = ["rescore_capability_strict.py"] + argv
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            mod.main()
    finally:
        sys.argv = old
    agg = json.loads((root / "analysis" / "parser_strict_sensitivity.json").read_text())
    pair = next(p for p in agg["per_pair"] if p["pair_id"] == "qwen_2b")
    assert pair["strict_delta"]["delta"] == pytest.approx(-0.75, abs=1e-6)
    assert pair["strict_delta"]["direction"] == "down"
