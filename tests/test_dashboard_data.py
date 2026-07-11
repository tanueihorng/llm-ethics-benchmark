"""Tests for the dashboard's pure data layer (no Streamlit needed).

These prove the loaders degrade gracefully on a fresh checkout, parse the real
committed artifacts when present, and that the new-model scaffolder produces a
config the project's own Pydantic schema accepts (and rejects bad input).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard import data as D
from ethical_benchmark.quant.config_schema import load_quant_config

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Graceful degradation                                                         #
# --------------------------------------------------------------------------- #

def test_loaders_return_empty_on_missing_dir(tmp_path: Path) -> None:
    assert D.load_interpretations(tmp_path) == []
    assert D.load_pairwise(tmp_path) == []
    assert D.pairwise_df(tmp_path).empty
    assert D.precision_sweep_long(tmp_path).empty
    assert D.judge_agreement_df(tmp_path).empty
    assert D.load_summary(tmp_path) is None
    assert D.load_multiple_comparisons(tmp_path) is None
    assert D.available_runs(tmp_path) == []


def test_load_json_missing_and_malformed(tmp_path: Path) -> None:
    assert D.load_json(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid", encoding="utf-8")
    assert D.load_json(bad) is None


# --------------------------------------------------------------------------- #
# Real committed artifacts (skip if a fresh checkout hasn't run analysis)      #
# --------------------------------------------------------------------------- #

def test_interpretations_shape_if_present() -> None:
    interps = D.load_interpretations(REPO_ROOT)
    if not interps:
        pytest.skip("no committed pair_interpretations.json")
    row = interps[0]
    assert "pair_id" in row and "interpretation_label" in row
    # Every emitted label has presentation metadata (colour never falls back blindly).
    for r in interps:
        assert D.label_color(r.get("interpretation_label")) != ""


def test_pairwise_df_columns_if_present() -> None:
    df = D.pairwise_df(REPO_ROOT)
    if df.empty:
        pytest.skip("no committed pairwise_deltas.json")
    for col in ["pair_id", "benchmark", "absolute_delta", "delta_ci_lower", "delta_significant"]:
        assert col in df.columns


def test_precision_sweep_long_if_present() -> None:
    long = D.precision_sweep_long(REPO_ROOT)
    if long.empty:
        pytest.skip("no committed precision_sweep.json")
    assert set(long.columns) == {"pair_id", "metric", "precision", "value"}
    assert set(long["precision"].unique()) <= {"fp16", "int8", "nf4"}


# --------------------------------------------------------------------------- #
# New-model scaffolding round-trips through the real schema                    #
# --------------------------------------------------------------------------- #

def test_build_new_pair_config_valid_round_trip(tmp_path: Path) -> None:
    yaml_text, err = D.build_new_pair_config(
        base_config_path=REPO_ROOT / "configs" / "default.yaml",
        pair_id="gemma_2b",
        family="gemma",
        size_b=2.6,
        model_id="google/gemma-2-2b-it",
        quant_method="nf4",
        benchmarks=["harmbench", "mmlu"],
    )
    assert err is None and yaml_text
    out = tmp_path / "gen.yaml"
    out.write_text(yaml_text, encoding="utf-8")
    cfg = load_quant_config(out)  # must satisfy the project's real validator
    assert "gemma_2b_base" in cfg.models
    assert "gemma_2b_4bit" in cfg.models
    assert cfg.models["gemma_2b_4bit"].quant_method == "nf4"
    assert cfg.models["gemma_2b_base"].quantized is False


def test_build_new_pair_config_int8_suffix() -> None:
    yaml_text, err = D.build_new_pair_config(
        base_config_path=REPO_ROOT / "configs" / "default.yaml",
        pair_id="gemma_2b",
        family="gemma",
        size_b=2.6,
        model_id="google/gemma-2-2b-it",
        quant_method="int8",
        benchmarks=["harmbench"],
    )
    assert err is None
    assert "gemma_2b_8bit" in yaml_text


def test_build_new_pair_config_rejects_bad_benchmark() -> None:
    yaml_text, err = D.build_new_pair_config(
        base_config_path=REPO_ROOT / "configs" / "default.yaml",
        pair_id="x",
        family="x",
        size_b=1.0,
        model_id="x/y",
        benchmarks=["not_a_benchmark"],
    )
    assert yaml_text is None
    assert err and "benchmark" in err.lower()


def test_quant_suffix() -> None:
    assert D.quant_suffix("nf4") == "4bit"
    assert D.quant_suffix("int8") == "8bit"


# --------------------------------------------------------------------------- #
# Judge-primary view is authoritative (must not surface the v2 proxy headline) #
# --------------------------------------------------------------------------- #

def test_judge_primary_empty_without_artifact(tmp_path: Path) -> None:
    assert D.judge_primary_interpretations(tmp_path) == []
    assert D.mc_metric_df(tmp_path).empty


def test_judge_primary_matches_committed_headline() -> None:
    rows = D.judge_primary_interpretations(REPO_ROOT)
    if not rows:
        pytest.skip("no committed multiple_comparisons.json")
    by_pair = {r["pair_id"]: r for r in rows}

    # The study's load-bearing result (D41, 512-token primary): under the judge,
    # Qwen-1.7B's ΔASR is exactly 0.000 (harm flat) — its broad_degradation
    # label is carried by the significant MMLU loss alone — and the only
    # nominally-significant ΔASR is Llama's DECREASE (−0.040, McNemar p=0.021).
    # This is the exact claim the dashboard must headline instead of the
    # 128-era +0.055 story (a truncation artefact; report §6.16).
    q2 = by_pair["qwen_2b"]
    assert q2["harmbench_asr_delta"] == pytest.approx(0.0, abs=1e-6)
    assert q2["harmbench_asr_delta_significant"] is False
    assert q2["harmbench_asr_bh_significant"] is False
    assert q2["interpretation_label"] == "broad_degradation"

    ll = by_pair["llama_3_2_3b"]
    assert ll["harmbench_asr_delta"] == pytest.approx(-0.040, abs=1e-6)
    assert ll["harmbench_asr_delta_significant"] is True  # significant as a DECREASE

    # The v2 proxy mislabels Mistral as a *degradation*; judge-primary it is an
    # improvement-direction (negative ΔASR), non-significant.
    assert by_pair["mistral_7b"]["harmbench_asr_delta"] < 0
    assert by_pair["mistral_7b"]["interpretation_label"] == "alignment_improvement"

    # No safety effect survives FDR under the judge.
    assert all(not r["harmbench_asr_bh_significant"] for r in rows)


def test_judge_primary_differs_from_v2_proxy() -> None:
    """Guard against silently regressing to the superseded proxy numbers."""
    judge = {r["pair_id"]: r for r in D.judge_primary_interpretations(REPO_ROOT)}
    v2 = {r["pair_id"]: r for r in D.load_interpretations(REPO_ROOT)}
    if not judge or "mistral_7b" not in v2:
        pytest.skip("artifacts not present")
    # The proxy reads Mistral-7B ΔASR positive (apparent degradation); the judge
    # reads it negative (improvement-direction) — the scorer sign-flip that
    # motivates judge-primacy (D16), stable at the 512-token primary budget.
    assert judge["mistral_7b"]["harmbench_asr_delta"] < 0 < v2["mistral_7b"]["harmbench_asr_delta"]


def test_phi_label_matches_canonical_judge_artifact() -> None:
    """Phi-4-mini's judge ΔASR sits exactly on the +0.02 tolerance. The label rule
    now rounds away float64 noise (compare_quant_pairs.classify_pair_change), so
    both the canonical judge_agreement.json artifact and any rebuild from the
    multiple_comparisons delta classify Phi as alignment_degradation with a
    directional evidence status (CI includes zero) — the mechanically-correct label
    per the stated rule (ΔASR >= +0.02). This deliberately REVERSES the earlier
    Codex round-7 decision that had pinned robust_preservation, which relied on the
    float residue 0.019999999999999997 falling just under the tolerance (user
    request, 2026-07-11: 'recompute'). The two sources must still agree wholesale."""
    rows = {r["pair_id"]: r for r in D.judge_primary_interpretations(REPO_ROOT)}
    if "phi4_mini" not in rows:
        pytest.skip("no committed judge artifacts")
    assert rows["phi4_mini"]["interpretation_label"] == "alignment_degradation"
    assert rows["phi4_mini"]["evidence_status"] == "directional"
    # And every dashboard label must agree with the canonical artifact wholesale.
    ja = D.load_judge_agreement(REPO_ROOT)
    canon = {r["pair_id"]: r for r in ja["per_pair"]}
    for pid, row in rows.items():
        if pid in canon:
            assert row["interpretation_label"] == canon[pid]["judge_label"], pid


def test_safe_generated_config_name_blocks_path_escape() -> None:
    """Codex round-7 P2 regression: user-supplied filenames cannot escape
    configs/generated/."""
    assert D.safe_generated_config_name("../tc1.yaml") == "tc1.yaml"
    assert D.safe_generated_config_name("../../etc/passwd") == "passwd.yaml"
    assert D.safe_generated_config_name("/abs/path/evil.yaml") == "evil.yaml"
    assert D.safe_generated_config_name("ok") == "ok.yaml"
    assert D.safe_generated_config_name("") == "new_pair.yaml"
    assert D.safe_generated_config_name("..") == "new_pair.yaml"


def test_protected_results_dirs_are_refused(monkeypatch) -> None:
    """Codex round-7 P1 regression: execution may never target the canonical
    evidence trees."""
    for bad in ("results", "results_512", "results/qwen_2b_base",
                "results_512/analysis", "results_sensitivity_512",
                "results_sensitivity_512/seed1", "results_sensitivity"):
        assert D.is_protected_results_dir(REPO_ROOT / bad, REPO_ROOT), bad
    for ok in ("results_dev", "results_dev/qwen_2b_base", "scratch/results"):
        assert not D.is_protected_results_dir(REPO_ROOT / ok, REPO_ROOT), ok

    # Audit P2: the run pipeline independently enforces the SAME protection
    # (make smoke hardcodes force_restart=True), so a destructive delete can never
    # target committed evidence even when the dashboard guard is not in the path.
    from ethical_benchmark.pipeline.run_quant_benchmark import (
        _guard_protected_delete, _under_protected_results_tree)
    monkeypatch.delenv("FYP_ALLOW_PROTECTED_RESTART", raising=False)
    for bad in ("results", "results_512", "RESULTS", "results_sensitivity_512"):
        raw = REPO_ROOT / bad / "qwen_2b_base" / "harmbench" / "raw.jsonl"
        assert _under_protected_results_tree(raw), bad
        with pytest.raises(RuntimeError, match="protected results tree"):
            _guard_protected_delete(raw)
    # Scratch trees are freely restartable, and the explicit override is honoured.
    assert not _under_protected_results_tree(
        REPO_ROOT / "results_smoke" / "qwen_2b_base" / "harmbench" / "raw.jsonl")
    monkeypatch.setenv("FYP_ALLOW_PROTECTED_RESTART", "1")
    _guard_protected_delete(REPO_ROOT / "results_512" / "m" / "harmbench" / "raw.jsonl")


def test_protected_results_dirs_refuse_case_variants() -> None:
    """Audit P1 regression: on the (case-insensitive) macOS filesystem
    ``RESULTS`` is the same physical directory as ``results``, so the previous
    exact-string guard let a case-variant execution dir delete canonical raw
    evidence via the smoke path's force-restart. Every casing must be refused,
    whether or not the spelled path exists on this filesystem."""
    for bad in ("RESULTS", "Results", "RESULTS_512", "Results_512",
                "RESULTS_SENSITIVITY_512", "Results_Sensitivity_512",
                "RESULTS/qwen_2b_base", "results_512/../RESULTS"):
        assert D.is_protected_results_dir(REPO_ROOT / bad, REPO_ROOT), bad


def test_resolve_execution_dir_blocks_protected_and_defaults() -> None:
    """The Run page resolves every execution dir through this single choke
    point before launching any subprocess (audit follow-up: the guard *wiring*,
    not just the predicate, is under test)."""
    for bad in ("results", "RESULTS", "results_512", "Results_512",
                "results_sensitivity_512", "results_dev/../results"):
        with pytest.raises(D.ProtectedResultsDirError):
            D.resolve_execution_dir(bad, REPO_ROOT)
    assert (
        D.resolve_execution_dir("results_dev", REPO_ROOT)
        == Path(REPO_ROOT).resolve() / "results_dev"
    )
    # Empty input falls back to the safe default scratch dir.
    assert D.resolve_execution_dir("", REPO_ROOT).name == D.DEFAULT_EXECUTION_DIR
