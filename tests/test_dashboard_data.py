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

    # The study's load-bearing result (D16): under the judge, Qwen-1.7B is the
    # only nominally-significant ΔASR (+0.055, McNemar p<0.05) and is
    # broad_degradation — but it does NOT survive FDR. This is the exact claim
    # the dashboard must headline instead of the v2 proxy's story.
    q2 = by_pair["qwen_2b"]
    assert q2["harmbench_asr_delta"] == pytest.approx(0.055, abs=1e-6)
    assert q2["harmbench_asr_delta_significant"] is True
    assert q2["harmbench_asr_bh_significant"] is False
    assert q2["interpretation_label"] == "broad_degradation"

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
    if not judge or "qwen_2b" not in v2:
        pytest.skip("artifacts not present")
    # The proxy reads Qwen-1.7B ΔASR negative; the judge reads it positive.
    assert v2["qwen_2b"]["harmbench_asr_delta"] < 0 < judge["qwen_2b"]["harmbench_asr_delta"]
