"""Tests for the fp16 -> INT8 -> NF4 precision-sweep analysis (T29).

Builds synthetic summary sidecars in a tmp results dir and checks the trend
computation (deltas vs fp16, monotonicity/cliff shape classification) and the
graceful INT8-pending handling — no GPU, no real runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import scripts.precision_sweep_analysis as ps  # noqa: E402  (added to path by conftest/rootdir)


def _write_summary(results_dir: Path, alias: str, bench: str, fname: str, **metrics) -> None:
    d = results_dir / alias / bench
    d.mkdir(parents=True, exist_ok=True)
    (d / fname).write_text(json.dumps({"metrics": metrics}), encoding="utf-8")


def _write_judge_sidecar(results_dir: Path, alias: str, backend: str, outcomes) -> None:
    """Writes a redacted HarmBench judge sidecar: {prompt_id: judge_harmful bool}."""
    d = results_dir / alias / "harmbench"
    d.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"prompt_id": pid, "judge_harmful": bool(v)}) for pid, v in outcomes.items()]
    (d / f"scores.judge.{backend}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _paired(b: int, c: int, concordant_harmful: int = 0, concordant_safe: int = 0):
    """Builds (fp16_outcomes, int8_outcomes) with exactly b appeared / c disappeared.

    b = fp16-safe -> INT8-harmful (appeared); c = fp16-harmful -> INT8-safe
    (disappeared). The two concordant cells pad the shared prompt set to a chosen n.
    Both dicts share every prompt id, so the McNemar pairing is over all of them.
    """
    fp16, int8 = {}, {}
    i = 0
    for _ in range(b):
        fp16[f"p{i}"], int8[f"p{i}"] = False, True
        i += 1
    for _ in range(c):
        fp16[f"p{i}"], int8[f"p{i}"] = True, False
        i += 1
    for _ in range(concordant_harmful):
        fp16[f"p{i}"], int8[f"p{i}"] = True, True
        i += 1
    for _ in range(concordant_safe):
        fp16[f"p{i}"], int8[f"p{i}"] = False, False
        i += 1
    return fp16, int8


def _populate(results_dir: Path, pair_prefix: str, judge_asr, mmlu) -> None:
    """Writes judge-ASR + MMLU summaries for fp16/int8/nf4 of one pair."""
    aliases = {"fp16": f"{pair_prefix}_base", "int8": f"{pair_prefix}_8bit", "nf4": f"{pair_prefix}_4bit"}
    for prec, alias in aliases.items():
        if judge_asr[prec] is not None:
            _write_summary(results_dir, alias, "harmbench", "summary.judge.harmbench_cls.json",
                           attack_success_rate=judge_asr[prec])
        if mmlu[prec] is not None:
            _write_summary(results_dir, alias, "mmlu", "summary.json", accuracy=mmlu[prec])


def test_graded_trend(tmp_path: Path) -> None:
    # ASR rises monotonically fp16<int8<nf4 with int8 intermediate -> "graded".
    _populate(tmp_path, "qwen_2b",
              judge_asr={"fp16": 0.10, "int8": 0.15, "nf4": 0.20},
              mmlu={"fp16": 0.62, "int8": 0.60, "nf4": 0.53})
    out = ps.analyse(tmp_path)
    asr = out["per_pair"]["qwen_2b"]["metrics"]["harmbench_asr_judge"]
    assert asr["delta_int8_vs_fp16"] == 0.05
    assert asr["delta_nf4_vs_fp16"] == 0.10
    assert asr["shape"] == "graded"
    assert out["per_pair"]["qwen_2b"]["int8_present"] is True


def test_cliff_at_nf4(tmp_path: Path) -> None:
    # INT8 ~ fp16, the whole drop is the 4-bit step -> "cliff_at_nf4".
    _populate(tmp_path, "qwen_4b",
              judge_asr={"fp16": 0.10, "int8": 0.10, "nf4": 0.30},
              mmlu={"fp16": 0.70, "int8": 0.70, "nf4": 0.69})
    asr = ps.analyse(tmp_path)["per_pair"]["qwen_4b"]["metrics"]["harmbench_asr_judge"]
    assert asr["shape"] == "cliff_at_nf4"


def test_non_monotonic(tmp_path: Path) -> None:
    _populate(tmp_path, "llama_3_2_3b",
              judge_asr={"fp16": 0.10, "int8": 0.30, "nf4": 0.12},
              mmlu={"fp16": 0.61, "int8": 0.60, "nf4": 0.57})
    asr = ps.analyse(tmp_path)["per_pair"]["llama_3_2_3b"]["metrics"]["harmbench_asr_judge"]
    assert asr["shape"] == "non_monotonic"


def test_int8_pending_is_graceful(tmp_path: Path) -> None:
    # Only fp16 + nf4 present (INT8 not run) -> int8 None, shape "int8_pending", no crash.
    _populate(tmp_path, "mistral_7b",
              judge_asr={"fp16": 0.385, "int8": None, "nf4": 0.345},
              mmlu={"fp16": 0.63, "int8": None, "nf4": 0.61})
    rec = ps.analyse(tmp_path)["per_pair"]["mistral_7b"]
    asr = rec["metrics"]["harmbench_asr_judge"]
    assert asr["int8"] is None
    assert asr["shape"] == "int8_pending"
    assert asr["delta_nf4_vs_fp16"] == -0.04  # fp16/nf4 delta still computed
    assert rec["int8_present"] is False


def test_accuracy_direction(tmp_path: Path) -> None:
    # MMLU accuracy falling monotonically (lower = worse) is still "graded".
    _populate(tmp_path, "phi4_mini",
              judge_asr={"fp16": 0.05, "int8": 0.05, "nf4": 0.05},
              mmlu={"fp16": 0.70, "int8": 0.69, "nf4": 0.66})
    mmlu = ps.analyse(tmp_path)["per_pair"]["phi4_mini"]["metrics"]["mmlu_accuracy"]
    assert mmlu["shape"] == "graded"
    assert mmlu["delta_nf4_vs_fp16"] == -0.04


def test_v2_proxy_prefers_v2_summary(tmp_path: Path) -> None:
    # The v2 proxy column must read summary.v2.json, not the runtime summary.json
    # (the original models' summary.json is the v1 scorer; reading it would mix
    # v1 and v2 across precisions). When both exist, prefer summary.v2.json.
    alias = "qwen_2b_base"
    _write_summary(tmp_path, alias, "harmbench", "summary.json", attack_success_rate=0.775)
    _write_summary(tmp_path, alias, "harmbench", "summary.v2.json", attack_success_rate=0.600)
    val = ps._read_metric(tmp_path, alias, "harmbench", "summary.v2.json", "attack_success_rate")
    assert val == 0.600  # v2 sidecar wins over the v1 runtime summary


def test_v2_proxy_falls_back_to_runtime_summary(tmp_path: Path) -> None:
    # If no v2 sidecar was built, fall back to summary.json rather than returning None.
    alias = "qwen_2b_8bit"
    _write_summary(tmp_path, alias, "harmbench", "summary.json", attack_success_rate=0.585)
    val = ps._read_metric(tmp_path, alias, "harmbench", "summary.v2.json", "attack_success_rate")
    assert val == 0.585  # graceful fallback when summary.v2.json is absent


def test_xstest_over_refusal_reads_v2_summary(tmp_path: Path) -> None:
    # The XSTest over-refusal column has the SAME v1/v2 mixing hazard as the
    # HarmBench proxy (the original models' summary.json is v1; INT8 ran v2), so it
    # must also read summary.v2.json (audit M8 / D36). Guard both the METRICS map
    # and the read behaviour.
    assert ps.METRICS["xstest_over_refusal"][1] == "summary.v2.json"
    alias = "qwen_2b_base"
    _write_summary(tmp_path, alias, "xstest", "summary.json", over_refusal_rate=0.032)    # v1
    _write_summary(tmp_path, alias, "xstest", "summary.v2.json", over_refusal_rate=0.052)  # v2
    val = ps._read_metric(tmp_path, alias, "xstest", "summary.v2.json", "over_refusal_rate")
    assert val == 0.052  # v2 wins; v1/v2 are not mixed across precisions


def test_int8_mcnemar_reproduces_documented_llama(tmp_path: Path) -> None:
    # The §6.15 / thesis-footnote INT8 headline: the fp16->INT8 Llama-3B HarmBench
    # ASR move (Δ=+0.040) is significant under BOTH judges and McNemar — classifier
    # b=9,c=1 -> exact p=0.022; GPT-4o b=8,c=0 -> exact p=0.008. This locks the
    # discordant counts AND the binomial p-values the thesis now cites from
    # results/analysis/precision_sweep.json (closing the reproducibility gap).
    cls_fp16, cls_int8 = _paired(9, 1, concordant_harmful=7, concordant_safe=183)   # n=200
    api_fp16, api_int8 = _paired(8, 0, concordant_harmful=8, concordant_safe=184)   # n=200
    _write_judge_sidecar(tmp_path, "llama_3_2_3b_base", "harmbench_cls", cls_fp16)
    _write_judge_sidecar(tmp_path, "llama_3_2_3b_8bit", "harmbench_cls", cls_int8)
    _write_judge_sidecar(tmp_path, "llama_3_2_3b_base", "api_judge", api_fp16)
    _write_judge_sidecar(tmp_path, "llama_3_2_3b_8bit", "api_judge", api_int8)

    rec = ps.analyse(tmp_path)["per_pair"]["llama_3_2_3b"]["harmbench_asr_int8_mcnemar"]
    cls = rec["harmbench_cls"]
    assert (cls["n"], cls["b"], cls["c"], cls["n_discordant"]) == (200, 9, 1, 10)
    # Exact two-sided binomial: 2 * (C(10,0)+C(10,1)) / 2^10 = 22/1024 = 0.021484375
    # (rounds to 0.0215; the thesis displays this as p=0.022).
    assert cls["p_value"] == 22 / 1024
    assert round(cls["p_value"], 4) == 0.0215
    assert cls["uncorrected_significant"] is True
    assert cls["delta"] == 0.04 and cls["direction"] == "harmful-ward"

    api = rec["api_judge"]
    assert (api["n"], api["b"], api["c"], api["n_discordant"]) == (200, 8, 0, 8)
    # Exact two-sided binomial: 2 * C(8,0) / 2^8 = 2/256 = 0.0078125 (-> 0.008).
    assert api["p_value"] == 2 / 256
    assert round(api["p_value"], 3) == 0.008
    assert api["uncorrected_significant"] is True
    assert api["delta"] == 0.04 and api["direction"] == "harmful-ward"


def test_int8_mcnemar_safe_ward_and_partial_judges(tmp_path: Path) -> None:
    # delta<0 (more disappeared than appeared) -> "safe-ward"; and a backend whose
    # sidecar is absent reports None rather than crashing (partial-judge coverage).
    fp16, int8 = _paired(1, 3, concordant_safe=10)
    _write_judge_sidecar(tmp_path, "mistral_7b_base", "harmbench_cls", fp16)
    _write_judge_sidecar(tmp_path, "mistral_7b_8bit", "harmbench_cls", int8)

    rec = ps.analyse(tmp_path)["per_pair"]["mistral_7b"]["harmbench_asr_int8_mcnemar"]
    cls = rec["harmbench_cls"]
    assert (cls["b"], cls["c"], cls["n_discordant"]) == (1, 3, 4)
    assert cls["delta"] < 0 and cls["direction"] == "safe-ward"
    assert cls["uncorrected_significant"] is False  # exact p=0.625
    assert rec["api_judge"] is None  # api_judge sidecar never written


def test_int8_mcnemar_missing_sidecars_are_none(tmp_path: Path) -> None:
    # No judge sidecars at all (only summary files) -> both backends None, no crash.
    _populate(tmp_path, "qwen_2b",
              judge_asr={"fp16": 0.10, "int8": 0.15, "nf4": 0.20},
              mmlu={"fp16": 0.62, "int8": 0.60, "nf4": 0.53})
    rec = ps.analyse(tmp_path)["per_pair"]["qwen_2b"]["harmbench_asr_int8_mcnemar"]
    assert rec["harmbench_cls"] is None
    assert rec["api_judge"] is None
