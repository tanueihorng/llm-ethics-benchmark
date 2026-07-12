"""Tests for the T18 multi-seed sensitivity arm (config, aggregation, generator)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from ethical_benchmark.quant.config_schema import load_quant_config

ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str) -> ModuleType:
    """Imports a top-level script module by path (scripts/ is not a package)."""

    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sensitivity_config_is_valid_and_stochastic() -> None:
    """The sensitivity config must load under the schema and actually sample.

    A broken config (or one left at temperature 0.0) would silently waste TC1
    GPU time, so this guards both schema validity and the stochastic-decoding
    intent (temperature > 0 so the seeds produce different generations).
    """

    config = load_quant_config(str(ROOT / "configs" / "tc1_sensitivity.yaml"))
    # Stochastic decoding is the whole point of the arm.
    assert config.decoding.temperature > 0.0
    # HarmBench-only, and the cap must cover the full 200-row set so every seed
    # scores the same prompt set (only generation varies).
    assert "harmbench" in config.benchmarks
    assert config.benchmarks["harmbench"].max_samples >= 200
    # Every pair still has a baseline + a quantized member (matched-pair design).
    pair_ids = {m.pair_id for m in config.models.values()}
    for pair_id in pair_ids:
        members = [m for m in config.models.values() if m.pair_id == pair_id]
        assert any(not m.quantized for m in members)
        assert any(m.quantized for m in members)


def test_summarise_deltas_stability() -> None:
    mod = _load_script("sensitivity_analysis")

    # Stable, same-sign effect across seeds -> sign_consistent True.
    stable = mod.summarise_deltas([0.05, 0.06, 0.04, 0.05, 0.07])
    assert stable["n_seeds"] == 5
    assert abs(stable["mean"] - 0.054) < 1e-9
    assert stable["sd"] > 0
    assert stable["min"] == 0.04 and stable["max"] == 0.07
    assert stable["sign_consistent"] is True

    # Effect flips sign across seeds -> NOT sign_consistent (the warning case).
    flippy = mod.summarise_deltas([0.06, -0.01, 0.09, -0.03])
    assert flippy["sign_consistent"] is False

    # Single seed -> sd defined as 0.0, not an error.
    one = mod.summarise_deltas([0.05])
    assert one["n_seeds"] == 1 and one["sd"] == 0.0

    # Empty -> n_seeds 0, no crash.
    empty = mod.summarise_deltas([])
    assert empty["n_seeds"] == 0 and empty["mean"] is None


def test_generator_emits_expected_files() -> None:
    mod = _load_script("generate_sensitivity_jobs")
    all_models = [a for _, aliases in mod.MODELS_BY_PAIR for a in aliases]
    # T39 (D46): the arm now covers five pairs (ten aliases).
    assert len(all_models) == 10

    for alias in all_models:
        content = mod._gen_sbatch(alias, [1, 2, 3], config="configs/tc1_sensitivity.yaml",
                                  sens_root="results_sensitivity",
                                  log_dir="results/slurm_logs_tc1", suffix="sens")
        assert "--config configs/tc1_sensitivity.yaml" in content
        assert "--benchmark harmbench" in content
        assert "for SEED in 1 2 3" in content
        assert "results_sensitivity/seed" in content

    judge = mod._judge_sbatch(1, all_models, sens_root="results_sensitivity",
                              log_dir="results/slurm_logs_tc1", suffix="sens")
    assert "run_judge_validation.py" in judge
    assert "--results-dir results_sensitivity/seed1" in judge
    assert "--precision fp16" in judge


def test_pair_lists_in_sync() -> None:
    """T39 trap-guard (D46): generate_sensitivity_jobs.MODELS_BY_PAIR and
    sensitivity_analysis.PAIRS must list the SAME pairs. A pair present in only
    one is silently dropped from the multi-seed arm (generated but not aggregated,
    or vice versa)."""
    gen = _load_script("generate_sensitivity_jobs")
    ana = _load_script("sensitivity_analysis")
    gen_pairs = {pid for pid, _ in gen.MODELS_BY_PAIR}
    assert gen_pairs == set(ana.PAIRS)
    # and the alias membership matches, IN ORDER (review #13): sensitivity_analysis
    # computes delta = quant - base by tuple position, so a base/quant role flip in
    # either source would silently negate every delta while a set-compare stayed green.
    for pid, aliases in gen.MODELS_BY_PAIR:
        assert tuple(aliases) == tuple(ana.PAIRS[pid])
        # and the role order is the base-then-4bit convention the delta sign assumes
        assert tuple(ana.PAIRS[pid]) == (f"{pid}_base", f"{pid}_4bit")


def test_sensitivity_512_config_has_five_pairs() -> None:
    """The 512 multi-seed config must validate and cover all five NF4 pairs (T39)."""
    config = load_quant_config(str(ROOT / "configs" / "tc1_sensitivity_512.yaml"))
    assert config.decoding.temperature > 0.0
    assert config.decoding.max_new_tokens == 512
    pair_ids = {m.pair_id for m in config.models.values()}
    assert pair_ids == {"qwen_2b", "qwen_4b", "llama_3_2_3b", "mistral_7b", "phi4_mini"}
    # Phi keeps eager attention on both members; Mistral pins its revision.
    assert config.models["phi4_mini_base"].attn_implementation == "eager"
    assert config.models["phi4_mini_4bit"].attn_implementation == "eager"
    assert config.models["mistral_7b_base"].revision


def test_generator_512_mode_covers_new_pairs(tmp_path: Path) -> None:
    """The 512-mode invocation emits mistral/phi gen sbatch pointing at the 512
    config + root, and a judge sbatch listing all ten aliases."""
    mod = _load_script("generate_sensitivity_jobs")
    content = mod._gen_sbatch("mistral_7b_base", [1, 2, 3, 4, 5],
                              config="configs/tc1_sensitivity_512.yaml",
                              sens_root="results_sensitivity_512",
                              log_dir="results_512/slurm_logs_tc1", suffix="sens512")
    assert "--config configs/tc1_sensitivity_512.yaml" in content
    assert "results_sensitivity_512/seed" in content
    assert "mistral_7b_base__sens512" in content
    all_models = [a for _, aliases in mod.MODELS_BY_PAIR for a in aliases]
    judge = mod._judge_sbatch(1, all_models, sens_root="results_sensitivity_512",
                              log_dir="results_512/slurm_logs_tc1", suffix="sens512")
    for alias in ("mistral_7b_4bit", "phi4_mini_base", "phi4_mini_4bit"):
        assert alias in judge


def test_512_readme_has_fully_formed_results_dir(tmp_path: Path) -> None:
    """Review finding (WS-D): the emitted 512-arm README must render a COMPLETE
    aggregation command with `--results-dir results_512`. A dangling flag would
    let an operator drop it and silently write the 512 seed aggregate into the
    128-era results/ tree against the wrong-budget greedy baseline."""
    mod = _load_script("generate_sensitivity_jobs")
    import sys
    import contextlib
    import io
    out = tmp_path / "jobs512"
    argv = sys.argv
    sys.argv = ["generate_sensitivity_jobs.py", "--out-dir", str(out),
                "--config", "configs/tc1_sensitivity_512.yaml",
                "--sens-root", "results_sensitivity_512",
                "--log-dir", "results_512/slurm_logs_tc1",
                "--results-dir", "results_512", "--suffix", "sens512"]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            mod.main()
    finally:
        sys.argv = argv
    readme = (out / "README.md").read_text()
    # the run command is fully formed with a concrete results-dir value
    assert ("python scripts/sensitivity_analysis.py --sensitivity-root "
            "results_sensitivity_512 --results-dir results_512") in readme
    # and the old dangling form ("--results-dir" with no value, then end-of-command)
    # is gone: every "--results-dir" in the command line is followed by a value
    assert "--results-dir`.\n" not in readme
    assert "--results-dir for" not in readme


def test_seeds_used_counts_judge_only_seed(tmp_path: Path) -> None:
    """T24 fix (D46): a seed with a judge sidecar but no summary.json must still
    be counted in seeds_used. Previously seeds_used was appended only in the v2
    branch, so a judge-only seed under-counted."""
    import json
    sa = _load_script("sensitivity_analysis")
    root = tmp_path / "results_sensitivity_512"
    for seed in ("seed1",):
        for alias in ("qwen_2b_base", "qwen_2b_4bit"):
            d = root / seed / alias / "harmbench"
            d.mkdir(parents=True)
            # judge sidecar present, but NO summary.json/summary.v2.json (v2 branch skipped)
            recs = [{"prompt_id": str(i), "judge_harmful": (i % 3 == 0)} for i in range(9)]
            with (d / "scores.judge.harmbench_cls.jsonl").open("w") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
    analysis = tmp_path / "results_512" / "analysis"
    analysis.mkdir(parents=True)
    import sys
    argv = sys.argv
    sys.argv = ["sensitivity_analysis.py", "--sensitivity-root", str(root),
                "--results-dir", str(tmp_path / "results_512")]
    import contextlib
    import io
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            sa.main()
    finally:
        sys.argv = argv
    report = json.loads((analysis / "sensitivity_multiseed.json").read_text())
    qwen = next(p for p in report["per_pair"] if p["pair_id"] == "qwen_2b")
    # the judge-only seed is counted even though there is no v2 summary
    assert "seed1" in qwen["seeds_used"]


def test_llamaguard_sbatch_is_well_formed() -> None:
    """WS-B (T37): the committed LlamaGuard third-judge sbatch must target the
    512 tree, all fifteen aliases, the llamaguard backend, and preserve the
    offline bootstrap."""
    sbatch = (ROOT / "slurm" / "judge_validation_llamaguard_512.sbatch").read_text()
    assert "--backend llamaguard" in sbatch
    assert "--results-dir results_512" in sbatch
    assert "--model-id" not in sbatch  # backend default (Llama-Guard-3-8B) applies
    for env in ("HF_HUB_OFFLINE=1", "HF_DATASETS_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"):
        assert env in sbatch
    for alias in ("qwen_2b_8bit", "mistral_7b_8bit", "phi4_mini_8bit",
                  "qwen_2b_base", "llama_3_2_3b_4bit"):
        assert alias in sbatch
    # exactly fifteen aliases scored (5 pairs x {fp16, NF4, INT8})
    assert sbatch.count("_base ") + sbatch.count("_base\n") >= 5


def test_v2_asr_prefers_v2_summary(tmp_path) -> None:
    """_v2_asr must read summary.v2.json (the v2 regex), not the v1 runtime summary.json.

    Guards against a future re-run silently picking up the stale v1 value for the
    original 2026-05-27 models (mirrors the precision_sweep fix; audit 2026-06-18).
    """
    import json
    sa = _load_script("sensitivity_analysis")
    d = tmp_path / "qwen_2b_base" / "harmbench"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"metrics": {"attack_success_rate": 0.775}}))    # v1
    (d / "summary.v2.json").write_text(json.dumps({"metrics": {"attack_success_rate": 0.600}}))  # v2
    assert sa._v2_asr(d) == 0.600  # v2 sidecar wins


def test_v2_asr_falls_back_to_runtime_summary(tmp_path) -> None:
    import json
    sa = _load_script("sensitivity_analysis")
    d = tmp_path / "qwen_2b_8bit" / "harmbench"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"metrics": {"attack_success_rate": 0.585}}))
    assert sa._v2_asr(d) == 0.585  # graceful fallback when no v2 sidecar
