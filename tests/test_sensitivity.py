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


def test_generator_emits_expected_files(tmp_path: Path) -> None:
    mod = _load_script("generate_sensitivity_jobs")
    out_dir = tmp_path / "jobs"

    written = []
    for alias in mod.ALL_MODELS:
        content = mod._gen_sbatch(alias, [1, 2, 3])
        assert "--config configs/tc1_sensitivity.yaml" in content
        assert "--benchmark harmbench" in content
        assert "for SEED in 1 2 3" in content
        assert "results_sensitivity/seed" in content
        written.append(content)
    assert len(written) == 6

    judge = mod._judge_sbatch(1, mod.ALL_MODELS)
    assert "run_judge_validation.py" in judge
    assert "--results-dir results_sensitivity/seed1" in judge
    assert "--precision fp16" in judge
