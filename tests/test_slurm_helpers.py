"""Tests for SLURM job generation and dry-run submission helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ethical_benchmark.cluster.generate_jobs import generate_job_scripts
from ethical_benchmark.cluster.submit_jobs import submit_all
from ethical_benchmark.quant.config_schema import QuantizationConfig


def test_generate_jobs_count(tmp_path: Path, quant_config_dict: dict) -> None:
    config = QuantizationConfig.model_validate(quant_config_dict)
    jobs_dir = tmp_path / "jobs"

    manifest = generate_job_scripts(
        config=config,
        config_path=tmp_path / "config.yaml",
        results_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        python_bin="python",
        script="run_quant_benchmark.py",
        seed=42,
        device="cpu",
        max_samples=5,
    )

    expected = sum(len(entry.benchmarks) for entry in config.models.values())
    assert len(manifest) == expected
    assert (jobs_dir / "manifest.json").exists()


def test_submit_all_dry_run(tmp_path: Path, quant_config_dict: dict) -> None:
    config = QuantizationConfig.model_validate(quant_config_dict)
    jobs_dir = tmp_path / "jobs"

    generate_job_scripts(
        config=config,
        config_path=tmp_path / "config.yaml",
        results_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        python_bin="python",
        script="run_quant_benchmark.py",
        seed=42,
        device="cpu",
        max_samples=5,
    )

    submissions = submit_all(jobs_dir=jobs_dir, dry_run=True)
    assert submissions
    assert all(item["submitted"] is False for item in submissions)
    assert (jobs_dir / "submitted_jobs.json").exists()


def test_generate_jobs_include_workdir_and_setup_commands(tmp_path: Path, quant_config_dict: dict) -> None:
    quant_config_dict["slurm"]["work_dir"] = "/tc1home/FYP/utan001/fyp_quant/repo"
    quant_config_dict["slurm"]["setup_commands"] = [
        "module load slurm",
        "module load anaconda",
        "source activate fyp-tc1",
    ]
    config = QuantizationConfig.model_validate(quant_config_dict)
    jobs_dir = tmp_path / "jobs"

    manifest = generate_job_scripts(
        config=config,
        config_path=tmp_path / "config.yaml",
        results_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        python_bin="python",
        script="run_quant_benchmark.py",
        seed=42,
        device="cpu",
        max_samples=5,
    )

    first_job = Path(manifest[0]["sbatch_path"]).read_text(encoding="utf-8")
    assert "cd /tc1home/FYP/utan001/fyp_quant/repo" in first_job
    assert "module load slurm" in first_job
    assert "module load anaconda" in first_job
    assert "source activate fyp-tc1" in first_job


def test_generate_jobs_resolves_relative_script_under_workdir(
    tmp_path: Path, quant_config_dict: dict
) -> None:
    """Relative --script paths must be anchored to work_dir so the sbatch
    works regardless of submission cwd. Plain shell quoting must be applied."""

    quant_config_dict["slurm"]["work_dir"] = "/tc1home/FYP/utan001/fyp_quant/repo"
    config = QuantizationConfig.model_validate(quant_config_dict)
    jobs_dir = tmp_path / "jobs"

    manifest = generate_job_scripts(
        config=config,
        config_path=tmp_path / "config.yaml",
        results_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        python_bin="python",
        script="run_quant_benchmark.py",
        seed=42,
        device="cpu",
        max_samples=5,
    )

    body = Path(manifest[0]["sbatch_path"]).read_text(encoding="utf-8")
    assert "/tc1home/FYP/utan001/fyp_quant/repo/run_quant_benchmark.py" in body


def test_generate_jobs_includes_log_dir_mkdir(
    tmp_path: Path, quant_config_dict: dict
) -> None:
    """sbatch prologue must create the log dir; SLURM silently drops output
    when --output= points at a missing directory."""

    config = QuantizationConfig.model_validate(quant_config_dict)
    jobs_dir = tmp_path / "jobs"

    manifest = generate_job_scripts(
        config=config,
        config_path=tmp_path / "config.yaml",
        results_dir=tmp_path / "results",
        jobs_dir=jobs_dir,
        python_bin="python",
        script="run_quant_benchmark.py",
        seed=42,
        device="cpu",
        max_samples=5,
    )

    body = Path(manifest[0]["sbatch_path"]).read_text(encoding="utf-8")
    assert "mkdir -p" in body
    assert "results/slurm_logs" in body


def test_slurm_time_rejects_bare_minutes(quant_config_dict: dict) -> None:
    """Bare-minute integer walltimes (e.g. '360') are ambiguous on SLURM and
    must be rejected at config-load time."""

    quant_config_dict["slurm"]["time"] = "360"
    with pytest.raises(ValidationError):
        QuantizationConfig.model_validate(quant_config_dict)


def test_slurm_time_accepts_colon_forms(quant_config_dict: dict) -> None:
    for value in ("60:00", "06:00:00", "1-00:00:00"):
        quant_config_dict["slurm"]["time"] = value
        QuantizationConfig.model_validate(quant_config_dict)  # no exception
