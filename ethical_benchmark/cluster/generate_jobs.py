"""Generates SLURM sbatch scripts for model x benchmark quantization runs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shlex
from typing import Any, Dict, List

from ethical_benchmark.quant.config_schema import QuantizationConfig, load_quant_config

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses CLI args for sbatch generation.

    Returns:
        Parsed CLI namespace.

    Side Effects:
        Reads command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Generate SLURM scripts for quantization benchmark matrix")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to quantization config")
    parser.add_argument("--results_dir", default="results", help="Result root directory")
    parser.add_argument("--jobs_dir", default="slurm/jobs", help="Directory for generated sbatch files")
    parser.add_argument("--python_bin", default="python", help="Python executable used in sbatch command")
    parser.add_argument("--script", default="run_quant_benchmark.py", help="Benchmark runner script path")
    parser.add_argument("--seed", type=int, default=42, help="Seed to pass to all jobs")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device policy")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample cap override")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configures logging.

    Args:
        level: Logging level.

    Side Effects:
        Configures global logger.
    """

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _sbatch_text(
    config: QuantizationConfig,
    job_name: str,
    output_log: str,
    error_log: str,
    command: str,
) -> str:
    """Builds sbatch script text from config defaults.

    Args:
        config: Quantization config.
        job_name: SLURM job name.
        output_log: Stdout log path.
        error_log: Stderr log path.
        command: Command line to execute.

    Returns:
        Script text.

    Side Effects:
        None.
    """

    slurm = config.slurm
    account_line = f"#SBATCH --account={slurm.account}\n" if slurm.account else ""
    gpu_line = f"#SBATCH --gres=gpu:{slurm.gpus}\n" if slurm.gpus > 0 else ""
    prologue_lines: List[str] = []

    if slurm.work_dir:
        prologue_lines.append(f"cd {shlex.quote(slurm.work_dir)}")
    # Ensure SLURM log dirs exist on the compute node; SLURM silently drops
    # output if --output= refers to a missing directory.
    prologue_lines.append(f"mkdir -p {shlex.quote(str(Path(output_log).parent))}")
    if str(Path(error_log).parent) != str(Path(output_log).parent):
        prologue_lines.append(f"mkdir -p {shlex.quote(str(Path(error_log).parent))}")
    prologue_lines.extend(slurm.setup_commands)
    prologue_block = "\n".join(prologue_lines)
    if prologue_block:
        prologue_block = f"{prologue_block}\n\n"

    return (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --partition={slurm.partition}\n"
        f"#SBATCH --qos={slurm.qos}\n"
        f"{account_line}"
        f"{gpu_line}"
        f"#SBATCH --cpus-per-task={slurm.cpus_per_task}\n"
        f"#SBATCH --mem={slurm.mem}\n"
        f"#SBATCH --time={slurm.time}\n"
        f"#SBATCH --output={output_log}\n"
        f"#SBATCH --error={error_log}\n\n"
        "set -euo pipefail\n\n"
        f"{prologue_block}"
        f"{command}\n"
    )


def generate_job_scripts(
    config: QuantizationConfig,
    config_path: Path,
    results_dir: Path,
    jobs_dir: Path,
    python_bin: str,
    script: str,
    seed: int,
    device: str,
    max_samples: int | None,
) -> List[Dict[str, Any]]:
    """Generates sbatch files for all model x benchmark combinations.

    Args:
        config: Quantization config.
        config_path: Path to configuration file to embed in job commands.
        results_dir: Result root path.
        jobs_dir: Output folder for sbatch files.
        python_bin: Python executable.
        script: Benchmark runner script path.
        seed: Random seed.
        device: Device policy.
        max_samples: Optional max sample override.

    Returns:
        Job manifest rows.

    Side Effects:
        Writes sbatch files and manifest JSON.
    """

    jobs_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config.slurm.log_dir)
    # Note: don't mkdir here — the path may live under a cluster-side FS that
    # doesn't exist on the submission machine. The sbatch prologue creates it.

    # Resolve the script path. If a relative `script` is supplied together with
    # a `work_dir`, the sbatch `cd`s into work_dir before running, so the path
    # must resolve from there. Anchor relative scripts under work_dir; leave
    # absolute paths alone.
    script_path = Path(script)
    if not script_path.is_absolute() and config.slurm.work_dir:
        resolved_script = str(Path(config.slurm.work_dir) / script_path)
    else:
        resolved_script = str(script_path)

    manifest: List[Dict[str, Any]] = []
    for model_alias, model_entry in config.models.items():
        for benchmark in model_entry.benchmarks:
            job_key = f"{model_alias}__{benchmark}".replace("/", "_").replace(".", "_")
            sbatch_path = jobs_dir / f"{job_key}.sbatch"
            stdout_log = str(log_dir / f"{job_key}.out")
            stderr_log = str(log_dir / f"{job_key}.err")

            cmd_parts = [
                shlex.quote(python_bin),
                shlex.quote(resolved_script),
                "--config", shlex.quote(str(config_path)),
                "--model", shlex.quote(model_alias),
                "--benchmark", shlex.quote(benchmark),
                "--output_dir", shlex.quote(str(results_dir)),
                "--seed", str(int(seed)),
                "--device", shlex.quote(device),
            ]
            if max_samples is not None:
                cmd_parts.extend(["--max_samples", str(int(max_samples))])

            command = " ".join(cmd_parts)
            text = _sbatch_text(
                config=config,
                job_name=job_key[:120],
                output_log=stdout_log,
                error_log=stderr_log,
                command=command,
            )
            sbatch_path.write_text(text, encoding="utf-8")

            manifest.append(
                {
                    "job_key": job_key,
                    "model_alias": model_alias,
                    "benchmark": benchmark,
                    "sbatch_path": str(sbatch_path),
                    "stdout_log": stdout_log,
                    "stderr_log": stderr_log,
                }
            )

    manifest_path = jobs_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    """CLI entrypoint for sbatch generation.

    Side Effects:
        Writes sbatch scripts and manifest to disk.
    """

    args = parse_args()
    setup_logging(args.log_level)

    config = load_quant_config(args.config)
    manifest = generate_job_scripts(
        config=config,
        config_path=Path(args.config),
        results_dir=Path(args.results_dir),
        jobs_dir=Path(args.jobs_dir),
        python_bin=args.python_bin,
        script=args.script,
        seed=args.seed,
        device=args.device,
        max_samples=args.max_samples,
    )
    LOGGER.info("Generated %d SLURM job scripts in %s", len(manifest), args.jobs_dir)


if __name__ == "__main__":
    main()
