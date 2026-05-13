"""Checks SLURM run progress and missing output artifacts."""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List

from ethical_benchmark.pipeline.run_quant_benchmark import build_run_paths
from ethical_benchmark.quant.config_schema import load_quant_config

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses CLI args for run status checks.

    Returns:
        Parsed CLI namespace.

    Side Effects:
        Reads command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Check quantization run status from outputs and SLURM queue")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to quantization config")
    parser.add_argument("--results_dir", default="results", help="Result root directory")
    parser.add_argument("--jobs_dir", default="slurm/jobs", help="Jobs directory")
    parser.add_argument("--output_json", default="slurm/check_status.json", help="Status report output path")
    parser.add_argument("--skip_squeue", action="store_true", help="Skip SLURM queue lookup")
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


def _load_submissions(jobs_dir: Path) -> List[Dict[str, Any]]:
    """Loads submitted job metadata when available.

    Args:
        jobs_dir: Jobs directory.

    Returns:
        Submission records list.

    Side Effects:
        Reads JSON from disk.
    """

    path = jobs_dir / "submitted_jobs.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _current_squeue_job_ids() -> set[str]:
    """Queries currently queued/running job IDs for the invoking user only.

    Returns:
        Set of job IDs reported by ``squeue`` for ``$USER``. Scoping to the
        current user avoids accidental collisions with other users' job IDs
        on a shared cluster.

    Side Effects:
        Executes ``squeue`` command.
    """

    user = os.environ.get("USER") or getpass.getuser()
    proc = subprocess.run(
        ["squeue", "-h", "-u", user, "-o", "%A"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return set()

    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def check_status(config_path: Path, results_dir: Path, jobs_dir: Path, skip_squeue: bool) -> Dict[str, Any]:
    """Builds run status report.

    Args:
        config_path: Quantization config path.
        results_dir: Result root directory.
        jobs_dir: Jobs directory.
        skip_squeue: If true, skip queue lookup.

    Returns:
        Status report dictionary.

    Side Effects:
        Reads summary files and optionally runs ``squeue``.
    """

    config = load_quant_config(config_path)
    submissions = _load_submissions(jobs_dir)
    running_ids = set() if skip_squeue else _current_squeue_job_ids()

    expected_runs: List[Dict[str, Any]] = []
    for model_alias, model_entry in config.models.items():
        for benchmark in model_entry.benchmarks:
            summary_path = build_run_paths(results_dir, model_alias, benchmark)["summary_json"]
            expected_runs.append(
                {
                    "model_alias": model_alias,
                    "benchmark": benchmark,
                    "summary_path": str(summary_path),
                    "completed": summary_path.exists(),
                }
            )

    completed = [row for row in expected_runs if row["completed"]]
    missing = [row for row in expected_runs if not row["completed"]]

    submission_lookup = {row.get("job_key"): row for row in submissions}
    for row in expected_runs:
        job_key = f"{row['model_alias']}__{row['benchmark']}".replace("/", "_").replace(".", "_")
        submission = submission_lookup.get(job_key)
        if submission:
            job_id = submission.get("job_id")
            row["job_id"] = job_id
            row["in_queue"] = bool(job_id and job_id in running_ids)
        else:
            row["job_id"] = None
            row["in_queue"] = False

    return {
        "num_expected_runs": len(expected_runs),
        "num_completed": len(completed),
        "num_missing": len(missing),
        "running_job_ids": sorted(running_ids),
        "runs": expected_runs,
    }


def main() -> None:
    """CLI entrypoint for status checker.

    Side Effects:
        Writes status report JSON.
    """

    args = parse_args()
    setup_logging(args.log_level)

    report = check_status(
        config_path=Path(args.config),
        results_dir=Path(args.results_dir),
        jobs_dir=Path(args.jobs_dir),
        skip_squeue=args.skip_squeue,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    LOGGER.info(
        "Run status: %d/%d completed. Report: %s",
        report["num_completed"],
        report["num_expected_runs"],
        output_path,
    )


if __name__ == "__main__":
    main()
