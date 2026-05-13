"""Submits generated SLURM jobs and records submission metadata."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List

LOGGER = logging.getLogger(__name__)


JOB_ID_PATTERN = re.compile(r"Submitted batch job\s+(\d+)")


def parse_args() -> argparse.Namespace:
    """Parses CLI args for SLURM submission.

    Returns:
        Parsed CLI namespace.

    Side Effects:
        Reads command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Submit generated SLURM benchmark jobs")
    parser.add_argument("--jobs_dir", default="slurm/jobs", help="Directory containing manifest.json")
    parser.add_argument("--dry_run", action="store_true", help="Print sbatch commands without submitting")
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


def load_manifest(jobs_dir: Path) -> List[Dict[str, Any]]:
    """Loads job manifest from disk.

    Args:
        jobs_dir: Jobs directory containing manifest.

    Returns:
        Manifest rows.

    Side Effects:
        Reads manifest JSON.
    """

    path = jobs_dir / "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def submit_all(jobs_dir: Path, dry_run: bool) -> List[Dict[str, Any]]:
    """Submits all jobs from manifest.

    Args:
        jobs_dir: Jobs directory.
        dry_run: Whether to skip actual submission.

    Returns:
        Submission records.

    Side Effects:
        Executes ``sbatch`` commands and writes submission manifest.
    """

    manifest = load_manifest(jobs_dir)
    submissions: List[Dict[str, Any]] = []

    for row in manifest:
        sbatch_path = row["sbatch_path"]
        if dry_run:
            LOGGER.info("DRY RUN: sbatch %s", sbatch_path)
            submissions.append({**row, "submitted": False, "job_id": None})
            continue

        proc = subprocess.run(
            ["sbatch", sbatch_path],
            check=False,
            capture_output=True,
            text=True,
        )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        job_id_match = JOB_ID_PATTERN.search(stdout)
        job_id = job_id_match.group(1) if job_id_match else None

        submission = {
            **row,
            "submitted": proc.returncode == 0,
            "job_id": job_id,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": proc.returncode,
        }
        submissions.append(submission)

    output_path = jobs_dir / "submitted_jobs.json"
    output_path.write_text(json.dumps(submissions, indent=2), encoding="utf-8")
    return submissions


def main() -> None:
    """CLI entrypoint for submission helper.

    Side Effects:
        Submits jobs and writes submitted job metadata.
    """

    args = parse_args()
    setup_logging(args.log_level)

    submissions = submit_all(jobs_dir=Path(args.jobs_dir), dry_run=args.dry_run)
    ok = sum(int(item["submitted"] or args.dry_run) for item in submissions)
    LOGGER.info("Submission helper processed %d jobs (%d successful or dry-run).", len(submissions), ok)


if __name__ == "__main__":
    main()
