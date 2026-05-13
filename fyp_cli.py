"""Unified convenience CLI for the quantization FYP workflow."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ethical_benchmark.analysis.compare_quant_pairs import (
    build_pairwise_report,
    compute_cross_family_consistency,
    compute_scale_sensitivity,
    summarize_pair_labels,
    write_csv,
)
from ethical_benchmark.cluster.check_runs import check_status
from ethical_benchmark.cluster.generate_jobs import generate_job_scripts
from ethical_benchmark.cluster.submit_jobs import submit_all
from ethical_benchmark.pipeline.run_quant_benchmark import execute_quant_benchmark
from ethical_benchmark.pipeline.run_quant_matrix import run_quant_matrix
from ethical_benchmark.quant.config_schema import load_quant_config

LOGGER = logging.getLogger(__name__)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """Adds common options shared across root parser and subcommands.

    Args:
        parser: Target argument parser.

    Side Effects:
        Mutates parser option definitions.
    """

    parser.add_argument("--config", default="configs/default.yaml", help="Path to quantization config")
    parser.add_argument("--results_dir", default="results", help="Results root directory")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FYP quantization workflow CLI")
    _add_common_options(parser)
    return parser


def parse_args() -> argparse.Namespace:
    """Parses unified CLI arguments.

    Returns:
        Parsed argument namespace.

    Side Effects:
        Reads process command-line arguments.
    """

    parser = _base_parser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run quick smoke benchmark")
    _add_common_options(smoke)
    smoke.add_argument("--model", "-m", default="qwen_0_8b_bf16")
    smoke.add_argument("--benchmark", "-b", default="harmbench", choices=["harmbench", "xstest", "mmlu"])
    smoke.add_argument("--max_samples", "-n", type=int, default=20)
    smoke.add_argument("--batch_size", type=int, default=2)
    smoke.add_argument("--seed", "-s", type=int, default=42)
    smoke.add_argument("--device", "-d", default="auto", choices=["auto", "cpu", "cuda"])

    run_one = subparsers.add_parser("run", help="Run one model x benchmark")
    _add_common_options(run_one)
    run_one.add_argument("--model", "-m", required=True)
    run_one.add_argument("--benchmark", "-b", required=True, choices=["harmbench", "xstest", "mmlu"])
    run_one.add_argument("--max_samples", "-n", type=int, default=None)
    run_one.add_argument("--batch_size", type=int, default=None)
    run_one.add_argument("--seed", "-s", type=int, default=42)
    run_one.add_argument("--device", "-d", default="auto", choices=["auto", "cpu", "cuda"])
    run_one.add_argument("--no-resume", dest="resume", action="store_false")
    run_one.set_defaults(resume=True)
    run_one.add_argument("--force_restart", action="store_true")

    matrix = subparsers.add_parser("matrix", help="Run full or filtered matrix")
    _add_common_options(matrix)
    matrix.add_argument("--model", "-m", action="append", default=None)
    matrix.add_argument("--benchmark", "-b", action="append", choices=["harmbench", "xstest", "mmlu"])
    matrix.add_argument("--max_samples", "-n", type=int, default=None)
    matrix.add_argument("--batch_size", type=int, default=None)
    matrix.add_argument("--seed", "-s", type=int, default=42)
    matrix.add_argument("--device", "-d", default="auto", choices=["auto", "cpu", "cuda"])
    matrix.add_argument("--no-resume", dest="resume", action="store_false")
    matrix.add_argument("--reuse", dest="reuse_loaded_model", action="store_true")
    matrix.add_argument("--no-reuse", dest="reuse_loaded_model", action="store_false")
    matrix.add_argument("--force_restart", action="store_true")
    matrix.set_defaults(resume=True, reuse_loaded_model=True)

    analyze = subparsers.add_parser("analyze", help="Run pairwise quantization analysis")
    _add_common_options(analyze)
    analyze.add_argument("--output_dir", "-o", default="results/analysis")

    cluster_gen = subparsers.add_parser("cluster-generate", help="Generate SLURM scripts")
    _add_common_options(cluster_gen)
    cluster_gen.add_argument("--jobs_dir", "-j", default="slurm/jobs")
    cluster_gen.add_argument("--python_bin", default="python")
    cluster_gen.add_argument("--script", default="run_quant_benchmark.py")
    cluster_gen.add_argument("--seed", "-s", type=int, default=42)
    cluster_gen.add_argument("--device", "-d", default="auto", choices=["auto", "cpu", "cuda"])
    cluster_gen.add_argument("--max_samples", "-n", type=int, default=None)

    cluster_submit = subparsers.add_parser("cluster-submit", help="Submit generated SLURM scripts")
    _add_common_options(cluster_submit)
    cluster_submit.add_argument("--jobs_dir", "-j", default="slurm/jobs")
    cluster_submit.add_argument("--dry_run", action="store_true")

    cluster_check = subparsers.add_parser("cluster-check", help="Check cluster run status")
    _add_common_options(cluster_check)
    cluster_check.add_argument("--jobs_dir", "-j", default="slurm/jobs")
    cluster_check.add_argument("--output_json", "-o", default="slurm/check_status.json")
    cluster_check.add_argument("--skip_squeue", action="store_true")

    return parser.parse_args()


def _run_analysis(config_path: Path, results_dir: Path, output_dir: Path) -> None:
    """Runs pairwise analysis and writes outputs.

    Args:
        config_path: Config path.
        results_dir: Results directory.
        output_dir: Analysis output directory.

    Side Effects:
        Writes JSON/CSV analysis artifacts.
    """

    config = load_quant_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairwise_rows = build_pairwise_report(config=config, results_dir=results_dir)
    label_rows = summarize_pair_labels(pairwise_rows)
    scale_report = compute_scale_sensitivity(config=config, labels=label_rows)
    cross_family_report = compute_cross_family_consistency(config=config, labels=label_rows)

    (output_dir / "pairwise_deltas.json").write_text(json.dumps(pairwise_rows, indent=2), encoding="utf-8")
    (output_dir / "pair_interpretations.json").write_text(json.dumps(label_rows, indent=2), encoding="utf-8")
    (output_dir / "quantization_analysis_summary.json").write_text(
        json.dumps(
            {
                "pairwise_count": len(pairwise_rows),
                "label_count": len(label_rows),
                "scale_sensitivity": scale_report,
                "cross_family": cross_family_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    write_csv(pairwise_rows, output_dir / "pairwise_deltas.csv")
    write_csv(label_rows, output_dir / "pair_interpretations.csv")


def main() -> None:
    """CLI entrypoint.

    Side Effects:
        Executes selected workflow command and writes outputs.
    """

    args = parse_args()
    _setup_logging(args.log_level)

    config_path = Path(args.config)
    results_dir = Path(args.results_dir)
    config = load_quant_config(config_path)

    if args.command == "smoke":
        execute_quant_benchmark(
            config=config,
            model_alias=args.model,
            benchmark_name=args.benchmark,
            output_dir=results_dir,
            seed=args.seed,
            device=args.device,
            resume=False,
            force_restart=True,
            max_samples_override=args.max_samples,
            batch_size_override=args.batch_size,
        )
        LOGGER.info("Smoke run complete.")
        return

    if args.command == "run":
        execute_quant_benchmark(
            config=config,
            model_alias=args.model,
            benchmark_name=args.benchmark,
            output_dir=results_dir,
            seed=args.seed,
            device=args.device,
            resume=args.resume,
            force_restart=args.force_restart,
            max_samples_override=args.max_samples,
            batch_size_override=args.batch_size,
        )
        LOGGER.info("Single run complete.")
        return

    if args.command == "matrix":
        selected_models = args.model if args.model else list(config.models.keys())
        benchmark_filter = {item.lower() for item in args.benchmark} if args.benchmark else None

        run_quant_matrix(
            config=config,
            output_dir=results_dir,
            seed=args.seed,
            device=args.device,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            selected_models=selected_models,
            benchmark_filter=benchmark_filter,
            resume=args.resume,
            force_restart=args.force_restart,
            reuse_loaded_model=args.reuse_loaded_model,
        )
        LOGGER.info("Matrix run complete.")
        return

    if args.command == "analyze":
        _run_analysis(config_path=config_path, results_dir=results_dir, output_dir=Path(args.output_dir))
        LOGGER.info("Analysis complete.")
        return

    if args.command == "cluster-generate":
        manifest = generate_job_scripts(
            config=config,
            config_path=config_path,
            results_dir=results_dir,
            jobs_dir=Path(args.jobs_dir),
            python_bin=args.python_bin,
            script=args.script,
            seed=args.seed,
            device=args.device,
            max_samples=args.max_samples,
        )
        LOGGER.info("Generated %d job scripts.", len(manifest))
        return

    if args.command == "cluster-submit":
        submissions = submit_all(jobs_dir=Path(args.jobs_dir), dry_run=args.dry_run)
        LOGGER.info("Processed %d submissions.", len(submissions))
        return

    if args.command == "cluster-check":
        report = check_status(
            config_path=config_path,
            results_dir=results_dir,
            jobs_dir=Path(args.jobs_dir),
            skip_squeue=args.skip_squeue,
        )
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info(
            "Status: %d/%d completed.", report["num_completed"], report["num_expected_runs"]
        )
        return


if __name__ == "__main__":
    main()
