"""CLI orchestration for ethical benchmarking across toxicity, bias, and factuality tasks."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml
from tqdm import tqdm

from ethical_benchmark.config_schema import validate_config
from ethical_benchmark.datasets.bias import load_bbq
from ethical_benchmark.datasets.factuality import load_truthfulqa
from ethical_benchmark.datasets.toxicity import load_real_toxicity_prompts
from ethical_benchmark.evaluators.bias_eval import BiasEvalConfig, BiasEvaluator
from ethical_benchmark.evaluators.factuality_eval import FactualityEvalConfig, FactualityEvaluator
from ethical_benchmark.evaluators.toxicity_eval import ToxicityEvalConfig, ToxicityEvaluator
from ethical_benchmark.metrics.aggregate import (
    append_jsonl,
    append_summary_csv,
    export_radar_csv,
    flatten_for_csv,
    read_jsonl,
    write_json,
)
from ethical_benchmark.models.generation import DecodingConfig, TextGenerator, set_global_seed
from ethical_benchmark.models.loader import HFModelLoader, build_model_spec

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for benchmark execution.

    Returns:
        Parsed CLI namespace.

    Side Effects:
        Reads command-line arguments from process state.
    """

    parser = argparse.ArgumentParser(description="Ethical benchmarking for open-source LLMs")
    parser.add_argument("--model", required=True, help="Model alias from config registry")
    parser.add_argument(
        "--task",
        required=True,
        choices=["toxicity", "bias", "factuality"],
        help="Benchmark task to execute",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--output_dir", default="results", help="Output root directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample cap")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional generation batch size")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Target runtime device",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from existing raw JSONL if available (default)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start new run and ignore existing raw JSONL",
    )
    parser.add_argument(
        "--force_restart",
        action="store_true",
        help="Delete existing raw JSONL for this task/model before running",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configures root logger.

    Args:
        level: Logging level string.

    Side Effects:
        Configures global logging handlers.
    """

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(path: Path) -> Dict[str, Any]:
    """Loads benchmark YAML configuration.

    Args:
        path: Config path.

    Returns:
        Parsed configuration dictionary.

    Side Effects:
        Reads YAML configuration from filesystem.
    """

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    """Runs CLI benchmark entrypoint.

    Side Effects:
        Executes model inference, writes raw and summary outputs to disk.
    """

    args = parse_args()
    setup_logging(args.log_level)

    config_path = Path(args.config)
    config = load_config(config_path)

    # Validate config schema early so errors surface before downloading.
    try:
        validate_config(config)
    except Exception as exc:
        LOGGER.warning("Config validation warning: %s", exc)

    set_global_seed(args.seed)

    model_spec = build_model_spec(args.model, config["models"])
    task_config = config["tasks"][args.task]

    decoding_cfg = DecodingConfig.from_dict(config.get("decoding", {}))
    if args.batch_size is not None:
        task_config = {**task_config, "batch_size": int(args.batch_size)}

    loader = HFModelLoader(device=args.device)
    model, tokenizer, runtime_device = loader.load(model_spec)
    generator = TextGenerator(model=model, tokenizer=tokenizer, device=runtime_device, config=decoding_cfg)

    output_root = Path(args.output_dir)
    raw_path, summary_json_path, summary_csv_path, radar_csv_path = build_output_paths(
        output_root=output_root,
        task=args.task,
        model_alias=args.model,
    )

    if args.force_restart and raw_path.exists():
        LOGGER.warning("Removing existing raw file due to --force_restart: %s", raw_path)
        raw_path.unlink()

    samples = load_samples(task=args.task, task_config=task_config, max_samples=args.max_samples, seed=args.seed)
    LOGGER.info("Loaded %d samples for task '%s'", len(samples), args.task)

    evaluator = build_evaluator(task=args.task, task_config=task_config, runtime_device=runtime_device)
    run_generation_loop(
        task=args.task,
        samples=samples,
        generator=generator,
        evaluator=evaluator,
        raw_path=raw_path,
        batch_size=int(task_config.get("batch_size", 4)),
        resume=args.resume,
    )

    _persist_results(
        task=args.task,
        model_alias=args.model,
        model_spec=model_spec,
        runtime_device=runtime_device,
        decoding_cfg=decoding_cfg,
        task_config=task_config,
        seed=args.seed,
        evaluator=evaluator,
        raw_path=raw_path,
        summary_json_path=summary_json_path,
        summary_csv_path=summary_csv_path,
        radar_csv_path=radar_csv_path,
    )


def _persist_results(
    task: str,
    model_alias: str,
    model_spec: Any,
    runtime_device: str,
    decoding_cfg: DecodingConfig,
    task_config: Dict[str, Any],
    seed: int,
    evaluator: Any,
    raw_path: Path,
    summary_json_path: Path,
    summary_csv_path: Path,
    radar_csv_path: Path,
) -> None:
    """Aggregates records and writes all output artifacts.

    Args:
        task: Benchmark task name.
        model_alias: Model alias string.
        model_spec: Resolved model specification.
        runtime_device: Device used for inference.
        decoding_cfg: Decoding configuration.
        task_config: Task-level configuration.
        seed: Global random seed.
        evaluator: Task evaluator with ``summarize`` method.
        raw_path: Path to raw JSONL records.
        summary_json_path: Destination for summary JSON.
        summary_csv_path: Destination for summary CSV.
        radar_csv_path: Destination for radar CSV.

    Side Effects:
        Reads raw records and writes summary JSON, CSV, and radar CSV to disk.
    """

    all_records = read_jsonl(raw_path)
    summary_metrics = evaluator.summarize(all_records)

    summary_payload: Dict[str, Any] = {
        "task": task,
        "model": {
            "alias": model_alias,
            "hf_id": model_spec.hf_id,
            "runtime_device": runtime_device,
        },
        "config": {
            "seed": seed,
            "decoding": decoding_cfg.__dict__,
            "task": task_config,
        },
        "num_records": len(all_records),
        "metrics": summary_metrics,
    }

    write_json(summary_payload, summary_json_path)

    csv_row = {
        "task": task,
        "model_alias": model_alias,
        "model_hf_id": model_spec.hf_id,
        "runtime_device": runtime_device,
        "num_records": len(all_records),
    }
    csv_row.update(flatten_for_csv(summary_metrics))
    append_summary_csv(csv_row, summary_csv_path)

    radar_values = build_radar_dimensions(task=task, metrics=summary_metrics)
    export_radar_csv(radar_values, radar_csv_path)

    LOGGER.info("Raw records: %s", raw_path)
    LOGGER.info("Summary JSON: %s", summary_json_path)
    LOGGER.info("Summary CSV: %s", summary_csv_path)
    LOGGER.info("Radar CSV: %s", radar_csv_path)


def build_output_paths(
    output_root: Path,
    task: str,
    model_alias: str,
) -> Tuple[Path, Path, Path, Path]:
    """Constructs output paths for raw and summary artifacts.

    Args:
        output_root: Output root directory.
        task: Benchmark task.
        model_alias: Model alias.

    Returns:
        Tuple of raw JSONL, summary JSON, summary CSV, and radar CSV paths.

    Side Effects:
        None.
    """

    safe_alias = model_alias.replace("/", "_")
    raw_path = output_root / "raw" / f"{task}__{safe_alias}.jsonl"
    summary_json_path = output_root / "summary" / f"{task}__{safe_alias}.json"
    summary_csv_path = output_root / "summary" / f"{task}_summary.csv"
    radar_csv_path = output_root / "summary" / f"{task}__{safe_alias}__radar.csv"
    return raw_path, summary_json_path, summary_csv_path, radar_csv_path


def load_samples(
    task: str,
    task_config: Dict[str, Any],
    max_samples: int | None,
    seed: int,
) -> Sequence[Any]:
    """Loads dataset samples for selected task.

    Args:
        task: Task name.
        task_config: Task-specific configuration.
        max_samples: Optional override for sample count.
        seed: Deterministic seed.

    Returns:
        Sequence of task sample objects.

    Side Effects:
        May download benchmark datasets from Hugging Face cache/backends.

    Raises:
        ValueError: If task is unsupported.
    """

    sample_limit = max_samples if max_samples is not None else task_config.get("max_samples")

    if task == "toxicity":
        return load_real_toxicity_prompts(
            dataset_name=task_config.get("dataset_name", "allenai/real-toxicity-prompts"),
            split=task_config.get("split", "train"),
            max_samples=sample_limit,
            seed=seed,
        )

    if task == "bias":
        return load_bbq(
            dataset_name=task_config.get("dataset_name", "heegyu/bbq"),
            split=task_config.get("split", "test"),
            config_name=task_config.get("config_name"),
            max_samples=sample_limit,
            seed=seed,
        )

    if task == "factuality":
        return load_truthfulqa(
            dataset_name=task_config.get("dataset_name", "truthful_qa"),
            config_name=task_config.get("config_name", "multiple_choice"),
            split=task_config.get("split", "validation"),
            max_samples=sample_limit,
            seed=seed,
        )

    raise ValueError(f"Unsupported task '{task}'.")


def build_evaluator(task: str, task_config: Dict[str, Any], runtime_device: str) -> Any:
    """Builds task evaluator from config.

    Args:
        task: Task name.
        task_config: Task-specific configuration.
        runtime_device: Runtime device string.

    Returns:
        Task evaluator instance.

    Side Effects:
        May initialize external classifier pipelines for specific tasks.

    Raises:
        ValueError: If task is unsupported.
    """

    if task == "toxicity":
        eval_cfg = ToxicityEvalConfig.from_dict(task_config.get("evaluation", {}))
        return ToxicityEvaluator(config=eval_cfg, device=runtime_device)

    if task == "bias":
        eval_cfg = BiasEvalConfig.from_dict(task_config.get("evaluation", {}))
        return BiasEvaluator(config=eval_cfg)

    if task == "factuality":
        eval_cfg = FactualityEvalConfig.from_dict(task_config.get("evaluation", {}))
        return FactualityEvaluator(config=eval_cfg)

    raise ValueError(f"Unsupported task '{task}'.")


def run_generation_loop(
    task: str,
    samples: Sequence[Any],
    generator: TextGenerator,
    evaluator: Any,
    raw_path: Path,
    batch_size: int,
    resume: bool,
) -> None:
    """Executes batched generation and evaluation with resume support.

    Args:
        task: Task name.
        samples: Sequence of sample dataclasses.
        generator: Shared text generator.
        evaluator: Task evaluator.
        raw_path: Destination JSONL file.
        batch_size: Batch size for generation.
        resume: Whether to skip already processed sample IDs.

    Side Effects:
        Appends per-sample records to ``raw_path``.
    """

    existing_records = read_jsonl(raw_path) if resume else []
    processed_ids = {str(item["sample_id"]) for item in existing_records}

    remaining_samples = [sample for sample in samples if str(sample.sample_id) not in processed_ids]
    LOGGER.info(
        "Task '%s' progress: %d existing records, %d remaining",
        task,
        len(existing_records),
        len(remaining_samples),
    )

    total = len(samples)
    completed = len(existing_records)
    failed_batches = 0

    batches = list(batched(remaining_samples, batch_size=max(1, batch_size)))
    progress_bar = tqdm(
        batches,
        desc=f"Generating [{task}]",
        unit="batch",
        initial=len(existing_records) // max(1, batch_size),
        total=len(existing_records) // max(1, batch_size) + len(batches),
    )

    for batch in progress_bar:
        try:
            prompts = [sample.prompt for sample in batch]
            responses = generator.generate_batch(prompts)
            batch_records = evaluator.evaluate_batch(batch, responses)
            append_jsonl(batch_records, raw_path)

            completed += len(batch)
            progress_bar.set_postfix(done=completed, total=total, failed=failed_batches)
        except Exception as exc:
            failed_batches += 1
            failed_ids = [str(s.sample_id) for s in batch]
            LOGGER.error(
                "Batch failed (samples %s): %s. Skipping batch.",
                failed_ids[:3],
                exc,
            )
            progress_bar.set_postfix(done=completed, total=total, failed=failed_batches)
            continue

    if failed_batches:
        LOGGER.warning(
            "Task '%s' completed with %d failed batches out of %d total.",
            task,
            failed_batches,
            len(batches),
        )


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    """Yields fixed-size slices of a sequence.

    Args:
        items: Sequence to iterate.
        batch_size: Desired batch size.

    Yields:
        Sequence slices.

    Side Effects:
        None.
    """

    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def build_radar_dimensions(task: str, metrics: Dict[str, Any]) -> Dict[str, float]:
    """Builds scalar dimensions for radar plotting.

    Args:
        task: Task name.
        metrics: Task summary metrics.

    Returns:
        Dictionary of dimension names and normalized scalar values.

    Side Effects:
        None.
    """

    if task == "toxicity":
        mean_toxicity = metrics.get("mean_toxicity")
        safety_score = 1.0 - float(mean_toxicity) if mean_toxicity is not None else 0.0
        return {"toxicity_safety": max(0.0, min(1.0, safety_score))}

    if task == "bias":
        accuracy = float(metrics.get("accuracy") or 0.0)
        bias_gap = float(metrics.get("bias_gap") or 0.0)
        fairness = 1.0 - min(1.0, abs(bias_gap))
        return {
            "bias_accuracy": max(0.0, min(1.0, accuracy)),
            "bias_fairness": max(0.0, min(1.0, fairness)),
        }

    if task == "factuality":
        objective = metrics.get("objective", {})
        mc_accuracy = float(objective.get("mc_accuracy") or 0.0)
        return {"factuality_accuracy": max(0.0, min(1.0, mc_accuracy))}

    return {}


if __name__ == "__main__":
    main()
