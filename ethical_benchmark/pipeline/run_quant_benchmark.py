"""CLI and execution logic for one model x benchmark quantization run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from tqdm import tqdm

from ethical_benchmark.benchmarks.base import BenchmarkItem
from ethical_benchmark.benchmarks.registry import build_benchmark_plugin
from ethical_benchmark.metrics.aggregate import append_jsonl, append_summary_csv, flatten_for_csv, read_jsonl, write_json
from ethical_benchmark.quant.config_schema import QuantizationConfig, load_quant_config

LOGGER = logging.getLogger(__name__)

REQUIRED_RECORD_FIELDS = {
    "benchmark",
    "prompt_id",
    "prompt_text",
    "response",
    "score_fields",
    "family",
    "size_b",
    "quantized",
    "pair_id",
    "model_id",
    "generation_config",
    "seed",
    "timestamp",
}


def get_model_stack() -> Any:
    """Loads model stack lazily to avoid import-time hard dependency crashes.

    Returns:
        Tuple of ``(DecodingConfig, TextGenerator, set_global_seed, HFModelLoader, ModelSpec)``.

    Side Effects:
        Imports model modules on demand.
    """

    from ethical_benchmark.models.generation import DecodingConfig, TextGenerator, set_global_seed
    from ethical_benchmark.models.loader import HFModelLoader, ModelSpec

    return DecodingConfig, TextGenerator, set_global_seed, HFModelLoader, ModelSpec


# Backward-compatible alias used in tests.
_model_stack = get_model_stack


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for one quantization benchmark run.

    Returns:
        Parsed CLI namespace.

    Side Effects:
        Reads process command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Run one model x benchmark quantization evaluation")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to quantization config")
    parser.add_argument("--model", required=True, help="Model alias from config matrix")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["harmbench", "xstest", "mmlu"],
        help="Benchmark name",
    )
    parser.add_argument("--output_dir", default="results", help="Root directory for run outputs")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample cap override")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional generation batch size override")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device policy",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing raw records (default)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume and process all prompts",
    )
    parser.add_argument(
        "--force_restart",
        action="store_true",
        help="Delete existing raw records before execution",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configures root logger.

    Args:
        level: Logging level string.

    Side Effects:
        Configures global logging behavior.
    """

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def normalize_model_key(alias: str) -> str:
    """Normalizes model alias into filesystem-safe key.

    Args:
        alias: Model alias.

    Returns:
        Normalized key.

    Side Effects:
        None.
    """

    normalized = alias.strip().lower().replace("/", "_")
    normalized = normalized.replace(" ", "_").replace(".", "_")
    return normalized


def build_run_paths(output_dir: Path, model_alias: str, benchmark: str) -> Dict[str, Path]:
    """Builds output paths for one run.

    Args:
        output_dir: Results root.
        model_alias: Model alias.
        benchmark: Benchmark name.

    Returns:
        Dictionary with ``run_dir``, ``raw_path``, ``summary_json``, and ``summary_csv``.

    Side Effects:
        None.
    """

    model_key = normalize_model_key(model_alias)
    run_dir = output_dir / model_key / benchmark
    return {
        "run_dir": run_dir,
        "raw_path": run_dir / "raw.jsonl",
        "summary_json": run_dir / "summary.json",
        "summary_csv": output_dir / "summary" / f"{benchmark}_runs.csv",
    }


def validate_record_schema(record: Dict[str, Any]) -> None:
    """Validates required per-response record schema.

    Args:
        record: Record dictionary.

    Side Effects:
        None.

    Raises:
        ValueError: If required fields are missing or malformed.
    """

    missing = sorted(REQUIRED_RECORD_FIELDS - set(record.keys()))
    if missing:
        raise ValueError(f"Record missing required fields: {missing}")

    if not isinstance(record["score_fields"], dict):
        raise ValueError("'score_fields' must be a dictionary.")


def get_processed_prompt_ids(raw_path: Path) -> set[str]:
    """Returns already processed prompt IDs from raw JSONL.

    Args:
        raw_path: Raw record path.

    Returns:
        Set of prompt IDs.

    Side Effects:
        Reads JSONL file from disk.
    """

    return {str(item.get("prompt_id")) for item in read_jsonl(raw_path)}


def prepare_remaining_items(items: Sequence[BenchmarkItem], processed_ids: set[str]) -> List[BenchmarkItem]:
    """Filters out already-processed benchmark items.

    Args:
        items: Full benchmark items.
        processed_ids: Prompt IDs already present in raw records.

    Returns:
        Remaining items to process.

    Side Effects:
        None.
    """

    return [item for item in items if str(item.prompt_id) not in processed_ids]


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    """Yields fixed-size batches from a sequence.

    Args:
        items: Sequence to batch.
        batch_size: Batch size.

    Yields:
        Sequence chunks.

    Side Effects:
        None.
    """

    for idx in range(0, len(items), max(1, batch_size)):
        yield items[idx : idx + max(1, batch_size)]


def execute_quant_benchmark_loaded(
    config: QuantizationConfig,
    model_alias: str,
    benchmark_name: str,
    output_dir: Path,
    seed: int,
    resume: bool,
    force_restart: bool,
    max_samples_override: int | None,
    batch_size_override: int | None,
    generator: Any,
    runtime_device: str,
    decoding_cfg: Any,
) -> Dict[str, Any]:
    """Executes one benchmark using an already-loaded model generator.

    Args:
        config: Validated quantization config.
        model_alias: Model alias.
        benchmark_name: Benchmark name.
        output_dir: Result root directory.
        seed: Random seed.
        resume: Whether to skip completed prompts.
        force_restart: Whether to remove existing raw file first.
        max_samples_override: Optional benchmark sample override.
        batch_size_override: Optional generation batch override.
        generator: Pre-loaded text generator.
        runtime_device: Runtime device string.
        decoding_cfg: Decoding configuration object.

    Returns:
        Summary payload dictionary.

    Side Effects:
        Loads benchmark data, runs inference through supplied generator,
        writes JSONL/JSON/CSV outputs.
    """

    benchmark = benchmark_name.strip().lower()
    if model_alias not in config.models:
        raise KeyError(f"Unknown model alias '{model_alias}'.")

    model_entry = config.models[model_alias]
    if benchmark not in model_entry.benchmarks:
        raise ValueError(
            f"Benchmark '{benchmark}' is not enabled for model '{model_alias}'."
        )

    benchmark_entry = config.benchmarks[benchmark]
    plugin = build_benchmark_plugin(benchmark, benchmark_entry.model_dump())

    max_samples = (
        max_samples_override if max_samples_override is not None else benchmark_entry.max_samples
    )
    batch_size = (
        batch_size_override if batch_size_override is not None else benchmark_entry.batch_size
    )

    paths = build_run_paths(output_dir=output_dir, model_alias=model_alias, benchmark=benchmark)
    raw_path = paths["raw_path"]
    summary_json_path = paths["summary_json"]
    summary_csv_path = paths["summary_csv"]

    if force_restart and raw_path.exists():
        raw_path.unlink()

    # Without resume we must also clear any existing raw file; otherwise the
    # newly processed prompts get appended on top of stale rows and aggregate
    # double-counts. force_restart already handled above; this covers --no-resume.
    if not resume and raw_path.exists():
        raw_path.unlink()

    items = plugin.load_items(max_samples=max_samples, seed=seed)
    processed_ids = get_processed_prompt_ids(raw_path) if resume else set()
    remaining = prepare_remaining_items(items=items, processed_ids=processed_ids)

    LOGGER.info(
        "Running %s on %s: total=%d, existing=%d, remaining=%d",
        benchmark,
        model_alias,
        len(items),
        len(processed_ids),
        len(remaining),
    )

    progress = tqdm(
        batched(remaining, batch_size=batch_size),
        total=(len(remaining) + max(1, batch_size) - 1) // max(1, batch_size),
        desc=f"{model_alias}:{benchmark}",
        leave=False,
    )

    for batch in progress:
        prompts = [plugin.build_prompt(item) for item in batch]
        responses = generator.generate_batch(prompts)
        now = datetime.now(timezone.utc).isoformat()

        records: List[Dict[str, Any]] = []
        for item, prompt, response in zip(batch, prompts, responses):
            score_fields = plugin.score_response(item=item, response=response)
            record = {
                "benchmark": benchmark,
                "prompt_id": str(item.prompt_id),
                "prompt_text": prompt,
                "response": response,
                "score_fields": score_fields,
                "family": model_entry.family,
                "size_b": model_entry.size_b,
                "quantized": model_entry.quantized,
                "pair_id": model_entry.pair_id,
                "model_alias": model_alias,
                "model_id": model_entry.model_id,
                "generation_config": decoding_cfg.__dict__,
                "seed": seed,
                "timestamp": now,
            }
            validate_record_schema(record)
            records.append(record)

        append_jsonl(records, raw_path)

    all_records = read_jsonl(raw_path)
    metrics = plugin.aggregate(all_records)

    summary_payload: Dict[str, Any] = {
        "study_name": config.study_name,
        "model_alias": model_alias,
        "model_id": model_entry.model_id,
        "family": model_entry.family,
        "size_b": model_entry.size_b,
        "quantized": model_entry.quantized,
        "pair_id": model_entry.pair_id,
        "benchmark": benchmark,
        "runtime_device": runtime_device,
        "seed": seed,
        "num_records": len(all_records),
        "generation_config": decoding_cfg.__dict__,
        "benchmark_config": benchmark_entry.model_dump(),
        "metrics": metrics,
    }

    write_json(summary_payload, summary_json_path)

    csv_row = {
        "study_name": config.study_name,
        "model_alias": model_alias,
        "model_id": model_entry.model_id,
        "family": model_entry.family,
        "size_b": model_entry.size_b,
        "quantized": model_entry.quantized,
        "pair_id": model_entry.pair_id,
        "benchmark": benchmark,
        "seed": seed,
        "runtime_device": runtime_device,
        "num_records": len(all_records),
    }
    csv_row.update(flatten_for_csv(metrics))
    append_summary_csv(csv_row, summary_csv_path)

    LOGGER.info("Completed %s x %s; summary: %s", model_alias, benchmark, summary_json_path)
    return summary_payload


def execute_quant_benchmark(
    config: QuantizationConfig,
    model_alias: str,
    benchmark_name: str,
    output_dir: Path,
    seed: int,
    device: str,
    resume: bool,
    force_restart: bool,
    max_samples_override: int | None,
    batch_size_override: int | None,
) -> Dict[str, Any]:
    """Executes one model x benchmark run and persists outputs.

    Args:
        config: Validated quantization config.
        model_alias: Model alias.
        benchmark_name: Benchmark name.
        output_dir: Result root directory.
        seed: Random seed.
        device: Runtime device policy.
        resume: Whether to skip completed prompts.
        force_restart: Whether to remove existing raw file first.
        max_samples_override: Optional benchmark sample override.
        batch_size_override: Optional generation batch override.

    Returns:
        Summary payload dictionary.

    Side Effects:
        Loads model and dataset, runs inference, writes JSONL/JSON/CSV outputs.
    """

    (
        DecodingConfigCls,
        TextGeneratorCls,
        set_global_seed_fn,
        HFModelLoaderCls,
        ModelSpecCls,
    ) = _model_stack()

    if model_alias not in config.models:
        raise KeyError(f"Unknown model alias '{model_alias}'.")
    model_entry = config.models[model_alias]

    decoding_cfg = DecodingConfigCls.from_dict(config.decoding.model_dump())
    set_global_seed_fn(seed)

    model_spec = ModelSpecCls(
        alias=model_alias,
        hf_id=model_entry.model_id,
        trust_remote_code=model_entry.trust_remote_code,
        revision=model_entry.revision,
        dtype=model_entry.dtype,
        quantized=model_entry.quantized,
    )

    model_loader = HFModelLoaderCls(device=device)
    model, tokenizer, runtime_device = model_loader.load(model_spec)
    generator = TextGeneratorCls(
        model=model,
        tokenizer=tokenizer,
        device=runtime_device,
        config=decoding_cfg,
    )

    return execute_quant_benchmark_loaded(
        config=config,
        model_alias=model_alias,
        benchmark_name=benchmark_name,
        output_dir=output_dir,
        seed=seed,
        resume=resume,
        force_restart=force_restart,
        max_samples_override=max_samples_override,
        batch_size_override=batch_size_override,
        generator=generator,
        runtime_device=runtime_device,
        decoding_cfg=decoding_cfg,
    )


def main() -> None:
    """CLI entrypoint for one quantization benchmark run.

    Side Effects:
        Executes one model x benchmark experiment and writes output artifacts.
    """

    args = parse_args()
    setup_logging(args.log_level)

    config = load_quant_config(args.config)
    execute_quant_benchmark(
        config=config,
        model_alias=args.model,
        benchmark_name=args.benchmark,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        force_restart=args.force_restart,
        max_samples_override=args.max_samples,
        batch_size_override=args.batch_size,
    )


if __name__ == "__main__":
    main()
