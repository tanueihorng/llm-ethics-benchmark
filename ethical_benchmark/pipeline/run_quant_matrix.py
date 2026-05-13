"""CLI for executing the full quantization benchmark matrix."""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
import sys
from typing import Dict, List

from ethical_benchmark.pipeline.run_quant_benchmark import (
    execute_quant_benchmark,
    execute_quant_benchmark_loaded,
    get_model_stack,
    setup_logging,
)
from ethical_benchmark.quant.config_schema import QuantizationConfig, load_quant_config

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for matrix execution.

    Returns:
        Parsed CLI namespace.

    Side Effects:
        Reads command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Run the full model x benchmark quantization matrix")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to quantization config")
    parser.add_argument("--output_dir", default="results", help="Result root directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device policy",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample cap override")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional batch size override")
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=["harmbench", "xstest", "mmlu"],
        help="Optional benchmark filter; repeatable",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Optional model alias filter; repeatable",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume partial runs (default)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume",
    )
    parser.add_argument("--force_restart", action="store_true", help="Delete prior raw outputs first")
    parser.add_argument(
        "--reuse_loaded_model",
        action="store_true",
        default=True,
        help="Reuse one model load across benchmarks for each model (default)",
    )
    parser.add_argument(
        "--no-reuse_loaded_model",
        dest="reuse_loaded_model",
        action="store_false",
        help="Disable model reuse and load per run",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def _resolve_models(all_models: List[str], selected: List[str] | None) -> List[str]:
    """Resolves selected model aliases.

    Args:
        all_models: All model aliases from config.
        selected: Optional requested aliases.

    Returns:
        Ordered model alias list.

    Side Effects:
        None.

    Raises:
        KeyError: If selected alias is unknown.
    """

    if not selected:
        return list(all_models)

    unknown = [item for item in selected if item not in all_models]
    if unknown:
        raise KeyError(f"Unknown model aliases: {unknown}")

    return list(selected)


def _selected_benchmarks_for_model(
    config: QuantizationConfig,
    model_alias: str,
    benchmark_filter: set[str] | None,
) -> List[str]:
    """Returns ordered benchmark list for a model with optional filtering.

    Args:
        config: Quantization config.
        model_alias: Model alias.
        benchmark_filter: Optional benchmark name filter.

    Returns:
        Ordered benchmark names.

    Side Effects:
        None.
    """

    benchmarks = config.models[model_alias].benchmarks
    if benchmark_filter is None:
        return list(benchmarks)
    return [name for name in benchmarks if name in benchmark_filter]


def _release_model_memory() -> None:
    """Attempts to release CUDA memory between model loops.

    Side Effects:
        Runs ``gc.collect`` and calls ``torch.cuda.empty_cache`` when torch is
        already imported and CUDA is available.
    """

    gc.collect()

    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return

    try:
        cuda = getattr(torch_mod, "cuda", None)
        if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
            empty_cache = getattr(cuda, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
    except Exception:  # pragma: no cover
        return


def run_quant_matrix(
    config: QuantizationConfig,
    output_dir: Path,
    seed: int,
    device: str,
    max_samples: int | None,
    batch_size: int | None,
    selected_models: List[str],
    benchmark_filter: set[str] | None,
    resume: bool,
    force_restart: bool,
    reuse_loaded_model: bool,
) -> None:
    """Runs selected matrix combinations.

    Args:
        config: Quantization config.
        output_dir: Output root directory.
        seed: Random seed.
        device: Device policy.
        max_samples: Optional sample cap override.
        batch_size: Optional generation batch override.
        selected_models: Ordered model aliases.
        benchmark_filter: Optional benchmark filter.
        resume: Resume mode.
        force_restart: Force restart mode.
        reuse_loaded_model: Whether to reuse one model load per model.

    Side Effects:
        Executes inference and writes output artifacts.
    """

    runs: List[Dict[str, str]] = []
    for model_alias in selected_models:
        for benchmark in _selected_benchmarks_for_model(config, model_alias, benchmark_filter):
            runs.append({"model": model_alias, "benchmark": benchmark})

    LOGGER.info("Planned runs: %d", len(runs))

    if not reuse_loaded_model:
        for idx, run in enumerate(runs, start=1):
            LOGGER.info("[%d/%d] %s x %s", idx, len(runs), run["model"], run["benchmark"])
            execute_quant_benchmark(
                config=config,
                model_alias=run["model"],
                benchmark_name=run["benchmark"],
                output_dir=output_dir,
                seed=seed,
                device=device,
                resume=resume,
                force_restart=force_restart,
                max_samples_override=max_samples,
                batch_size_override=batch_size,
            )
        return

    (
        DecodingConfigCls,
        TextGeneratorCls,
        set_global_seed_fn,
        HFModelLoaderCls,
        ModelSpecCls,
    ) = get_model_stack()

    current_idx = 0
    for model_alias in selected_models:
        benchmarks = _selected_benchmarks_for_model(config, model_alias, benchmark_filter)
        if not benchmarks:
            continue

        model_entry = config.models[model_alias]
        set_global_seed_fn(seed)
        decoding_cfg = DecodingConfigCls.from_dict(config.decoding.model_dump())

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

        LOGGER.info(
            "Loaded model once for reuse: %s (%d benchmark runs)",
            model_alias,
            len(benchmarks),
        )

        # Use try/finally so an exception in any benchmark still releases this
        # model's memory before the next model loads. On TC1 with mem: 10G,
        # stacking two large models OOMs immediately.
        try:
            for benchmark in benchmarks:
                current_idx += 1
                LOGGER.info("[%d/%d] %s x %s", current_idx, len(runs), model_alias, benchmark)
                execute_quant_benchmark_loaded(
                    config=config,
                    model_alias=model_alias,
                    benchmark_name=benchmark,
                    output_dir=output_dir,
                    seed=seed,
                    resume=resume,
                    force_restart=force_restart,
                    max_samples_override=max_samples,
                    batch_size_override=batch_size,
                    generator=generator,
                    runtime_device=runtime_device,
                    decoding_cfg=decoding_cfg,
                )
        finally:
            del model
            del tokenizer
            del generator
            _release_model_memory()


def main() -> None:
    """CLI entrypoint for matrix execution.

    Side Effects:
        Executes all selected model x benchmark combinations and writes outputs.
    """

    args = parse_args()
    setup_logging(args.log_level)

    config = load_quant_config(args.config)
    selected_models = _resolve_models(list(config.models.keys()), args.model)
    benchmark_filter = {item.lower() for item in args.benchmark} if args.benchmark else None

    run_quant_matrix(
        config=config,
        output_dir=Path(args.output_dir),
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

    LOGGER.info("Matrix execution complete.")


if __name__ == "__main__":
    main()
