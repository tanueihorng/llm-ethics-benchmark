"""Prepare datasets and model weights for TC1 offline-mode runs.

Run on an approved networked local/staging machine with the target environment,
then transfer the prepared Hugging Face cache to TC1 without credentials.
See docs/TC1_CLUSTER_RUNBOOK.md section 4.6 for cache transfer and offline smoke
validation. Do not run this custom script on the TC1 head node: load_dataset can
prepare datasets as well as download files, and package-install permission is
not blanket authorization for that work.

The script reads configured datasets and model_ids from configs/tc1.yaml and
populates the local Hugging Face cache. It does not change SLURM offline flags.

Usage:
    python scripts/prefetch_tc1.py --config configs/tc1.yaml
    python scripts/prefetch_tc1.py --datasets-only
    python scripts/prefetch_tc1.py --models-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import yaml

LOGGER = logging.getLogger("prefetch_tc1")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _collect_dataset_targets(config: dict) -> List[Tuple[str, str, List[str] | None, str | None]]:
    """Returns a list of (dataset_name, split, subjects-or-None, config_name-or-None) tuples to fetch."""

    targets: List[Tuple[str, str, List[str] | None, str | None]] = []
    benchmarks = config.get("benchmarks", {})
    for name, bench in benchmarks.items():
        # Skip benchmarks that load from a local CSV (e.g. XSTest); no HF fetch needed.
        if bench.get("local_csv"):
            LOGGER.info("Skipping HF prefetch for benchmark %s (uses local_csv).", name)
            continue
        dataset_name = str(bench.get("dataset_name", "")).strip()
        split = str(bench.get("split", "train")).strip()
        if not dataset_name:
            continue
        # Apply default for HarmBench since the dataset requires a config_name
        # but our YAML may not specify it explicitly.
        default_config_name = "standard" if dataset_name == "walledai/HarmBench" else None
        config_name = bench.get("config_name", default_config_name)
        subjects = bench.get("subjects")
        if isinstance(subjects, list) and subjects:
            # MMLU pattern: subjects are themselves dataset configs, override config_name
            targets.append((dataset_name, split, [str(s) for s in subjects], None))
        else:
            targets.append((dataset_name, split, None, config_name))
    return targets


def _collect_model_targets(config: dict) -> List[str]:
    """Returns the unique set of HF model_ids referenced in the config."""

    models = config.get("models", {})
    unique: Set[str] = set()
    for entry in models.values():
        model_id = str(entry.get("model_id", "")).strip()
        if model_id:
            unique.add(model_id)
    return sorted(unique)


def _prefetch_datasets(targets: Iterable[Tuple[str, str, List[str] | None, str | None]]) -> None:
    from datasets import load_dataset  # lazy import; not needed for --models-only

    for dataset_name, split, subjects, config_name in targets:
        if subjects:
            for subject in subjects:
                LOGGER.info("Caching dataset %s/%s (split=%s)", dataset_name, subject, split)
                load_dataset(dataset_name, subject, split=split)
        elif config_name:
            LOGGER.info("Caching dataset %s/%s (split=%s)", dataset_name, config_name, split)
            load_dataset(dataset_name, config_name, split=split)
        else:
            LOGGER.info("Caching dataset %s (split=%s)", dataset_name, split)
            load_dataset(dataset_name, split=split)


def _prefetch_models(model_ids: Iterable[str]) -> None:
    from huggingface_hub import snapshot_download  # lazy import

    for model_id in model_ids:
        LOGGER.info("Caching model weights for %s", model_id)
        # snapshot_download mirrors the full repo into ~/.cache/huggingface/hub.
        # It is the programmatic equivalent of `huggingface-cli download`.
        snapshot_download(repo_id=model_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tc1.yaml"),
        help="Path to the TC1 quantization config (default: configs/tc1.yaml).",
    )
    parser.add_argument(
        "--datasets-only",
        action="store_true",
        help="Cache only benchmark datasets; skip model weights.",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Cache only model weights; skip benchmark datasets.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help=(
            "Also cache the HarmBench judge classifier "
            "(cais/HarmBench-Llama-2-13b-cls) for T20 judge validation."
        ),
    )
    parser.add_argument(
        "--judge-model-id",
        default="cais/HarmBench-Llama-2-13b-cls",
        help="HF model id of the judge classifier to cache with --judge.",
    )
    return parser.parse_args()


def main() -> int:
    _setup_logging()
    args = parse_args()

    if args.datasets_only and args.models_only:
        LOGGER.error("--datasets-only and --models-only are mutually exclusive.")
        return 2

    if not args.config.exists():
        LOGGER.error("Config not found at %s", args.config)
        return 1

    config = _load_config(args.config)

    if not args.models_only:
        dataset_targets = _collect_dataset_targets(config)
        LOGGER.info("Will cache %d dataset target(s).", len(dataset_targets))
        _prefetch_datasets(dataset_targets)

    if not args.datasets_only:
        model_ids = _collect_model_targets(config)
        LOGGER.info("Will cache %d model repo(s): %s", len(model_ids), model_ids)
        _prefetch_models(model_ids)

    if args.judge:
        LOGGER.info("Caching judge classifier: %s", args.judge_model_id)
        _prefetch_models([args.judge_model_id])

    LOGGER.info("Prefetch complete. Subsequent SLURM jobs can run with HF_*_OFFLINE=1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
