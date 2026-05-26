"""Pre-cache datasets and model weights for TC1 offline-mode runs.

Run this ONCE on the TC1 HEAD NODE (not on a compute node) after activating
the fyp-tc1 conda environment:

    module load anaconda
    source activate fyp-tc1
    cd /tc1home/FYP/utan001/fyp_quant/repo
    huggingface-cli login   # paste your HF token (one-time, for Llama gating)
    python scripts/prefetch_tc1.py

The script reads the configured datasets and model_ids from configs/tc1.yaml,
then triggers a download into the default Hugging Face cache
(~/.cache/huggingface/). Once cached, all SLURM jobs will read from the local
cache and run with HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1, and
TRANSFORMERS_OFFLINE=1 set in the sbatch setup_commands.

Per the TC1 user guide (page 16): installing packages and downloading files
on the head node is the expected workflow ("conda install", "pip install"
are explicitly demonstrated there). What is forbidden is running compute
("executing coding and occupying high CPU and Memory usage"). This script
performs only network I/O and minimal CPU work, so it is policy-compliant.

Usage:
    python scripts/prefetch_tc1.py                     # default config
    python scripts/prefetch_tc1.py --config configs/tc1.yaml
    python scripts/prefetch_tc1.py --datasets-only     # skip model weights
    python scripts/prefetch_tc1.py --models-only       # skip datasets
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


def _collect_dataset_targets(config: dict) -> List[Tuple[str, str, List[str] | None]]:
    """Returns a list of (dataset_name, split, subjects-or-None) tuples to fetch."""

    targets: List[Tuple[str, str, List[str] | None]] = []
    benchmarks = config.get("benchmarks", {})
    for name, bench in benchmarks.items():
        dataset_name = str(bench.get("dataset_name", "")).strip()
        split = str(bench.get("split", "train")).strip()
        if not dataset_name:
            continue
        subjects = bench.get("subjects")
        if isinstance(subjects, list) and subjects:
            targets.append((dataset_name, split, [str(s) for s in subjects]))
        else:
            targets.append((dataset_name, split, None))
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


def _prefetch_datasets(targets: Iterable[Tuple[str, str, List[str] | None]]) -> None:
    from datasets import load_dataset  # lazy import; not needed for --models-only

    for dataset_name, split, subjects in targets:
        if subjects:
            for subject in subjects:
                LOGGER.info("Caching dataset %s/%s (split=%s)", dataset_name, subject, split)
                load_dataset(dataset_name, subject, split=split)
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

    LOGGER.info("Prefetch complete. Subsequent SLURM jobs can run with HF_*_OFFLINE=1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
