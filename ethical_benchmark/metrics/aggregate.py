"""Aggregation helpers for benchmark metrics and result persistence."""

from __future__ import annotations

import csv
import gzip
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


def compute_bootstrap_ci(
    values: np.ndarray,
    num_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Optional[Dict[str, float]]:
    """Computes non-parametric bootstrap confidence interval of the mean.

    Args:
        values: Numeric samples.
        num_resamples: Number of bootstrap resamples.
        confidence: Confidence level in (0, 1).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with mean, lower, and upper CI bounds, or ``None`` for empty input.

    Side Effects:
        None.
    """

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None

    rng = np.random.default_rng(seed)
    means = []
    n = len(values)
    for _ in range(num_resamples):
        sample = rng.choice(values, size=n, replace=True)
        means.append(float(np.mean(sample)))

    alpha = 1.0 - confidence
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))

    return {
        "mean": float(np.mean(values)),
        "lower": lower,
        "upper": upper,
        "confidence": confidence,
    }


def append_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    """Appends dictionary records to a JSONL file.

    Supports both plain ``.jsonl`` and compressed ``.jsonl.gz`` paths.

    Args:
        records: Iterable of serializable dictionaries.
        path: Destination JSONL path (``.jsonl`` or ``.jsonl.gz``).

    Side Effects:
        Creates parent directories and appends lines to file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    if str(path).endswith(".gz"):
        with gzip.open(path, "at", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with path.open("a", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Reads JSONL records from disk.

    Supports both plain ``.jsonl`` and compressed ``.jsonl.gz`` paths.

    Args:
        path: Source JSONL path.

    Returns:
        Parsed list of dictionaries. Empty list if file does not exist.

    Side Effects:
        Reads from filesystem.
    """

    if not path.exists():
        return []

    output: List[Dict[str, Any]] = []

    if str(path).endswith(".gz"):
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = open(path, "r", encoding="utf-8")  # noqa: SIM115

    try:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            output.append(json.loads(stripped))
    finally:
        handle.close()

    return output


def write_json(data: Dict[str, Any], path: Path) -> None:
    """Writes a dictionary to JSON file.

    Args:
        data: Serializable dictionary payload.
        path: Destination path.

    Side Effects:
        Creates parent directories and writes JSON file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def append_summary_csv(row: Dict[str, Any], path: Path) -> None:
    """Appends a summary row to CSV, creating headers when needed.

    Args:
        row: Flat dictionary row.
        path: Destination CSV path.

    Side Effects:
        Creates CSV file and appends one row.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    fieldnames = sorted(row.keys())
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def flatten_for_csv(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flattens nested dictionaries to one-level key map for CSV export.

    Args:
        data: Nested dictionary.
        prefix: Current key prefix.

    Returns:
        Flattened dictionary.

    Side Effects:
        None.
    """

    output: Dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            output.update(flatten_for_csv(value, prefix=new_key))
        else:
            output[new_key] = value
    return output


def export_radar_csv(summary: Dict[str, float], path: Path) -> None:
    """Exports radar chart friendly ``dimension,value`` CSV.

    Args:
        summary: Mapping from dimension to scalar score.
        path: Destination CSV path.

    Side Effects:
        Writes CSV file for visualization pipelines.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dimension", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])
