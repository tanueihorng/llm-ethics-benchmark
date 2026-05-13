"""Registry for benchmark plugin construction."""

from __future__ import annotations

from typing import Any, Dict

from ethical_benchmark.benchmarks.base import BenchmarkPlugin
from ethical_benchmark.benchmarks.harmbench import HarmBenchPlugin
from ethical_benchmark.benchmarks.mmlu import MMLUPlugin
from ethical_benchmark.benchmarks.xstest import XSTestPlugin


def build_benchmark_plugin(name: str, config: Dict[str, Any]) -> BenchmarkPlugin:
    """Builds a benchmark plugin by name.

    Args:
        name: Benchmark name.
        config: Benchmark-specific configuration.

    Returns:
        Benchmark plugin instance.

    Side Effects:
        None.

    Raises:
        ValueError: If benchmark name is unsupported.
    """

    normalized = name.strip().lower()
    if normalized == "harmbench":
        return HarmBenchPlugin(config)
    if normalized == "xstest":
        return XSTestPlugin(config)
    if normalized == "mmlu":
        return MMLUPlugin(config)

    raise ValueError(f"Unsupported benchmark '{name}'.")
