"""Task evaluators for ethical benchmarking.

This package intentionally avoids eager imports of every evaluator module.
Some backends, especially toxicity classification, can pull in heavyweight
ML runtime dependencies at import time. Keeping these imports lazy makes
lightweight unit tests and non-toxicity workflows much more reliable.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BiasEvalConfig",
    "BiasEvaluator",
    "FactualityEvalConfig",
    "FactualityEvaluator",
    "ToxicityEvalConfig",
    "ToxicityEvaluator",
]


_EXPORT_MAP = {
    "BiasEvalConfig": ("ethical_benchmark.evaluators.bias_eval", "BiasEvalConfig"),
    "BiasEvaluator": ("ethical_benchmark.evaluators.bias_eval", "BiasEvaluator"),
    "FactualityEvalConfig": (
        "ethical_benchmark.evaluators.factuality_eval",
        "FactualityEvalConfig",
    ),
    "FactualityEvaluator": (
        "ethical_benchmark.evaluators.factuality_eval",
        "FactualityEvaluator",
    ),
    "ToxicityEvalConfig": (
        "ethical_benchmark.evaluators.toxicity_eval",
        "ToxicityEvalConfig",
    ),
    "ToxicityEvaluator": (
        "ethical_benchmark.evaluators.toxicity_eval",
        "ToxicityEvaluator",
    ),
}


def __getattr__(name: str) -> Any:
    """Lazily resolves evaluator exports on first access."""

    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
