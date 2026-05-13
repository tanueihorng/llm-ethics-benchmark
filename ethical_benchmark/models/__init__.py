"""Model loading and generation utilities.

Imports are resolved lazily so lightweight tooling can inspect the package
without forcing Torch initialization during module import.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DecodingConfig",
    "HFModelLoader",
    "ModelSpec",
    "TextGenerator",
    "build_model_spec",
    "set_global_seed",
]


_EXPORT_MAP = {
    "DecodingConfig": ("ethical_benchmark.models.generation", "DecodingConfig"),
    "TextGenerator": ("ethical_benchmark.models.generation", "TextGenerator"),
    "set_global_seed": ("ethical_benchmark.models.generation", "set_global_seed"),
    "HFModelLoader": ("ethical_benchmark.models.loader", "HFModelLoader"),
    "ModelSpec": ("ethical_benchmark.models.loader", "ModelSpec"),
    "build_model_spec": ("ethical_benchmark.models.loader", "build_model_spec"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolves model exports on first access."""

    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
