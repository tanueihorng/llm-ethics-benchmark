"""Unified Hugging Face model loader for sub-10B open-source language models."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Specification for loading a model from Hugging Face.

    Args:
        alias: Human-readable alias used in configs and CLI.
        hf_id: Hugging Face model identifier.
        trust_remote_code: Whether remote model code is allowed.
        revision: Optional model revision.
        dtype: Preferred precision. One of ``auto``, ``float16``, ``bfloat16``, ``float32``.
    """

    alias: str
    hf_id: str
    trust_remote_code: bool = False
    revision: Optional[str] = None
    dtype: str = "auto"


class HFModelLoader:
    """Loads causal language models and tokenizers from Hugging Face.

    The loader centralizes device placement and dtype resolution to ensure
    consistent experimental settings across models.
    """

    def __init__(self, device: str = "auto", low_cpu_mem_usage: bool = True) -> None:
        """Initializes the model loader.

        Args:
            device: Target device. One of ``auto``, ``cpu``, or ``cuda``.
            low_cpu_mem_usage: Enables memory-efficient model loading.

        Side Effects:
            None.
        """

        self.device = device
        self.low_cpu_mem_usage = low_cpu_mem_usage

    def load(self, spec: ModelSpec) -> Tuple[Any, Any, str]:
        """Loads a tokenizer and causal LM for a given model spec.

        Args:
            spec: Model specification describing what to load.

        Returns:
            Tuple containing:
            1. Loaded causal language model.
            2. Loaded tokenizer.
            3. Resolved runtime device string (``cpu`` or ``cuda``).

        Side Effects:
            Downloads and caches model assets from Hugging Face if not present.
        """

        runtime_device = self._resolve_runtime_device()
        torch_dtype = self._resolve_dtype(spec.dtype, runtime_device)

        LOGGER.info(
            "Loading model alias='%s' hf_id='%s' on device='%s' dtype='%s'",
            spec.alias,
            spec.hf_id,
            runtime_device,
            torch_dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_id,
            revision=spec.revision,
            trust_remote_code=spec.trust_remote_code,
            use_fast=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "revision": spec.revision,
            "trust_remote_code": spec.trust_remote_code,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
        }

        if runtime_device == "cuda":
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = None

        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **model_kwargs)

        if runtime_device == "cpu":
            model.to("cpu")

        model.eval()
        return model, tokenizer, runtime_device

    def _resolve_runtime_device(self) -> str:
        """Resolves runtime device from loader policy and hardware availability.

        Returns:
            ``cuda`` when available and requested/auto-selected, else ``cpu``.
        """

        if self.device == "cpu":
            return "cpu"
        if self.device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _resolve_dtype(dtype: str, runtime_device: str) -> torch.dtype:
        """Maps string dtype preferences to ``torch.dtype`` values.

        Args:
            dtype: Dtype policy string.
            runtime_device: Resolved runtime device.

        Returns:
            A ``torch.dtype`` instance.

        Raises:
            ValueError: If dtype string is unsupported.
        """

        normalized = dtype.strip().lower()
        if normalized == "auto":
            if runtime_device == "cuda":
                return torch.float16
            return torch.float32

        lookup = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }

        if normalized not in lookup:
            raise ValueError(f"Unsupported dtype '{dtype}'.")

        return lookup[normalized]


def build_model_spec(alias: str, registry: Dict[str, Dict[str, Any]]) -> ModelSpec:
    """Builds ``ModelSpec`` from config registry.

    Args:
        alias: Model alias requested by user.
        registry: Model registry loaded from configuration.

    Returns:
        Parsed ``ModelSpec`` for the alias.

    Side Effects:
        None.

    Raises:
        KeyError: If alias is missing from the registry.
    """

    if alias not in registry:
        raise KeyError(f"Unknown model alias '{alias}'. Available: {sorted(registry.keys())}")

    entry = registry[alias]
    return ModelSpec(
        alias=alias,
        hf_id=entry["hf_id"],
        trust_remote_code=bool(entry.get("trust_remote_code", False)),
        revision=entry.get("revision"),
        dtype=entry.get("dtype", "auto"),
    )
