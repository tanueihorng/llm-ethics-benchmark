"""Unified Hugging Face model loader for sub-10B open-source language models."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import logging
from typing import Any, Dict, Optional, Tuple

LOGGER = logging.getLogger(__name__)


def _get_torch() -> Any:
    """Imports torch lazily to avoid import-time runtime crashes."""

    return import_module("torch")


def _get_transformer_classes() -> Tuple[Any, Any]:
    """Imports Hugging Face auto classes lazily."""

    transformers = import_module("transformers")
    return transformers.AutoModelForCausalLM, transformers.AutoTokenizer


def _build_bnb_4bit_config() -> Any:
    """Builds a BitsAndBytesConfig for nf4 4-bit loading.

    Returns:
        ``transformers.BitsAndBytesConfig`` instance.

    Raises:
        ImportError: If ``bitsandbytes`` / ``accelerate`` are not installed.
    """

    transformers = import_module("transformers")
    torch = _get_torch()
    return transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


@dataclass(frozen=True)
class ModelSpec:
    """Specification for loading a model from Hugging Face.

    Args:
        alias: Human-readable alias used in configs and CLI.
        hf_id: Hugging Face model identifier.
        trust_remote_code: Whether remote model code is allowed.
        revision: Optional model revision.
        dtype: Preferred precision. One of ``auto``, ``float16``, ``bfloat16``, ``float32``.
        quantized: When True, load the checkpoint in 4-bit via bitsandbytes nf4.
    """

    alias: str
    hf_id: str
    trust_remote_code: bool = False
    revision: Optional[str] = None
    dtype: str = "auto"
    quantized: bool = False


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
        AutoModelForCausalLM, AutoTokenizer = _get_transformer_classes()

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

        if spec.quantized:
            if runtime_device != "cuda":
                raise RuntimeError(
                    f"Model '{spec.alias}' is marked quantized=true but runtime device is "
                    f"'{runtime_device}'. 4-bit bitsandbytes loading requires CUDA."
                )
            model_kwargs["quantization_config"] = _build_bnb_4bit_config()
            LOGGER.info("Loading '%s' with bitsandbytes 4-bit (nf4) quantization.", spec.alias)

        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **model_kwargs)

        if spec.quantized and not getattr(model, "is_loaded_in_4bit", False):
            LOGGER.warning(
                "Model '%s' requested 4-bit but is_loaded_in_4bit is False after load. "
                "Verify the checkpoint and bitsandbytes installation.",
                spec.alias,
            )

        if runtime_device == "cpu":
            model.to("cpu")

        model.eval()
        return model, tokenizer, runtime_device

    def _resolve_runtime_device(self) -> str:
        """Resolves runtime device from loader policy and hardware availability.

        Returns:
            ``cuda`` when available and requested/auto-selected, else ``cpu``.
        """

        torch = _get_torch()
        if self.device == "cpu":
            return "cpu"
        if self.device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _resolve_dtype(dtype: str, runtime_device: str) -> Any:
        """Maps string dtype preferences to ``torch.dtype`` values.

        Args:
            dtype: Dtype policy string.
            runtime_device: Resolved runtime device.

        Returns:
            A ``torch.dtype`` instance.

        Raises:
            ValueError: If dtype string is unsupported.
        """

        torch = _get_torch()
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
        quantized=bool(entry.get("quantized", False)),
    )
