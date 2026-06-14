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


def _build_bnb_4bit_config(compute_dtype: Any) -> Any:
    """Builds a BitsAndBytesConfig for nf4 4-bit loading.

    Args:
        compute_dtype: Torch dtype used for 4-bit compute.

    Returns:
        ``transformers.BitsAndBytesConfig`` instance.

    Raises:
        ImportError: If ``bitsandbytes`` / ``accelerate`` are not installed.
    """

    transformers = import_module("transformers")
    return transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _quantization_active(model: Any) -> bool:
    """Reports whether a loaded model shows any sign of active quantization.

    Robust across transformers versions: older builds expose
    ``is_loaded_in_4bit``; newer ones surface ``is_quantized`` and/or an
    ``hf_quantizer`` attribute. Used to refuse a run where a checkpoint marked
    ``quantized=true`` silently loaded in full precision (which would mislabel
    fp16 results as 4-bit and corrupt the baseline-vs-4bit comparison).

    Args:
        model: A loaded Hugging Face model.

    Returns:
        ``True`` if any quantization signal is present.

    Side Effects:
        None.
    """

    return (
        bool(getattr(model, "is_loaded_in_4bit", False))
        or bool(getattr(model, "is_quantized", False))
        or getattr(model, "hf_quantizer", None) is not None
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
        attn_implementation: Optional attention backend passed to
            ``from_pretrained`` (e.g. ``eager`` for Phi-4-mini on a V100). When
            None the kwarg is omitted entirely, leaving the load call identical
            to the existing models.
    """

    alias: str
    hf_id: str
    trust_remote_code: bool = False
    revision: Optional[str] = None
    dtype: str = "auto"
    quantized: bool = False
    attn_implementation: Optional[str] = None


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
            model_kwargs["quantization_config"] = _build_bnb_4bit_config(torch_dtype)
            LOGGER.info("Loading '%s' with bitsandbytes 4-bit (nf4) quantization.", spec.alias)

        # Only inject attn_implementation when explicitly set. When None, the
        # kwarg is omitted so the from_pretrained call for the existing models
        # (Qwen/Llama/Mistral) is byte-identical to before this field existed.
        if spec.attn_implementation:
            model_kwargs["attn_implementation"] = spec.attn_implementation
            LOGGER.info(
                "Loading '%s' with attn_implementation='%s'.", spec.alias, spec.attn_implementation
            )

        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **model_kwargs)

        if spec.quantized and not _quantization_active(model):
            # Refuse to proceed if quantization did not actually engage: a model
            # that silently loaded in fp16 but is recorded as quantized=True would
            # contaminate the entire baseline-vs-4bit comparison.
            raise RuntimeError(
                f"Model '{spec.alias}' requested 4-bit quantization but no "
                f"quantization is active after load (is_loaded_in_4bit / "
                f"is_quantized / hf_quantizer all absent). Refusing to proceed: "
                f"results would be fp16 mislabelled as 4-bit. Check the "
                f"checkpoint and the bitsandbytes installation."
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
        attn_implementation=entry.get("attn_implementation"),
    )
