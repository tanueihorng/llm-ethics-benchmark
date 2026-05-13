"""Text generation utilities with deterministic controls for benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import logging
import random
from typing import Any, Dict, Iterable, List

import numpy as np

LOGGER = logging.getLogger(__name__)


def _get_torch() -> Any:
    """Imports torch lazily to avoid import-time runtime crashes."""

    return import_module("torch")


@dataclass(frozen=True)
class DecodingConfig:
    """Decoding parameters shared across all benchmarked models.

    Args:
        max_new_tokens: Maximum number of generated tokens per prompt.
        temperature: Sampling temperature. Use ``0.0`` for greedy decoding.
        top_p: Nucleus sampling threshold when sampling is enabled.
        repetition_penalty: Penalizes repeated token generation.
        max_input_tokens: Maximum input context length.
        use_chat_template: Whether to format user prompts as chat messages.
    """

    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    max_input_tokens: int = 1024
    use_chat_template: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecodingConfig":
        """Constructs decoding config from a dictionary.

        Args:
            data: Dictionary containing decoding keys.

        Returns:
            DecodingConfig instance.

        Side Effects:
            None.
        """

        return cls(
            max_new_tokens=int(data.get("max_new_tokens", 128)),
            temperature=float(data.get("temperature", 0.0)),
            top_p=float(data.get("top_p", 1.0)),
            repetition_penalty=float(data.get("repetition_penalty", 1.0)),
            max_input_tokens=int(data.get("max_input_tokens", 1024)),
            use_chat_template=bool(data.get("use_chat_template", True)),
        )


def set_global_seed(seed: int) -> None:
    """Sets pseudo-random seeds for reproducible experiments.

    Args:
        seed: Integer random seed.

    Side Effects:
        Mutates RNG state for Python, NumPy, and Torch.
    """

    torch = _get_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TextGenerator:
    """Batched text generator around a Hugging Face causal LM."""

    def __init__(self, model: Any, tokenizer: Any, device: str, config: DecodingConfig) -> None:
        """Initializes the generation wrapper.

        Args:
            model: Loaded Hugging Face causal LM.
            tokenizer: Corresponding tokenizer.
            device: Runtime device string.
            config: Generation configuration.

        Side Effects:
            Mutates tokenizer padding configuration.
        """

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config

        # Left padding avoids shifting prompt tokens for autoregressive models.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_batch(self, prompts: Iterable[str]) -> List[str]:
        """Generates model responses for a batch of prompts.

        Args:
            prompts: Input prompts.

        Returns:
            Generated text responses in input order.

        Side Effects:
            Runs model inference on configured device.
        """

        prompt_list = [self._format_prompt(prompt) for prompt in prompts]
        if not prompt_list:
            return []

        inputs = self.tokenizer(
            prompt_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_input_tokens,
        )

        model_device = self.model.device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        do_sample = self.config.temperature > 0.0
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": do_sample,
            "top_p": self.config.top_p if do_sample else None,
            "temperature": self.config.temperature if do_sample else None,
            "repetition_penalty": self.config.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

        torch = _get_torch()
        with torch.no_grad():
            generated = self.model.generate(**inputs, **generation_kwargs)

        input_length = inputs["input_ids"].shape[1]
        outputs: List[str] = []
        for output_ids in generated:
            new_token_ids = output_ids[input_length:]
            text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
            outputs.append(self._postprocess(text))

        return outputs

    def _format_prompt(self, prompt: str) -> str:
        """Applies model-specific chat formatting when configured.

        Args:
            prompt: Raw user prompt.

        Returns:
            Prompt string compatible with tokenizer/model expectations.
        """

        if self.config.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.debug("Chat template failed; using raw prompt. Error: %s", exc)
        return prompt

    @staticmethod
    def _postprocess(text: str) -> str:
        """Normalizes generated text for downstream evaluators.

        Args:
            text: Raw decoded text.

        Returns:
            Cleaned text response.
        """

        cleaned = text.strip()
        markers = ["assistant", "Assistant"]
        for marker in markers:
            if cleaned.startswith(f"{marker}:"):
                return cleaned.split(":", 1)[1].strip()
        return cleaned
