"""Text generation utilities with deterministic controls for benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import logging
import random
from typing import Any, Dict, Iterable, List, Optional

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
        assistant_prefill: Optional text placed at the start of the assistant's
            reply before generation (prefill attack arm, T87/D105). ``None``
            leaves prompts exactly as before. The returned response is the
            prefill followed by the model's continuation.
    """

    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    max_input_tokens: int = 1024
    use_chat_template: bool = True
    assistant_prefill: Optional[str] = None

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
            assistant_prefill=data.get("assistant_prefill") or None,
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

        prompts = list(prompts)
        prompt_list = [self._format_prompt(prompt) for prompt in prompts]
        if not prompt_list:
            return []

        # Left-truncate: for a chat-templated prompt the generation-prompt tail
        # (the most recent turn + the assistant cue) lives at the END, so an
        # over-length prompt must drop tokens from the FRONT, not the back. The
        # default right-truncation would silently cut the assistant cue.
        # When the chat template was applied it already emits the model's special
        # tokens (including BOS for Llama/Mistral, whose templates render one); the
        # default add_special_tokens=True would then prepend a SECOND BOS. Suppress
        # special tokens on the templated path (the template is the single source of
        # truth for the prompt format); keep them for the raw-prompt fallback so a
        # non-templated model still receives its canonical BOS.
        # A prefilled prompt always differs from the raw prompt, but
        # _format_prompt refuses to prefill without a chat template, so the
        # prefill arm is always on the templated path.
        templated = bool(self.config.use_chat_template) and (
            bool(self.config.assistant_prefill)
            or any(fmt != raw for fmt, raw in zip(prompt_list, prompts))
        )
        self.tokenizer.truncation_side = "left"
        inputs = self.tokenizer(
            prompt_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_input_tokens,
            add_special_tokens=not templated,
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
        prefill_ids = self._prefill_token_ids() if self.config.assistant_prefill else None
        outputs: List[str] = []
        for output_ids in generated:
            new_token_ids = output_ids[input_length:]
            if prefill_ids is not None:
                # Decode prefill and continuation together. Decoding the
                # continuation alone loses its leading space on SentencePiece
                # tokenizers (Mistral strips one leading space on decode).
                ids = new_token_ids.tolist() if hasattr(new_token_ids, "tolist") else list(new_token_ids)
                text = self.tokenizer.decode(prefill_ids + ids, skip_special_tokens=True).strip()
            else:
                text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
            outputs.append(self._postprocess(text))

        return outputs

    def _prefill_token_ids(self) -> List[int]:
        """Token ids whose decoding is exactly the configured prefill.

        Raises:
            RuntimeError: If the tokenizer cannot round-trip the prefill text.
        """

        prefill = str(self.config.assistant_prefill)
        ids = list(self.tokenizer.encode(prefill, add_special_tokens=False))
        decoded = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        if decoded != prefill:
            raise RuntimeError(
                f"Tokenizer does not round-trip the prefill (got {decoded!r})."
            )
        return ids

    def _format_prompt(self, prompt: str) -> str:
        """Applies model-specific chat formatting, plus the prefill when configured.

        Args:
            prompt: Raw user prompt.

        Returns:
            Prompt string compatible with tokenizer/model expectations.
        """

        templated = self._apply_template(prompt)
        if not self.config.assistant_prefill:
            return templated
        return self._apply_prefill(prompt, templated)

    def _apply_prefill(self, prompt: str, templated: str) -> str:
        """Renders the prompt with the assistant reply already started.

        Renders ``[user, assistant(prefill)]`` through the model's own chat
        template and cuts the string right after the prefill, which is what
        transformers' ``continue_final_message`` does. This puts the prefill
        exactly where the model's reply would begin, including template-specific
        spacing (Mistral inserts a space after ``[/INST]``; Qwen3 keeps its
        empty think block).

        Raises:
            RuntimeError: If no chat template was applied, or the rendered
                prefill does not extend the plain generation prompt.
        """

        prefill = str(self.config.assistant_prefill)
        if templated == prompt:
            raise RuntimeError(
                "assistant_prefill requires an applied chat template; refusing to "
                "prefill a raw prompt."
            )
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": prefill},
        ]
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        cut = rendered.rfind(prefill)
        if cut < 0:
            raise RuntimeError("Chat template dropped the assistant prefill.")
        prefilled = rendered[: cut + len(prefill)]
        if not prefilled.startswith(templated):
            raise RuntimeError(
                "Prefilled prompt does not extend the generation prompt; the "
                "template changes earlier context when an assistant turn is present."
            )
        return prefilled

    def _apply_template(self, prompt: str) -> str:
        """Applies the tokenizer chat template only (no prefill)."""

        if self.config.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                try:
                    return self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning(
                        "Chat template failed; FELL BACK to the raw prompt "
                        "(no template applied). Error: %s", exc,
                    )
            except Exception as exc:  # pragma: no cover
                LOGGER.warning(
                    "Chat template failed; FELL BACK to the raw prompt "
                    "(no template applied). Error: %s", exc,
                )
        return prompt

    def prompt_was_templated(self, prompt: str) -> bool:
        """Provenance: whether the chat template actually transformed ``prompt``.

        Returns True when ``use_chat_template`` is enabled and
        ``apply_chat_template`` produced a string different from the raw prompt;
        False when templating was disabled or silently fell back to the raw
        prompt (see ``_format_prompt``). This lets each ``raw.jsonl`` record carry
        a per-prompt flag for which prompt path the model actually saw, so a
        silent template failure is auditable from the artifacts and not only the
        run-time WARNING.
        """

        return bool(self.config.use_chat_template) and self._apply_template(prompt) != prompt

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
