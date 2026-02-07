"""Dataset loader for RealToxicityPrompts used in toxicity evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

from datasets import load_dataset

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToxicitySample:
    """Container for a toxicity benchmark sample.

    Args:
        sample_id: Stable sample identifier.
        prompt: Prompt string to pass to the language model.
        prompt_toxicity: Optional toxicity score attached to the prompt.
        metadata: Additional dataset metadata.
    """

    sample_id: str
    prompt: str
    prompt_toxicity: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_real_toxicity_prompts(
    dataset_name: str = "allenai/real-toxicity-prompts",
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> List[ToxicitySample]:
    """Loads RealToxicityPrompts and converts rows to ``ToxicitySample``.

    Args:
        dataset_name: Hugging Face dataset identifier.
        split: Dataset split.
        max_samples: Optional cap on number of examples.
        seed: Random seed used for deterministic shuffling.

    Returns:
        List of toxicity samples.

    Side Effects:
        Downloads dataset from Hugging Face if not cached.

    Raises:
        ValueError: If max_samples is negative or dataset_name is empty.
    """

    if not dataset_name or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string.")
    if max_samples is not None and max_samples < 0:
        raise ValueError(f"max_samples must be non-negative, got {max_samples}.")
    if not split or not split.strip():
        raise ValueError("split must be a non-empty string.")

    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.shuffle(seed=seed)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    samples: List[ToxicitySample] = []
    for idx, row in enumerate(dataset):
        prompt_payload = row.get("prompt", "")
        if isinstance(prompt_payload, dict):
            prompt_text = str(prompt_payload.get("text", "")).strip()
            prompt_score = _safe_float(prompt_payload.get("toxicity"))
        else:
            prompt_text = str(prompt_payload).strip()
            prompt_score = _safe_float(row.get("prompt_toxicity"))

        if not prompt_text:
            continue

        metadata: Dict[str, Any] = {
            "challenging": row.get("challenging"),
        }

        samples.append(
            ToxicitySample(
                sample_id=str(row.get("id", row.get("comment_id", idx))),
                prompt=prompt_text,
                prompt_toxicity=prompt_score,
                metadata=metadata,
            )
        )

    scored_count = sum(1 for s in samples if s.prompt_toxicity is not None)
    LOGGER.info(
        "Toxicity dataset loaded: %d samples (%d with prompt toxicity scores)",
        len(samples),
        scored_count,
    )
    return samples


def _safe_float(value: Any) -> Optional[float]:
    """Casts values to float where possible.

    Args:
        value: Arbitrary object.

    Returns:
        ``float`` if conversion succeeds, otherwise ``None``.
    """

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
