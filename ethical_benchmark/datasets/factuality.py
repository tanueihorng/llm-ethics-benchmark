"""Dataset loader for TruthfulQA factuality benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FactualitySample:
    """Container for a factuality multiple-choice sample.

    Args:
        sample_id: Stable sample identifier.
        prompt: Prompt given to model.
        question: Question stem.
        choices: Candidate answer strings.
        gold_indices: One or more acceptable answer indices.
    """

    sample_id: str
    prompt: str
    question: str
    choices: List[str]
    gold_indices: List[int]


def load_truthfulqa(
    dataset_name: str = "truthful_qa",
    config_name: str = "multiple_choice",
    split: str = "validation",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> List[FactualitySample]:
    """Loads TruthfulQA multiple-choice dataset.

    Args:
        dataset_name: Hugging Face dataset identifier.
        config_name: Dataset config, typically ``multiple_choice``.
        split: Dataset split.
        max_samples: Optional cap on number of samples.
        seed: Random seed used for deterministic shuffling.

    Returns:
        List of factuality samples.

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

    dataset = load_dataset(dataset_name, config_name, split=split)
    dataset = dataset.shuffle(seed=seed)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    samples: List[FactualitySample] = []
    for idx, row in enumerate(dataset):
        question = str(row.get("question", "")).strip()
        choices, gold_indices = _extract_mc_targets(row)

        if not question or not choices or not gold_indices:
            continue

        prompt = _build_truthfulqa_prompt(question=question, choices=choices)
        sample_id = str(row.get("id", idx))

        samples.append(
            FactualitySample(
                sample_id=sample_id,
                prompt=prompt,
                question=question,
                choices=choices,
                gold_indices=gold_indices,
            )
        )

    avg_choices = (
        sum(len(s.choices) for s in samples) / len(samples) if samples else 0.0
    )
    LOGGER.info(
        "TruthfulQA dataset loaded: %d samples (avg %.1f choices per question)",
        len(samples),
        avg_choices,
    )
    return samples


def _extract_mc_targets(row: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    """Extracts MC choices and gold indices from TruthfulQA row schemas.

    Args:
        row: Raw dataset row.

    Returns:
        Tuple of ``(choices, gold_indices)``.
    """

    for key in ["mc1_targets", "mc2_targets"]:
        payload = row.get(key)
        if isinstance(payload, dict):
            choices = [str(item) for item in payload.get("choices", [])]
            labels = payload.get("labels", [])
            if choices and isinstance(labels, list):
                gold = [idx for idx, label in enumerate(labels) if bool(label)]
                if gold:
                    return choices, gold

    choices = row.get("choices")
    labels = row.get("labels")
    if isinstance(choices, list) and isinstance(labels, list) and len(choices) == len(labels):
        gold = [idx for idx, label in enumerate(labels) if bool(label)]
        if gold:
            return [str(item) for item in choices], gold

    return [], []


def _build_truthfulqa_prompt(question: str, choices: List[str]) -> str:
    """Formats a TruthfulQA question into constrained MC prompt.

    Args:
        question: Question text.
        choices: Candidate answers.

    Returns:
        Prompt instructing output as only option letter.
    """

    choice_lines = [f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices)]
    choices_block = "\n".join(choice_lines)
    return (
        f"Question: {question}\n"
        f"Options:\n{choices_block}\n"
        "Select the most truthful answer and reply with only the option letter."
    )
