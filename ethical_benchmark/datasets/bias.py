"""Dataset loader for BBQ (Bias Benchmark for QA)."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, List, Optional

from datasets import load_dataset

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BiasSample:
    """Container for a social bias multiple-choice sample.

    Args:
        sample_id: Stable sample identifier.
        prompt: Prompt given to the model.
        question: Core question text.
        choices: Candidate answer options.
        gold_index: Ground-truth answer index.
        axis: Demographic axis/category label.
        stereotyped_index: Optional stereotyped answer index.
    """

    sample_id: str
    prompt: str
    question: str
    choices: List[str]
    gold_index: int
    axis: str
    stereotyped_index: Optional[int] = None


def load_bbq(
    dataset_name: str = "heegyu/bbq",
    split: str = "test",
    config_name: Optional[str] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> List[BiasSample]:
    """Loads BBQ and formats examples for QA prompting.

    Args:
        dataset_name: Hugging Face dataset identifier.
        split: Dataset split.
        config_name: Optional dataset configuration/subset.
        max_samples: Optional cap on number of samples.
        seed: Random seed used for deterministic shuffling.

    Returns:
        List of social bias samples.

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

    if config_name:
        dataset = load_dataset(dataset_name, config_name, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    dataset = dataset.shuffle(seed=seed)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    samples: List[BiasSample] = []
    for idx, row in enumerate(dataset):
        question = str(row.get("question", row.get("query", ""))).strip()
        context = str(row.get("context", "")).strip()
        choices = _extract_choices(row)
        if not question or len(choices) < 2:
            continue

        gold_index = _parse_choice_index(
            row.get("label", row.get("answer", row.get("correct_answer"))),
            len(choices),
        )
        if gold_index is None:
            continue

        stereotyped_index = _parse_choice_index(
            row.get(
                "stereotyped_index",
                row.get("stereotyped_answer", row.get("stereotype_label")),
            ),
            len(choices),
        )

        axis = str(
            row.get(
                "category",
                row.get("demographic", row.get("target_group", "unknown")),
            )
        )

        prompt = _build_bbq_prompt(context=context, question=question, choices=choices)

        samples.append(
            BiasSample(
                sample_id=str(row.get("example_id", row.get("id", idx))),
                prompt=prompt,
                question=question,
                choices=choices,
                gold_index=gold_index,
                axis=axis,
                stereotyped_index=stereotyped_index,
            )
        )

    axes = {s.axis for s in samples}
    stereo_count = sum(1 for s in samples if s.stereotyped_index is not None)
    LOGGER.info(
        "BBQ dataset loaded: %d samples across %d axes (%d with stereotyped labels)",
        len(samples),
        len(axes),
        stereo_count,
    )
    return samples


def _extract_choices(row: Dict[str, Any]) -> List[str]:
    """Extracts answer choices from varying BBQ schemas.

    Args:
        row: Raw dataset row.

    Returns:
        Ordered list of candidate choices.
    """

    if all(key in row for key in ["ans0", "ans1", "ans2"]):
        return [str(row["ans0"]), str(row["ans1"]), str(row["ans2"])]

    if isinstance(row.get("choices"), list):
        return [str(item) for item in row["choices"]]

    if isinstance(row.get("answers"), list):
        return [str(item) for item in row["answers"]]

    extracted: List[str] = []
    for key in sorted(row):
        if re.fullmatch(r"ans\d+", key):
            extracted.append(str(row[key]))

    return extracted


def _parse_choice_index(value: Any, num_choices: int) -> Optional[int]:
    """Parses choice index from integers, letters, or option text.

    Args:
        value: Candidate label value.
        num_choices: Number of answer choices.

    Returns:
        Parsed zero-based index or ``None`` when unavailable.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        if 0 <= value < num_choices:
            return value
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.isdigit():
        parsed = int(text)
        if 0 <= parsed < num_choices:
            return parsed
        return None

    letter_match = re.search(r"\b([A-Z])\b", text.upper())
    if letter_match:
        index = ord(letter_match.group(1)) - ord("A")
        if 0 <= index < num_choices:
            return index

    return None


def _build_bbq_prompt(context: str, question: str, choices: List[str]) -> str:
    """Builds a constrained multiple-choice prompt for BBQ.

    Args:
        context: Context passage.
        question: Question string.
        choices: Candidate answer choices.

    Returns:
        Prompt instructing the model to output only option letter.
    """

    choice_lines = [f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices)]
    choices_block = "\n".join(choice_lines)
    context_block = f"Context: {context}\n" if context else ""
    return (
        f"{context_block}Question: {question}\n"
        f"Options:\n{choices_block}\n"
        "Select the best answer and reply with only the option letter."
    )
