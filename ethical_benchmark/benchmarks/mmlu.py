"""MMLU subset plugin for capability evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset

from ethical_benchmark.benchmarks.base import BenchmarkItem, BenchmarkPlugin
from ethical_benchmark.benchmarks.utils import parse_choice_index, shuffle_and_limit


@dataclass(frozen=True)
class MMLUConfig:
    """Configuration for MMLU plugin.

    Args:
        dataset_name: Hugging Face dataset ID.
        split: Dataset split.
        subjects: Subject subset names.
    """

    dataset_name: str = "cais/mmlu"
    split: str = "test"
    subjects: List[str] | None = None


DEFAULT_SUBJECTS: List[str] = [
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "high_school_world_history",
    "high_school_macroeconomics",
    "human_aging",
]


class MMLUPlugin(BenchmarkPlugin):
    """Benchmark plugin for MMLU subset capability measurement."""

    name = "mmlu"

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes MMLU plugin.

        Args:
            config: Benchmark configuration dictionary.

        Side Effects:
            None.
        """

        subjects = config.get("subjects")
        subject_list = [str(item) for item in subjects] if isinstance(subjects, list) else DEFAULT_SUBJECTS

        self.config = MMLUConfig(
            dataset_name=str(config.get("dataset_name", "cais/mmlu")),
            split=str(config.get("split", "test")),
            subjects=subject_list,
        )

    def load_items(self, max_samples: int | None, seed: int) -> List[BenchmarkItem]:
        """Loads MMLU subset items.

        Args:
            max_samples: Optional maximum number of items across subjects.
            seed: Random seed.

        Returns:
            List of benchmark items.

        Side Effects:
            May download dataset files.
        """

        rows: List[BenchmarkItem] = []

        for subject in self.config.subjects or DEFAULT_SUBJECTS:
            dataset = load_dataset(self.config.dataset_name, subject, split=self.config.split)
            for idx, row in enumerate(dataset):
                question = str(row.get("question", "")).strip()
                choices = row.get("choices", row.get("options", []))
                answer = row.get("answer", row.get("target"))

                if not question or not isinstance(choices, list) or len(choices) == 0:
                    continue

                gold_index = _parse_gold_index(answer=answer, num_choices=len(choices))
                if gold_index is None:
                    continue

                prompt_id = str(row.get("id", f"{subject}_{idx}"))
                prompt = _build_mc_prompt(question=question, choices=[str(c) for c in choices])
                rows.append(
                    BenchmarkItem(
                        prompt_id=prompt_id,
                        prompt_text=prompt,
                        metadata={
                            "subject": subject,
                            "gold_index": gold_index,
                            "num_choices": len(choices),
                        },
                    )
                )

        return shuffle_and_limit(rows, max_samples=max_samples, seed=seed)

    def score_response(self, item: BenchmarkItem, response: str) -> Dict[str, Any]:
        """Scores MMLU response by exact option match.

        Args:
            item: MMLU item.
            response: Generated response.

        Returns:
            Score dictionary with parsed answer and correctness.

        Side Effects:
            None.
        """

        num_choices = int(item.metadata.get("num_choices", 4))
        gold_index = int(item.metadata["gold_index"])
        predicted_index = parse_choice_index(response=response, num_choices=num_choices)

        return {
            "predicted_index": predicted_index,
            "gold_index": gold_index,
            "is_answered": predicted_index is not None,
            "is_correct": predicted_index == gold_index if predicted_index is not None else False,
            "subject": item.metadata.get("subject"),
        }

    def aggregate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregates MMLU records.

        Args:
            records: Per-response run records.

        Returns:
            Capability summary metrics.

        Side Effects:
            None.
        """

        if not records:
            return {
                "num_samples": 0,
                "answered_count": 0,
                "correct_count": 0,
                "answered_rate": None,
                "accuracy": None,
                "subject_breakdown": {},
            }

        total = len(records)
        answered_count = sum(int(item["score_fields"]["is_answered"]) for item in records)
        correct_count = sum(int(item["score_fields"]["is_correct"]) for item in records)

        by_subject: Dict[str, Dict[str, int]] = {}
        for item in records:
            subject = str(item["score_fields"].get("subject", "unknown"))
            bucket = by_subject.setdefault(subject, {"total": 0, "answered": 0, "correct": 0})
            bucket["total"] += 1
            bucket["answered"] += int(item["score_fields"]["is_answered"])
            bucket["correct"] += int(item["score_fields"]["is_correct"])

        subject_breakdown = {
            subject: {
                "num_samples": stats["total"],
                "answered_rate": stats["answered"] / stats["total"] if stats["total"] else None,
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else None,
            }
            for subject, stats in by_subject.items()
        }

        return {
            "num_samples": total,
            "answered_count": answered_count,
            "correct_count": correct_count,
            "answered_rate": answered_count / total,
            "accuracy": correct_count / total,
            "subject_breakdown": subject_breakdown,
        }


def _parse_gold_index(answer: Any, num_choices: int) -> int | None:
    """Parses gold answer index for MMLU rows.

    Args:
        answer: Raw answer field.
        num_choices: Number of options.

    Returns:
        Zero-based correct index or ``None``.

    Side Effects:
        None.
    """

    if isinstance(answer, int):
        if 0 <= answer < num_choices:
            return answer
        return None

    if isinstance(answer, str):
        answer_clean = answer.strip().upper()
        if answer_clean.isdigit():
            idx = int(answer_clean)
            if 0 <= idx < num_choices:
                return idx
        if len(answer_clean) == 1 and "A" <= answer_clean <= "Z":
            idx = ord(answer_clean) - ord("A")
            if 0 <= idx < num_choices:
                return idx

    return None


def _build_mc_prompt(question: str, choices: List[str]) -> str:
    """Builds a strict multiple-choice prompt.

    Args:
        question: Question text.
        choices: Option texts.

    Returns:
        Prompt that asks for option letter only.

    Side Effects:
        None.
    """

    lines = [f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices)]
    choices_block = "\n".join(lines)
    return (
        f"Question: {question}\n"
        f"Options:\n{choices_block}\n"
        "Choose the best answer and reply with only the option letter."
    )
