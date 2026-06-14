"""ARC-Challenge plugin — second capability benchmark (T23 / weakness W5).

ARC-Challenge (Clark et al., 2018; ``allenai/ai2_arc`` config ``ARC-Challenge``)
is a structurally different capability measure from MMLU: grade-school science
questions designed to resist retrieval/co-occurrence shortcuts. It shares MMLU's
multiple-choice, exact-match scoring machinery, so adding it turns the study's
capability anchor into a two-benchmark composite — directly strengthening the
alignment-vs-capability disambiguation that the rest of the study rests on.

Row schema differs from MMLU: ``choices`` is a ``{"text": [...], "label": [...]}``
dict and the gold answer is ``answerKey`` (a label, e.g. "A" or "3"). Options are
re-labelled A, B, C, … in the prompt for parity with the MMLU prompt format, so
the two capability benchmarks present an identical answer interface to the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from ethical_benchmark.benchmarks.base import BenchmarkItem, BenchmarkPlugin
from ethical_benchmark.benchmarks.utils import parse_choice_index, shuffle_and_limit


@dataclass(frozen=True)
class ARCConfig:
    """Configuration for the ARC-Challenge plugin.

    Args:
        dataset_name: Hugging Face dataset ID.
        config_name: HF dataset config (the ARC subset, e.g. "ARC-Challenge").
        split: Dataset split.
    """

    dataset_name: str = "allenai/ai2_arc"
    config_name: str = "ARC-Challenge"
    split: str = "test"


# Single aggregation bucket: ARC-Challenge has no subjects, but reusing the
# MMLU-style subject_breakdown keeps the capability summaries uniform.
_BUCKET = "arc_challenge"


class ARCChallengePlugin(BenchmarkPlugin):
    """Benchmark plugin for ARC-Challenge capability measurement."""

    name = "arc"

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes the ARC plugin.

        Args:
            config: Benchmark configuration dictionary.

        Side Effects:
            None.
        """

        self.config = ARCConfig(
            dataset_name=str(config.get("dataset_name", "allenai/ai2_arc")),
            config_name=str(config.get("config_name", "ARC-Challenge")),
            split=str(config.get("split", "test")),
        )

    def load_items(self, max_samples: int | None, seed: int) -> List[BenchmarkItem]:
        """Loads ARC-Challenge items.

        Args:
            max_samples: Optional maximum number of items.
            seed: Random seed for deterministic ordering.

        Returns:
            List of benchmark items.

        Side Effects:
            May download dataset files.
        """

        dataset = load_dataset(
            self.config.dataset_name, self.config.config_name, split=self.config.split
        )

        rows: List[BenchmarkItem] = []
        for idx, row in enumerate(dataset):
            question = str(row.get("question", "")).strip()
            choices = row.get("choices") or {}
            texts = choices.get("text") if isinstance(choices, dict) else None
            labels = choices.get("label") if isinstance(choices, dict) else None
            answer_key = row.get("answerKey")

            if not question or not isinstance(texts, list) or len(texts) == 0:
                continue
            if not isinstance(labels, list) or len(labels) != len(texts):
                continue

            gold_index = _gold_index_from_label(answer_key=answer_key, labels=labels)
            if gold_index is None:
                continue

            prompt_id = str(row.get("id", f"arc_{idx}"))
            prompt = _build_mc_prompt(question=question, choices=[str(t) for t in texts])
            rows.append(
                BenchmarkItem(
                    prompt_id=prompt_id,
                    prompt_text=prompt,
                    metadata={
                        "subject": _BUCKET,
                        "gold_index": gold_index,
                        "num_choices": len(texts),
                    },
                )
            )

        return shuffle_and_limit(rows, max_samples=max_samples, seed=seed)

    def score_response(self, item: BenchmarkItem, response: str) -> Dict[str, Any]:
        """Scores an ARC response by exact option match.

        Args:
            item: ARC item.
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
            "subject": item.metadata.get("subject", _BUCKET),
        }

    def aggregate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregates ARC records into capability metrics.

        Args:
            records: Per-response run records.

        Returns:
            Capability summary metrics (accuracy + answered rate).

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
            subject = str(item["score_fields"].get("subject", _BUCKET))
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


def _gold_index_from_label(answer_key: Any, labels: List[Any]) -> Optional[int]:
    """Resolves the zero-based gold index by matching ``answerKey`` to a label.

    Handles both letter labels (["A","B","C","D"]) and numeric labels
    (["1","2","3","4"]); falls back to letter/number coercion if the exact
    label is absent. Returns ``None`` if it cannot be resolved.
    """

    if answer_key is None:
        return None
    key = str(answer_key).strip()
    label_strs = [str(label).strip() for label in labels]

    if key in label_strs:
        return label_strs.index(key)

    key_upper = key.upper()
    # Letter key against letter labels.
    if len(key_upper) == 1 and "A" <= key_upper <= "Z":
        idx = ord(key_upper) - ord("A")
        if 0 <= idx < len(label_strs):
            return idx
    # Numeric key (1-based in ARC's numeric scheme) → zero-based index.
    if key.isdigit():
        idx = int(key) - 1
        if 0 <= idx < len(label_strs):
            return idx
    return None


def _build_mc_prompt(question: str, choices: List[str]) -> str:
    """Builds a strict multiple-choice prompt (identical format to the MMLU plugin).

    Kept byte-for-byte equivalent to ``mmlu._build_mc_prompt`` so the two
    capability benchmarks present an identical answer interface to the model.
    """

    lines = [f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices)]
    choices_block = "\n".join(lines)
    return (
        f"Question: {question}\n"
        f"Options:\n{choices_block}\n"
        "Choose the best answer and reply with only the option letter."
    )
