"""XSTest plugin for over-refusal evaluation on benign prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset

from ethical_benchmark.benchmarks.base import BenchmarkItem, BenchmarkPlugin
from ethical_benchmark.benchmarks.utils import is_malformed_response, is_refusal_response, shuffle_and_limit


@dataclass(frozen=True)
class XSTestConfig:
    """Configuration for XSTest plugin.

    Args:
        dataset_name: Hugging Face dataset ID.
        split: Dataset split.
        benign_only: Restrict to benign prompts.
    """

    dataset_name: str = "allenai/xstest-response"
    split: str = "test"
    benign_only: bool = True


class XSTestPlugin(BenchmarkPlugin):
    """Benchmark plugin for over-refusal measurement."""

    name = "xstest"

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes XSTest plugin.

        Args:
            config: Benchmark configuration dictionary.

        Side Effects:
            None.
        """

        self.config = XSTestConfig(
            dataset_name=str(config.get("dataset_name", "allenai/xstest-response")),
            split=str(config.get("split", "test")),
            benign_only=bool(config.get("benign_only", True)),
        )

    def load_items(self, max_samples: int | None, seed: int) -> List[BenchmarkItem]:
        """Loads benign prompts for over-refusal testing.

        Args:
            max_samples: Optional maximum number of items.
            seed: Random seed.

        Returns:
            List of benchmark items.

        Side Effects:
            May download dataset files.
        """

        dataset = load_dataset(self.config.dataset_name, split=self.config.split)

        rows: List[BenchmarkItem] = []
        for idx, row in enumerate(dataset):
            prompt = _extract_prompt_text(row)
            if not prompt:
                continue

            benign = _is_benign_prompt(row)
            if self.config.benign_only and not benign:
                continue

            prompt_id = str(row.get("id", row.get("prompt_id", f"xstest_{idx}")))
            rows.append(
                BenchmarkItem(
                    prompt_id=prompt_id,
                    prompt_text=prompt,
                    metadata={
                        "benign": benign,
                        "category": row.get("category", row.get("label")),
                    },
                )
            )

        return shuffle_and_limit(rows, max_samples=max_samples, seed=seed)

    def score_response(self, item: BenchmarkItem, response: str) -> Dict[str, Any]:
        """Scores one response for over-refusal.

        Args:
            item: XSTest item.
            response: Generated response.

        Returns:
            Score dictionary containing refusal and malformed flags.

        Side Effects:
            None.
        """

        malformed = is_malformed_response(response)
        refusal = False if malformed else is_refusal_response(response)
        benign_answer = (not malformed) and (not refusal)

        return {
            "is_refusal": refusal,
            "benign_answer": benign_answer,
            "malformed": malformed,
            "benign": bool(item.metadata.get("benign", True)),
        }

    def aggregate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregates XSTest records.

        Args:
            records: Per-response run records.

        Returns:
            Over-refusal summary metrics.

        Side Effects:
            None.
        """

        if not records:
            return {
                "num_samples": 0,
                "benign_prompt_count": 0,
                "over_refusal_count": 0,
                "benign_answer_count": 0,
                "malformed_output_count": 0,
                "over_refusal_rate": None,
                "benign_answer_rate": None,
                "malformed_rate": None,
            }

        benign_records = [item for item in records if item["score_fields"].get("benign", True)]
        total = len(benign_records)
        if total == 0:
            return {
                "num_samples": len(records),
                "benign_prompt_count": 0,
                "over_refusal_count": 0,
                "benign_answer_count": 0,
                "malformed_output_count": 0,
                "over_refusal_rate": None,
                "benign_answer_rate": None,
                "malformed_rate": None,
            }

        refusal_count = sum(int(item["score_fields"]["is_refusal"]) for item in benign_records)
        benign_answer_count = sum(int(item["score_fields"]["benign_answer"]) for item in benign_records)
        malformed_count = sum(int(item["score_fields"]["malformed"]) for item in benign_records)

        return {
            "num_samples": len(records),
            "benign_prompt_count": total,
            "over_refusal_count": refusal_count,
            "benign_answer_count": benign_answer_count,
            "malformed_output_count": malformed_count,
            "over_refusal_rate": refusal_count / total,
            "benign_answer_rate": benign_answer_count / total,
            "malformed_rate": malformed_count / total,
        }


def _extract_prompt_text(row: Dict[str, Any]) -> str:
    """Extracts prompt text from possible XSTest schemas.

    Args:
        row: Dataset row.

    Returns:
        Prompt text or empty string.

    Side Effects:
        None.
    """

    candidates = [
        row.get("prompt"),
        row.get("query"),
        row.get("instruction"),
        row.get("user_prompt"),
    ]

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    return ""


def _is_benign_prompt(row: Dict[str, Any]) -> bool:
    """Determines whether a row should be treated as benign.

    Args:
        row: Dataset row.

    Returns:
        ``True`` if the row appears benign.

    Side Effects:
        None.
    """

    keys = [
        row.get("label"),
        row.get("type"),
        row.get("safety_label"),
        row.get("is_harmful"),
    ]

    for value in keys:
        if isinstance(value, bool):
            return not value
        if isinstance(value, (int, float)):
            return value <= 0
        if isinstance(value, str):
            lowered = value.lower().strip()
            if lowered in {"safe", "benign", "harmless", "non-harmful", "non_harmful"}:
                return True
            if lowered in {"unsafe", "harmful", "adversarial", "jailbreak"}:
                return False

    # Default conservative choice for XSTest over-refusal tasks.
    return True
