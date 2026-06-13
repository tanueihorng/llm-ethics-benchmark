"""XSTest plugin for over-refusal evaluation on benign prompts."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ethical_benchmark.benchmarks.base import BenchmarkItem, BenchmarkPlugin
from ethical_benchmark.benchmarks.utils import is_malformed_response, is_refusal_response, shuffle_and_limit

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class XSTestConfig:
    """Configuration for XSTest plugin.

    Args:
        local_csv: Optional path to a local XSTest CSV file (e.g.
            data/xstest_v2_prompts.csv). When set, this takes precedence
            over the HF dataset and the plugin loads directly from disk.
            The CSV is expected to contain columns: id, prompt, type, label,
            focus, note (the canonical schema from paul-rottger/xstest).
        dataset_name: Hugging Face dataset ID (only used if local_csv is unset).
        split: Dataset split (only used if local_csv is unset).
        config_name: Optional HF dataset config name (only used if local_csv
            is unset).
        benign_only: Restrict to benign (label="safe") prompts.
    """

    local_csv: str | None = None
    dataset_name: str = "paul-rottger/xstest-prompts"
    split: str = "prompts"
    config_name: str | None = None
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
            local_csv=config.get("local_csv"),
            dataset_name=str(config.get("dataset_name", "paul-rottger/xstest-prompts")),
            split=str(config.get("split", "prompts")),
            config_name=config.get("config_name"),
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
            May download dataset files (only when local_csv is unset).
        """

        if self.config.local_csv:
            dataset = self._load_local_csv(self.config.local_csv)
        else:
            # Lazy import so the plugin can run without `datasets` installed
            # when only the local CSV path is used.
            from datasets import load_dataset

            if self.config.config_name:
                dataset = load_dataset(
                    self.config.dataset_name,
                    self.config.config_name,
                    split=self.config.split,
                )
            else:
                dataset = load_dataset(self.config.dataset_name, split=self.config.split)

        rows: List[BenchmarkItem] = []
        considered = 0
        unclassified = 0
        for idx, row in enumerate(dataset):
            prompt = _extract_prompt_text(row)
            if not prompt:
                continue

            considered += 1
            verdict = _classify_benign(row)
            if verdict is None:
                # No recognised safety signal in this row; fall back to benign
                # (conservative for over-refusal) but track it so a fully
                # unrecognised schema is surfaced rather than silently counting
                # every prompt as benign.
                unclassified += 1
                benign = True
            else:
                benign = verdict
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

        if considered and unclassified == considered:
            LOGGER.warning(
                "XSTest: none of the %d loaded prompts carried a recognised "
                "safety signal (label/type/id) — every prompt defaulted to "
                "benign. The dataset schema may have changed; over-refusal "
                "denominators may be wrong.",
                considered,
            )
        elif unclassified:
            LOGGER.warning(
                "XSTest: %d of %d prompts had no recognised safety signal and "
                "defaulted to benign.",
                unclassified,
                considered,
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

        # Over-refusal is measured on benign prompts only. A record whose benign
        # flag is missing cannot be confirmed benign, so it is excluded from the
        # denominator (conservative) rather than silently counted as benign.
        benign_records = [item for item in records if item["score_fields"].get("benign", False)]
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


    @staticmethod
    def _load_local_csv(path: str) -> List[Dict[str, Any]]:
        """Loads XSTest prompts from a local CSV file.

        Args:
            path: Path to the CSV file (relative paths resolved from CWD).

        Returns:
            List of dict rows, one per CSV row.

        Side Effects:
            Reads file from disk.

        Raises:
            FileNotFoundError: If the CSV does not exist.
        """

        csv_path = Path(path)
        if not csv_path.is_absolute():
            csv_path = Path.cwd() / csv_path
        if not csv_path.exists():
            raise FileNotFoundError(
                f"XSTest local_csv not found: {csv_path}. "
                f"Expected the canonical xstest_prompts.csv (e.g. data/xstest_v2_prompts.csv)."
            )
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


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


def _classify_benign(row: Dict[str, Any]) -> Optional[bool]:
    """Classifies a row's safety from any recognised schema signal.

    Args:
        row: Dataset row.

    Returns:
        ``True``/``False`` when a recognised signal (label/type/id convention)
        determines safety; ``None`` when no signal is present, so the caller can
        decide the fallback and surface that the schema was not understood.

    Side Effects:
        None.
    """

    keys = [
        row.get("label"),
        row.get("type"),
        row.get("safety_label"),
        row.get("is_harmful"),
        row.get("category"),
    ]

    # Track whether the row carried *any* usable safety field. A present field
    # that names a non-contrast category (e.g. XSTest v2 "homonyms") is a
    # recognised SAFE signal by convention — not an unrecognised row — so it must
    # not trip the schema-drift warning. Only a row with no usable signal at all
    # returns None.
    has_signal = False
    for value in keys:
        if isinstance(value, bool):
            return not value
        if isinstance(value, (int, float)):
            return value <= 0
        if isinstance(value, str):
            lowered = value.lower().strip()
            if not lowered:
                continue
            has_signal = True
            if lowered in {"safe", "benign", "harmless", "non-harmful", "non_harmful"}:
                return True
            if lowered in {"unsafe", "harmful", "adversarial", "jailbreak"}:
                return False
            # XSTest v2 (paul-rottger/xstest-prompts) encodes safety in the
            # `type` field: unsafe prompts are prefixed with `contrast_`, safe
            # prompts use the unprefixed category name (e.g. `homonyms`,
            # `figurative_language`, `definitions`).
            if lowered.startswith("contrast_") or lowered.startswith("contrast "):
                return False

    # XSTest id convention as a last-resort cue: "v1_X" / "v2_X" are safe; the
    # contrast set (unsafe) uses ids like "v1_contrast_X" / "v2_contrast_X".
    for id_key in ("id", "id_v1", "id_v2", "prompt_id"):
        id_val = row.get(id_key)
        if isinstance(id_val, str) and id_val.strip():
            has_signal = True
            if "contrast" in id_val.lower():
                return False

    # A row that carried a usable signal but no unsafe/contrast marker is safe by
    # convention (the prior behaviour for XSTest v2 safe categories). A row with
    # no usable signal at all is genuinely unrecognised -> None, so the caller can
    # surface possible schema drift.
    if has_signal:
        return True
    return None


def _is_benign_prompt(row: Dict[str, Any]) -> bool:
    """Boolean view of :func:`_classify_benign`.

    Rows with no recognised safety signal default to benign — the conservative
    choice for an over-refusal task — but callers that need to detect schema
    drift should use :func:`_classify_benign` and inspect the ``None`` case.
    """

    verdict = _classify_benign(row)
    return True if verdict is None else verdict
