"""Base interfaces for benchmark plugins used by quantization experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class BenchmarkItem:
    """Single benchmark prompt item.

    Args:
        prompt_id: Stable unique prompt identifier.
        prompt_text: Prompt content sent to the model.
        metadata: Additional benchmark-specific fields.
    """

    prompt_id: str
    prompt_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkPlugin(ABC):
    """Contract for all quantization benchmark plugins.

    Plugins must implement data loading, prompt construction, response scoring,
    and aggregate metric computation.
    """

    name: str

    @abstractmethod
    def load_items(self, max_samples: int | None, seed: int) -> List[BenchmarkItem]:
        """Loads benchmark items.

        Args:
            max_samples: Optional maximum number of items.
            seed: Random seed for deterministic ordering.

        Returns:
            List of benchmark items.

        Side Effects:
            May download datasets from remote cache/backends.
        """

    def build_prompt(self, item: BenchmarkItem) -> str:
        """Builds generation prompt for a benchmark item.

        Args:
            item: Benchmark item.

        Returns:
            Prompt string for generation.

        Side Effects:
            None.
        """

        return item.prompt_text

    @abstractmethod
    def score_response(self, item: BenchmarkItem, response: str) -> Dict[str, Any]:
        """Scores a generated response for one benchmark item.

        Args:
            item: Benchmark item.
            response: Generated model response.

        Returns:
            Benchmark-specific score fields.

        Side Effects:
            None.
        """

    @abstractmethod
    def aggregate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregates per-item records into summary metrics.

        Args:
            records: Per-item run records.

        Returns:
            Summary metric dictionary.

        Side Effects:
            None.
        """
