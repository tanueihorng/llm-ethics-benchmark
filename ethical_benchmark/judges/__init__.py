"""Judge-model validation helpers for derived scoring pipelines.

Judge validation is a derived sensitivity check over the v2 refusal scorer.
It never mutates TC1-original raw outputs or the v2 sidecars; results are
written to ``scores.judge.<name>.jsonl`` / ``summary.judge.<name>.json``.
See ``ethical_benchmark/judges/validation.py`` for details and D11/D12/D13
in ``docs/PROJECT_LOG.md`` for the governing decisions.
"""

from ethical_benchmark.judges.validation import (
    HARMBENCH_CLS_PROMPT,
    SCORER_VERSION,
    HarmBenchJudgeBackend,
    JudgeBackend,
    JudgeInput,
    JudgeResult,
    LlamaGuardJudgeBackend,
    OpenAICompatibleJudgeBackend,
    parse_yes_no,
    run_judge_validation,
)

__all__ = [
    "HARMBENCH_CLS_PROMPT",
    "SCORER_VERSION",
    "HarmBenchJudgeBackend",
    "JudgeBackend",
    "JudgeInput",
    "JudgeResult",
    "LlamaGuardJudgeBackend",
    "OpenAICompatibleJudgeBackend",
    "parse_yes_no",
    "run_judge_validation",
]
