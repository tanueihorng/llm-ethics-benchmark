---
name: fyp-second-judge
description: Use in fyp_quant when adding, reviewing, or planning a second independent HarmBench judge such as LlamaGuard or an API yes/no classifier.
---

# FYP Second Judge

Use this skill for an explicitly requested judge extension or robustness review. The existing HarmBench classifier
sidecars are primary for current report claims; a second judge is an optional
cross-check unless the project log says otherwise.

## Start

1. Apply `AGENTS.md`; search `docs/PROJECT_LOG.md` for entries relevant to this task, without loading its full history.
2. Resolve the current judge artifacts and relevant task from the request and log; earlier second-judge tasks may already be complete.
3. Use the matching `docs/agent_tasks/` packet if needed; do not rerun a completed study by default.

## Scope

- Add or review a second HarmBench-only judge backend.
- Write new redacted sidecars named `scores.judge.<name>.jsonl` and `summary.judge.<name>.json`.
- Extend agreement analysis without overwriting existing raw, v2, or `harmbench_cls` judge artifacts.
- Use TC1 compute through `sbatch`; do not run user compute code on the head node.

## Guardrails

- Do not reuse the HarmBench classifier prompt as a generic second-judge rubric.
- Do not persist prompt text, behavior text, generation text, or raw model responses in new sidecars.
- Document backend choice, rubric, and limitations in `docs/PROJECT_LOG.md`.

## Finish

Run the narrow checks first, then the harness:

```bash
pytest tests/test_judge_validation.py
python scripts/run_judge_validation.py --help
python scripts/judge_agreement.py --help
make agent-check
```
