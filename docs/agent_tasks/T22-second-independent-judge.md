# Agent Task Packet: T22 Second Independent Judge

> **STATUS: COMPLETED 2026-06-14 (D25/D26) — HISTORICAL PACKET.** Do not execute. 128-era framing (pre-D41); the primary study is now the 512-token tree and the second judge has been run at 512 too. Kept as a point-in-time record only.

## Objective

Add and run a second independent HarmBench judge as a robustness cross-check against `cais/HarmBench-Llama-2-13b-cls`.

## Likely Files

- `ethical_benchmark/judges/validation.py`
- `scripts/run_judge_validation.py`
- `scripts/judge_agreement.py`
- `slurm/judge_validation.sbatch`
- `tests/test_judge_validation.py`
- `docs/PROJECT_LOG.md`

## Allowed Actions

- Add a new judge backend with a strict yes/no rubric.
- Write redacted `scores.judge.<name>.jsonl` and `summary.judge.<name>.json` sidecars.
- Compare judges using IDs, booleans, counts, rates, and agreement metrics.
- Run TC1 work through `sbatch`.

## Forbidden Actions

- Do not run user compute code on the TC1 head node.
- Do not reuse the HarmBench classifier prompt as if it were a LlamaGuard/API rubric.
- Do not overwrite existing raw, v2, or HarmBench-classifier judge sidecars.
- Do not persist prompt text, behavior text, generation text, or raw responses in new sidecars.

## Verification

```bash
pytest tests/test_judge_validation.py
python scripts/run_judge_validation.py --help
python scripts/judge_agreement.py --help
make agent-check
```

## Done Criteria

- Backend choice and prompt/rubric are documented in `docs/PROJECT_LOG.md`.
- Sidecars are redacted and idempotent.
- Agreement output distinguishes HarmBench classifier vs the second judge.
- Report-worthy changes regenerate the docx.
