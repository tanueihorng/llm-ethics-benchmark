# TC1 Agent Checklist

Generated: 2026-09-07 10:50:03 UTC+8

## Boundary

- Human handles Wi-Fi, VPN, MFA, and any interactive password step.
- Agent handles repo checks, Git sync, sbatch submission, squeue/seff monitoring, and redacted artifact validation after login.
- TC1 head node rule: no user compute code. Use `sbatch` for GPU work.

## Pre-Login Local Checks

```bash
python fyp_cli.py agent-status
make agent-check
git status -sb
```

## TC1 Head Node Sequence

```bash
cd /tc1home/FYP/utan001/fyp_quant/repo
git pull --ff-only
squeue -u utan001
```

## If Re-Running Or Extending Judge Validation

```bash
# Prepare the cache on an approved networked local/staging machine; see runbook section 4.6.
# Transfer the cache to TC1 before submitting; do not run dataset preparation on the head node.
sbatch slurm/judge_validation.sbatch
squeue -u utan001
seff <JOBID>
```

## Back On Mac After SCP

```bash
# Select the budget/scorer and arguments for the current task; primary evidence is results_512/.
python scripts/judge_agreement.py --help
# Rebuild only the requested derived analysis; do not substitute proxy pair labels for judge ASR.
make agent-check
python scripts/generate_handoff.py
```

## Guardrails

- Do not copy raw HarmBench prompt/response text into chat, logs, or handoffs.
- Preserve `raw.jsonl` and `summary.json`; write new scoring as sidecars.
- If report-worthy content changes, regenerate the report and log it in PROJECT_LOG.
