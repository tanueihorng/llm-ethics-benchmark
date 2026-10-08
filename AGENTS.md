# AGENTS.md

Repository instructions for fyp_quant. Keep `AGENTS.md` and `CLAUDE.md` identical.

## Start with the task

1. Inspect `git status --short`; preserve existing edits and untracked work. Do not delete files, split the project log, or rewrite Git history as cleanup.
2. Run `python fyp_cli.py agent-status` for current state. Search `docs/PROJECT_LOG.md` for the task/topic and superseding decisions; read matching entries, not the full history.
3. Read affected code and callers. Use `docs/REPO_GUIDE.md` and `make help` for navigation; use the task-specific skill below when relevant.

`docs/PROJECT_LOG.md` is authoritative for status, tasks, decisions and changes. `docs/HANDOFF.md`, `docs/AGENT_DASHBOARD.md` and `todo.md` are secondary views. Verify dated claims against live artifacts. Never infer that an old deadline means an action was completed.

## Record and verify changes

- Add one newest-first PROJECT_LOG §4 row per change: `When (UTC+8) | Files | Change | Why / Notes | Report? | Who`. Use `YYYY-MM-DD HH:MM`, actual touched paths, and whether `make report` ran. Bump metadata date/author.
- Record decisions with rationale in §3, follow-ups in §2, and current-state changes in §1. Tick completed tasks and record their durable outcome; preserve historical entries. Do not commit or push without the matching log update.
- Start with focused checks; broaden for demonstrated integration risk. Fix in-scope failures without asking permission for each retry. Distinguish pre-existing failures and checks not run; passing checks prove only what they test.
- Harness changes require `.agents/skills/fyp-harness-maintenance/` and regeneration of affected generated harness documents. Pure instruction/dependency cleanup does not require a report rebuild unless report content changes.

## Research and artifact integrity

- TC1-original `raw.jsonl` and `summary.json` are immutable. Write post-hoc scores to derived redacted sidecars. Never expose raw HarmBench prompts, behaviors, generations or responses in chat, logs or docs; use IDs, labels and aggregates.
- Follow `configs/artifact_policy.yaml`. Never weaken a pattern to clear a finding; correct the claim or add a justified allow entry. Preserve redaction, immutability and stale-text safeguards.
- Do not invent facts, numbers, commands, paths or citations. Separate session-verified evidence, historical records and inference.
- Read the primary report artifact before describing its contents. Trace claims inward to supporting sources; resolve citation identity/title/author and label preprints. Named methods require bibliography entries in the same edit (D54/D56/D61).
- Pair presence checks with omission checks, or disclose missing coverage. Derive coverage denominators independently of the extractor under audit. Recheck delegated findings against primary artifacts; make verification reproducible.
- Every audit is a fresh, independent audit of the current artifact. Prior audits, log summaries of their findings and earlier worklists are unverified leads: open them only after completing your own checks, then reconcile. An unchanged hash exempts nothing, and each reported finding needs evidence gathered in this audit.
- When results, primary configuration or significance claims change, search for the retired claim across README, results card, log, builders, decks, dashboard and figures (D42). Correct current views, label historical snapshots, and self-test affected stale-text checks.

## Reports

Resolve the deliverable in `configs/report_versions.yaml` (`make reports-list`). Canonical, submitted and review copies have different purposes; do not promote a comparison draft or choose the final-report base implicitly. Edit builders, never generated DOCX files. Preserve frozen submission copies and historical builders.

Reports and presentations stay off GitHub (D94). Report and deck builders, rendered documents, `docs/**/*.html`, `docs/viva/`, `docs/drafts/` and `world3d/` are gitignored and exist only locally; never `git add -f` them. CI runs only the core suite, so run the report suite locally.

Regenerate the affected deliverable for changes to methodology, architecture, experimental setup, results, limitations or appendices. Canonical content uses `scripts/build_fyp_report_v5.js` and `make report`; other variants use their registered builders. Inspect the source and regenerate if unsure whether content changed. Record regeneration in PROJECT_LOG.

## TC1 operations

Read `docs/TC1_CLUSTER_RUNBOOK.md`. Run evaluation, dataset preparation, analysis, script generation and validation locally or through `sbatch`, never on the head node. Use direct `sbatch`, not `make cluster-submit` or `srun`; preserve bootstrap and offline flags.

Head-node setup permits only documented package installation, file transfer and SLURM administration. Download permission does not permit arbitrary Python. Run `scripts/prefetch_tc1.py` on approved networked local/staging infrastructure, then transfer the prepared cache (§4.6). Do not start expensive experiments or overwrite scorer sidecars for an instruction audit.

## Task routing

- Harness: `.agents/skills/fyp-harness-maintenance/`
- Report/evidence audit: `.agents/skills/fyp-report-audit/`; the original version is kept as `.agents/skills/fyp-report-audit-old/` for use when requested by name
- Judge extensions: `.agents/skills/fyp-second-judge/`
- Meetup story: `.agents/skills/fyp-meetup-brief/`
- Task packets: relevant entries under `docs/agent_tasks/`
- Specialist review definitions: `.codex/agents/`; use when authorized and useful

## Project Overview

Matched fp16, NF4 and INT8 loading across HarmBench, XSTest, MMLU and ARC-Challenge. Primary study: `results_512/`; 128-token comparison: `results/`. HarmBench is classifier-primary. XSTest keeps its prespecified v2 scorer and separately identified validation-informed sensitivity analysis. Never substitute regex non-refusal proxy interpretations for judge-primary ASR.

Primary comparison family: `results_512/analysis/multiple_comparisons.json`; judge-strict sensitivity uses its explicitly named family. Resolve other evidence from the claim registry and target document.

`ethical_benchmark/` contains research code; `fyp_cli.py` is the CLI; `dashboard/data.py` serves the dashboard. Installation, commands, script categories and app boundaries are in `docs/REPO_GUIDE.md`. Historical instruction rationale remains in `docs/archive/agent-instructions-2026-09-06.md` and `docs/archive/agent-instructions-2026-09-13-pre-cleanup.md`.
