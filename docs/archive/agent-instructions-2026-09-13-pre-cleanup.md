# AGENTS.md

Repository instructions for coding agents working on fyp_quant.

AGENTS.md and CLAUDE.md share the same rules. Update both together.

## Context and current state

`docs/PROJECT_LOG.md` is the authoritative status, task, decision and changelog record. Read only the relevant sections or task entries; do not load the entire history at startup. For a live summary, use `python fyp_cli.py agent-status`. Search the log for the task ID or topic, then read the matching entry and any superseding decisions. Routine edits need the affected files and their callers, not a full project tour.

`docs/HANDOFF.md` is a dated secondary bridge; `todo.md` holds tactical resumption detail. Verify important claims against live artifacts. Neither overrides the project log. Historical instruction narratives are in `docs/archive/agent-instructions-2026-09-06.md`; consult only for the relevant incident.

## Change record

Every repository change includes one newest-first row in PROJECT_LOG §4: `When (UTC+8) | Files | Change | Why / Notes | Report? | Who`. Use `YYYY-MM-DD HH:MM`, list actual touched paths, and record whether `make report` ran. Bump the log metadata date and author. Record decisions with rationale in §3; add new follow-ups to §2. When closing an action, tick it and move its durable outcome into the changelog. Update §1 when current state changes. Do not commit or push without the matching log update.

## Research and artifact integrity

- Treat TC1-original `raw.jsonl` and `summary.json` as immutable. Post-hoc scoring writes derived redacted sidecars. Do not expose raw HarmBench prompt, behavior, generation or response text in logs, docs or chat; use IDs, labels and aggregates.
- Use `configs/artifact_policy.yaml` for immutability, redaction and stale-text checks. Never weaken a review pattern to clear a hit; correct the claim or add a justified allow entry. Preserve the safeguards when changing the harness.
- Do not invent facts, numbers, commands, paths or citations. Distinguish evidence verified this session, recalled records and inference. Say what was not checked. A passing gate proves only the properties it tests.
- Read the primary report artifact before stating its content. Verify claims inward from the sentence to its source, including whether the source supports the wording. Resolve citation identifiers and confirm title/author; label preprints. Named methods need bibliography entries in the same edit (D54/D56/D61).
- Pair presence checks with omission checks, or disclose that omissions were not covered. Derive coverage denominators independently of the extractor being audited. Recheck other agents' findings against primary artifacts; make the evidence trail independently reproducible.
- When a result, primary configuration or significance claim changes, search the whole claim surface for the retired claim itself (D42), including README, results card, log, builders, decks, dashboard and figures. Fix current-facing claims; label historical snapshots. Extend and self-test stale-text checks where appropriate.

## Reports

Resolve the requested document and its builder in `configs/report_versions.yaml`; do not assume every report request targets the canonical line. Edit the builder, never a generated DOCX. The canonical report uses `scripts/build_fyp_report_v5.js` and `make report`; separate rewrite/submission variants have their own registered builders and targets.

Regenerate the affected deliverable for changes to methodology, architecture, experimental setup, results, limitations or appendices; run `make report` for changes affecting the canonical report. Pure instruction cleanup or refactors with no report-content effect do not require regeneration. If unsure whether content changed, inspect the affected report source and regenerate. Record regeneration in the log.

## TC1 operations

Use `docs/TC1_CLUSTER_RUNBOOK.md` for cluster work. All evaluation, dataset preparation, analysis, script generation and validation run locally or through `sbatch`, never on the head node. Use direct `sbatch`, not `make cluster-submit` or `srun`. Preserve the environment bootstrap and offline flags in job scripts.

Head-node setup permits the documented package installation, file-transfer and SLURM administration activities. Do not infer permission for arbitrary Python from permission to download files. `scripts/prefetch_tc1.py` calls `load_dataset` and may prepare datasets: run it on an approved networked local/staging machine and transfer the prepared cache; see runbook §4.6. Never start expensive experiments or overwrite existing scorer sidecars merely to audit instructions.

## Verification and routing

Run checks relevant to changed behavior. For harness changes, use `.agents/skills/fyp-harness-maintenance/`; run its required gate and regenerate affected harness documents. For other work, focused tests/builds and artifact inspection are the starting point; broaden for demonstrated integration risk. Continue fixing in-scope failures without asking for approval for each local retry.

- Report/evidence audit: `.agents/skills/fyp-report-audit/`.
- Judge extensions: `.agents/skills/fyp-second-judge/`.
- Meetup story: `.agents/skills/fyp-meetup-brief/`.
- Task packets: `docs/agent_tasks/`, selected by the current task rather than a fixed old task number.
- Isolated specialist reviews: `.codex/agents/`, when the task benefits from one.

## Project Overview

fyp_quant compares matched fp16, NF4 and INT8 model loading across HarmBench, XSTest, MMLU and ARC-Challenge. The primary study is the 512-token tree `results_512/`; `results/` retains the 128-token comparison. HarmBench is classifier-primary; XSTest retains its prespecified v2 scorer with separately identified validation-informed sensitivity analysis. Do not substitute regex non-refusal proxy interpretations for judge-primary ASR.

Use `results_512/analysis/multiple_comparisons.json` for the primary family, and the explicitly named judge-strict family for that sensitivity analysis. Resolve other evidence from the claim registry and target document rather than copying historical headline numbers into instructions.

Code: `ethical_benchmark/` contains benchmarks, model loading, pipeline, analysis, cluster helpers and config validation. `fyp_cli.py` is the CLI; `Makefile` lists local commands; `dashboard/data.py` is the dashboard data layer. Permanent architecture and operational detail lives in `docs/REPO_GUIDE.md` and the task-specific runbooks.
