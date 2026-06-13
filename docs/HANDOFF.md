# Handoff

Last refreshed: 2026-06-09 (~17:55 UTC+8) by Claude (hand-authored via /fyp-agent-handoff).

## Mode

Fresh-session recovery — for the next Claude/Codex session or a new context window.

## Objective

Continue the FYP. The one live thread is the **T18 multi-seed sensitivity run**: it is built and pushed but **not yet run on TC1**. When its results come back, interpret them and fold them into the report. Everything else is in a finished, consistent state.

## Source Of Truth

- Read `/Users/tanueihorng/fyp_quant/AGENTS.md`, then `/Users/tanueihorng/fyp_quant/docs/PROJECT_LOG.md` (§1 status, §2 open actions, §3 decisions D1–D21, §4 changelog).
- Run `git status -sb` and `git log --oneline -5` for live Git state — do **not** trust any hard-coded commit/ahead count.
- This file is a bridge, not a second project log. Verify claims below from the repo.

## Current State (durable)

- **Study complete + judge-validated (D16).** Authoritative headline (judge-primary, from `results/analysis/judge_agreement.json`): Qwen 1.7B = **broad_degradation** (ΔASR +0.055, CI [+0.010,+0.100], McNemar p=0.027; ΔMMLU −0.087); Qwen 4B = **alignment_degradation but directional** (ΔASR +0.025, n.s., McNemar p=0.18); Llama 3B = **broad_degradation** (ΔASR 0.000; ΔMMLU −0.043 sig). NF4 never reduces harmful compliance in any pair.
- **D19 (this session):** two-layer reporting — `classify_pair_change` label unchanged; new `label_evidence_status` (confirmed/directional/null) + McNemar exact test (`mcnemar_exact_test`) + per-subject MMLU artifact (`results/analysis/mmlu_subject_breakdown.{json,csv}`). Analysis CSVs normalised to LF.
- **D20 (this session):** T18 sensitivity arm built as turnkey TC1 infra — `configs/tc1_sensitivity.yaml` (HarmBench, T=0.7), `slurm/jobs_tc1_sensitivity/*` (per-model generation looping seeds 1–5 + per-seed judge), `scripts/sensitivity_analysis.py`. **RUN IS PENDING on TC1** (user executes; head-node Python forbidden). `results_sensitivity/` is gitignored.
- **D21 (this session):** fixed the Agent Harness GitHub Actions CI — `check_immutable_artifacts` now tolerates absent (gitignored) raw artifacts (CI/fresh-clone case) while still failing on mutation. The earlier red CI / failure emails are resolved.
- The previously-uncommitted agent-harness layer (D17/D18) is now committed + pushed.
- **193 tests pass; `make agent-check` 8/8.** Supervisor email drafted at `docs/email_drZhang_2026-06-09.md` (T1, not yet sent).

## What Changed / What Was Claimed

- Decisions D19, D20, D21 added to `docs/PROJECT_LOG.md` §3 (all verified-from-repo, not just claimed).
- Code: `ethical_benchmark/analysis/compare_quant_pairs.py`, `scripts/judge_agreement.py`, `scripts/mmlu_subject_breakdown.py` (new), `scripts/sensitivity_analysis.py` (new), `scripts/generate_sensitivity_jobs.py` (new), `ethical_benchmark/harness/agent.py`, tests (`test_quant_analysis.py`, `test_sensitivity.py` new, `test_agent_harness.py`), report source `scripts/build_fyp_report.js` (+ regenerated docx).

## Verification To Run

```bash
git status -sb && git log --oneline -5      # live Git state
pytest -q                                   # expect 193 passed
make agent-check                            # expect 8/8 pass
python scripts/judge_agreement.py           # regenerates judge headline + McNemar/evidence
python scripts/sensitivity_analysis.py      # ONLY meaningful after T18 has run on TC1
```

## Risks / Things To Distrust

- **T18 has not run.** Do not claim sensitivity results exist until `results_sensitivity/` is populated (on TC1/Mac) and `sensitivity_analysis.py` has produced `results/analysis/sensitivity_multiseed.{json,csv}`. The outcome is genuinely open (effect may hold or wash out across seeds — both are valid findings).
- The CI tolerance for absent artifacts (D21) is **intentional** — do not "re-tighten" it back to failing on absence.
- `raw.jsonl` / `summary.json` are immutable TC1-original artifacts; post-hoc scoring/judge work uses redacted sidecars only.
- Verify Git state live; never rely on a written ahead-count.

## Next Actions (ordered)

1. **Run T18 on TC1** (user): `git pull --ff-only`, then `sbatch slurm/jobs_tc1_sensitivity/qwen_2b_base__sens.sbatch` + `qwen_2b_4bit__sens.sbatch` (load-bearing pair first; optionally the qwen_4b + llama pairs for full symmetry), then the `judge_sens_seed{1..5}.sbatch` jobs, then SCP `results_sensitivity/` to the Mac and run `python scripts/sensitivity_analysis.py`. (If only the Qwen 1.7B pair is generated, edit `--models` in the judge sbatch to `qwen_2b_base qwen_2b_4bit`.)
2. **Interpret** `sensitivity_multiseed`: is the Qwen 1.7B ΔASR stable across seeds? Fold into report §6.5 + Ch9 and `make report` (then it becomes report-worthy).
3. **Outstanding tasks (§2):** T1 (send the drafted email to Dr. Zhang), T3 (`MyTCinfo`), T15 (final submission). Optional robustness: T22 (second judge), T19/T23 (more scale points / second capability benchmark).
4. Confirm the Agent Harness CI run on the latest commit is green (GitHub Actions tab).

## Privacy / Artifact Guardrails

No raw HarmBench prompt/response text in chat, docs, or commits. Use IDs, counts, labels, aggregate metrics, and redacted sidecars. Preserve `raw.jsonl`/`summary.json` as TC1-original.
