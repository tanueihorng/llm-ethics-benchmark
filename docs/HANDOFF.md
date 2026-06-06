# Handoff

Last refreshed: 2026-06-06 16:30 UTC+8 by Claude (verified against live `git status -sb`, `pytest -q` = 178 pass, 6 judge sidecars present). User is stopping for the day; next session resumes here.

## Mode

Fresh-session recovery for Codex, Claude Code, or another coding agent.

## Objective

Start the next session from the repo state, then improve the result/discussion analysis and decide whether to add robustness extensions such as a second independent judge or more model pairs.

## Source Of Truth

- Read `/Users/tanueihorng/fyp_quant/AGENTS.md`.
- Read `/Users/tanueihorng/fyp_quant/docs/PROJECT_LOG.md`.
- Run `git status -sb` for live Git state.
- Treat this file as a bridge, not as a replacement for `docs/PROJECT_LOG.md`.

## Current State

- D16 is the key project decision: the official HarmBench classifier is now the primary HarmBench ASR scorer; the v2 regex scorer is a secondary non-refusal-rate proxy.
- TC1 HarmBench judge validation has completed: job `61047`, fp16, 32 GB V100, `n=200 x 6`, zero parse errors.
- Main judge-primary result: Qwen 1.7B is the only statistically significant HarmBench safety worsening under the official judge, and it also loses significant MMLU capability, so the correct headline label is `broad_degradation`.
- Qwen 4B is directionally worse under the judge but not significant; Llama is flat on judge ASR with capability loss.
- The D16 judge-primary work is on `origin/main`. For any later local edits, verify live with `git status -sb` and `git log --oneline origin/main..HEAD`.
- `docs/FYP_Report_2026-05-27.docx` is generated from `scripts/build_fyp_report.js`; do not hand-edit the docx.
- The handoff skill is now committed project-level at `.claude/skills/fyp-agent-handoff/SKILL.md` (Claude Code loads it automatically in this repo; mirrors the Codex user-level copy at `~/.codex/skills/`). `.claude/settings.local.json` is gitignored and stays machine-local.

## What Changed / What Was Claimed

- The HarmBench judge sidecars are the authoritative HarmBench ASR scoring layer:
  - `results/*/harmbench/scores.judge.harmbench_cls.jsonl`
  - `results/*/harmbench/summary.judge.harmbench_cls.json`
  - `results/analysis/judge_agreement.{json,csv}`
- Raw TC1 evidence remains immutable:
  - `results/*/*/raw.jsonl`
  - `results/*/*/summary.json`
- v2 sidecars remain useful as a proxy/non-refusal comparison, but not the HarmBench headline.
- `docs/PROJECT_LOG.md` records the full D16 story, open actions, decisions, and changelog.

## Verification To Run

```bash
git status -sb
git log --oneline origin/main..HEAD
rg -n "NEXT: push|push the latest|pending judge|v2 primary|RESULTS PENDING" docs scripts README.md AGENTS.md CLAUDE.md -g '!docs/HANDOFF.md'
python scripts/judge_agreement.py
pytest -q
git diff --check
stat -f '%z %Sm %N' docs/FYP_Report_2026-05-27.docx
```

Expected high-level outcomes:

- No current-facing stale instruction says the judge is pending or that v2 is primary.
- No current-facing stale instruction says the latest D16 work still needs pushing.
- Historical changelog / Appendix G rows may mention prior states; treat those as history, not current instructions.
- `judge_agreement.py` reproduces the D16 judge-primary labels and CIs.
- Tests pass.
- Report regenerates cleanly with `make report` (the docx byte count drifts a few bytes per build due to zip nondeterminism — a small size delta is not a stale-doc signal; check content/revision rows, not exact bytes).

## Risks / Things To Distrust

- Stale prose is the main risk now, especially in report appendices, README, AGENTS, CLAUDE, and PROJECT_LOG status rows.
- Do not trust old v2-headline claims over `results/analysis/judge_agreement.{json,csv}`.
- Do not add a second judge by reusing the wrong backend default; verify the resolved model id and prompt/rubric.
- Do not expand model coverage until the current write-up is stable enough for supervisor review.

## Next Actions

1. Finish any remaining stale-text cleanup and commit/push it.
2. Strengthen Chapter 6 discussion: make the judge-vs-v2 disagreement central, explain why the scoring layer changed while the raw model outputs remain valid, and state the Qwen 1.7B broad-degradation conclusion cleanly.
3. Improve robustness presentation: effect-size table, bootstrap-CI interpretation, family-dependent judge agreement, and limitations of relying on one official classifier.
4. Decide on optional extension:
   - Best first extension: a second independent judge for HarmBench outputs.
   - Lower priority: more model pairs or stochastic/multi-seed decoding sensitivity.
5. Handle non-technical closeout: T1 supervisor email, T3 `MyTCinfo`, T15 final submission.

## Privacy / Artifact Guardrails

- Do not print or copy raw HarmBench prompts/responses into handoffs, logs, diagnostics, or chat.
- Use IDs, counts, labels, aggregate metrics, and redacted sidecars only.
- Preserve `raw.jsonl` and `summary.json` as TC1-original artifacts.
- New scoring or judging must write derived sidecars, never overwrite raw evidence.
