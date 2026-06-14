# Handoff

Last refreshed: 2026-06-14 (~16:00 UTC+8) by Claude (hand-authored via /fyp-agent-handoff).

## Mode

Fresh-session recovery — for the next Claude/Codex session or a new context window.

## Objective

The study is complete, judge-validated, and now hardened against both of its Priority-1 weaknesses that needed new runs (W1 via multi-seed, W3 via a second judge). The remaining work is **non-compute**: send the supervisor email (T1) and submit the report (T15). There is **no pending TC1 compute**.

## Source Of Truth

- Read `/Users/tanueihorng/fyp_quant/AGENTS.md`, then `/Users/tanueihorng/fyp_quant/docs/PROJECT_LOG.md` (§1 status, §2 open actions, §3 decisions D1–D26, §4 changelog).
- Run `git status -sb` and `git log --oneline -8` for live Git state — never trust a hard-coded commit/ahead count.
- This file is a bridge, not a second project log. Verify the claims below from the repo.

## Current State (durable)

- **Study complete + judge-validated (D16) + double-robustness-checked.** Authoritative judge-primary headline (`results/analysis/judge_agreement.json`): Qwen 1.7B = **broad_degradation** (greedy ΔASR +0.055, CI [+0.010,+0.100], McNemar p=0.027; ΔMMLU −0.087); Qwen 4B = alignment_degradation but directional (ΔASR +0.025, n.s.); Llama 3B = broad_degradation (ΔASR 0.000; ΔMMLU −0.043 sig). NF4 never reduces harmful compliance in any pair.
- **T18 multi-seed sensitivity (D23):** Qwen 1.7B HarmBench re-run at T=0.7 across seeds 1–5, judge-scored. ΔASR mean **+0.024** (range [−0.010,+0.055], not sign-consistent) vs greedy +0.055 → the headline is the upper end of a decode-dependent range; tempered, not overturned. Artifact `results/analysis/sensitivity_multiseed.{json,csv}`; raw `results_sensitivity/` gitignored.
- **T21 Ch6/7 strengthening (D24):** §6.6.2 per-category judge ASR breakdown (Qwen 1.7B ASR rises 5/6 harm categories → headline broad-based; `results/analysis/harmbench_category_breakdown.{json,csv}`), §7.6 deployment implications, §7.7 positioning vs prior work, and **verified citations** replacing the old "Anonymous Authors" placeholders.
- **T22 second judge (D25 infra / D26 run):** gpt-4o re-scored all 6 models (`--backend api_judge`, local, 0 parse errors). Agrees with the primary classifier at **κ 0.69–0.94**; reproduces Qwen 1.7B ΔASR +0.045 (vs +0.055) in direction/magnitude but borderline on significance (McNemar p=0.122). **W3 substantially resolved.** Converges with T18: effect real but modest. Artifacts `results/*/harmbench/scores.judge.api_judge.*` + `results/analysis/judge_pairwise_agreement.{json,csv}` (redacted).
- **Report renamed** to `docs/FYP_Report_2026-06-14.docx` (was 2026-05-27). **218 tests pass; `make agent-check` 8/8.**

## What Changed / What Was Claimed (verify from repo)

- Decisions **D23–D26** in §3; **T18/T21/T22 ticked** in §2; **T24/T25** added (audit nit; stale Appendix D test inventory).
- Report source `scripts/build_fyp_report.js` (new §6.6.1/6.6.2/6.12-R4/7.6/7.7, verified refs, output renamed). New scripts: `harmbench_category_breakdown.py`, `judge_pairwise_agreement.py`. Backend wiring in `ethical_benchmark/judges/validation.py` + `scripts/run_judge_validation.py` (`--backend api_judge`).
- `git status -sb` should be clean and in sync with `origin/main` (everything above is committed + pushed). Untracked, by design: 5 presentation artifacts (`.claude/launch.json`, `.claude/skills/html-diagram/`, `docs/Codex_Meetup_*`, `docs/fyp_architecture.html`) and the gitignored email drafts (`docs/email_*.md`).

## Verification To Run

```bash
git status -sb && git log --oneline -8        # live Git state
pytest -q                                      # expect 218 passed
make agent-check                               # expect 8/8 pass
python scripts/judge_agreement.py              # primary judge headline + McNemar/evidence
python scripts/sensitivity_analysis.py         # T18 multi-seed aggregate (byte-identical regen)
python scripts/judge_pairwise_agreement.py     # T22 judge-vs-judge (needs api_judge sidecars, present)
```

## Risks / Things To Distrust

- **Both robustness checks (T18, T22) cover only the load-bearing Qwen 1.7B finding deeply.** They corroborate its *direction* but show *significance is borderline/decode-and-judge-dependent*. Do not over-state the +0.055 as a large/certain effect — the honest framing (now in the report) is "real but modest, ~+0.045–0.055".
- **api_judge sends benchmark text to an external API** (D25, accepted) — written sidecars stay redacted; the key is read from `OPENAI_API_KEY` (in the user's `~/.zshrc`, NOT this tool's shell). `openai` is installed in conda `base` (no `.venv`).
- Repo is **PUBLIC**; email drafts are gitignored and intentionally remain only in history (settled — do not re-raise).
- `raw.jsonl`/`summary.json` are immutable TC1-original; all judge/v2/sensitivity layers are redacted sidecars.
- `pairwise_deltas.{json,csv}` stores the **v2** ASR by design; the headline uses the **judge** numbers from `judge_agreement.json`.
- Appendix D test inventory is stale (178 vs real 218) — tracked as **T25**, not a live error elsewhere.

## Next Actions (ordered)

1. **T1 — send the supervisor email.** Draft at `docs/email_drZhang_2026-06-13.md` (gitignored, local-only; results-in-hand follow-up to the 2026-05-23 email that was already sent). User is finalising wording.
2. **T15 — submit `docs/FYP_Report_2026-06-14.docx`.**
3. **T3 — run `MyTCinfo` on TC1** (storage quota; non-blocking).
4. Optional/later: open-weight LlamaGuard cross-check (`--backend llamaguard`, TC1, for full reproducibility), **T23** (ARC-Challenge second capability benchmark, W5), **T25** (refresh Appendix D), **T24** (`sensitivity_analysis.py` provenance nit).

## Privacy / Artifact Guardrails

No raw HarmBench prompt/response text in chat, docs, or commits. Use IDs, counts, labels, aggregate metrics, redacted sidecars. Preserve `raw.jsonl`/`summary.json` as TC1-original. `results_sensitivity/` stays gitignored; only redacted aggregates are committed. Email drafts (`docs/email_*.md`) are gitignored — never commit them.
