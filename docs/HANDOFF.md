# Handoff

Last refreshed: 2026-06-14 (~13:10 UTC+8) by Claude (hand-authored via /fyp-agent-handoff).

## Mode

Fresh-session recovery — for the next Claude/Codex session or a new context window.

## Objective

The study is complete, judge-validated, and now multi-seed-robustness-checked. The remaining work is **non-compute**: commit/push this session's T18 write-up, then send the supervisor email (T1) and submit the report (T15). There is **no pending TC1 compute**.

## Source Of Truth

- Read `/Users/tanueihorng/fyp_quant/AGENTS.md`, then `/Users/tanueihorng/fyp_quant/docs/PROJECT_LOG.md` (§1 status, §2 open actions, §3 decisions D1–D23, §4 changelog).
- Run `git status -sb` and `git log --oneline -5` for live Git state — do **not** trust any hard-coded commit/ahead count.
- This file is a bridge, not a second project log. Verify the claims below from the repo.

## Current State (durable)

- **Study complete + judge-validated (D16) + multi-seed-checked (D23).** Authoritative judge-primary headline (from `results/analysis/judge_agreement.json`): Qwen 1.7B = **broad_degradation** (greedy ΔASR +0.055, CI [+0.010,+0.100], McNemar p=0.027; ΔMMLU −0.087); Qwen 4B = **alignment_degradation but directional** (ΔASR +0.025, n.s.); Llama 3B = **broad_degradation** (ΔASR 0.000; ΔMMLU −0.043 sig). NF4 never reduces harmful compliance in any pair.
- **D23 (this session) — T18 ran on TC1.** Multi-seed (T=0.7, top-p 0.8, seeds 1–5) HarmBench sensitivity arm for the load-bearing **Qwen 1.7B pair**, judge-scored. Result: judge ΔASR **mean +0.024, sd 0.028, range [−0.010,+0.055], not sign-consistent** (4/5 seeds positive) vs greedy +0.055 (which = the max seed). The 4-bit ASR is seed-stable (~0.170); the baseline is the noisy member (0.115–0.180). This **tempers but does not overturn** the headline — the greedy +0.055 is the upper end of a decode-dependent range; the ΔMMLU half of the dual-degradation claim is unaffected. Redacted aggregate at `results/analysis/sensitivity_multiseed.{json,csv}`; raw `results_sensitivity/` is gitignored.
- **Report folded in:** new §6.6.1 (W1 robustness check), rewritten §6.5 caveat, Ch8 limitation + Ch9 future-work flipped from "future" to "partially completed". `make report` regenerated `docs/FYP_Report_2026-05-27.docx`.
- **Correctness audit (this session):** 3 independent read-only audits (code-quality, results/methodology cross-check, repo-hygiene) + an own numeric reconciliation — **all PASS**. T18 numbers recompute exactly from the 10 raw judge summaries (0 parse errors, 200/200 samples); greedy headline unchanged; aggregates redacted; `results_sensitivity/` correctly gitignored; 215 tests; `git diff --check` clean. One low-severity provenance nit logged as **T24** (no committed number affected).
- **215 tests pass; `make agent-check` 8/8.**

## What Changed / What Was Claimed (verify from repo)

- **Uncommitted on the Mac** (working tree only — `git status -sb`):
  - `M scripts/build_fyp_report.js`, `M docs/FYP_Report_2026-05-27.docx`, `M docs/PROJECT_LOG.md`
  - `?? results/analysis/sensitivity_multiseed.{json,csv}` (redacted — should be committed)
  - `?? docs/email_drZhang_2026-06-13.md` (T1 draft; user is refining)
- Decisions **D23** added to §3; **T18 ticked** `[x]` in §2; **T24** added (audit nit); §1 metadata bumped.
- Pre-existing untracked presentation artifacts (not from this session, referenced in earlier changelog rows): `.claude/launch.json`, `.claude/skills/html-diagram/`, `docs/Codex_Meetup_Agent_Harness_Report.{docx,html}`, `docs/fyp_architecture.html` — need an intentionality decision (commit vs gitignore).

## Verification To Run

```bash
git status -sb && git log --oneline -5           # live Git state
pytest -q                                         # expect 215 passed
make agent-check                                  # expect 8/8 pass
python scripts/sensitivity_analysis.py            # regenerates sensitivity_multiseed.{json,csv} (should be byte-identical)
python scripts/judge_agreement.py                 # regenerates judge headline + McNemar/evidence
```

## Risks / Things To Distrust

- **T18 covers only the Qwen 1.7B pair.** The symmetric Qwen 4B + Llama sensitivity infra exists but was **not run** (they stay greedy-only). `sensitivity_multiseed.json` correctly shows `n_seeds=0` / `n/a` for those two pairs — that is expected, not a bug. A full-matrix extension is future work (folds into T19).
- **Do not re-tighten the CI immutable-artifacts tolerance (D21)** or treat a small docx byte-delta as staleness.
- `raw.jsonl` / `summary.json` are immutable TC1-original artifacts; the judge/v2/sensitivity layers are redacted sidecars only.
- `pairwise_deltas.{json,csv}` still stores the **v2** HarmBench ASR by design (D16) — the report's Ch6 headline correctly cites the **judge** numbers from `judge_agreement.json`. Don't mistake the v2 CSV for the headline source.
- Verify Git state live; never rely on a written ahead-count.

## Next Actions (ordered)

1. **Commit + push this session's T18 work** (when the user asks): `scripts/build_fyp_report.js`, `docs/FYP_Report_2026-05-27.docx`, `docs/PROJECT_LOG.md`, `docs/HANDOFF.md`, and the redacted `results/analysis/sensitivity_multiseed.{json,csv}`. Decide separately on the untracked presentation artifacts + email draft. Branch off `main` if not already on a feature branch.
2. **T1 — send the supervisor email.** Draft at `docs/email_drZhang_2026-06-13.md` (results-in-hand follow-up to the 2026-05-23 email, which was already sent). User is refining the wording.
3. **T15 — submit the interim report** (T18 is now folded in).
4. **T3 — run `MyTCinfo` on TC1** (storage quota; non-blocking).
5. Optional/later: **T24** (sensitivity_analysis.py provenance nit — do before any full-matrix T18 extension), full-matrix T18 (Qwen 4B + Llama), **T22** (second independent judge), **T19/T23** (more scale points / second capability benchmark).

## Privacy / Artifact Guardrails

No raw HarmBench prompt/response text in chat, docs, or commits. Use IDs, counts, labels, aggregate metrics, and redacted sidecars. Preserve `raw.jsonl`/`summary.json` as TC1-original. `results_sensitivity/` stays gitignored (raw harmful generations); only the redacted aggregate is committed.
