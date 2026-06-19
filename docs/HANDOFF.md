# Handoff

Last refreshed: 2026-06-18 by Claude (hand-authored via /fyp-agent-handoff).

## Mode

Fresh-session recovery / wrap-up. The research, code, report, audit, reuse deliverables, and a standalone thesis are all **complete and on `main`**. What remains is non-research: email the supervisor, submit, and a licence decision.

## Objective

Send the supervisor email (T1) and submit the FYP document (T15); optionally settle the `LICENSE` decision. No experiments, analysis, or report rewrites are outstanding.

## Source Of Truth

- Read `/Users/tanueihorng/fyp_quant/AGENTS.md`, then `/Users/tanueihorng/fyp_quant/docs/PROJECT_LOG.md` (§1 status; §3 decisions **D1–D36**; §4 changelog). Tactical resume buffer: `/Users/tanueihorng/fyp_quant/todo.md`.
- Run `git status -sb` and `git log --oneline -8` for live Git state — never trust a hard-coded commit/ahead count.

## Current State (durable, verify before acting)

- **`main` @ `87920c7`, in sync with origin.** 295 tests pass; `make agent-check` 8/8.
- **The study (durable headline, judge-primary D16):** under four-bit NF4, Qwen-1.7B is the only significant ΔASR (+0.055, modest/borderline/judge-dependent); the INT8 precision point (D35, report §6.15) shows the effect is **not bit-width-graded** — capability loss is a clean cliff at 4-bit, and a second, both-judge + McNemar-significant ASR move appears on **Llama-3B specifically at INT8** and reverts at NF4 (caveated: ~8–9 prompts, one pair).
- **Full-repo audit (D36) verdict:** nothing invalidates the results — every primary HarmBench ASR is classifier-scored, the v2 regex is the demoted foil only, 120 raw artefacts hash-match.
- **Two deliverable documents exist:**
  - `docs/FYP_Report_2026-06-14.docx` — the interim report (`make report`; source `scripts/build_fyp_report.js`).
  - `docs/FYP_Thesis_2026-06-18.docx` — a SEPARATE full thesis (`make thesis`; source `scripts/build_fyp_thesis.js`), IEEE-cited with all 18 sources browse-verified against arXiv. The two builders are independent.
- **Reuse package:** `pip install -e .` (`pyproject.toml`), `CITATION.cff`, and `docs/{QUICKSTART,REPRODUCIBILITY,RESULTS_CARD,paper_outline,THESIS_OUTLINE}.md`.

## What Changed (recent, verified) — see PROJECT_LOG §4

- D35 / T29: INT8 precision point run + folded into report §6.15 (merged `48330d4`).
- D36: full-repo scorer-integrity audit + non-invalidating fixes (tests 282→295).
- Reuse/dissemination deliverables + README refresh.
- New standalone thesis (`build_fyp_thesis.js`) with IEEE in-text `[n]` citations, a strengthened §2.4 research gap, and every reference verified against its arXiv source (PROJECT_LOG §4 rows 2026-06-18 20:00 / 20:45 / 21:15).

## Verification To Run

```bash
git status -sb && git log --oneline -3     # expect main @ 87920c7, in sync
pytest -q                                   # expect 295 passed
make agent-check                            # expect 8/8 pass
make thesis && make report                  # both docx rebuild cleanly (independent)
```

## Risks / Things To Distrust

- **`make report` and `make thesis` are independent** — editing one does not change the other; don't conflate them.
- **The thesis cover says "Final Report — Thesis."** If the actual submission milestone is the *interim*, relabel it (one line in `scripts/build_fyp_thesis.js`).
- **No `LICENSE` yet** — the repo is all-rights-reserved (blocks reuse). This is a USER decision pending an NTU FYP IP-policy check; do not add one unilaterally (`docs/REPRODUCIBILITY.md §6`).
- Do **not** re-inflate the INT8 Llama finding (it is deliberately caveated) or describe INT8 as "8-bit NF4" (it is LLM.int8, a different method).
- The v2 regex is a demoted foil, never a primary result; all reported HarmBench ASR is classifier-scored (audited study-wide, D36).

## Next Actions (ordered)

1. **T1 — email Dr. Zhang** (`docs/email_drZhang_2026-06-13.md`, gitignored, local-only). Submission-critical.
2. **T15 — submit.** Choose the report or the thesis per the milestone; rebuild with `make report` / `make thesis` if edited.
3. **LICENSE decision** (optional, enables reuse) — confirm NTU FYP IP policy, then add MIT/Apache-2.0 + set `license` in `pyproject.toml`/`CITATION.cff`; optionally archive to Zenodo for a DOI.
4. **T3 — `MyTCinfo`** on TC1 (storage quota). Optional.
5. Optional thesis polish + research follow-ups are listed in `todo.md`.

## Privacy / Artifact Guardrails

No raw HarmBench/XSTest prompt or response text in chat, docs, or commits — IDs, counts, labels, aggregates, redacted sidecars only. `raw.jsonl`/`summary.json` are immutable TC1 originals (gitignored; hash-pinned in `results/raw_artifact_manifest.sha256`) — never reopen. Email drafts (`docs/email_*.md`) are gitignored — never commit.
