# Handoff

## Mode

Executor → next session.

## Objective

Finish the report line: three style residuals, the freeze checklist, and the outward tasks that
only the student can do.

## Source Of Truth

- Read `/Users/tanueihorng/fyp_quant/AGENTS.md` (identical to `CLAUDE.md` below the header).
- Read `/Users/tanueihorng/fyp_quant/docs/PROJECT_LOG.md` — §1, §2 (T47 especially), and the
  §4 changelog rows dated **2026-07-25 through 2026-08-02**.
- Run `git status -sb` and `git log --oneline -1` for live Git state. Do not trust any count below.
- The prior packet `docs/agent_tasks/T47-T52-report-line-handoff.md` is still broadly accurate,
  **except** that it predates the gap-6 work and the pagination gate described here.

## Current State

**Verified from the repo this session.**

- The whole v6 / rewrite / audit line — roughly a fortnight of document work that had been sitting
  uncommitted — was committed and pushed as `d6fbd03` on 2026-08-02. The working tree is clean.
- `Examples of Past FYP Papers/` (110 MB of past students' reports and copyrighted papers) is now
  **gitignored**. It was untracked but not ignored, and this repository is public. Keep it local.
- `docs/rewritten_fyp_report_v1.docx` is the selected submission candidate (D59, T52). It is
  **57 A4 pages** as rendered by the bundled LibreOffice, up from 56 before this session.
- `docs/FYP_Report_2026-07-01_v5.docx` and `scripts/build_fyp_report_v5.js` are **byte-identical
  to HEAD** and must stay that way.
- Gates at the end of this session: `make verify-claims` **319/319**, `make agent-check` **9/9**,
  `pytest` **519 passed**, front-matter pagination **73/73**.

## What Changed This Session

**T47 gap 6 is closed.** §6.2 of the rewrite now carries **Table 6.2**: one published figure per
prior study ([37], [10], [35], [36], [39]) placed beside this study's own, with the scale
conversion and the chief comparability obstacle stated per row. It replaces a wholly qualitative
discussion. Full provenance, including the twelve wording defects found and fixed before anything
entered the builder, is in `docs/audits/2026-08-02_prior_work_numbers/PROVENANCE.md`; the selection
rule that was fixed **before** the data was read is in `SELECTION_PREREG.md` beside it.

- `scripts/build_rewritten_fyp_report_v1.py` — Table 6.2, revised §6.2 prose, corrected
  front-matter page lists. Rebuilt.
- `scripts/build_fyp_report_v6.js` — two citation fixes (below). This is the shared reference
  source the rewrite reads, so both artifacts were rebuilt.
- `scripts/report_tools/check_pagination.py` (new) + `tests/test_report_pagination.py` (new).
- `tests/test_rewritten_report_v1.py` — two new tests pinning the imported figures and barring the
  refuted ones. Both mutation-proven to fire.
- `docs/PROJECT_LOG.md` — §1 metadata, §2 T47 status, one §4 row.

**Two citation defects were found and fixed** in `build_fyp_report_v6.js`:

- **[36] Q-resafe** is genuinely archival but was cited with the *preprint* title and no locus.
  The PMLR v267 version of record is titled "Assessing Safety Risks and Quantization-aware Safety
  Patching for Quantized Large Language Models" — no `Q-resafe:` prefix — at pp. 9728–9746.
- **[35] HarmLevelBench** is **non-archival**. `in Proc.` asserted a proceedings volume that does
  not exist. Now labelled a preprint with the workshop named as a presentation venue (D56).

**A latent hazard was closed.** The rewrite's Table of Contents, List of Figures and List of Tables
page numbers are hand-typed, and only the TOC self-corrects in Word — so any edit that changes body
length drifts them silently. This was risk #5 in the previous packet ("pagination has never been
verified by any check here"). `check_pagination.py` renders the docx and reads each page's own
printed number; on first run it caught **3 pre-existing drifts** in the committed build (§1.5 said
2 / was 3; §2.5 said 8 / was 9; §2.6 said 9 / was 10). All corrected.

## Verification To Run

```bash
git status -sb
```

```bash
make verify-claims
```

```bash
make agent-check
```

```bash
python scripts/report_tools/check_pagination.py
```

```bash
git status --porcelain -- scripts/build_fyp_report_v5.js docs/FYP_Report_2026-07-01_v5.docx
```

Expect 319/319, 9/9, 519 tests, 73/73 pagination, and **no output at all** from the last command.
If v5 has changed, stop and restore it with `git checkout HEAD -- <path>` before anything else.

## Risks / Things To Distrust

1. **The primary sources are not in the repo and never will be.** `Examples of Past FYP Papers/`
   is gitignored, so a fresh clone cannot re-read the papers behind Table 6.2 or the corpus study.
   Two of the five sources ([37], [39]) exist only as local PDFs; the other three are re-readable
   from arXiv. `PROVENANCE.md` records the locating evidence precisely so a verifier can find each
   figure again, but it is not a substitute for the source. Do not re-derive these numbers from the
   provenance file — go back to the paper.
2. **`v5` carries the same two citation defects** at its own numbering, `[14]` and `[15]`. It was
   left frozen deliberately, on the student's standing instruction. It is flagged, not fixed.
   Decide explicitly rather than discovering it later.
3. **A green harness has repeatedly been a poor predictor of semantic correctness here.** This
   session is another data point: every gate was green while §6.2 still had no published figure in
   it, and two of the first-draft numbers were wrong before independent verification caught them.
4. **Pagination is checked against LibreOffice, not Word.** The numbers were originally derived
   from a LibreOffice render, so the comparison is self-consistent, but Word may paginate
   differently. The freeze step still needs a human in Word.
5. **Do not batch verification.** D57 exists because batched judgement measurably flips findings.
   Reading may be batched; judging one claim against its source may not.

## Next Actions

1. **T47 (a)** — the three marginal style residuals are all that remains of T47. Everything
   substantive is closed.
2. **T50 freeze checklist** — open the chosen docx in Word, `Ctrl+A` then `F9`, confirm the Table
   of Contents populates, export the PDF. The TOC is a real Word field and is blank in every
   non-Word renderer.
3. **Student-only, in rough priority:** T1 (July email to Dr. Zhang, voice-locked and ready),
   T15 (submit), T34 (Overleaf upload), T3 (`MyTCinfo`), and deleting the OpenAI key now that T35
   is complete.
4. **Open research, not blocking:** T30b, a second annotator for both human gold sets.

## Do Not Touch / Do Not Revert

- `scripts/build_fyp_report_v5.js` and `docs/FYP_Report_2026-07-01_v5.docx` — frozen for
  side-by-side comparison at the student's explicit request.
- `results/` and `results_512/` `raw.jsonl` / `summary.json` — immutable TC1 originals.
- The `epistemic_hedge` patterns in `configs/artifact_policy.yaml`. Clear a hit by rewording or by
  an `allow` entry that quotes the exempted span — never by weakening a pattern.
- The front-matter page numbers in `build_rewritten_fyp_report_v1.py` are now correct against the
  render. If you change body length, re-run `check_pagination.py` and fix them rather than
  guessing.

## Privacy / Artifact Guardrails

No raw HarmBench prompt or response text in any document, log, or message. Use IDs, counts, labels
and the redacted `scores.judge.*` sidecars. Raw outputs and v2 sidecars are never modified.
