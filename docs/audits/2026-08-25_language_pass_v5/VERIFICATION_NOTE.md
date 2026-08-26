# Verification Note — v5 Language Pass (2026-08-25)

All checks below were **run this session** against the built artifacts
(`docs/rewritten_fyp_report_v5.docx`, `docs/rewritten_fyp_interim_v5.docx`).
Commands are listed so any reviewer can re-derive every line.

## Numerical values

- Build-time gate: the v5 builder asserts that the numerical-token multiset of
  the built document is identical to the v4 source (no token lost, none
  invented; tabbed front-matter page-number lines excluded as layout metadata,
  verified separately by the pagination checker).
- Independent rendered-document check: `tests/test_rewritten_report_v5.py::
  test_numbers_citations_questions_headings_preserved` compares v4 vs v5 DOCX
  directly. **Passing for both the report and interim pair.**
- Spot values confirmed unchanged in the rendered text: −0.090, −0.0316
  (shown as −0.032 with unrounded −0.0316), −0.048, q = 0.008, q = 0.0122,
  p = 0.0963, κ = 0.59, κ = 0.11, n = 200/250/300/1,172, 512-token budget,
  128-token comparison, MDE table (Appendix B) untouched.

## Citations

- Bracket-marker sequence v4→v5: **byte-identical** (build assertion + test).
  No reference was added, removed, renumbered, or moved to a different claim.
- `tests/test_ieee_citation_order.py` (first-use IEEE order over body AND table
  cells): **19/19 passing**, including both v5 documents.
- `tests/test_citation_completeness.py` (usage→presence: every named method in
  a builder carries its bibliography entry): **22/22 passing**, including the
  new v5 builder.
- Printed bibliography: 61 entries, all inherited from the v1 reference block
  (asserted at build time by normalised-prefix membership); the three entries
  dropped as uncited were already dropped by the v4 renumbering and are
  unchanged from v4.

## Research questions, headings, structure

- RQ1–RQ5 wording: **verbatim equal** to v4 (test-asserted).
- All Heading-styled paragraphs: **identical list** v4→v5 (test-asserted);
  section numbering unchanged.
- Paragraph count: unchanged (test-asserted).

## Tables and figures

- Every table cell equals its v4 counterpart except two PROSE cells declared in
  `KNOWN_TABLE_CELL_CHANGES` (Table 6.2 "Present study" row; Table 5.2
  generation-truncation row) — test-asserted. No value, axis, interval, legend,
  or model label changed; no figure image was regenerated.

## Conclusions and claim strength

- `make verify-claims`: **93/93 artifact-derived result checks pass**.
- Surface verification (`make verify-claims` second phase): **606/607 pass**.
  The single failure is `surfaces:coverage` for `docs/statistics_explainer.html`
  — the pre-existing, separately-tracked T69 gap (file created 2026-08-25
  01:14, before this pass; owner decides its lifecycle). It is unrelated to the
  v5 line and predates this session's edits.
- Contradiction guards extended for the new vocabulary are proven to bite:
  `tests/test_claim_registry.py::test_cleared_phrasing_bites_on_wrong_count`,
  `…_bites_on_hedged_count`, `…_cleared_unnegated_asr_postcorrection_fails`.
- Non-significant results remain non-equivalence claims everywhere: the pass
  rewrote them as "no statistically significant difference was detected" /
  "inconclusive" formulations; `tests/test_claim_fitness.py` (43 checks) and
  the `epistemic_hedge` scan over the new builder both pass.
- Locked scope sentences (`tests/test_report_scope_invariants.py`): **34/34
  passing**; two candidate edits that would have touched locked wording were
  rejected during review (see LEFT_TECHNICAL.md items 2–3).

## Pagination and rendering

- LibreOffice render of the final report: **60 physical / 51 printed pages**;
  all **73/73** front-matter contents/list entries match the render
  (`scripts/report_tools/check_pagination.py`, 0 mismatches). The v5 builder
  carries the 33 page-number overrides that the lengthened prose requires;
  Microsoft Word remains the submission pagination authority (D62) and will
  re-field the TOC on open.
- `tests/test_report_pagination.py`: **7/7 passing**.
- Visual spot-check of rendered pages (Abstract p. iii, §4.7 p. 28, References)
  confirmed clean typography and the revised wording on the page.

## Full test suite

- `python3 -m pytest tests/`: **763 passed, 0 failed**, with exactly one
  deselection: `tests/test_claim_registry.py::
  test_all_registered_surfaces_satisfy_their_profiles`, which fails only on the
  same pre-existing T69 unregistered-file gap (verified: its failure detail
  names `docs/statistics_explainer.html` and nothing else).

## Not covered by this pass (stated scope)

- The v4 artifacts themselves are untouched (non-destructive line), so v4, v3,
  v1, compact, and progress documents keep their original wording.
- The LaTeX mirrors under `fyp_submission/report_latex/rewritten_*` still
  record the v4 line; pointing them at v5 (or at whichever line becomes the
  final report for T15) is left as an explicit follow-up decision, recorded in
  PROJECT_LOG §2.
- Figure-internal text (e.g. Figure 4.2's in-image title) was not altered.
