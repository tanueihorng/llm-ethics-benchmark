# Content-only correction — 1 October 2026

The student clarified that the staleness request authorized content updates while preserving the existing design. The initial sweep added visible overlays and Word context sections; those additions exceeded that scope and have been removed. D102 records the content-only constraint.

Eighteen added HTML overlays and their CSS were removed. The existing snapshot banners, text blocks and status cards carry the relevant content updates. The additional architecture-guide callout and two Word context pages were removed. The three Word companions were rebuilt from their sources with updated text in existing paragraphs and footers. The meetup builder copies the frozen June package and changes only three named text anchors; it adds no paragraphs, sections, tables, styles or drawings. Both PowerPoint decks already retained their original design and required no further edit in this correction.

## Verification

- [52/52 checks passed](verification.json). Original design fingerprints were derived independently from the backups captured before the initial sweep: **19 HTML files, three Word documents and two PowerPoint decks**. All 24 match their original design definitions.
- HTML stylesheets, element structure and layout attributes match. The new overlay elements and their CSS are absent. Content-specific title/ARIA-label/metadata text is outside the formatting comparison; existing inline styles, classes and other attributes remain covered.
- Every Word and PowerPoint part under `word/` or `ppt/` is inventoried and compared: formatting/shape XML, section/page-break structure, relationships and embedded media. Only text values are excluded. The three editable chart workbooks match in all parts after normalizing their identified creation/modification timestamps; chart data and styles are retained.
- The scientific/staleness verifier was updated to check the restored designs and inline content instead of the removed panels. Its [35 checks pass](../2026-10-01_nonreport_staleness/verification.json), including fresh primary-family/equivalence checks, presence/omission checks and source preservation. The preceding version and its output are preserved here as historical `before_content_only_*` files.
- All **48 current Word pages** rendered. [The render comparison](render_review.json) found 40 page bodies identical to the previously reviewed renders, with automatic footer page fields outside the pixel comparison. All eight changed bodies were inspected at full size. Word formatting and field structure match independently across the full packages. Native Word/PowerPoint and mobile behavior were not retested.
- [Repository guards](guard-check.log) and [claim checks](verify-claims.log) were rerun after the correction. The earlier 853-test run is recorded in the sweep record; those tests were not repeated for this presentation-content correction.
- [The final protected manifest](protected_after.json) matches all **221 files** in `submission/` and `fyp_submission/` by SHA-256, size and nanosecond modification time, including additions/deletions. The 107 excluded research DOCX/PDF/TeX artifacts also retain their original hashes and timestamps. No research report builder, raw artifact or scorer sidecar was edited.
- The metadata-only [report-version refresh](report-versions.log) reports 45 documents, 26 change records and zero problems. No `make report`, experiment, commit or push ran. Existing unrelated edits were preserved.

Run from the repository root:

```sh
python docs/audits/2026-10-01_content_only_restore/verify.py
python docs/audits/2026-10-01_nonreport_staleness/verify.py
python scripts/agent_check.py --skip-pytest
make verify-claims
```

This record verifies the requested content/design boundary and artifact preservation; it does not certify report readiness or every scientific claim in historical snapshots.
