# Rewritten Interim v4 — claim-to-source entailment audit

Post-remediation audit target: `docs/rewritten_fyp_interim_v4.docx`, SHA-256
`e25cf66f9988b0f1f25c478af74eacdf657089c2aa69729b1514e7393d6cdf08`.

## Outcome

The current v4 is **source-entailment clean within the disclosed coverage
boundary**. The regenerated rendered document contains:

- 61 references;
- 93 citation-bearing units: 80 prose sentences and 13 table rows; and
- 142 expanded reference occurrences.

Every unit has a human verdict and evidence locator in `ledger.csv`. The
post-remediation ledger contains **0 P1 findings and 0 P2 findings**. Eight P3
rows remain labelled as qualified synthesis, inference, external-validity
extrapolation, or source-coverage nuance; none is a contradiction or unsupported
attribution.

## Repairs applied

1. **Corrected-comparisons literature gap.** Chapter 2 §2.7 now states that
   Kharinaev et al. declare 31 paired comparisons and apply a multiplicity
   correction [10], while [8] and [9] emphasise aggregate retention across many
   tasks. The false claim that none of [8]–[10] used corrected comparisons is
   gone.
2. **Scoped the introduction's prevalence statement.** The sentence now names
   quantization evaluations [8], [9] next to the perplexity/benchmark-accuracy
   statement. The following sentence describes behavioural and trustworthiness
   effects, which all three cited studies examine.
3. **Added calibration-method citations.** The calibrated-method side now cites
   [5], [6], [22], and the load-time NF4/LLM.int8 side cites [24], [25].
4. **Replaced an unsupported “typically” generalisation.** Chapter 6 now
   enumerates ordinary quantization [10], [29], an active decoding attack [30],
   and a deliberately prepared checkpoint [33].
5. **Narrowed human-evidence terminology.** “Human ground truth” was replaced
   with “human annotations” or “the human annotator” on all three v4 surfaces.
6. **Updated No Free Labels.** Reference [59] now records acceptance in the
   Conference on Language Modeling (COLM), 2026, while retaining
   arXiv:2503.05061. Acceptance was checked against the official COLM 2026
   accepted-papers list.
7. **Bound the Kharinaev comparison to its exact source location.** Table 6.2
   and the adjacent explanation now identify the IEEE version of record,
   Table 14.

## Corrections to the first audit record

The initial audit contained two errors that were caught before report
remediation and are corrected here rather than silently erased:

- **Kharinaev was not a version mismatch.** Visual inspection of physical page
  19 shows the exact 0.571, 0.536, 4.500, 3.561, −0.187 and 8.622 values in the
  IEEE version of record's Table 14. Text extraction had omitted the table and
  led to the false conclusion that the values existed only in arXiv v2 Table 9.
  The report's values and polarity were already correct; only an explicit
  Table-14 locator was added.
- **Reference [9] does include trustworthiness/ethical evaluation.** Its
  trustworthiness section and Ethics Benchmark analyse quantized-model
  behaviour, including changes in whether models answer ethical questions. The
  original “reference [9] has no safety construct” finding was too categorical.
  The report now uses the more exact “behavioural and trustworthiness effects.”

These corrections follow the repository rule that the evidence trail is itself
a claim and must be re-derived from the primary artifact.

## Source coverage and boundary

- **52 of 61** references have identity-matched local PDFs.
- Correct local full texts remain absent for nine classic statistical sources:
  [47]–[51], [54], [56], [60] and [61]. Their titles, authors and procedure
  identities were checked against authoritative DOI or publisher records. Their
  report uses are graded as method-identity citations; this audit does not claim
  page-level full-text review for those nine.
- `/Users/tanueihorng/Downloads/Introduction_to_Bootstrap.pdf` is not Efron and
  Tibshirani's book. It is a different 2011 article by Stephanie Fook Chong and
  remains explicitly rejected in `source_map.csv`.
- The reverse-direction pass is persisted in `reverse_direction.csv`; all three
  originally actionable rows are now marked resolved.

## Reproducible files

- `audit_manifest.json` — target hash and independent counts
- `inventory.csv` / `inventory.json` — every citation-bearing unit extracted
  from the rendered DOCX
- `source_map.csv` — bibliography-to-local-source mapping and missing coverage
- `ledger.csv` / `ledger_summary.json` — every entailment verdict and evidence
  trail
- `reverse_direction.csv` — reverse-direction findings and their resolutions

The audit does not replace the project's artifact-derived result locks, and it
does not claim that all 61 primary full texts were locally available. Within
that stated boundary, the current v4 has no unresolved P1 or P2
source-entailment finding.
