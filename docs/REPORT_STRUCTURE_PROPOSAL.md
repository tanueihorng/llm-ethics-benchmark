# Report structure — measurement and restructure proposal

**Status:** PROPOSAL. Nothing in this document has been applied to any report builder.
No prose, number, label or claim lock has been changed. Each item below needs an
explicit go/no-go from the author before it is executed.

**Date:** 2026-07-25 · **Scope:** `docs/FYP_Report_2026-07-01_v5.docx` and its five
mirrors · **Trigger:** external reader feedback that the document does not read as an
academic research report, plus the measurement in §1.

Related: [PROJECT_LOG.md](PROJECT_LOG.md) (source of truth), [THESIS_OUTLINE.md](THESIS_OUTLINE.md)
(the project's own chapter blueprint, which this proposal measures against).

---

## 1. Measurement

Word counts extracted from the quoted strings in `scripts/build_fyp_report_v5.js`
(so they include captions and table cell text; treat as ±10% approximate). Figure and
table counts are exact. Figure counts are stated **after** the 2026-07-25 diagram
addition — before it, Chapters 1–5 contained no figure at all.

| Chapter | Words | % of body | Figures | Tables |
|---|---:|---:|---:|---:|
| 1 Introduction | 1,344 | 3.9% | 0 | 0 |
| 2 Literature Review | 1,271 | 3.7% | 0 | 0 |
| 3 Methodology | 3,235 | 9.4% | 1 | 4 |
| 4 System Design and Implementation | 1,875 | 5.4% | 1 | 2 |
| 5 Experimental Setup | 1,920 | 5.5% | 1 | 2 |
| **6 Results and Analysis** | **18,836** | **54.4%** | 6 | 5 |
| 7 Discussion and Threats to Validity | 2,218 | 6.4% | 0 | 0 |
| 8 Limitations | 1,390 | 4.0% | 0 | 0 |
| 9 Future Work | 1,423 | 4.1% | 0 | 0 |
| 10 Conclusion | 1,086 | 3.1% | 0 | 0 |
| **Body total** | **34,598** | 100% | 9 | 13 |
| *Appendix G — Document Revision History* | *3,603* | — | 0 | 1 |
| *Appendix H — Scoring-Methodology Correction Record* | *990* | — | 0 | 1 |
| *Appendices A–F combined* | *981* | — | 0 | 3 |

Reproduce with:

```bash
python3 -c "import re,sys; src=open('scripts/build_fyp_report_v5.js',encoding='utf-8').read(); lines=src.split(chr(10)); b=[(i,re.search(r'H1\(\"([^\"]+)\"',l).group(1)) for i,l in enumerate(lines) if re.match(r'\s*H1\(\"',l)]; b.append((len(lines),'EOF')); [print(f'{n[:44]:<46}{sum(len(t.split()) for t in re.findall(chr(34)+r'((?:[^'+chr(34)+r'\\\\]|\\\\.)*)'+chr(34), chr(10).join(lines[s:e]))):>7}') for (s,n),(e,_) in zip(b,b[1:])]"
```

### What the measurement says

1. **Results is 54% of the body.** The conventional share for an empirical CS
   dissertation is 25–35%. Chapter 6 is larger than Chapters 1–5 and 7–10 combined.
2. **Literature Review is 3.7%** — 1,271 words. It is smaller than Future Work, and
   about a third of the size of the revision-history appendix. The conventional share
   is 15–20%, and it is the chapter an examiner reads to decide whether the author
   knows the field.
3. **Process material inside the deliverable.** Appendix G (revision history) plus
   Appendix H (correction record) total ~4,600 words — more than the literature
   review, discussion and conclusion combined.
4. **Total length** is ~34,600 words of body plus ~5,600 of appendices. This has not
   been checked against the CCDS page/word limit; see P0.

The rigor dimensions are not the problem. Threats to validity (Ch7), multiplicity
correction, three independent judges, two human gold sets, pre-registration, pinned
revisions and 93 machine-checked claim locks are all present and are stronger than
typical FYP practice. The issue is that a reader meets none of that until page ~40.

---

## 2. Proposals

Ordered by ratio of examiner-visible improvement to risk. **P0 is a prerequisite for
P1 and P2** — there is no point expanding one chapter if the document is already over
the limit.

### P0 — Check the length limit first  *(no writing; 10 minutes)*

Confirm the CCDS FYP page/word limit and whether appendices count toward it. Every
proposal below is either "add words" (P1) or "move words" (P2, P3); which of those is
allowed depends entirely on the answer. **Blocking for P1.**

### P1 — Expand Chapter 2 from ~1,300 to ~4,000 words  *(highest value)*

Restructure into three threads, each closing with an explicit "…but none of these
does X" gap statement, then a fourth short section that states the gap this study
fills:

1. Quantization and its behavioural effects (what is known, what is contested).
2. Safety benchmarks and red-teaming — HarmBench, XSTest, the threat models they
   encode and the ones they do not.
3. LLM-as-judge evaluation and its validity — chance-corrected agreement, human
   grounding, multi-judge designs.

Most of this material already exists in the report, scattered through §6.12, Chapter 7
and the reference list (32 references, all bracket-cited). This is substantially a
*relocation and framing* exercise rather than new research.

- **Risk:** low. Additive; touches no claim-locked sentence.
- **Effort:** the largest single writing task here.
- **Watch:** `tests/test_citation_completeness.py` requires any newly named method to
  land with its bibliography entry in the same edit (D54), and `ieee_citations`
  enforces first-use bracket order — renumbering is automatic but must be re-verified.

### P2 — Split Chapter 6  *(highest structural impact)*

Two options; **(a) is recommended.**

**(a) Split into two chapters.** Chapter 6 becomes *Results* (the main fp16-vs-NF4
study, per-pair findings, multiplicity correction — roughly 7,000 words) and a new
Chapter 7 becomes *Scorer Validation and Sensitivity* (judge agreement, human gold
sets, multi-seed, generation-length, precision sweep — roughly 11,000 words). This
reads as two distinct contributions, which is what they are, and it makes the
validation work visible in the table of contents instead of buried at §6.12.

**(b) Keep one chapter, demote the tail.** Add a one-page "Results at a glance"
opening and move §6.14–§6.16 detail into appendices.

- **Risk:** medium. Chapter renumbering cascades through cross-references and both
  LaTeX mirrors; `tests/test_tex_sync.py` will fire on any docx/tex divergence.
  Mitigation: move whole blocks, never rewrite sentences in the same edit, and run
  `make verify-claims` between the move and any prose polish.
- **Effort:** mechanical but wide — 6 docx builders plus 2 `.tex` mirrors.

### P3 — Move the revision history out of the report  *(cheapest win)*

Delete Appendix G (3,603 words) from the report builders. It duplicates PROJECT_LOG §4,
which is the actual source of truth, and a changelog inside an academic deliverable is
the single most likely reason a reader concludes the document is a work log.

**Keep Appendix H** (Scoring-Methodology Correction Record). It is a genuine
research-integrity artifact and it is defensible at viva — but trim it to about one
page and point to the repo for the full trail.

- **Risk:** low. No claim lock references Appendix G; the only appendix a lock touches
  is Appendix A (it asserts the embedded YAML reproduces `configs/tc1_512.yaml`).
- **Effort:** small — delete one array from six builders and both mirrors.

### P4 — Optional polish

- A "Results at a glance" table on the first page of Chapter 6: five pairs × four
  benchmarks, delta + significance, one line each. Useful even if P2(a) is taken.
- Reconcile the **model-size inconsistency found while building the figures**: the
  report's Table 3.1 lists Llama-3.2-3B as "3 B" while the thesis Table 3.1 lists it
  as "3.2". Both are defensible roundings of ~3.21 B, but they should not disagree
  across two documents describing the same study. This is why the new study-design
  figure carries no parameter counts.

---

## 3. What is NOT proposed

- **No change to any result, number, label, significance verdict or interpretation.**
  The findings are sound and multiply validated; this is entirely a document-structure
  proposal.
- **No change to the pre-registered analysis or the BH-FDR family composition.**
- **No removal of Appendix H.** Disclosing a scoring-methodology correction is a
  strength, not an embarrassment, and removing it after the fact would be worse than
  never having written it.

---

## 4. Execution order if approved

1. P0 (length limit) — answer first.
2. P3 (delete Appendix G) — smallest diff, verifies the six-builder + two-mirror loop.
3. P1 (expand Chapter 2) — the writing task.
4. P2 (split Chapter 6) — the wide mechanical change, done last so it moves settled text.
5. Regenerate all six docx, recompile both PDFs, refresh the Overleaf zips, then
   `make verify-claims` and `make agent-check`.
