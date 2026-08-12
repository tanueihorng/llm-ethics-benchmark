# T51 — enumeration audit of the submission candidate

**Date:** 2026-08-13 · **Target:** `docs/rewritten_fyp_report_v1.docx` (the T52/D59 submission
candidate) · **Method:** enumeration, not pattern matching.

T51 exists because three successive audit rounds each found paraphrases of one defect class
after the previous round had closed it: remediation was string-based while the defect is
semantic. On 2026-08-13 the same shape appeared three more times in one session (a scan that
enumerated only builders, a pattern set that carried one hedge word, a priority list that
named two closed tasks). This audit therefore enumerates rather than greps.

## Enumeration basis — declared, so the blind spots are visible

**Unit:** every sentence in the **rendered** `.docx`, not the builder (D56: the builder is not
the artifact). Body paragraphs, table cells and captions. Nothing sampled.

**Extraction:** `word/document.xml` → paragraphs in document order, heading context tracked,
table membership tracked, sentence split guarded against decimals, abbreviations and bracket
citations. **886 units → 1 450 sentences (460 in table cells).**

> One extractor bug was found and fixed mid-audit, and it is recorded because it would have
> invalidated the result: `<w:t[^>]*>` also matches `<w:tab>`, so table-of-contents field
> markup was being captured as body text. The corrected pattern is `<w:t(?:\s[^>]*)?>`. This is
> the same defect shape the audit was convened to find — a pattern loose enough to match the
> wrong thing — occurring in the audit's own instrument.

**Partition:**

| Class | Count | Treatment |
|---|---|---|
| A — contains a numeral | 568 | every numeric token checked against artifact-derived values |
| B — asserts without a numeral | 182 | read individually; no lock or number check can reach these |
| — bibliography entries | 99 | out of scope here (claim-to-source axis, audited 2026-07-26 / 2026-08-02) |
| C — structural, definitional, front matter | 601 | out of scope, counted |
| **In scope** | **750** | |

**Declared blind spots.** (1) Whether a cited *paper* supports the sentence citing it is the
D56 claim-to-source axis, audited separately — this audit checks claims against **this study's
artifacts**. (2) Prior-work figures in §2 and §6.2 come from other papers by design and were
verified in `docs/audits/2026-08-02_prior_work_numbers/`. (3) Sentence splitting is heuristic;
a mis-split changes unit boundaries, not coverage.

## Mechanical layer — result

**765 numeric tokens across 568 numeric sentences. Every number that should come from this
study traces to a committed artifact value.** The 44 unmatched tokens are all explicable and
none is a claim about results: table-of-contents page numbers and section numbers, the academic
year, dependency version floors in Appendix A, and prior-work figures from cited papers.

Artifact basis: 28 files under `results_512/analysis/` and `generated/`, 1 148 distinct numeric
values, expanded to accepted surface forms (percentage, percentage-point and 0–3 dp roundings).

**This is necessary but not sufficient, and the distinction is the point of T51:** matching a
*value* does not prove correct *attribution*. "κ 0.59" matching some artifact number does not
show it is bound to the right scorer. The 93 claim locks assert those bindings for load-bearing
claims; class B is where neither mechanism reaches.

## Findings

### F1 (P3, FIXED) — a selective ranking summary in §5.2

**Was:** "The API judge and classifier agree most strongly on the Llama pair, where kappa
exceeds 0.94 for both aliases. *Agreement is lower on Phi and Qwen*, but the per-pair
harmful-ward significance conclusion is unchanged."

**Measured** (`judge_pairwise_agreement.json`, per-alias κ, ascending): phi4_mini_base 0.678,
qwen_4b_4bit 0.744, qwen_4b_base 0.768, **mistral_7b_4bit 0.777**, qwen_2b_base 0.785,
qwen_2b_4bit 0.800, phi4_mini_4bit 0.826, **mistral_7b_base 0.827**, llama_base 0.942,
llama_4bit 0.954.

**Defect.** Not false — everything non-Llama is lower than Llama — but naming Phi and Qwen
implies Mistral is not in that group, and `mistral_7b_4bit` (0.777) sits below three of the four
Qwen aliases while `mistral_7b_base` (0.827) is the highest non-Llama alias. `phi4_mini_4bit`
(0.826) is likewise near the top of the band, so "lower on Phi" holds for one Phi alias and not
the other. **Aggravating: the per-alias κ values appear in no table.** Table 5.1 carries per-pair
deltas under three judges, not per-alias agreement, so the reader has only the range "0.68 to
0.95" and this sentence — there is nothing to check the summary against.

**Fix** (adds rigour rather than deleting, per D56): the sentence now states the actual
structure — one lower band from 0.68 to 0.83, minimum on Phi-4-mini at fp16 with the Qwen3-4B
pair next, and no other family systematically separated within that band.

## Claims verified against artifacts and found correct

Spot-verification was not sampled from the easy cases; it targeted every checkable ranking,
universal and count claim outside tables (81 sentences), plus the highest-risk class-B
assertions.

| Claim | Artifact check | Verdict |
|---|---|---|
| "nine prompts move from harmful to safe and one moves from safe to harmful" (Llama) | `b=1, c=9, δ=−0.040, p=0.0215` | correct |
| "No pair shows a statistically significant increase in over-refusal, and four pairs are non-significant" | 1 significant (Phi, a *decrease*), 4 non-significant | correct, and precisely worded |
| "Exactly three contrasts survive BH-FDR, none a HarmBench contrast" | qwen_2b MMLU, llama ARC, phi over-refusal | correct |
| "The API judge and Llama Guard do not find a significant harmful-ward NF4 pair" | API judge *does* find llama significant at −0.035 — a **decrease** | correct **because** "harmful-ward" is present; a looser sentence here would be false |
| "Llama Guard's per-pair directions agree for four pairs; Qwen3-1.7B is the exception … decreases by 0.050" | 4 same-sign; qwen_2b clf 0.000 vs LG −0.050 | correct |
| "The lowest agreement occurs on Phi" (Llama Guard) | 0.362 / 0.380, both lowest | correct |
| "Qwen3-4B is the largest harmful-ward point estimate" | +0.040, max of {0.000, +0.040, −0.040, −0.020, +0.020} | correct |
| "Llama has the only upward over-refusal estimate" | +0.016; Mistral is exactly 0.000, not upward | correct |
| "Mistral-7B ARC is the only positive point estimate, at +0.0094" | only positive of five | correct |
| "every INT8 HarmBench contrast at 512 tokens is non-significant under both judges" | 10/10 `uncorrected_significant: false` | correct |
| "sixteen prompts flip toward harmful compliance and sixteen toward safety, McNemar p = 1.000" | `b=16, c=16, p=1.0` | correct |
| "regex recalls only 3.2% of full refusals; the judge recalls 96.8%" | 2/63 and 61/63 | correct |
| "Every model revision in the primary configuration is pinned" | 10/10 in `configs/tc1_512.yaml` | correct |
| "Across all five pairs, the arm does not produce a stable significant harmful-ward effect" | qwen_4b has 1/5 significant seeds and all five positive — **carried by the word "stable"** | correct as written; see note |

**Note on the multi-seed sentence.** It is accurate, but it is the sentence in this document
that depends most on a single adjective. `qwen_4b` has one individually significant seed
(+0.050, p = 0.0129) and all five seed deltas positive (mean +0.029). "Stable" is doing the work
that makes the sentence true. It was left as written because it *is* true and the underlying
per-seed evidence is committed in `sensitivity_multiseed.json`, but a reader who drops the
adjective gets a false statement — the same trap the T1 email hit on 2026-07-25, where "no seed
anywhere shows a significant increase" had to be scoped to qwen_2b. **Do not paraphrase this
sentence without re-reading the artifact.**

## Verdict

**One P3 defect in 750 enumerated claims, fixed. No P1, no P2. No result, number, significance
verdict, label or conclusion changed.**

The document's numeric layer is fully traceable and its prose is consistently hedged at the
resolution the design supports — several sentences are correct *only* because of a qualifier
("harmful-ward", "stable", "four pairs"), which is a sign of careful writing rather than luck,
but also the reason paraphrasing this document is dangerous.

## Standing method (T51's actual deliverable)

Enumeration replaces pattern-matching for this defect class. Concretely: **enumerate the units
from the artifact, declare the basis and the blind spots, partition so that every unit is either
checked or explicitly out of scope, and count what you did not check.** A grep can only return
what someone already thought to look for; that failure mode has now recurred often enough in
this repo — D42, D49, T56, and three separate instances on 2026-08-13 — to be treated as the
default expectation rather than a surprise.
