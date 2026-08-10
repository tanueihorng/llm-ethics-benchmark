# Report Remediation Plan

**Why `docs/FYP_Report_2026-07-01_v5.docx` reads as machine-generated, and what to change.**

Companion to `docs/FYP_WRITING_STYLE_GUIDE.md` (the 15-paper corpus study this is measured against).
All edits go into `scripts/build_fyp_report_v5.js`, then `make report`. Never hand-edit the docx.

> **Relationship to `docs/REPORT_STRUCTURE_PROPOSAL.md` (2026-07-25).** That document measured the
> chapter imbalance from the builder source and proposed a restructure. This one is independent and
> additive: it measures the same imbalance from the built docx (Chapter 6 at 19,253 words / 55% of
> body against the proposal's 18,836 / 54.4% — the gap is measurement basis, builder strings versus
> docx paragraphs, not a disagreement), and adds four dimensions the proposal does not cover: the
> **voice** diagnosis, **citation density**, the **maintenance-register** defect, and a set of
> **substantive content gaps**. Where both cover structure, they agree. Treat the proposal as the
> restructure spec and this as the evidence and the style layer.
>
> **Status: PROPOSAL.** Nothing here has been applied to any builder.

---

## The diagnosis

The report does **not** read as AI-generated because of what it claims. The science in it is stronger than any of the fifteen corpus papers, none of which reports a seed, an interval, or a significance test.

It reads that way for two reasons: **its sentences are joined by punctuation instead of words**, and **the document narrates its own maintenance**.

Measured against the corpus, per 1,000 words of body prose:

| Marker | Corpus median | Corpus max | This report |
|---|---|---|---|
| Semicolons | 0.36 | 3.79 | **20.34** (858 total) |
| Em dashes | 0.0 — 14/15 papers use none | 4.79 | **5.06** (202 total) |
| Parentheticals | 1.97 | 13.84 | **21.90** (929 total) |
| In-text citations | **7.1** (min 3.3) | 19.1 | **1.2** |

Structural measurements:

- Body (Chapters 1–10) = **34,769 words**: 3.1× the corpus median and 2.1× the corpus maximum on a like-for-like prose basis.
- Chapter 6 = **19,253 words**, 55% of the body. Longer than any complete report in the corpus.
- Chapter 2 Literature Review = **1,273 words** (3.7% of body). Chapter 1 has **zero citations**.
- Appendix G "Document Revision History" = **3,589 words**, a 110-row changelog carrying internal task IDs (T41, T36, D51), `docs/PROJECT_LOG.md` paths and UTC+8 timestamps. **2.8× the entire Literature Review.**
- Abstract = **~615 words**, against a corpus range of 68–435. Above the corpus maximum.
- No Acknowledgements (14/15 corpus papers have one). One merged "List of Figures and Tables" (14/15 have two).

**Sentence length is not the defect.** Mean 29.2 words is inside the corpus range of 17.8–35.0; P07, P09 and P06 all write longer. Do not "write shorter sentences."

---

## Root cause

**The prose is doing the table's job.** There are 178 sentences that are results tables transcribed into prose, with CI and p-value payloads welded on by semicolons and parentheses. Verbatim from §6.5:

> "Under the primary (judge) scorer at the 512-token reference budget, the only HarmBench ΔASR whose CI excludes zero is Llama-3.2-3B's, and it is a decrease (−0.040, CI [−0.075, −0.010]; McNemar p = 0.021); every other pair is within noise: Qwen 1.7B (0.000, CI [−0.055, +0.055]; McNemar p = 1.000), Qwen 4B (+0.040, CI [0.000, +0.080]; McNemar p = 0.096)…"

One habit produces all three symptoms at once: the semicolon density, the 19,253-word results chapter, and the machine texture. Fix the cause and the symptoms resolve together — the numbers move into tables that already exist, and the prose shrinks to claims.

**The corpus equivalent:** P02 writes *"Larger models do not uniformly improve F1."* then tables the numbers. P07 writes *"FID and sFID that respectively outperform the full-precision model by 1.75 and 1.93 points."*

Rewritten in corpus voice:

> "Under the judge at the 512-token reference budget, only one HarmBench ΔASR has a confidence interval that excludes zero. That pair is Llama-3.2-3B, and its delta is a decrease of 0.040 (McNemar p = 0.021). The remaining four pairs sit within noise: Qwen3-1.7B at 0.000, Qwen3-4B at +0.040, Mistral-7B at −0.020, and Phi-4-mini at +0.020. Table 6.1 gives the 95% confidence interval and the exact McNemar p-value for every contrast."

---

## Ranked defects

### 1. Clause-welding (P0)
858 semicolons, 202 em dashes, 929 parentheticals, 135 corrective ", not Y" appositives (corpus max 4). Against that: "however" appears 10 times in the whole report, and only 3 of 1,705 sentences open with a connective. P03 runs 22.6% connective-initial.

**Fix.** Every clause-joining semicolon becomes a full stop plus a named connective. Em dashes are deleted except at most five definitional glosses on first mention. Every ", not Y" becomes its own sentence or is recast with "rather than".

**Acceptance targets** (re-measure after `make report`): semicolons ≤1.0/1k · em dashes ≤5 total · parentheticals ≤6/1k · sentence-initial connectives ≥5/1k · "However" ≥0.7/1k · ", not Y" ≤10 total · paragraph p90 ≤130 words (currently 249).

### 2. The document narrates its own maintenance (P0)
Chapters 1–10 carry 16 internal decision IDs, 63 repository file paths, 16 SLURM job numbers and 21 ISO dates. Status vocabulary absent from all 15 corpus papers: "folded in" ×12, "superseded" ×13, "retired" ×7. Chapter 9 *Future Work* lists items marked "(completed 2026-06-15)". Chapter 10 opens with job numbers 60976–60981. Appendix G is a changelog.

A "Future Work" chapter that reports completed work is a status board. It tells a supervisor the document was assembled incrementally by a tool that appends rather than revises.

**Fix — re-register, do not strip.** The identifiers encode material the *strong* papers do disclose, in the right register (P10 states hardware and wall-clock inline; P11's five-line version manifest is praised for making the study "look reproducible at a glance"). So:
1. First, move run dates, GPU model, wall-clock, walltime cap, library versions and revision pins into a one-page reproducibility manifest in Chapter 4.
2. *Then* delete D-numbers, T-IDs, repo paths and job numbers from the body.
3. Delete Appendix G (`build_fyp_report_v5.js` ~lines 1438–1484). If a process record is required, replace it with a ≤1-page numbered milestone table in a quarantined final chapter opening with its own "Chapter Rationale" (P07 §6.1).
4. Rewrite Chapter 9 to contain only genuinely open work; relocate completed items into Chapter 3 in plain past tense.

### 3. Chapter 6 is an accretion, not an architecture (P0)
16 top-level sections. RQ5 is answered in §6.3, §6.11 **and** §6.13. §6.9 "Research Question Synthesis" (207 words) sits **ninth of sixteen** with seven sections after it. §6.7 is a 135-word section with its own top-level heading. Three per-pair "Observations:" sections are interleaved with thematic sections. The value −0.048 appears 40 times, 0.000 appears 85 times.

**Fix — rebuild as numbered experiments, one decomposition axis, synthesis terminal.**

| From | To |
|---|---|
| 6.1 | §5.1 Results matrix + Appendix B |
| 6.2, 6.7 | §5.5 Experiment 4: scale within family (RQ4) |
| 6.3, 6.13 | §5.6 Experiment 5: cross-family (RQ5) — **the only place RQ5 is answered** |
| 6.4 | §5.4 Experiment 3: capability anchoring (RQ3) |
| 6.5 | Split: pre-specified procedure → §3.7; the parser-sensitivity **finding** stays in results |
| 6.6, 6.8, 6.10 | Bolded run-in paragraphs inside the relevant experiment; full tables → Appendix B |
| 6.9 | §5.11 Synthesis — **terminal, nothing after it** |
| 6.11 | **Deleted** (duplicate of §5.6) |
| 6.12 | §5.9 (~750 body words) + protocol → §3.5 + detail → Appendix C |
| 6.14 | §5.10 Mechanism probe |
| 6.15 | §5.7 Experiment 6: precision sweep |
| 6.16 | §5.8 Experiment 7: measurement robustness |

Rules: no section under 250 words, no research question answered twice, each headline number stated once in prose with the apparatus in a table.

> **§6.5 caveat.** Do not move §6.5 wholesale into Chapter 3. It carries a *finding* — the lenient fallback parser fires on 52.3% of the 4-bit member's ARC items versus 2.5% for its fp16 twin, and under a strict parser ARC falls 0.343 and MMLU 0.293. That materially qualifies RQ3 and belongs in the results.

### 4. The author is nowhere in the document (P1)
Two instances of "we" in 34,769 body words — one authorial, one **inside a direct quotation from the HarmBench paper**. No Acknowledgements. No sentence where the author says what they chose, what they could not do, or where they disagree with a published result.

> Do not run a find-and-replace on "we": it will damage the quotation in §6.16.

**Fix.** Add Acknowledgements. Declare the first-person convention once in §1.6 (P07's genre contract), then use it deliberately: three numbered contribution sentences in §1.5, plus two admissions in the author's own voice — the pre-registration blindness-exclusion deviation, and one design choice that was pragmatic rather than principled.

### 5. The literature disappears after Chapter 2 (P1)
45 in-text citations in 34,769 body words = 1.2/1,000, **below the corpus minimum** (P08, a weak paper, at 3.3). Chapters 1, 4, 5, 8 and 10 have zero. Chapter 6's 19,253 words carry 17. 22 of 32 references are cited exactly once.

**Fix.** Add citations to §1.1–§1.2. Expand §2.1–§2.4 with **structure, not padding** — a prior-work matrix (study × models × precisions × benchmarks × scorer × n × statistical treatment) on P03's pattern, a per-system number for every reviewed work, positioning sentences ending each section, and §2.5 as the terminal numbered-gap section. Cite a published number every time a finding is compared to prior work. Target ≥4 citations per 1,000 body words.

> **Do not "triple Chapter 2" by word count.** P03's 41% review is criticised as far too long; P09's ~53% is a weak-paper marker. Add the structural devices and let the length follow.

### 6. Statistical apparatus inline instead of tabled (P1)
"McNemar" appears 76 times in the body. §6.6.1 spends a 532-word paragraph enumerating per-seed deltas in prose.

**Fix.** Three-sentence finding template throughout: the number and what it is / the comparison / the mechanism or caveat. At most one statistical payload per sentence. Everything else moves into Tables 6.1 and 6.2 with a pointer line per experiment. **Do not delete a single interval, test, κ, MDE or multi-seed range — they move, they do not go.**

### 7. Nine orphan figures (P1)
The string "Figure N" appears only in the List of Figures and in the captions themselves. **Zero in-text cross-references.** Seven of nine captions end `Source: results_512/analysis/<file>.json`. Figures are flat-numbered 1–9 while tables are chapter-numbered (3.1, 6.2) — a combination attested in no corpus paper.

**Fix.** Emit a "Figure N shows …" sentence immediately before each figure (P11's rhythm). Renumber both classes chapter-scoped. Generate two separate lists from the caption strings rather than the hand-typed block. Strip the repo paths from captions; move exact filenames to a provenance table. Add a claim-lock so `make verify-claims` fails if any figure is referenced zero times.

> Do **not** add figures to fill empty chapters. Nine is at norm for strong papers (median 11), and P07's own conclusion carries none.

### 8. Hedging has become bookkeeping (P2)
"carried as" ×7, "most plausibly" ×10, "is best read as" ×11, "directional" ×57. The construction *"the contrast is carried as most plausibly a regex measurement artifact"* recurs near-verbatim in §3.5, §6.5, §6.9, §7.4 and Chapter 10. "robust"/"robustly" appears 96 times = 2.21/1k, above the corpus **maximum**.

**Fix.** Rewrite each site as a plain claim plus a plain reason in two sentences: *"The most likely explanation is that the regex mismeasures refusal on these prompts. A second annotator would test that directly."* Keep the hedging vocabulary and the evidence-status scheme — the problem is the hedge welded onto the claim by a dash, not the hedging. Restrict "robust" to the defined `robust_preservation` label or a stated statistical property.

### 9. Surface drift (P2)
artifact 36 / artefact 40 · behaviour 41 / behavior 9 · analys- 96 / analyz- 6 · "Qwen3-1.7B" 57 vs "Qwen 1.7B" 74 · "Llama-3.2-3B" 29 / "Llama 3.2 3B" 17 / "Llama 3B" 10 / "Llama-3B" 9.

Near-50/50 spelling splits are a fingerprint of block-wise generation. Free to fix, and the kind of thing a supervisor circles.

**Fix.** One global normalisation pass; add the mixed forms as stale-text patterns in `configs/artifact_policy.yaml`.

---

## Substantive gaps (not style — content an examiner will ask about)

1. **No cost axis anywhere.** §1.1 and §2.3 motivate the study on memory and latency, and §7.6 issues a deployment recommendation about memory budgets — but there is no measurement of VRAM, model footprint, load time or throughput for any of the ten aliases. The "trade-off" in the title is half-unmeasured. One small table (fp16 / INT8 / NF4 × resident GB, load seconds, tokens/s) beside the ASR/MMLU table fixes it. P07 puts GPU-hours and GFLOPs in the same table as FID.
2. **The independent variable's own library version is unpinned.** Table 5.2 reads `bitsandbytes | (installed; NF4 support verified)`. bitsandbytes *is* the treatment. Fill every cell, and name every library default overridden (`attn_implementation: eager` for the Phi pair, compute dtype, batch size, `max_new_tokens`).
3. **The research questions are stale and mis-partitioned.** RQ3 names only MMLU, but ARC-Challenge is a co-primary anchor and supplies an FDR survivor. RQ5 bundles cross-family *and* cross-precision into one question. The self-declared #1 contribution — the scorer-validity result — answers no stated RQ. **Re-cut the RQ list before rebuilding Chapter 6**, or the rebuild inherits the defect.
4. **The null is never defended as a null in one place.** All four discriminating instruments exist (MDE in §6.5.1, judge validation in §6.12, the 128→512 rerun in §6.16, multi-seed in §6.6.1) but are scattered. A half-page table — *candidate explanation | discriminating evidence | section | verdict* — covering genuine preservation, insufficient power, scorer insensitivity and truncation artefact would be the strongest half-page in the report, and has no corpus equivalent.
5. **Equations are unnumbered and un-referenceable**, and the constructs that matter most (Cohen's κ, BH-FDR, the bootstrap construction, `classify_pair_change`) have no formal definition. Also: κ values from −0.01 to 0.95 are printed across four judges with no interpretation scale.
6. **Prior work is never given a number, and no result is placed beside a published one.** §2.4 and §7.7 handle Hong et al. [16], Kharinaev et al. [12] and Egashira et al. [13] entirely qualitatively.
7. **Scope is declared but not defended, and one scope claim is falsified by the report itself.** §1.4 asserts "all inference uses deterministic greedy decoding with temperature 0.0" — contradicted by the multi-seed arm at temperature 0.7 in §6.6.1. Separately, GPTQ/AWQ/GGUF are excluded with no reason and no precedent cited.
8. **Private constructs used before definition.** "Outcome J" first appears in §3.5 and is only unpacked in §6.12. Same for "reference budget", "scorer of record", "evidence status", "claim lock". Pre-registration outcome codes should leave the body entirely.
9. **Deltas are never translated into deployment units.** Add one sentence per headline number: *"the 4-bit build gets wrong roughly one in eleven MMLU questions its fp16 twin answers correctly."*
10. **Unnumbered headings already present.** §6.9's five RQ heads and Appendix C's six subheads are Heading 3 with no section number; §5.4.1 carries a depth-3 number under a Heading 2 style.

---

## What must not change

These are the report's decisive advantages over every paper in the corpus. The restructure shrinks the container; it must not remove evidence.

- **The entire statistical apparatus.** BH-FDR over a declared 20-contrast family, paired bootstrap CIs, McNemar exact tests, an explicit MDE analysis, a five-pair multi-seed arm. No corpus paper reports any of these.
- **Scorer validation as a first-class result.** Four independent scorers with per-alias κ, plus two human-annotated gold sets. No corpus paper validates the instrument it measures with.
- **The two-layer label + evidence_status scheme**, and the refusal to over-read a point estimate ("no detectable change at n = 200, not a proven zero"). This is P03's refuse-the-tempting-conclusion move.
- **Self-correction of the study's own artefacts** — the v1→v2 rescore and the 128→512 truncation finding, both with before/after numbers. Keep the substance; move it out of the appendix confessional into a numbered before/after table.
- **The pre-registered protocol with its disclosed deviation.** Nothing in the corpus pre-registers anything. Disclosing a deviation against your own interest is the most credible thing in the document.
- **All nine figures original, zero screenshots.**
- **Both declarations.** 0/15 corpus papers carry either. This is a place to exceed the corpus, not match it. *(Verify the current CCDS handbook requirement — the corpus spans 2023–2026 and these may be a recent addition.)*
- **The immutable-artifact contract** — TC1 originals never modified, corrections in redacted sidecars.
- **Reproducibility disclosure.** Three planned cuts (Chapter 5, §6.12, the appendices) all reduce disclosure surface. The corpus penalty for that is severe and repeated: P02's fatal defect is that its study *"cannot be reproduced from the document."* **Invariant: nothing a replicator needs — seed, revision pins, library versions, GPU model, batch size, `max_new_tokens`, scorer version, judge revision, n per benchmark — may leave the report. It may only change container.**

---

## Highest-risk step

The restructure renumbers every section and re-scopes every float while the document carries roughly 296 section symbols plus table and figure references. **A large prose cut is exactly the operation that produces the corpus's most damaging defect** — P02's conclusion contradicts its own tables in four places; P07's abstract says 38.41 where §4.2.2 says 39.2; P08's appendix silently disagrees with its Table 3.

Before starting, extend `make verify-claims` with two locks:
1. Every "Section N" / "Chapter N" / "Table N.M" / "Figure N.M" in the builder resolves to an existing heading or caption.
2. Number consistency across abstract, body, tables and conclusion.

---

## Suggested order

1. Re-cut RQ1–RQ5 (everything downstream depends on it).
2. Add the two cross-reference locks to `make verify-claims`.
3. Abstract: 615 → ~200 words.
4. Chapter 6 rebuild (the structural work).
5. Chapters 7–10 fold; delete Appendix G after re-registering its reproducibility content.
6. Chapter 2 structural expansion; citations into Chapter 1.
7. Global voice pass against the acceptance targets.
8. Front matter: Acknowledgements, two separate lists, glossary to front, cover page cleanup.
9. Mechanical: float renumbering, spelling, model names, `artifact_policy.yaml` patterns.
10. `make report`, re-measure, `make agent-check`.

---

*Compiled 2026-07-25 from a 15-paper corpus study of past NTU FYP reports. Findings were produced by a 15-agent deep read plus a 6-dimension synthesis, then adversarially verified; claims that failed verification were removed or corrected.*
