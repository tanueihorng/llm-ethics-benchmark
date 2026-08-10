# How NTU Research-Type FYP Reports Are Written

**A corpus study of 15 past NTU FYP reports (2023/24 – 2025/26), for use in writing this project's report.**

Source corpus: `Examples of Past FYP Papers/` (15 PDFs, 26–96 pages).
Method: full text extraction + 152 rendered pages inspected visually + per-paper deep read + mechanical measurement of every countable property. Every rule below names the paper(s) it comes from.

> **Read this first.** Eight of the fifteen papers are weak. A practice is only a pattern worth copying if the *strong* papers do it. Where the corpus is **worse** than this project (statistics, scorer validation, pre-registration), this guide says so and tells you to keep what you have. Do not regress to corpus norm.

---

## 1. The corpus, classified

| ID | Paper | Genre | Tier |
|---|---|---|---|
| P07 | Additive Quantization for Truly Tiny Compressed Diffusion Models | research | **exemplary** |
| P01 | AI Deployment on Edge Device | hybrid | solid |
| P02 | VulnDetect AI (CCDS25-0560) | hybrid | solid |
| P03 | LLMs for Time-Series Applications (CCDS25-0849) | research | solid |
| P10 | Building Multi-Modal Intelligence (CCDS24-0792) | hybrid | solid |
| P11 | Benchmarking AI Workflows / DietCoke | hybrid | solid |
| P15 | AgentScript | hybrid | solid |
| P04 | A Novel Recommender System (SCSE23-0498) | research | weak |
| P05 | AI Monitoring at the Edge | development | weak |
| P06 | Security for LLMs (CCDS24-0587) | research | weak |
| P08 | Model Compression Techniques (SCSE21-0187) | hybrid | weak |
| P09 | Long-Term Memory Methods for LLMs | research | weak |
| P12 | RAGLens (CCDS24-0592) | hybrid | weak |
| P13 | Defences Against Hallucination | development | weak |
| P14 | Performance Optimization of LLM Inference | hybrid | weak |

**Genre test that actually works:** look at what the *results chapter* contains. Research = numbers that answer a stated question. Development = evidence that the system works (screenshots, feature walkthroughs). P07 declares its genre outright in §1.3 — *"a work of scientific research… rather than a specific software or hardware deliverable"* — and then legitimately has no implementation chapter at all.

**P07 is the closest model available to this project**: a research-type FYP about quantization, 48 body pages, 42 references, **zero appendices**.

---

## 2. Measured norms

Counts that distinguish good from bad, and counts that don't.

| Property | Strong papers | Weak papers | Verdict |
|---|---|---|---|
| Pages | median 58 | median 61 | **does not separate quality** |
| Figures | median **11** | median **18.5** | weakly *negative* — more figures ≠ better |
| Tables | median 6 (research-genre 13) | median 9.5 | does not separate |
| References | median **42** | median 30 | mild positive signal |
| In-text citations / 1,000 words | corpus median **7.1**, range 3.3–19.1 | | **strong signal** |
| Prose body words | median 11,237, max 16,376 | | |
| Abstract | median **198** words, range 68–435 | | |

> **Do not chase figure count.** The corpus-wide figure median of 16 is inflated by screenshot-heavy weak development reports (P05 has 94, P12 has 40). Among papers worth imitating it is 11. Make each figure argue something instead.

**Universal conventions (near-unanimous):**
- IEEE numeric `[n]` citation — **15/15**
- Acknowledgements — **14/15**
- Separate *List of Figures* and *List of Tables* — **14/15**
- Figure captions **below** the figure — **324 of 345 measured captions (94%)**
- Table captions: genuinely split in this corpus (P07 above; P06, P05, P08 above; P04, P10 below). Pick one and hold it. IEEE convention is above.
- Declaration of Originality — **0/15**. GenAI-use declaration — **0/15**. List of Abbreviations — **0/15**.

---

## 3. Chapter architecture

Research-type reports run **4–6 chapters**. The only 10-chapter report in the corpus is P05, a development report whose chapters are named after build stages.

**P07 (exemplary):** Introduction 10% · Literature Review 19% · **Methodology 23% (largest)** · Experiments 19% · Conclusion + Limitations 4% · Project Timeline & Retrospective 12.5%.

**P03:** Introduction · Literature Review · Experiment Design and Methodology · Experiment Results and Analysis · Discussion and Conclusion.

Three structural rules the strong papers share:

**Rule 1 — name chapters after argument stages, not build stages.** P07 and P03 have *no implementation chapter at all*. Implementation detail lives inside methodology or in an appendix.

**Rule 2 — the results chapter holds exactly one decomposition axis.**
- P03 §4.1–4.6: one subsection per artefact under test, then **§4.7 "Overall Comparisons and Baseline Analysis"**.
- P15 §5.1–5.4: one subsection per numbered experiment (`Experiment 1: …`), each motivated by the previous one's finding.
- P07 §4.2.1–4.2.5: one per evaluation, ending in the ablation.

None of them interleaves per-unit with per-theme sections. Interleaving is the signature of a document that grew by appending.

**Rule 3 — synthesis goes last, always.** P03 §4.7 is 7 of 7; P11 §4.4 is 4 of 4; P01 §5.6 is 6 of 6. Nothing follows it inside the chapter.

**Where limitations live:** folded into the discussion or conclusion chapter, not given their own. P07 §5.1 "Limitations and Future Work" is ~250 words inside a 2-page conclusion. P15 §6.4 ends the Discussion. The only standalone Limitations chapter in the corpus (P01) is 292 words. There is no minimum chapter length — P07's conclusion chapter is deliberately thin *because the discussion was done inline*.

**Process/retrospective content:** legitimate, but quarantined. P07 Chapter 6 "Project Timeline and Retrospective" opens with **§6.1 Chapter Rationale** explaining why the chapter exists and announcing its register switch: *"This chapter is written from a first-person perspective as it describes my personal journey of discovery."* It sits after the conclusion so it cannot contaminate the research narrative, and it gives abandoned work a **numbered results table** (Table 6.1) rather than a changelog.

---

## 4. Voice — the highest-value dimension

This is what separates writing that reads as human from writing that doesn't. It is not vocabulary and **it is not sentence length**.

**The mechanism: the corpus joins clauses with WORDS at the head of a new sentence. Machine-written prose joins them with PUNCTUATION inside one sentence.**

Measured per 1,000 prose words:

| Marker | Corpus median | Corpus max |
|---|---|---|
| Semicolons | 0.36 | 3.79 |
| Em dashes | **0.0** (14/15 papers use none) | 4.79 |
| Parentheticals | 1.97 | 13.84 |
| Sentence-initial connectives | ~5.1 | P03 runs 22.6% of sentences |

Sentence *length* is not a defect: the corpus mean ranges 17.8–35.0 words. P07, P09 and P06 all write long. They just don't weld.

**What the corpus actually does:**

- **Contrast and consequence get their own sentence opener.** "However," "Therefore," "Although," "Despite," "In addition," "Furthermore,"
- **Hedge in a plain separate sentence**, never welded on with a dash:
  - P03: *"However, this observation should be interpreted with caution."*
  - P07: *"This may possibly indicate a flaw in the metrics used."*
  - P01: *"One potential reason for this could be that the TensorRT INT8 calibration may be better aligned with the Jetson's GPU inference optimisations."*
- **State the finding first, then the numbers.** P02: *"Larger models do not uniformly improve F1."* P15: *"AgentScript achieves 39.5%."*
- **Compute the delta for the reader.** P07: *"FID and sFID that respectively outperform the full-precision model by 1.75 and 1.93 points."*
- **Admit failure in the first person.** P03: *"Furthermore, I was unable to obtain valid results for CT under this configuration."* P07: *"No consideration has been made to whether this is a principled way to implement learning rate decay."*

**First person:** the corpus does not avoid it. P07 uses 83 "we" plus 22 "I"; P10 uses 229; P01 uses 25 "I". What the deep read flags as the anti-pattern is *unannounced* editorial "we" in a single-authored report (P04 152, P05 238) and the lone unexplained slip (P11). **Declare the convention once, then use it deliberately.** P15's five instances of "we" — all in the abstract and the numbered contributions — turn the pronoun into a marker of authorial contribution.

**Anti-pattern — epistemic bookkeeping.** P14 repeats *"does not conclusively rule out the potential for…"* near-verbatim twice; it reads as a defensive formula, not a judgement. A hedge repeated as a stock phrase is worse than no hedge.

---

## 5. Results, tables and statistics

**Table conventions from P07 (booktabs, the cleanest in the corpus):**
- Horizontal rules only, no vertical rules, no shading.
- Caption **above**, numbered by chapter, and carrying the measurement protocol, not just a label: *"Table 4.4: Latency and FLOPs of the method on LDM-4 ImageNet 256 × 256. Latency measured for generation of 4 images at 20 inference time-steps using the DDIM sampler."*
- **Directional arrows in every header** (IS↑, FID↓, Precision↑) so the reader never has to remember which way is good.
- **Baseline is always the first row.**
- Big matrices get **internal midrules grouping rows into a story** — Table 4.5's ablation reads top-to-bottom as `+ PEFT (AdamW)` / `+ SeMI-PV` / `- KAQ` / `+ KAQ + LAQ`.
- Prior published methods are interleaved as **ordinary rows** alongside your own, so comparison needs no cross-referencing.
- Declare the bolding rule in the caption (P10: *"The highest scores excluding GPT-4o are bolden"*).

**Pair an absolute table with a delta table** (P02 Tables VII and VIII). The delta table is where the story is.

**Caption states the finding, not the contents** (P10): *"Table 4.5: Performance comparison of various LMMs on Video-MME. LongVA achieves state-of-the-art results among 7B models."* A table-skimming examiner still gets the argument.

**How the strong papers write a results paragraph** (P07's fixed order):
1. Restate the exact experimental conditions.
2. Walk the table, computing deltas *for* the reader.
3. Name the competitor that beats you and why the comparison is unfair.
4. Flag what is unexplained.

**Anomalies get a mechanism, not silence.** P01 reports the ONNX-INT8 model being *slower* than FP32 as *"contrary to expectations"* and attributes it to CPU/GPU kernel fallback. P07 uses a qualitative sample grid as evidence **against its own preferred metric**: *"a subjective evaluation of outputs with human eye-power shows the AdamW-based method to achieve worse alignment… This may possibly indicate a flaw in the metrics used."*

### Statistics: the corpus is bad, do not copy it

**Not one of the fifteen papers reports a seed, a repeated run, an error bar, or any significance test — including P07.** The dominant weak-paper failure is exactly this:

- P04: *"performed significantly better… across every recommendation and conversation metric"* on single unreplicated runs, no seed, no CI, no test.
- P06: forty configurations spanning 136.33–142.33/200 ranked by their means, while the chosen optimum scores 143/136/136 across three runs — a spread wider than the entire range being optimised.
- P12: nine decimal places, rankings asserted off differences of 0.001.
- P08: `n=1` everywhere while 0.02 pp gaps are signed and interpreted.

If your study has BH-FDR correction, bootstrap CIs, McNemar tests, multi-seed sensitivity and κ agreement, **you are already above every paper in this corpus.** Keep all of it. The work is presenting it so the rigour is visible to someone skimming — put the apparatus in the table and the claim in the sentence.

---

## 6. Literature review and citations

**Target: ~7 in-text citations per 1,000 words** (corpus median; minimum observed 3.3).

**Organisation:** by technique or by axis, never chronologically and never as an annotated bibliography. P07's Chapter 2 is a five-link causal chain where each section ends in a **positioning sentence** rather than a summary, so the gap accumulates: *"It is this category that we place ourselves in."*

**Paragraph shape for each prior work** (P01, enforced mechanically):
> [what the study did + citation] → [what it establishes for this project] → ["However / While …" naming precisely what it does not cover]

**Give every reviewed system a number.** P15 attaches a headline figure plus citation to each prior system, which is what makes its own baseline table feel earned rather than dropped in.

**Comparison matrices.** P03 builds **two orthogonal checkmark matrices** of prior work (pipeline × adaptation strategy; pipeline × conversion stage), places each at the head of the section it organises, and then **reuses one in the Discussion to make an argument**. This is how a taxonomy earns its page count. P09's variant: rows = benchmarks, columns = capability dimensions — *the empty column becomes your contribution or your stated scope limit*.

**The research gap — the single most important paragraph in the review.** Three attested patterns:

*P03 §2.5 "Critical Synthesis and Research Gaps"* — a rigid three-move shape:
1. Recap what each prior section established, by section number.
2. *"However, despite the growing number of proposed pipelines, a key gap in the existing literature is the absence of consistent experimental evaluation across studies."*
3. *"To address this gap, the experimental component of this study conducts…"*

*P07 §1.2* — absence claim with an explicit **scope quantifier** plus why it is hard: *"vector quantization has never been applied to diffusion models before, which pose unique challenges owing to their iterative reverse process and convolutional U-Net architecture."*

*P15 §2.8* — **numbered** gaps, each naming a specific *methodological failure* of prior work, mapped 1:1:1 onto numbered contributions and numbered experiments: *"Prior comparative work typically changes both the model and the control modality simultaneously, making it impossible to attribute performance differences to either factor."*

> **Do not grow a literature review by word count.** P03's 41% review is criticised in the deep read as *"far too long"* with textbook-level sections that *"add nothing a CCDS examiner needs"*; P09's ~53% is a weak-paper marker. What separates strong reviews is **structure** — matrices, positioning sentences, per-system numbers, a terminal numbered gap. Prescribe those and let the length follow.

**Citation mechanics:** brackets attach to the noun they qualify, before punctuation. Prefer `EfficientDM [23]` over bare `[23] raises the issue…` — citation-as-grammatical-subject (P07's dominant habit) forces the reader to hold reference numbers in working memory. Density is front-loaded: heavy in Introduction and Literature Review, one-per-paragraph in Methodology where a technique is imported, protocol-justifying in Results.

**Reference quality:** P05's 66 of 74 references are blogs and Medium posts — a reference count that flatters a thin literature base. P13 cites vendor marketing pages as evidence *in a report about evaluating vendors*.

---

## 7. Figures

**Five registers that do real work** (all from P07 unless noted):

1. **Headline result chart in the Introduction, on page 3, before any method text.** Fig 1.1: your metric on y, your controlled variable on x, one line per published competitor, dashed horizontal line for the unmodified baseline. The whole contribution legible in one glance.
2. **Diagnostic figure that DERIVES a design choice**, not one that reports a result. Fig 3.3 plots per-layer quantization MSE against bits/parameter over a parameter-count bar chart — and the design rule is read straight off it. The most transferable figure idea in the corpus.
3. **Conceptual explainer diagrams**, original, drawn in a vector tool. P02's Fig 3 uses real decision diamonds, branch labels, and explicitly drawn failure paths (*"Return None (graceful degradation)"*).
4. **Qualitative exhibit used as counter-evidence** against your own metric (P07 Fig 4.1).
5. **Comparison chart with a reference line.** P02 Fig 4 puts a red dashed line at the published baseline (48.0%) — *that line turns a bar chart into an argument.*

**Conventions:**
- Caption **below**, sentence case, multi-sentence, telling the reader what to look at: *"Figure 3.3: Top: Overall contribution to the quantization error versus the number of bits used per parameter… Bottom: Total count of parameters by layer."*
- **Every figure gets a "Figure N shows…" walkthrough sentence** (P11's rhythm: figure → *"Figure N shows X"* → one interpretive sentence). P15 — a *solid* paper — never once writes the word "Figure" in its running text, and the deep read calls that the thing that *"makes the report look unfinished on its most important pages."*
- Data labels on bars, so the reader never reads an axis (P11, P03).
- Original figures only. Screenshot albums belong to the weak development reports (P05 ~55 raster code screenshots; P13's six figures are all lifted, including one still carrying an *"a16z Enterprise"* watermark).

**Figure defects the corpus keeps making** — check yours: broken cross-references (P07, the exemplary paper, ships at least three), duplicate captions (P01 has two "Fig. 13"), inconsistent legend colour mapping between panels of the same figure (P07 Fig 3.3), truncated axes that exaggerate gaps (P03 Figs 3–4), and captions separated from their figure by a page break (P02 Fig 4).

---

## 8. Front matter and the abstract

**Sequence** (P07, cleanest in the corpus): Title page → Abstract → Acknowledgements → Contents → List of Tables → List of Figures. P01 is faulted for putting the TOC before the abstract.

**Title page fields** (P02, 2026): NTU crest → project code → title → author → `Supervisor: …` / `Co-Supervisor: …` → college → *"Submitted in Partial Fulfilment of the Requirements for the Degree of …"* → year. Nothing else. No corpus title page carries an email address, a document date, or a prose blurb.

**Page numbering:** roman for front matter, arabic restarting at 1 on Chapter 1 (P07, P01, P05, P08). **Running header:** project code flush right on every page (P01, P05, P06, P02).

**Abstract — target ~200 words**, structured field → gap → move → design counts → **headline numbers** → implication.

Weak abstracts omit the findings entirely: P06's 238 words never name a model or a result and read like an unchanged proposal abstract; P08's 69 words are uncitable; P12's 247 words are pure problem framing. P01's, by contrast, states its headline with numbers: *"INT8 quantized models achieved up to 4 times size reduction and faster inference speeds with about 4% loss in accuracy"* — though it never names the metric, which is its own defect.

**Acknowledgements — 90–130 words, first person.** Cover the supervisor (naming the *specific* help — P03 thanks a research scientist for advice on when reproducing an experiment was worth it, which quietly justifies that report's triage policy), the compute provider (P11 acknowledges NTU/NSCC for the resources; if your results depend on a cluster, say so), and a short closing line to family.

**Appendices.** P07 has **none**. What belongs there: configuration and job scripts, full per-pair result tables behind body figures, validation detail, and framework-extension instructions (P12's Appendices 1–3 are singled out as *"exactly the appendix a supervisor wants"*). What must never: an internal changelog, decision IDs, task IDs, repository paths, or build timestamps. Nothing comparable exists in any of the fifteen papers.

---

## 9. The anti-pattern catalogue

Ranked by how often it appears and how badly it damages credibility.

1. **Statistical illiteracy** — n=1 plus "significantly better" (P04, P06, P08, P12). The corpus's defining failure.
2. **The conclusion contradicts the tables** — P06's conclusion claims viability its own results refute; P12's §8.1 names a winner its Table 5 bolds against; P02's conclusion contradicts its results in four places *"because it was written before the final numbers landed and never re-synced."* **A large prose cut is exactly the operation that produces this defect — re-verify every number after restructuring.**
3. **Internal numeric contradiction** — the same baseline reported at two values (P04, P08); arithmetic errors inside headline tables (P08's "5.66 (36%)" against a baseline of 8.79 ms, which is 64%).
4. **Results chapters with no results** — P13's "Results and Discussion" has five pages, zero numbers, zero tables. P09 has three consecutive results pages with zero body prose.
5. **Broken cross-references and numbering gaps** — P09's tables run 3.1, 3.2, 3.4. P08 has three off-by-one figure references.
6. **Metric defined by phrase instead of formula** — P06 defines "True Positive Rate" as the sum of the confusion-matrix diagonals (that is accuracy) and propagates the error through nine captions and five headings.
7. **Tuning and evaluating on the same set** with no held-out split (P06).
8. **Confound identified, then ignored** — P12 names a chunk-size mismatch and still declares that arm the winner.
9. **Tutorial-level background** re-teaching convolution and dropout from blog sources (P01 ~5 pages, P05).
10. **Future tense in a completed results chapter** — P12's *"RAGAs will evaluate"* makes finished work read as a proposal.
11. **Surface drift** — P12 writes "SML" for SLM six times including inside two table captions.

---

## 10. The transferable moves, in priority order

1. **End the literature review with a numbered gap statement** that names a specific methodological failure of prior work, and map gaps 1:1:1 onto contributions and experiments (P15, P03, P01, P07).
2. **Put one headline comparison figure on page 3**, with a dashed baseline reference line (P07, P02).
3. **Organise results as numbered experiments**, each motivated by the previous one's finding, synthesis last (P15, P03).
4. **Close with an RQ → claim → evidence → verdict table.** P02 sets three quantitative success criteria and never revisits them; P04 is faulted for having no table mapping objectives to evidence. No corpus paper does this well — it is a free differentiator.
5. **Report the number that hurts.** P11's accuracy collapses to 35.5%, the prose localises the cause, and a fourth configuration recovers it to 43.4%. P10 prints human agreement (0.75) below its automated figure (0.89) without spin.
6. **Report failed configurations as first-class results** with their evidence (P01's three failed quantization paths; P07's Table 6.1 negative-results post-mortem).
7. **Explain counter-intuitive numbers mechanistically** rather than hiding them (P01, P07).
8. **State the confound-control sentence explicitly** — one per held-constant factor (P02: *"web search is disabled so that model differences reflect reasoning quality rather than retrieval differences"*).
9. **Disclose the environment as a bare version manifest** (P11 §4.1.2: OS, CUDA, Python, library versions) and name every library default you overrode.
10. **Report a cost axis in the same table as the quality axis** — GPU-hours, wall-clock, model size, memory (P07, P01, P11). Without it, a "trade-off" study measures only one side.
11. **Justify what you excluded as carefully as what you included, and cite a precedent for the omission** (P07).
12. **Define your analysis instrument in the method chapter, before any data** (P15's failure taxonomy; P03's reproduction gate).
13. **Confess non-principled implementation choices in one sentence, without apology** (P07). It costs nothing and buys credibility for everything you don't flag.
14. **Translate deltas into deployment units** (P14): *"a +0.05 ASR delta means roughly one additional harmful completion per twenty adversarial prompts served."*
15. **When a result is null, name the competing explanations and say which artefact discriminates each** (P06's right move in a weak paper; P14: *"deflate your own convenient null result"*).

---

*Compiled 2026-07-25 from a 15-paper corpus study. Companion document: `docs/REPORT_REMEDIATION_PLAN.md`.*
