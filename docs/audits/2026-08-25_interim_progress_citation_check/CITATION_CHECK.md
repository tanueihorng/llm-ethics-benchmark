# Citation-to-Paper Verification — `docs/rewritten_fyp_interim_progress.docx`

Date: 2026-08-25 (UTC+8). Executed by Claude (session audit). Artifact SHA-256: `96f0e930d3f1018637ab1fab1b8fd02a0c82ef0d542a1c0799b9fd6dfa5bca5d` (mtime 2026-08-25 00:02).

## Scope and question

The user asked: for every citation in the submitted interim progress docx, does this repo hold the actual cited paper? If yes, read each paper and write a summary. This document records the verification; summaries live in `PAPER_SUMMARIES.md`.

## Method (and how each step was self-tested)

1. **Bibliography census** — extracted all paragraphs with `python-docx`; references are the trailing block of `Reference`-styled paragraphs under the "References" Heading 1. Two independent counts agree: (a) `Reference`-styled paragraph count = 57; (b) leading-bracket numbers `[n]` enumerate contiguously 1–57, no gaps or duplicates.
2. **In-text citation census** — every visible text layer scanned: body paragraphs and tables (block-level walk), then a raw-XML pass over ALL `<w:p>` elements (which additionally covers text-box content), minus `Reference`-styled entries so bibliography markers cannot inflate the count. Footnotes/endnotes/header/footer XML parts contain no bracketed citations. **Corrected mid-audit (see finding C1):** the first passes counted single-number brackets only and missed IEEE *range* citations (`[4]–[7]`); after expanding ranges, the census agrees with the repo's own gate `tests/test_ieee_citation_order.py`, which passed on this exact artifact while the first census contradicted it. Three ranges exist, all in §1.1: `[1]–[3]`, `[4]–[7]`, `[8]–[10]`.
3. **Paper matching** — page 1–2 text of every candidate PDF under `Sources/` and `tmp/source_audit_external/` extracted (`pdftotext`) and fuzzy-matched against each entry's quoted title; **all 50 matched rows were then manually confirmed by reading each PDF's own page-1 title/author line against its bibliography entry** (not by trusting filenames). A duplicate mirror tree exists under `Examples of Past FYP Papers/Verified Literature Sources - Quantization and LLM Safety/`; canonical paths below point at `Sources/`.
4. **Absence sweep for unmatched titles** — all 150 PDFs in the repository (excluding `.git`, `node_modules`) were text-extracted and probed for distinctive phrases of the seven unmatched classics ("introduction to the bootstrap", "sampling error of the difference between correlated", "controlling the false discovery rate", "coefficient of agreement for nominal scales", "differences in proportions for the paired-sample", "measurement of observer agreement for categorical", "two-one-sided tests procedure"). Zero hits. This is an executed enumeration, not an assumption (D61).

## Result summary

| Question | Answer |
|---|---|
| References in the submitted docx | **57** ([1]–[57], contiguous) |
| Cited in body text | **57 of 57** (all cited; three citations are IEEE ranges `[1]–[3]`, `[4]–[7]`, `[8]–[10]` in §1.1) |
| With the actual paper in this repo | **50** of 57 (identity-verified) |
| Without any local paper copy | **7** of 57: [47], [48], [49], [50], [51], [54], [56] (statistical-methods classics) |
| Other findings | Coverage gap D2, stale aux artifacts D4, and this audit's own corrected false findings C1 below |

## Findings

- **D1 (coverage):** 50/57 citations resolve to an identity-verified local PDF. Full table below.
- **D2 (missing papers):** refs [47] Efron & Tibshirani (bootstrap book), [48] McNemar 1947, [49] Benjamini & Hochberg 1995, [50] Cohen 1960, [51] Connor 1987, [54] Landis & Koch 1977, [56] Schuirmann 1987 have **no copy of the actual paper anywhere in the repo**. For [48] and [56], `tmp/source_audit_external/ref_46_mcnemar.pdf` and `ref_54_schuirmann.pdf` exist but are **HTML pages saved with a .pdf extension** (`file(1)`: "HTML document text"), not papers. No summaries are provided for these seven because there is no local full text to read; they were excluded honestly rather than summarized from memory (D61). This persists the gap T62's ledger already recorded for the v4-era statistical classics.
- **C1 (corrected false finding, recorded per D61):** this audit initially reported two defects that are NOT real: (a) "refs [5] and [6] are never cited anywhere" and (b) "exactly one first-use ordering violation: (27, 2)". Original wording is quoted here so later readers do not inherit a silent rewrite. Both claims were artifacts of the audit's own extractor counting single-number brackets only and missing the three IEEE range citations `[1]–[3]`, `[4]–[7]`, `[8]–[10]` in §1.1 — `[5]` and `[6]` are cited inside `[4]–[7]`. The repo gate `tests/test_ieee_citation_order.py` passed on this exact artifact throughout; the contradiction was the trigger that exposed the extractor bug. With ranges expanded: all 57 references are cited, and the first-use sequence is strictly ascending with zero violations. The two "independent" census passes that had agreed on the false findings shared the same blind spot, so their agreement proved nothing about range coverage — a live instance of D61's denominator rule.
- **D4 (stale auxiliary artifacts):** the two HTML-as-.pdf files under `tmp/source_audit_external/` should be deleted or replaced with real PDFs if McNemar/Schuirmann full texts are wanted locally.

## Full mapping table

| Ref | Bibliography entry (truncated) | Repo PDF (canonical path) | Verification |
|---|---|---|---|
| [1] | J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, and S. Han… | `Sources/[05] Lin et al. - AWQ - Activation-aware weight quantization for on-device LLM compression and acceleration.pdf` | identity-verified (PDF page-1 title/author) |
| [2] | E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, “GPTQ: Accurate post-training quantizatio… | `Sources/[06] Frantar et al. - GPTQ - Accurate post-training quantization for generative pre-trained transformers.pdf` | identity-verified (PDF page-1 title/author) |
| [3] | T. Dettmers and L. Zettlemoyer, “The case for 4-bit precision: k-bit inference scaling laws,” i… | `Sources/[07] Dettmers et al. - The case for 4-bit precision - k-bit inference scaling laws.pdf` | identity-verified (PDF page-1 title/author) |
| [4] | A. Yang et al. (Qwen Team), “Qwen3 technical report,” Alibaba Group, 2025. arXiv:2505.09388. | `Sources/[01] Yang et al. - Qwen3 technical report.pdf` | identity-verified (PDF page-1 title/author) |
| [5] | A. Grattafiori et al. (Meta AI), “The Llama 3 herd of models,” Meta AI, 2024. arXiv:2407.21783. | `Sources/[02] Grattafiori et al. - The Llama 3 herd of models.pdf` | identity-verified (PDF page-1 title/author) |
| [6] | A. Abouelenin et al., “Phi-4-Mini technical report: Compact yet powerful multimodal language mo… | `Sources/[03] Abouelenin et al. - Phi-4-Mini technical report - Compact yet powerful multimodal language models via Mixture-of-LoRAs.pdf` | identity-verified (PDF page-1 title/author) |
| [7] | A. Q. Jiang et al., “Mistral 7B,” Mistral AI, 2023. arXiv:2310.06825. | `Sources/[51] Jiang et al. - Mistral 7B.pdf` | identity-verified (PDF page-1 title/author) |
| [8] | R. Jin, J. Du, W. Huang, et al., “A comprehensive evaluation of quantization strategies for lar… | `Sources/[08] Jin et al. - A comprehensive evaluation of quantization strategies for large language models.pdf` | identity-verified (PDF page-1 title/author) |
| [9] | S. Li, X. Ning, L. Wang, T. Liu, X. Shi, S. Yan, G. Dai, H. Yang, and Y. Wang, “Evaluating quan… | `Sources/[09] Li et al. - Evaluating quantized large language models.pdf` | identity-verified (PDF page-1 title/author) |
| [10] | A. Kharinaev, V. Moskvoretskii, E. Shvetsov et al., “Investigating the impact of quantization m… | `Sources/[10.1] Kharinaev et al. - Investigating the impact of quantization methods on the safety and reliability of large language models (IEEE Access version of record).pdf` | identity-verified (PDF page-1 title/author) |
| [11] | P. Röttger, H. R. Kirk, B. Vidgen, G. Attanasio, F. Bianchi, and D. Hovy, “XSTest: A test suite… | `Sources/[18] Röttger et al. - XSTest - A test suite for identifying exaggerated safety behaviours in large language models.pdf` | identity-verified (PDF page-1 title/author) |
| [12] | M. Mazeika et al., “HarmBench: A standardized evaluation framework for automated red teaming an… | `Sources/[24.1] Mazeika et al. - HarmBench - A standardized evaluation framework for automated red teaming and robust refusal (PMLR v235 version of record).pdf` | identity-verified (PDF page-1 title/author) |
| [13] | Y. Huang, L. Sun, et al., “Position: TrustLLM: Trustworthiness in large language models,” in Pr… | `Sources/[20] Huang et al. - Position - TrustLLM - Trustworthiness in large language models.pdf` | identity-verified (PDF page-1 title/author) |
| [14] | B. Wang et al., “DecodingTrust: A comprehensive assessment of trustworthiness in GPT models,” i… | `Sources/[21] Wang et al. - DecodingTrust - A comprehensive assessment of trustworthiness in GPT models.pdf` | identity-verified (PDF page-1 title/author) |
| [15] | Z. Zhang et al., “SafetyBench: Evaluating the safety of large language models,” in Proc. Associ… | `Sources/[22] Zhang et al. - SafetyBench - Evaluating the safety of large language models.pdf` | identity-verified (PDF page-1 title/author) |
| [16] | L. Ouyang et al., “Training language models to follow instructions with human feedback,” in Adv… | `Sources/[23] Ouyang et al. - Training language models to follow instructions with human feedback.pdf` | identity-verified (PDF page-1 title/author) |
| [17] | A. Zou, Z. Wang, N. Carlini, M. Nasr, J. Z. Kolter, and M. Fredrikson, “Universal and transfera… | `Sources/[25] Zou et al. - Universal and transferable adversarial attacks on aligned language models.pdf` | identity-verified (PDF page-1 title/author) |
| [18] | E. Perez, S. Huang, F. Song, T. Cai, R. Ring, J. Aslanides, A. Glaese, N. McAleese, and G. Irvi… | `Sources/[26] Perez et al. - Red teaming language models with language models.pdf` | identity-verified (PDF page-1 title/author) |
| [19] | D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, “Measuring m… | `Sources/[27] Hendrycks et al. - Measuring massive multitask language understanding.pdf` | identity-verified (PDF page-1 title/author) |
| [20] | P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, “Think yo… | `Sources/[28] Clark et al. - Think you have solved question answering Try ARC, the AI2 reasoning challenge.pdf` | identity-verified (PDF page-1 title/author) |
| [21] | S. Badshah and H. Sajjad, “Quantifying the capabilities of LLMs across scale and precision,” 20… | `Sources/[29] Badshah et al. - Quantifying the capabilities of LLMs across scale and precision.pdf` | identity-verified (PDF page-1 title/author) |
| [22] | G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han, “SmoothQuant: Accurate and efficient… | `Sources/[30] Xiao et al. - SmoothQuant - Accurate and efficient post-training quantization for large language models.pdf` | identity-verified (PDF page-1 title/author) |
| [23] | E. Frantar and D. Alistarh, “SparseGPT: Massive language models can be accurately pruned in one… | `Sources/[31] Frantar et al. - SparseGPT - Massive language models can be accurately pruned in one-shot.pdf` | identity-verified (PDF page-1 title/author) |
| [24] | T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient finetuning of quant… | `Sources/[32] Dettmers et al. - QLoRA - Efficient finetuning of quantized LLMs.pdf` | identity-verified (PDF page-1 title/author) |
| [25] | T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, “LLM.int8(): 8-bit matrix multiplication… | `Sources/[33] Dettmers et al. - LLM.int8() - 8-bit matrix multiplication for transformers at scale.pdf` | identity-verified (PDF page-1 title/author) |
| [26] | C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in… | `Sources/[34] Guo et al. - On calibration of modern neural networks.pdf` | identity-verified (PDF page-1 title/author) |
| [27] | I. Proskurina, L. Brun, G. Metzler, and J. Velcin, “When quantization affects confidence of lar… | `Sources/[11] Proskurina et al. - When quantization affects confidence of large language models.pdf` | identity-verified (PDF page-1 title/author) |
| [28] | K. Marchisio, S. Dash, H. Chen, D. Aumiller, A. Üstün, S. Hooker, and S. Ruder, “How does quant… | `Sources/[13] Marchisio et al. - How does quantization affect multilingual LLMs.pdf` | identity-verified (PDF page-1 title/author) |
| [29] | Y. Belkhiter, G. Zizzo, and S. Maffeis, “HarmLevelBench: Evaluating harm-level compliance and t… | `Sources/[35] Belkhiter et al. - HarmLevelBench - Evaluating harm-level compliance and the impact of quantization on model alignment.pdf` | identity-verified (PDF page-1 title/author) |
| [30] | K. Chen, J. Zhang, J. Hu et al., “Assessing safety risks and quantization-aware safety patching… | `Sources/[36] Chen et al. - Assessing safety risks and quantization-aware safety patching for quantized large language models.pdf` | identity-verified (PDF page-1 title/author) |
| [31] | J. Hong, J. Duan, C. Zhang, Z. Li, C. Xie, K. Lieberman, J. Diffenderfer, B. R. Bartoldson, A. … | `Sources/[37] Hong et al. - Decoding compressed trust - Scrutinizing the trustworthiness of efficient LLMs under compression.pdf` | identity-verified (PDF page-1 title/author) |
| [32] | W. Shao et al., “OmniQuant: Omnidirectionally calibrated quantization for large language models… | `Sources/[38] Shao et al. - OmniQuant - Omnidirectionally calibrated quantization for large language models.pdf` | identity-verified (PDF page-1 title/author) |
| [33] | P. Dong, H. Li, and S. Guo, “Durable quantization conditioned misalignment attack on large lang… | `Sources/[39] Dong et al. - Durable quantization conditioned misalignment attack on large language models.pdf` | identity-verified (PDF page-1 title/author) |
| [34] | S. Li, L. Yao, L. Zhang, and Y. Li, “Safety layers in aligned large language models: The key to… | `Sources/[40] Li et al. - Safety layers in aligned large language models - The key to LLM security.pdf` | identity-verified (PDF page-1 title/author) |
| [35] | A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, and N. Nanda, “Refusal in l… | `Sources/[15] Arditi et al. - Refusal in language models is mediated by a single direction.pdf` | identity-verified (PDF page-1 title/author) |
| [36] | X. Qi, A. Panda, K. Lyu, X. Ma, S. Roy, A. Beirami, P. Mittal, and P. Henderson, “Safety alignm… | `Sources/[16] Qi et al. - Safety alignment should be made more than just a few tokens deep.pdf` | identity-verified (PDF page-1 title/author) |
| [37] | K. Egashira, M. Vero, R. Staab, J. He, and M. Vechev, “Exploiting LLM quantization,” in Proc. N… | `Sources/[12] Egashira et al. - Exploiting LLM quantization.pdf` | identity-verified (PDF page-1 title/author) |
| [38] | A. Souly et al., “A StrongREJECT for empty jailbreaks,” in Proc. NeurIPS Datasets and Benchmark… | `Sources/[41] Souly et al. - A StrongREJECT for empty jailbreaks.pdf` | identity-verified (PDF page-1 title/author) |
| [39] | T. Xie et al., “SORRY-Bench: Systematically evaluating large language model safety refusal,” in… | `Sources/[42] Xie et al. - SORRY-Bench - Systematically evaluating large language model safety refusal.pdf` | identity-verified (PDF page-1 title/author) |
| [40] | P. Chao et al., “JailbreakBench: An open robustness benchmark for jailbreaking large language m… | `Sources/[43] Chao et al. - JailbreakBench - An open robustness benchmark for jailbreaking large language models.pdf` | identity-verified (PDF page-1 title/author) |
| [41] | S. Han et al., “WildGuard: Open one-stop moderation tools for safety risks, jailbreaks, and ref… | `Sources/[44] Han et al. - WildGuard - Open one-stop moderation tools for safety risks, jailbreaks, and refusals of LLMs.pdf` | identity-verified (PDF page-1 title/author) |
| [42] | J. Cui, W.-L. Chiang, I. Stoica, and C.-J. Hsieh, “OR-Bench: An over-refusal benchmark for larg… | `Sources/[19] Cui et al. - OR-Bench - An over-refusal benchmark for large language models.pdf` | identity-verified (PDF page-1 title/author) |
| [43] | Y. Wang, H. Li, X. Han, P. Nakov, and T. Baldwin, “Do-Not-Answer: Evaluating safeguards in LLMs… | `Sources/[45] Wang et al. - Do-Not-Answer - Evaluating safeguards in LLMs.pdf` | identity-verified (PDF page-1 title/author) |
| [44] | S. Kim et al., “Prometheus 2: An open source language model specialized in evaluating other lan… | `Sources/[46] Kim et al. - Prometheus 2 - An open source language model specialized in evaluating other language models.pdf` | identity-verified (PDF page-1 title/author) |
| [45] | L. Zheng et al., “Judging LLM-as-a-judge with MT-Bench and Chatbot Arena,” in Proc. NeurIPS Dat… | `Sources/[47] Zheng et al. - Judging LLM-as-a-judge with MT-Bench and Chatbot Arena.pdf` | identity-verified (PDF page-1 title/author) |
| [46] | A. Wei, N. Haghtalab, and J. Steinhardt, “Jailbroken: How does LLM safety training fail?,” in A… | `Sources/[48] Wei et al. - Jailbroken - How does LLM safety training fail.pdf` | identity-verified (PDF page-1 title/author) |
| [47] | B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York: Chapman & Hall, 1993… | — | NO local copy (see D2) |
| [48] | Q. McNemar, “Note on the sampling error of the difference between correlated proportions or per… | — | NO local copy (see D2) |
| [49] | Y. Benjamini and Y. Hochberg, “Controlling the false discovery rate: A practical and powerful a… | — | NO local copy (see D2) |
| [50] | J. Cohen, “A coefficient of agreement for nominal scales,” Educational and Psychological Measur… | — | NO local copy (see D2) |
| [51] | R. J. Connor, “Sample size for testing differences in proportions for the paired-sample design,… | — | NO local copy (see D2) |
| [52] | D. Card, P. Henderson, U. Khandelwal, R. Jia, K. Mahowald, and D. Jurafsky, “With little power … | `Sources/[55] Card et al. - With little power comes great responsibility.pdf` | identity-verified (PDF page-1 title/author) |
| [53] | R. Artstein and M. Poesio, “Survey article: Inter-coder agreement for computational linguistics… | `Sources/[56] Artstein et al. - Survey article - Inter-coder agreement for computational linguistics.pdf` | identity-verified (PDF page-1 title/author) |
| [54] | J. R. Landis and G. G. Koch, “The measurement of observer agreement for categorical data,” Biom… | — | NO local copy (see D2) |
| [55] | H. Inan et al., “Llama Guard: LLM-based input-output safeguard for human-AI conversations,” Met… | `Sources/[58] Inan et al. - Llama Guard - LLM-based input-output safeguard for human-AI conversations.pdf` | identity-verified (PDF page-1 title/author) |
| [56] | D. J. Schuirmann, “A comparison of the two one-sided tests procedure and the power approach for… | — | NO local copy (see D2) |
| [57] | O. Sainz, J. Campos, I. García-Ferrero, J. Etxaniz, O. Lopez de Lacalle, and E. Agirre, “NLP ev… | `Sources/[60] Sainz et al. - NLP evaluation in trouble - On the need to measure LLM data contamination for each benchmark.pdf` | identity-verified (PDF page-1 title/author) |

## Paper summaries

`PAPER_SUMMARIES.md` contains one structured summary per local paper (50 total): problem, approach, key findings with anchors, publication-status grading, a mechanical **Coverage** bullet stating exactly what each summary rests on, and where the report cites it. Compiled from five batch files under `summaries/`, upgraded for 12 papers by sentence-level reads under `deepreads/`, then re-verified (see v2 record below).

## v2 upgrade record (2026-08-25, later the same day)

**Depth upgrade.** 12 of the 50 summaries were replaced by sentence-level reads (one agent per paper): deep reads [10] Kharinaev, [11] XSTest, [12] HarmBench, [29] HarmLevelBench, [30] Chen et al., [33] Dong et al. (the load-bearing subset); full every-sentence passes on evidence-chosen high-usage papers [8] Jin, [9] Li, [21] Badshah, [27] Proskurina, [28] Marchisio, [31] Hong. Each produced: a per-section coverage table (READ-FULL/SKIMMED/UNREADABLE with extract-line anchors), an upgraded anchored summary with Limitations and FYP-relevance bullets, a bullet-by-bullet discrepancy check against the prior summary, and a grep self-test for every quoted number.

**Coverage bullets.** All 50 sections now carry a Coverage bullet: upgraded sections state their read counts + pointer to the audit trail; the other 38 state the honest mechanical basis (whole-PDF pdftotext, N pages / ~W words) plus batch-protocol depth (abstract, core method, main results; appendices not individually summarized).

**Corrections applied via the discrepancy checks** (spliced into PAPER_SUMMARIES.md):
- **[10]** "8-, 4-, and 2-bit" over-reach fixed — the paper targets 4-bit and 2-bit weight-only PTQ ("3 precision levels", levels unnamed; 8-bit dismissed as widely studied).
- **[9]** ethics-accuracy-rise claim scoped to <7B models only; AWQ's refusal-restoring effect added.
- **[21]** bitsandbytes scope corrected (FP16/FP32 baselines are PyTorch dtypes).
- **[28]** "128 English calibration samples" scoped to SmoothQuant/GPTQ only.
- Publication-status upgrades from PDF-internal evidence: [30] = ICML 2025 PMLR 267; [27] = Findings of NAACL 2024 pp. 1918–1928; [28] = Findings of EMNLP 2024 pp. 15928–15947.

**Submitted-docx defect found during upgrades:** bibliography entry [33]'s author order is wrong — it reads "P. Dong, H. Li, and S. Guo" but the ICLR 2025 PDF title block lists Haowei Li first (Li/Dong equal contribution). Flagged as an open item; not silently repaired in the docx.

**Cross-artifact numbering warning for future claim checks:** several readers verified citations against `scripts/build_fyp_report_v5.js` (canonical v5), where bracket numbers differ from this interim docx (e.g., v5's [8]=QLoRA vs interim [8]=Jin; v5's [30]=Wilcoxon vs interim [30]=Chen). All published "Cited in the report at" values are stamped exclusively from the submitted-docx census (`cited_in_expanded.json`); builder-derived verdicts were discarded as out-of-scope.

**Unresolved externally this session:** arXiv IDs in entries [8] (2402.16775), [9] (2402.18158), [27] (2405.00632), [28] (2407.03211), [30] (2506.20251 page range), [31] (2403.15447), [11] (2308.01263) do not appear in their PDFs' extracted text and could not be API-resolved because the web-search tool failed authentication all session (`api key invalid`). They are recorded UNRESOLVED, not verified.

**Verification layers executed:** (1) writer self-tests (grep proofs per quoted number, coverage tables); (2) parent spot-checks — every base summary's quoted numbers mechanically swept against extracts, ~32 manually re-derived, zero hard mismatches ([37]'s "1,500" = paper's "1.5k"); (3) independent verification wave by fresh agents (number audit of upgraded sections, coverage-bullet audit vs `coverage_stats.json`, strict entailment spot-check) — **collected 2026-08-25, results below.**

### Verification wave results (all three agents reported)

**Number audit** (`tmp/verify_numbers.md`, refs 8,9,10,11,21,27,28,29,30,31,33): 165 checks — 159 FOUND-VERBATIM, 4 FOUND-VARIANT (benign arithmetic derivations/trend hedges), **2 misattributions found → both independently re-verified by the parent against the raw extracts → both fixed at source**:
- [8] "Qwen-14B 8-bit GPTQ (15.91 GB)" was LLM.int8()'s cell; GPTQ = **17.92 GB** (Table 8 INT-8 row reads LLM.int8 15.91 / GPTQ 17.92). Fixed in `deepreads/fullref_8.md` incl. its self-test row, with the original error quoted in place.
- [27] "BLOOM-7.1B ARC CE 7.94→+0.83" sat in a different family's 7B block (family markers lost to pdftotext); BLOOM-7.1B's own ARC CE cell is **15.57 ↑1.06** (L267, directly after the "7.1B" marker). Re-labelled with the uncertainty stated explicitly.

**Coverage audit** (`tmp/verify_coverage.md`): mechanical pages/words 50/50 exact vs `coverage_stats.json`; deep-read count bullets 8/10 exact, 2 compressions ([28],[30] qualified status rows dropped by the compiler's exact-match counting) → root cause fixed in `compile_summaries_v2.py::coverage_counts` (scoped to the Coverage-table section, leading-keyword matching) and recompiled — all bullets now agree with their tables; non-upgraded honesty statements 40/40. The audit also caught that [10]'s upgrade had been stranded by a file-write race (landed after the previous compile) — resolved by the recompile.

**Entailment spot check** (`tmp/verify_entailment.md`, refs 10,11,29,31,33): **56/56 sentence units ENTAILED, 0 overreach, 0 contradicted**, judged anchor-by-anchor including cell-level table verification; four wording nuances and two evaluative glosses recorded inside clean entries.

**[12] HarmBench** (replacement reader, `deepreads/deepref_12.md`): main text §1–§6 sentence-level READ-FULL (paper has no §7), appendices covered per depth contract, D.3 headline values double-anchored in two independent extractions (510 behaviors; 18 methods vs 33 LLMs; GCG/PAIR/AutoDAN on Llama-2 class; GPT-4 white-box cells are genuine "-" not-run cells). Parent spot-checked 510 / "18 red teaming methods and 33 LLMs" / R2D2 32.5 row / PAIR 39.3 verbatim. Precision notes recorded: arXiv ID unprintable in the PMLR PDF; §5.2 prose R2D2 figures differ from single-panel table pairs and must be quoted as prose.

## Self-test record

- Bibliography count derived twice (style-based and bracket-enumeration) — agree at 57.
- In-text census: first derived twice (python-docx block walk; raw XML `<w:t>` join) — but both passes shared one blind spot (no range-citation expansion), so their agreement was not evidence. **The decisive cross-check was against an extractor with a different failure surface:** the repo's own `tests/test_ieee_citation_order.py`, which passed while the first census reported defects. After adding range expansion, the audit census and the repo gate agree exactly: 57/57 cited, zero ordering violations.
- All 50 matched PDFs eyeballed at page-1 level against their entries (author + title), not merely fuzzy-scored.
- Absence claim backed by a full sweep of 150 repo PDFs for distinctive title phrases (zero hits), plus `file(1)` confirmation that the only near-candidates are HTML, not PDFs.
- Known limitation: matching verifies *identity* (the repo holds the cited paper). Whether each sentence's *use* of a paper is faithful is the separate entailment question (T62/T67 discipline, D56); this pass did not re-adjudicate sentence-level entailment for this artifact.
- Meta-note kept deliberately (D61): the false findings this audit produced and then corrected are recorded in C1 with original wording, because "two independent passes agreed" is only as strong as the assumption that their bugs are uncorrelated.
