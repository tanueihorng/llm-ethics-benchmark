# Deep Read — [10] Kharinaev et al. (IEEE Access VoR)

Primary artifact: `docs/audits/2026-08-25_interim_progress_citation_check/tmp/extract/ref_10.txt` (pdftotext of `Sources/[10.1] ... (IEEE Access version of record).pdf`). Read sequentially, offsets 1→1850, in seven contiguous chunks; the read tool reports EOF at line 1850 (`wc -l` prints 1849 because the last line has no trailing newline). Alternate `Sources/[10.2] ... (arXiv v2).pdf` used only to cross-check the arXiv identifier.

**Extraction caveats (affect every row below):** the PDF is two-column, so pdftotext locally interleaves the columns. Examples: "C. BENCHMARKS AND EVALUATION" appears at line 366 *before* the "IV. EXPERIMENTAL PROCEDURE" header at line 373; §II-E's header (line 255) precedes §II-D's (line 257); Appendix B's GPTQ block (1080–1095) precedes the "B. QUANTIZATION METHODS" header (1097); several §V subsubsection labels (`a:`/`b:`/`c:`/`d:`) appear out of order. All text is legible — there is **no garbled text anywhere in the extract**, so no section is marked UNREADABLE. Crucially, **none of Tables 1–15 has its numeric body in the extract — captions only**; every number quoted below comes from running prose, and table-level values could not be independently checked from this extract.

## Coverage table

| Paper section | Extract lines (approx range) | Status (READ-FULL / SKIMMED / UNREADABLE) | Note |
|---|---|---|---|
| Front matter: dates, DOI, title, authors, affiliations, funding | 1–19 | READ-FULL | Received 29 May 2026, accepted 12 June 2026, published 15 June 2026, current version 30 June 2026; DOI on line 2. |
| Abstract + Index Terms | 21–34 | READ-FULL | Index Terms end with "SMILES" (line 34) — an apparent copyediting artefact, unrelated to the paper's content. |
| I. Introduction (incl. §I-A Open-Ended Safety Evaluation and Dataset Design, contributions list, roadmap) | 36–192 | READ-FULL | Contributions bullets at 148–185; roadmap sentence names Section VII then IX, skipping VIII (190–192). |
| II. Related Work (§II-A PTQ & Safety, §II-B QAT & Safety, §II-C Complementary Compression, §II-D Prior Findings, §II-E Alignment & Safety Considerations) | 193–297 | READ-FULL | Column interleaving reorders II-D/E headers (255 vs 257); all prose present. |
| III. OpenSafetyMini: Creating Challenging Safety Dataset (incl. III-1 Category-Level Filtering) | 269–372 (headers at 298, 325) | READ-FULL | Deflection-score filtration prose sits at 276–282 due to interleaving; Table 3/Figure 2 captions only. |
| IV. Experimental Procedure (§IV-A Models, §IV-B Quantization Procedures, §IV-C Benchmarks & Evaluation: open-ended human + judge, multiple-choice, trustworthiness) | 354–495 (header at 373) | READ-FULL | QAT method descriptions (STE/LSQ, QLoRA) appear at 354–363, before the IV-B header at 391, again from interleaving. |
| V. Results (§V-A Open-Ended Safety with subsubsections 4-bit / 2-bit / model size / QAT / OSM-vs-XSafety discrimination; §V-B LLM-as-a-Judge Reliability; §V-C Multiple-Choice Safety; §V-D Trustworthiness; §V-E Statistical Analysis: bootstrap, McNemar, results) | 487–841 | READ-FULL | Header order scrambled by column interleave (e.g., "B. LLM-AS-A-JUDGE RELIABILITY" at 530, "a: 4-BIT PRECISION" at 550, "1) AUTOMATIC EVALUATION" at 586). Tables 6–10 caption-only. |
| VI. Discussion (§VI-A QUIK Robustness … §VI-H Reconciling Conflicting Findings) | 842–938 | READ-FULL | Eight lettered subsections; VI-D header appears at 828, before VI-C at 866 (interleave). |
| VII. Limitations | 939–956 | READ-FULL | Two bullets; second bullet claims the evaluation "currently focuses on post-training quantization", inconsistent with the paper's own QAT experiments (see Limitations & caveats below). |
| VIII. Ethical Considerations | 957–978 | READ-FULL | Covers provocative dataset content and release of unsafe model outputs for research. |
| IX. Conclusion | 979–992 | READ-FULL | Headline counts: 21,328 human evaluations; 66 settings; 4 benchmarks; 5 modern LLMs; 4 PTQ; 2 QAT; 3 bit ranges (985–991). |
| Appendix A. Datasets (XSAFETY, SafetyBench, HotPotQA) | 994–1057 | READ-FULL | Under the unnumbered "APPENDIX." header at 993. |
| Appendix B. Quantization Methods (GPTQ, AWQ, QUIK, AQLM, QuIP/QuIP#) | 1059–1188 | READ-FULL | GPTQ block precedes the B header (interleave); equations rendered as ASCII (e.g., eq. 2 at 1126) but legible. |
| Appendix C. Why PTQ Preserves Safety Alignment: A Mechanistic Perspective | 1189–1255 | READ-FULL | Salient-direction account of PTQ-vs-QAT asymmetry. |
| Appendix D. Prompts (default XSafety judge prompt; safety criteria) | 1256–1307 | READ-FULL | Table 11 (criteria list) caption-only at 1265. |
| Appendix E. Technical Details (hardware, software, compute, licenses) | 1313–1336 | READ-FULL | 4×A40 for inference; T4 via Kaggle for AlignScore; ≈80 + ≈40 GPU-hours. |
| Appendix F. Benchmarks Complete Results (SafetyBench → Table 12; HotPotQA → Table 13) | 1337–1348 | READ-FULL | Pointer text only; Tables 12–13 bodies absent from extract. |
| Appendix G. Prompts (ambiguity scoring; OSM judge prompt; SafetyBench prompt; HotPotQA prompt; human-eval instructions) | 1349–1556 | READ-FULL | Judge prompt is guided-reasoning with JSON answer template; tie-break hierarchy given at 1394–1397. |
| Acknowledgment | 1558–1567 | READ-FULL | Credits Alex Tyulyupo for the deflection-scoring filtration procedure. |
| References [1]–[64] | 1568–1783 | READ-FULL | [43] cited for Mistral is actually the Mixtral paper and [44] for Qwen2.5 is the Qwen2.5-Math report — citation quirks inside the paper itself, noted here for the audit trail. |
| Author biographies | 1785–1848 | READ-FULL | Six author bios; final page number 96793 at line 1848. |

## Upgraded summary

- **Bibliography entry:** [10] A. Kharinaev, V. Moskvoretskii, E. Shvetsov et al., "Investigating the impact of quantization methods on the safety and reliability of large language models," IEEE Access, vol. 14, pp. 96771–96793, 2026, doi: 10.1109/ACCESS.2026.3703899. arXiv:2502.15799.
- **Repo PDF:** `Sources/[10.1] Kharinaev et al. - Investigating the impact of quantization methods on the safety and reliability of large language models (IEEE Access version of record).pdf` (alternate, cross-check only: `Sources/[10.2] ... (arXiv v2).pdf`)
- **Publication status:** Published IEEE Access version of record: vol. 14, 2026; received 29 May 2026, accepted 12 June 2026, date of publication 15 June 2026, current version 30 June 2026 (extract line 1); DOI 10.1109/ACCESS.2026.3703899 matches (line 2); pp. 96771–96793 (lines 63, 1848); Creative Commons Attribution 4.0 license (line 60). arXiv:2502.15799 confirmed on the cover of the alternate PDF ("arXiv:2502.15799v2 [cs.CR] 29 Jun 2025").
- **Problem & context:** Quantized deployment is expanding, but safety evaluation of quantized models has leaned on older architectures, PTQ-only techniques, limited bit ranges, and outdated datasets no longer challenging enough for modern models, judged either by multiple-choice tests or an LLM-as-a-Judge paradigm that may misalign with human judgment (lines 76–83). The authors argue closed-book, single-choice evaluation obscures efficiency-vs-risk trade-offs and blocks fair comparison across deployment contexts (lines 53–70).
- **Approach:** The paper introduces OpenSafetyMini, a human-curated English open-ended safety dataset of 1,067 prompts spanning 8 risk categories (lines 126–127, 149–150), built by scoring candidate prompts from XSAFETY [37] with a GPT-4o "deflection score" (10 independent evaluations per prompt), removing all prompts with average deflection score below 50, then manually reviewing survivors (lines 276–282; prompt in App. G1, lines 1350–1366). Retained prompts were labelled by five independent annotators by majority vote with a conservative tie-break hierarchy unsafe > ambiguous > safe > error (lines 125–126, 1394–1400). They release 4,268 unique labelled question-answer pairs from >20,000 human evaluations (conclusion: 21,328 evaluations; lines 24, 153, 421–422, 985–986). Evaluation covers 66 quantized variants — 4 PTQ methods × 5 modern LLMs across 3 precision levels plus 2 QAT methods (LSQ-variant STE, QLoRA) on 1 model across 3 precision settings (lines 25–26, 139–141, 155–156): PTQ = AWQ, QUIK, AQLM, QUIP# (lines 393–403); models = LLaMA 3.1 8B Instruct, Mistral 7B Instruct v0.2, Qwen2.5 at 1.5B/7B/14B, plus an abliterated LLaMA 3.1 8B as worst-case reference (lines 380–389). Four benchmarks: XSafety + OpenSafetyMini open-ended (human panels of five linguistic experts and Gemma 2 27B LLM-as-a-Judge with reported 92% agreement with humans, lines 413–440), SafetyBench multiple-choice (5-shot, logit-based answer selection, lines 442–458), and HotPotQA factuality under RAG scored by AlignScore and rule-based In-Accuracy (lines 462–484).
- **Key findings:** Both PTQ and QAT degrade safety alignment, with weight-updating QAT degrading it more severely than PTQ — extreme QAT settings fall below even the abliterated LLaMA baseline whose refusal directions were removed (Contributions, lines 157–171; also lines 631–636). Closed-ended benchmarks systematically misrepresent risk: gaps of up to 35 percentage points are invisible to multiple-choice testing (lines 172–177); concretely, the abliterated LLaMA scores identically to FP16 on SafetyBench (73.90 vs 73.90) despite a 32-percentage-point gap in human-rated safe behaviour (58.67% vs 90.59%; lines 720–726), and STE at 2-bit scores 51.50 on SafetyBench while producing safe responses only 16.25% of the time (a 35-point discrepancy), with STE 4-bit at 71.40 vs 53% safe (lines 729–735). Human evaluation anchor: 4-bit QUIK is statistically indistinguishable from FP16 (safe rate 90.75% vs 90.59%, Table 6 discussion, lines 516–524) while 2-bit QUIP# doubles unsafe responses and yields 6.28% unevaluable error outputs (lines 525–529). Headline discrimination case: on XSafety, LLaMA 4-bit QUIK (93.21%) vs AWQ (93.18%) differ by 0.03 points and are statistically indistinguishable, yet on OpenSafetyMini they split to 93.25% vs 89.50% — a 3.75-point gap invisible to XSafety (lines 651–657). No single method wins everywhere (Mistral inverts the LLaMA 4-bit ranking: QUIK degrades significantly while AWQ and QUIP# do not differ from baseline, lines 556–562); precision-specialised methods excel at their target precision (QUIK/AWQ at 4-bit, AQLM/QUIP# at 2-bit; abstract lines 29–33, contributions lines 178–185). Statistics: 95% bootstrap CIs over 1,067 binary labels (10,000 resamples; typical width ±0.009 to ±0.010, ±0.013 for 2-bit QUIP# on larger Qwen, lines 760–767); McNemar χ² with continuity correction across 31 PTQ comparisons with Holm–Bonferroni correction at α = 0.05 finds 25 of 31 significant after correction (lines 768–792). Judge stability: five runs at temperature 0.5 over 1,068 prompts vary by ≤0.47 pp (FP16) and ≤1.44 pp (abliterated) (Table 8 discussion, lines 541–569). Human inter-rater agreement: Krippendorff α = 0.623 (lines 417–418).
- **Cited in the report at:** Verified against the report-of-record builder this session: the FYP report v5 cites this paper as reference **[12]** (not [10]) in Chapter 2 > 2.4 Quantization and Behavioural Effects (`scripts/build_fyp_report_v5.js` line 446, "Kharinaev et al. [12] evaluate 66 quantized variants…") and in Chapter 7 > 7.7 Positioning Against Prior Work (same file, line 1004); bibliography entry at line 1080. No occurrence exists in Chapter 1 §1.1.
- **Limitations & caveats:** (i) The dataset filter rests on GPT-estimated deflection scores; the authors concede possible valuable low-deflection prompts were excluded and that a GPT-4o calibration bias cannot be fully ruled out (lines 370–372, 940–948). (ii) QAT is evaluated on one model only, following prior work (lines 385–386), without safety-aware training objectives — the authors explicitly warn against reading their result as a blanket indictment of QAT (lines 890–901). (iii) Internal inconsistency in the published text: Limitations bullet 2 states the evaluation "currently focuses on post-training quantization" (lines 949–951), contradicting the paper's own QAT experiments (§IV-B, §V QAT results); treat that bullet as a stale remnant. (iv) Minor editorial slips: intro roadmap skips Section VIII (lines 190–192) and Index Terms include "SMILES" (line 34). (v) All data tables' bodies are absent from the text extract, so table-only values (Tables 1–15) remain unverified from this pass. (vi) The extract never states which three precision levels the PTQ grid spans — see discrepancy check.
- **Relevance to this FYP:** This is the closest published analogue to the FYP's core question (does compression change safety?) and its strongest methodological warning supports our design choices: multiple-choice-style or knowledge-probing signals can completely miss behavioural safety loss (their 73.90-vs-73.90 case), which is why our study measures behaviour (HarmBench ASR, XSTest over-refusal) alongside capability anchors rather than safety knowledge alone; their per-model ranking inversions (QUIK/AWQ flip between LLaMA and Mistral) directly motivate our per-pair, per-family analysis instead of assuming one quantizer verdict. Their setting differs materially from ours — different models (LLaMA-3.1-8B scale up to Qwen2.5-14B), different quantizers (AWQ/QUIK/AQLM/QUIP#, not BitsAndBytes NF4/LLM.int8 on-the-fly loading), a generative-judge scorer rather than our HarmBench classifier primary — so none of their numbers transfer to ours; the citable content is directional (PTQ can degrade open-ended safety; closed-ended tests underestimate it; no universal winner).

## Discrepancy check vs current summary

Current summary = `PAPER_SUMMARIES.md` lines 89–96. Verdict per bullet:

- **Bibliography entry — CONFIRMED.** Authors "ARTEM KHARINAEV, VIKTOR MOSKVORETSKII, EGOR SHVETSOV, …" (extract lines 7–8); "VOLUME 14, 2026" (line 45); first page 96771 (line 63), last page 96793 (line 1848); DOI 10.1109/ACCESS.2026.3703899 (line 2). The trailing "arXiv:2502.15799" does not appear in the VoR extract but is confirmed on the alternate PDF's cover stamp ("arXiv:2502.15799v2 [cs.CR] 29 Jun 2025"), consistent with its stated role as cross-check.
- **Repo PDF — CONFIRMED.** Both files present in `Sources/` (verified by directory listing this session); the primary filename matches the bullet.
- **Publication status — CONFIRMED.** Line 1 gives publication date 15 June 2026; line 2 the DOI; line 63 first page 96771; line 60 CC-BY 4.0. Additional detail available: acceptance 12 June 2026 and current-version date 30 June 2026 (line 1).
- **Problem & context — CONFIRMED.** Lines 76–83 state prior work used older architectures, PTQ-only techniques, limited bit ranges, and "outdated datasets that are insufficiently challenging for modern models", and rely on "multiple-choice assessments or the LLM-as-a-Judge paradigm … may not align well with human judgment". The bullet paraphrases this accurately.
- **Approach — CORRECTED in one element, otherwise CONFIRMED.**
  - CONFIRMED: OpenSafetyMini 1,067 prompts / 8 risk categories (lines 126–127); deflection-score filtering with GPT-4o, 10 evaluations each, removing average score below 50 (lines 276–279); five independent annotators, majority vote (125–126); 4,268 released pairs (24, 153, 421–422); PTQ methods AWQ/QUIK/AQLM/QUIP# (393–403); models LLaMA 3.1 8B Instruct, Mistral 7B Instruct v0.2, Qwen2.5 at 1.5B/7B/14B, QAT on LLaMA 3.1 8B (380–386); four benchmarks incl. SafetyBench and factuality scoring (406–410, 442–484); Gemma 2 27B judge at 92% human agreement (436–440).
  - CORRECTED: "at 8-, 4-, and 2-bit" is not supported by the extract. The paper says PTQ applied "across 3 precision levels" (line 139) and concludes "3 bit ranges" (line 991), but never names the three levels; both methods sections say the study targets **4-bit and 2-bit weight-only quantization** (lines 210–211: "specifically targeting 4-bit and 2-bit"; lines 1070–1075: focus on 4/2-bit "due to the fact that 8-bit quantization is more widely studied"); the results narrative discusses only int4/int2 (Figures 4–7, lines 500–509). "8-bit" appears only as a capability of AWQ/QUIK methods (lines 395, 398) and background (line 1110). The summary should say "across three precision levels (the evaluated results reported are 4-bit and 2-bit)" rather than asserting 8-bit runs. Note also that the implicit decomposition 4×5×3 + 2×1×3 = 66 is this auditor's arithmetic from lines 139–141, not a statement in the paper.
- **Key findings — CONFIRMED (all quantitative claims match).** QAT-worse-than-PTQ and below-abliterated-baseline: lines 157–171, 631–636; abliteration definition "removing 'refusal directions'" line 388. Up-to-35-point closed/open gap: lines 172–177, 729–732. QUIK 93.21% vs AWQ 93.18% on XSafety (0.03 pts) vs 93.25% vs 89.50% on OpenSafetyMini (3.75 pts), LLaMA 4-bit: lines 653–657 exactly. No-universal-winner and precision-specialised-method claims: lines 178–185 and 556–562. Krippendorff α 0.623: lines 417–418. One nuance worth adding, not an error: the α figure comes from §IV-C (human-evaluation protocol), and the QUIK/AWQ example belongs to §V-A-2-c "Revealing differences XSafety misses" (the bullet's "(§V)" is fine at section granularity).
- **Cited in the report at — CORRECTED.** The current value "Chapter 1: Introduction > 1.1 Background and Motivation" does not match the report of record. In `scripts/build_fyp_report_v5.js`, "[12]" (this paper) occurs only at line 446 (under H2 "2.4 Quantization and Behavioural Effects", line 443, within H1 "Chapter 2: Literature Review", line 428) and line 1004 (under H2 "7.7 Positioning Against Prior Work", line 1003); grep finds zero occurrences of "[12]" or "Kharinaev" between lines 377–420 (Chapter 1). Also note the in-report bracket number is [12], so any claim-lock keyed on "[10]" pointing at this paper would be checking a different source ([10] in v5 = Jin et al., comprehensive evaluation of quantization strategies, refs array entry 10 at line 1078).

Nothing else in the current summary was found wrong or over-reaching beyond the two items above.

## Self-test

Total extract length: read tool reports **1850 lines**, EOF confirmed at line 1850 after sequential chunked reads covering offsets 1–1850 with no skipped range (`wc -l` prints 1849 newline characters; the final line is unterminated). Every number quoted above was re-derived from the extract with the greps below (run this session, file `ref_10.txt`):

```text
$ grep -nE "Received 29 May|10\.1109/ACCESS\.2026\.3703899|Creative Commons Attribution 4\.0|^96771$|^96793$" ref_10.txt
1:Received 29 May 2026, accepted 12 June 2026, date of publication 15 June 2026, ...
2:Digital Object Identifier 10.1109/ACCESS.2026.3703899
60:2026 The Authors. This work is licensed under a Creative Commons Attribution 4.0 License.
63:96771
1848:96793

$ grep -nE "1,067|4,268|21,328|20,000|66 quantized" ref_10.txt        # excerpt
23:...we introduce OpenSafetyMini, a humancurated safety dataset with 1,067 challenging questions...
24:release human safety evaluations for four LLMs ..., totaling 4,268 annotated
25:question-answer pairs ... By assessing 66 quantized variants
421:in a total of over 20,000 human evaluations, or 4,268 unique
985:...collected 21,328 human evaluations of
764:replacement) over the 1,067 binary safe/unsafe labels per

$ grep -nE "92% agreement|0\.623|five linguistic experts" ref_10.txt
413:Human evaluation was conducted by five linguistic experts
418:measured using Krippendorff α, yielded a score of 0.623.
439:a high 92% agreement with human judgments,

$ grep -nE "93\.21|93\.18|0\.03 points|93\.25|89\.50|3\.75 points" ref_10.txt
654:QUIK (93.21%) and AWQ (93.18%) are separated by only
655:0.03 points—statistically indistinguishable (see Section V-E).
656:On OpenSafetyMini, QUIK holds at 93.25% while AWQ drops
657:to 89.50%, a gap of 3.75 points.

$ grep -nE "37%—five times|90\.75%|90\.59%|6\.28% errors" ref_10.txt
519:to 37%—five times the FP16 rate.
521:(90.75%) is within 0.5 percentage points of the FP16 baseline
524:(90.59%), and it actually produces fewer errors and ambiguous
527:6.28% errors—outputs so degraded they cannot be evaluated at

$ grep -nE "58\.67%|73\.90|32 percentage|51\.50|16\.25% of the time|35-point|71\.40|only 53%|35 percentage" ref_10.txt
176:safety gaps of up to 35 percentage points invisible to
722:stripped of its refusal mechanisms and rated 58.67% safe by
724:to FP16 on SafetyBench (73.90 vs. 73.90). A 32 percentagepoint gap ...
730:scores 51.50 on SafetyBench yet produces safe responses only
731:16.25% of the time in open-ended evaluation—a 35-point
732:discrepancy. STE at 4-bit scores 71.40 on SafetyBench while
733:generating safe responses only 53% of the time.

$ grep -nE "1,068 prompts|0\.47 percentage|1\.44 points|17\.0%|16\.0%" ref_10.txt
542:we ran five independent evaluations of 1,068 prompts on
566:highly stable. Safety rates vary by at most 0.47 percentage
567:points across the five FP16 runs and 1.44 points across the
572:Topic (17.0%) and Goal Hijacking (16.0%), ...

$ grep -nE "10,000 resamples|±0\.009 to ±0\.010|±0\.013|31 comparisons|Twenty-five of the 31|Holm Bonferroni|α = 0\.05" ref_10.txt
763:bootstrap confidence intervals (10,000 resamples with
766:typically ±0.009 to ±0.010—and widen as safety rates fall,
767:reaching ±0.013 for QUIP# at 2-bit on larger Qwen models.
775:continuity correction. We ran 31 comparisons across all PTQ
777:Holm Bonferroni procedure (α = 0.05).
789:Twenty-five of the 31 comparisons are statistically significant

$ grep -nE "14\.7 points|1\.3 point|5\.3 point|8\.0 point|93\.75% on|93\.06% on|6\.25 points" ref_10.txt
646:OpenSafetyMini records 14.7 points on LLaMA at 4-bit.
659:alone. Similarly, Mistral QUIP# 4-bit records only a 1.3 point
660:drop on XSafety, but a 5.3 point drop on OpenSafetyMini,
669:safety-aligned models: LLaMA FP16 scores 93.75% on
670:XSafety and 93.06% on OpenSafetyMini, a negligible
671:difference. For Mistral FP16 the gap widens as 6.25 points
698:XSafety records an 8.0 point drop from baseline while

$ grep -nE "deflection score.?below 50|less than 50 were filtered|safety-aligned LLaMA 3.1 8B|3 precision levels|3 precision settings|3 bit ranges" ref_10.txt   # plus context sed
141:3 precision settings, supplemented by human evaluations        [with line 139 "...across 3 precision"]
279:below 50. The prompt template is reflected in Appendix G1.
381:a diverse set of models: the safety-aligned LLaMA 3.1 8B
991:and 3 bit ranges. Our findings reveal that quantized models
1365:Promts with deflect rate less than 50 were filtered
[sed 989,991]: "66 settings on 4 distinct benchmarks, encompassing 5 modern / LLMs, 4 state-of-the-art PTQ techniques, 2 QAT techniques / and 3 bit ranges."

$ grep -nE "specifically targeting 4-bit and 2-bit|8-bit|Int-8" ref_10.txt    # basis for the CORRECTED verdict
210:on 4-bit
211:and 2-bit weight-only quantization [1], [2], [3].
395:quantization, enabling efficient 8-bit and 4-bit deployment      [AWQ capability]
398:space, supporting 8-bit and 4-bit formats.                       [QUIK capability]
1073:focus is due to the fact that 8-bit quantization is more widely
1110:precision and quantizing weights-only to Int-8, Int-4, and

$ grep -nE "28,000|11,435|113k|A40 GPUs|T4 GPU|80 total GPU-hours|40 total GPU-hours|200 samples for each of the 14" ref_10.txt
1031:In total, the XSAFETY benchmark includes 28,000
1032:sentences with 200 samples for each of the 14 safety scenarios
1037:safety of LLMs, comprising 11,435 diverse multiple-choice
1048:It contains 113k Wikipedia-based question-answer pairs with
1316:For LLM Inference - 4 x NVIDIA A40 GPUs (48GB VRAM
1317:each). For AlignScore Evaluation - 1 x NVIDIA T4 GPU
1326:LLM workflows: ≈ 80 total GPU-hours (A40). AlignScore
1327:evaluation: ≈ 40 total GPU-hours (T4)
```

Report-side verification (not the extract): `grep -n '\[12\]' scripts/build_fyp_report_v5.js` → lines 446, 1004 only; `awk` over the `refs` array (lines 1068–1092) shows entry 12 = Kharinaev et al. and entry 10 = Jin et al.; Chapter 1 spans lines 375–426 and contains no `[12]`/`Kharinaev` hit.

Not verified from this extract (stated explicitly, per the absence-claim discipline): the numeric contents of Tables 1–15 (bodies absent from pdftotext output), and which three precision levels constitute the "3 bit ranges"/"3 precision levels" grid — the extract names only 4-bit and 2-bit as studied weight-only targets. PROJECT_LOG.md was deliberately not touched, per this task's hard rule that no repo file other than the deep-read output may be modified; the delegating agent should log this deep read.
