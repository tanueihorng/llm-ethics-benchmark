# Chapter 3 design rationale: what each choice rests on

- **Checked file:** `/Users/tanueihorng/Downloads/final_report.docx`, SHA-256 `d88b2f66…85f6eb` (saved 2026-09-28 02:09). The report was **not edited**.
- **Method:** I listed each design choice stated in Chapter 3, found the sentence in Chapters 1–2 that justifies it, and checked the cited claim against the local PDF in `Sources/` (pdftotext extracts; line references are to those extracts). Report numbers `[n]` are the report's own reference numbers, not the `Sources/` file prefixes.
- **Verdicts:** **Supported** = the source says it. **Own reason** = your design reasoning or a practical constraint; no citation is needed if it is worded as yours. **Partial** = the reason is sound but the cited text only covers part of it, or a mismatch weakens it. **Missing** = Chapter 3 states the choice and no reason is given, although a source exists.
- **Second pass (same day):** the web sources [1], [2], [4]–[6], Llama Guard [15] and the statistics citations [55]–[68] were then checked; see "Remaining citations" at the end.

> **Status 2026-10-08 (read this first):** This audit was run on the Downloads copy at SHA-256 `d88b2f66…`. Codex then applied its findings natively in Word (PROJECT_LOG rows 01:23 and 01:43): the current working copy is `/Users/tanueihorng/Downloads/final_report.docx` (`08a05520…`), with backups `final_report_2026-10-08.docx` and `final_report_2026-10-08_before_rationale_013542.docx`. Evidence for the edits is in `tmp/report_audit/verification_123_20261008/` (`RATIONALE_NATIVE_PASS.md`, `rationale_edits.json`). Codex also opened the McKinsey PDF [2] and confirmed the figures, the Llama clause and the authors (row 01:16), which closes that open item. Section references were corrected: Statistical Analysis is **§3.5**, not §3.4. The builder is unsynchronised; never rebuild over the Downloads copy. Live follow-up: PROJECT_LOG T82.

## Summary table

| # | Chapter 3 choice | Reason given now (where) | Verdict | What to do before writing Ch3 |
|---|---|---|---|---|
| 1 | Compact models, 1.7–7.2B | Adoption of open-weight models [1], [2]; "compact" illustrated with Qwen3.8-27B [5], [6] (§1.1); V100/access constraints (§3.2) | **Partial** | State the adoption motivation as yours. The 27B example is about 4–16× larger than your models, so it does not illustrate *your* scale. Add the research reason: Li et al. [17] |
| 2 | 4-bit as the primary precision | QLoRA memory example [23] (Ch2 opening); capability retention [9], [16], [17] | **Missing as a reason** | Dettmers & Zettlemoyer [9] already cited; use it as the reason |
| 3 | NF4 via BitsAndBytes, not GPTQ/AWQ | No calibration set; quantizer never tuned on benchmark prompts [23], [26] (§2.2 ¶2) | **Supported (by inference) + own reason** | Fine. Optionally phrase as "computed from the weights alone" |
| 4 | INT8 = LLM.int8, analysed separately | Mixed-precision outlier method, so not "8-bit NF4" [26] (§2.1) | **Supported** | None |
| 5 | Same checkpoint revision for both conditions | Published quantized files may differ in conversion, tokenizer, template, revision (§2.3 ¶5) | **Own reason** | Fine as worded ("may differ") |
| 6 | Measure harmful compliance + capability + over-refusal together | SORRY-Bench "recognises that a model which refuses everything would be useless" [51] (§2.4 ¶6) | **Supported, understated** | SORRY-Bench explicitly recommends all three. Say so |
| 7 | HarmBench classifier as primary scorer | Behaviour-completion criterion [19] (§2.4 ¶2); keyword proxy weakness [50] | **Supported** | Optional: HarmBench reports 93.2% classifier agreement with human labels |
| 8 | DirectRequest only (no attacks) | Threat-model contrast with JailbreakBench [48] (§2.4 ¶4); scope (§1.4) | **Own reason, supported** | Optional: cite [19] for the name "Direct Request" |
| 9 | Greedy decoding, 512 new tokens | None in Ch3 §3.1 | **Missing** | HarmBench [19] uses greedy decoding and standardises 512 tokens. Cite it |
| 10 | XSTest; keyword rule primary + 3-class semantic judge as sensitivity | XSTest targets exaggerated refusal; compact enough for paired/manual inspection [20], [52] (§2.4 ¶6); 3-class labels [20] (§2.4 ¶7) | **Supported, scorer choice uncited** | XSTest [20] itself uses both scorers. Cite it for the scorer design |
| 11 | MMLU (6-subject subset) + ARC-Challenge | Capability context [21], [22] (§2.4 ¶8); subset for tractability (§1.4) | **Supported + own reason** | None (SORRY-Bench also names MMLU; see #6) |
| 12 | Human validation of both scorers | Evaluator studies compare with human labels [53], [54]; annotators disagree, α = 0.623 [18] (§2.5) | **Supported; one wording fix** | Do-Not-Answer wording; see #12 below |
| 13 | gpt-4o API judge + Llama Guard as sensitivity judges | LLM-judge agreement and biases [3]; reproducibility concern (§2.5 ¶2) | **Supported** | Optional: HarmBench's own warning about closed-source judges |
| 14 | Paired bootstrap, exact McNemar, BH, TOST, kappa, power | Method citations [55]–[67] (§3.5) | **Identification citations** | Only [58] spot-checked: supported |
| 15 | Scope narrower than attack studies | Prepared checkpoints [10], [37]; "jailbreak benchmarks" [47], [48] (§2.6 ¶4) | **Partial** | [47] is an analysis, not a benchmark |

## Details and evidence

### 1. Compact models — Partial
- **Your own motivation is legitimate.** Ch3 §3.2 already gives the practical constraints ("reproducible access, V100 compatibility, and the availability of instruction-tuned checkpoints"). Those are your constraints and need no citation.
- **Mismatch:** §1.1 ¶2 argues that "a comparatively compact open-weight model can approach the performance of a leading proprietary system" using Qwen3.8-27B [5], [6]. Your largest model is 7.2B. The paragraph supports "open-weight models are capable", not "models of 1.7–7B are worth studying". I did not check [5] or [6] (web sources).
- **Research reason available, verified:** Li et al. [17], Table 1 summary (09.txt line 80): trustworthiness recommendation "W8, W8A8, KV8 (< 7B); W4, W4A8, KV4 (≥ 7B)", within 2% accuracy loss (line 83). This means models below 7B were the ones where 4-bit was *not* recommended for trustworthiness, which is exactly your range. Caveat for wording: this is Li et al.'s recommendation for the quantizers and tasks they tested. Use it as "small models have been found more sensitive", not as a prediction for NF4.

### 2. Why 4-bit — Missing as a reason
- [9] abstract (07.txt lines 45–47): "4-bit precision is almost universally optimal for total model bits and zero-shot accuracy." Also §"Promising directions": "4-bit precision is currently bit-by-bit the most efficient precision" (line 529).
- Ch2 currently cites [9] only for capability retention. It can carry "4-bit is the efficiency sweet spot, so it is the setting people deploy". Caveat: [9] is about zero-shot accuracy per bit, not safety. "Most people deploy 4-bit" would itself be an uncited adoption claim; avoid it or source it.

### 3. NF4 load-time, no calibration — Supported by inference
- QLoRA [23] (32.txt line 23/79): NF4 is "information theoretically optimal for normally distributed weights", a fixed data type; scales are block-wise absmax.
- LLM.int8 [26] (33.txt lines 21, 60): a 175B checkpoint "can be loaded, converted to Int8, and used immediately for inference".
- Neither paper says the words "no calibration set". §2.2 ¶2's statement is a correct reading of the methods, not a quote. "Never tuned on the benchmark prompts" is your own reasoning and is fine.

### 6. Three-outcome design — SORRY-Bench supports all three parts
- SORRY-Bench [51], limitations discussion (42.txt lines 932–941): a model that always refuses would be "the 'safest'" on the benchmark but "not useful at all"; the authors "suggest putting together SORRY-Bench and language model utility benchmarks (e.g., Chatbot Arena…, MMLU…)" and "recommend users evaluate model over-refusal (e.g., on OR-Bench… and XSTest…)".
- Current §2.4 ¶6 says only that a model refusing everything "would be useless". §6.2's "stress that refusals must be judged against usefulness [20], [51]" is similarly loose. This is your strongest single citation for the overall design.

### 7. HarmBench classifier — Supported
- Criterion (24.1.txt line 1137 onward): instances must be "unambiguous and non-minimal"; criteria are applied in manual labelling and in classifier prompts (lines 1153–1154).
- Validation: test classifier "93.2%" agreement with human labels vs 88.6% for the validation classifier (line 1186); Table 4 average 93.19 (line 1131).
- Closed-source judges: HarmBench says using them "is far from ideal, because models can change under the hood without warning" (lines 466–470). This directly supports keeping the open classifier primary and gpt-4o as a sensitivity judge.

### 8. Direct Request — Supported
- HarmBench appendix (line 1373): "Direct Request: This baseline uses the behavior strings themselves as test cases." Ch1 §1.4 uses the name "HarmBench DirectRequest" without a citation.

### 9. Greedy decoding and 512 tokens — Missing (strongest gap)
- Ch3 §3.1 ¶2 states "greedy decoding with temperature 0.0 and a maximum of 512 new tokens" with no reason.
- HarmBench [19]: target models generate completions "deterministically using greedy decoding" (lines 187–189; also line 462). On length: "the number of tokens generated during evaluation can have a drastic effect on ASR computed by substring matching metrics", up to 30%, so they "standardize this parameter to N = 512" (lines 256–271).
- Caveat: the 30% sensitivity is reported for substring-matching metrics, not their classifier. Cite it for "512 follows HarmBench's standard", not for "512 makes the classifier stable".
- The repo config already records this reasoning: `configs/tc1_512.yaml` line 117, "512-token generation budget (HarmBench's reference default)". The report does not.

### 10. XSTest scorers — Supported, but the scorer choice is uncited
- XSTest [20]: 250 safe prompts across ten types, 200 unsafe contrast prompts (18.txt lines 28–30, 222).
- §4.5 Automated Model Evaluation (lines 346–356): XSTest itself uses (1) a string-match method that "cannot distinguish between partial and full refusal" and (2) GPT-4 classifying full compliance / full refusal / partial refusal; the automated methods "qualitatively recover the results of manual evaluation".
- So your keyword-primary + three-class-sensitivity design follows XSTest's own two methods. Ch2 does not say this.

### 11. MMLU and ARC — Supported
- MMLU [21]: 57 tasks/subjects (27.txt lines 20, 58). ARC [22]: Challenge Set = questions answered incorrectly by both a retrieval-based and a word co-occurrence algorithm (28.txt lines 16–22).

### 12. Human validation — Supported, one wording fix
- Kharinaev [18]: Krippendorff α "0.623" (10.1.txt line 299); five annotators (line 962); 66 quantized variants (line 33).
- Do-Not-Answer [53]: Longformer "achieves comparable overall results with GPT-4" (45.txt line 475); size "≤ 600M" (line 92). §2.5 ¶1 says it is comparable "to human and GPT-4 evaluation". The paper claims parity with GPT-4, with human labels as the reference. **Fix the wording.**
- MT-Bench [3]: strong LLM judges reach "over 80% agreement, the same level of agreement between humans" (47.txt line 30).

### 14. Statistics — spot check only
- Fagerland [58]: describes the exact conditional test as "overly conservative" (fag.txt lines 15, 274). §3.5's "can be conservative" is supported.
- The other method citations identify standard procedures and were not re-read in this pass.

### 15. Scope paragraph — Partial
- §2.6 ¶4: "jailbreak benchmarks evaluate adaptive attack procedures [47], [48]". [47] (Wei et al., *Jailbroken*) analyses why safety training fails (competing objectives, mismatched generalization). It is not a benchmark. [48] (JailbreakBench) is. Split the two.

## Suggested order before writing Chapter 3
1. **#9:** add the HarmBench reason for greedy decoding and 512 tokens.
2. **#6:** strengthen the SORRY-Bench sentence. It anchors the whole three-outcome design.
3. **#10:** cite XSTest's own two automated methods for your scorer design.
4. **#2 and #1:** give the reason for 4-bit ([9]) and for small models ([17]). Reword the 27B example or move it so it does not stand in for your model scale.
5. **#12 and #15:** the two wording fixes (Do-Not-Answer, *Jailbroken*).

Items 3, 4, 5, 7, 8, 11, 13 and 14 need no change for Chapter 3 to be accurate.

## Remaining citations (second pass, 2026-10-08)

### Web sources (Chapter 1)
| Ref | Claim in report | Checked against | Verdict |
|---|---|---|---|
| [1] AI Index 2025 | Closed-vs-open lead on Chatbot Arena fell from 8.04% (Jan 2024) to 1.70% (Feb 2025) | Stanford HAI 2025 report, Technical Performance page (exact sentence) | **Supported** |
| [1] AI Index 2025 | Strong performance across a wider range of model sizes | Same page: smallest model above 60% MMLU went from PaLM (540B, 2022) to Phi-3-mini (3.8B, 2024) | **Supported** (the Phi-3-mini fact is the more specific wording) |
| [4] AI Index 2026 | Gap widened again to 3.3% as of March 2026 | HAI Chapter 2 PDF, highlight 3: "As of March 2026, the top closed model leads the top open model by 3.3%, up from 0.5% in August 2024" | **Supported**. arXiv:2606.15708 title and first authors match the reference |
| [5] Qwen3.8-27B card | Downloadable, 27B parameters, Apache 2.0 | Hugging Face card: "Number of Parameters: 27B", licence apache-2.0, Aug 2026 | **Supported**. The card describes a vision-language model; the report does not say otherwise |
| [6] Artificial Analysis | Index v4.3.2 aggregates ten evaluations; Qwen3.8-27B (highest reasoning) 34; GPT-5.5 (same setting) 38 | artificialanalysis.ai, model pages: Qwen3.8 27B (Xhigh) 34; GPT-5.5 (Xhigh) 38; v4.3.2 lists 10 evaluations | **Supported as of 2026-10-08.** It is a live leaderboard, so the scores can change; keep the access date |
| [2] McKinsey/Mozilla | 700+ leaders and developers, 41 countries; 63% of organisations, 72% in tech; 60% lower implementation costs; category includes Llama | The primary page and PDF **timed out twice**. Search-result summaries corroborate 700+/41/63%/72%/60% and the April 2025 date | **Not verified from the primary source.** The Llama clause and the author list are unchecked. Re-open the PDF yourself before submission |

The Chapter 1 sources are accurate. The issue is the one in #1 above: they show that open models are capable, and the 27B example is larger than any model you test.

### Llama Guard [15]
- "Classifies both a user's request and a model's response against a taxonomy of safety risks" (Ch1) and "an open-weight safety classifier" (Ch3): the abstract describes an "input-output safeguard model" with "a safety risk taxonomy" for "both prompt and response classification", and the authors "publicly release" it. **Supported.**

### Statistics [55]–[68]
| Ref | Claim | Verdict |
|---|---|---|
| [56] Efron & Tibshirani 1985 | Percentile bootstrap interval | **Supported**: "The percentile method interval is just the interval between the 100·α and 100·(1−α) percentiles of the bootstrap distribution" |
| [55] Efron & Tibshirani 1993 (book) | Bootstrap | Identification only. Local copy is a 451-page scan; not re-read |
| [57] McNemar 1947 | Exact McNemar test | Identification only; no local copy. The test is McNemar's; "exact" is the binomial version, which [58] discusses |
| [58] Fagerland et al. | Exact conditional test "can be conservative" | **Supported** ("overly conservative") |
| [59] Benjamini & Hochberg | Controls the expected proportion of false discoveries, under assumptions | **Supported**: the summary defines FDR as "the expected proportion of falsely rejected hypotheses" and proves control "for independent test statistics". "Under its assumptions" is the right hedge |
| [60] Cohen 1960 | Kappa = agreement corrected for chance | **Supported** via the local 1986 Citation Classic commentary ("agreement corrected for chance"); the 1960 paper itself is not local |
| [61] Connor 1987, [62] SAS | Normal approximation for paired-proportion power | **Supported**: SAS PROC POWER `PAIREDFREQ`, `TEST=MCNEMAR METHOD=CONNOR` "uses an approximation from Connor (1987)"; the cited pages 7870–7873 cover PAIREDFREQ. Connor itself is not local |
| [63] Card et al. | Same sentence as [61], [62] | **Partial**: Card et al. argue NLP experiments are often underpowered and estimate power for paired classifier comparisons (McNemar) **by simulation**, not by a normal approximation. Cite it for "power matters / depends on agreement rate", not for the approximation |
| [64] Artstein & Poesio | Kappa depends on prevalence | **Supported**: §3 "Bias and Prevalence", the prevalence problem |
| [65] Landis & Koch | Verbal categories are not universal | **Supported**: they give the bands but call them "clearly arbitrary" |
| [66] Schuirmann, [67] Lakens | TOST; margin fixed beforehand | **Supported**: Lakens recommends prespecifying the equivalence bounds; Schuirmann defines TOST |
| [68] Sainz et al. | Contamination may inflate capability scores | **Supported**: "Contamination causes an overestimation of the performance" |

### One Chapter 3 wording point (not a citation error)
§3.5 says equivalence tests use "a practical margin fixed before the test is run". That matches `results_512/analysis/capability_equivalence.json` (margin 0.05 "fixed before computation"). The same file also says "The analysis as a whole is post hoc: it was added after the INT8 results were known". Chapters 4, 6 and 7 say this, but §3.5 does not. When you rewrite Chapter 3, state there that the equivalence analysis is post hoc, so the methods chapter does not read as if it were planned in advance.

### Updated to-do list from the second pass
- [63]: narrow its role in the power sentence (or move it to the sentence saying non-significance is not evidence of no effect).
- [2]: open the McKinsey PDF yourself and confirm the Llama clause and the author list.
- §3.5: add "post hoc" to the equivalence sentence.
