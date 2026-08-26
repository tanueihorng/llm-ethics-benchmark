# Full Read — [9] Li et al., Evaluating Quantized LLMs (ICML 2024)

Full every-line read of the pdftotext extract `docs/audits/2026-08-25_interim_progress_citation_check/tmp/extract/ref_9.txt` (read sequentially in four offset/limit passes; see Self-test). All anchors below are extract line numbers in that file, written `L###`.

## Coverage table

| Paper section | Extract lines (approx) | Status (READ-FULL / SKIMMED / UNREADABLE) | Note |
|---|---|---|---|
| Title, authors, affiliations/correspondence footnote | 1–5, 64–67 | READ-FULL | Author list and order match the bibliography entry for [9]. |
| Abstract | 6–27 | READ-FULL | States scope: PTQ on Weight/Activation/KV Cache, 11 families, 125M–180B, five task types. |
| §1 Introduction + Table 1 (key knowledge & bit-width recommendations) | 28–158 (Table 1 at 81–119) | READ-FULL | Table 1 carries the paper's headline claims and the per-task bit-width table. |
| §2.1 Quantization (Eq. 1–2, granularity choices, Figure 1) | 160–171, 304–447 | READ-FULL | Two-column extraction interleaves equation symbols and figure fragments; content recoverable. |
| §2.2 Benchmarks and Models (Table 2) | 170–303, 449–469 | READ-FULL | Table 2 cells reflowed to one column; all benchmark sizes and family lists recoverable. |
| §2.3 Statistical Analysis (AbsMax / Std / Kurtosis) | 489–503 | READ-FULL | — |
| §3 Basic NLP tasks: §3.1 setups | 512–523 | READ-FULL | — |
| §3.2 Effects of quantization on three tensor types (Figs 2–3, Tables 3–4) | 424–510, 525–615, 741–1017 | READ-FULL | Figure regions are bare axis numbers/legend tokens from column interleave; all prose and table values readable. |
| §3.3 Effects on different LLMs (+ Fig. 13 Spearman matrices) | 710–736, 2889–2989, 2500–2790 (Fig 13 data) | READ-FULL | Spearman correlation matrices appear as reflowed numeric rows; structure described in caption L2790. |
| §3.4 Effects on different tasks (+ Tables 8–9 PPL analysis, B.5–B.6 method results) | 695–708, 1079–1086, 2990–3086 | READ-FULL | Includes the accuracy-gain-after-quantization analysis (Mistral-7B RACE +3.5% under W8A8, L2995). |
| Appendix A (A.1 inference stages, A.2 format rationale, A.3 group sizes) | 1934–1985 | READ-FULL | One garble: "(¡= 4-bit)" at L1953 renders "≤ 4-bit"; context unambiguous. |
| Appendix B (B.1 datasets, B.2 PPL/Gen modes, Figs 8–12, Tables 7, 10–12) | 1987–2460, 2461–3347 | READ-FULL | Human-upper-bound figures per dataset included (e.g., Winogrande dev human 94.1%, L2004). |
| §4 Emergent abilities (§4.1 setups, §4.2 results incl. error taxonomy) | 1026–1096, 1097–1203 | READ-FULL | Error-type proportions quoted in prose at L1094–1096 and L1173; Figure 24 percentages at L4737–4771 are scrambled but consistent with prose. |
| Appendix C (C.1 datasets, C.2 metrics/templates, C.3 results: StrategyQA misguidance, GSM8K failure cases, few-shot ablation, AWQ/SQ comparison, Figs 14–23) | 3349–3431, 3505–4026, 4016–4470, Figs 4332–4718 | READ-FULL | Verbatim worked examples retained in extract; StrategyQA keyword-matching bias documented at L3517–3531. |
| §5 Trustworthiness (§5.1 setups, §5.2 results) | 1177–1215, 1205–1224 | READ-FULL | Covers Ethics/Hallucination/Adversarial Robustness only (see also D). |
| Appendix D (D.1 datasets, D.2 metrics, D.3 ethics, D.4 hallucination, D.5 adversarial robustness, Tables 16–18, Figs 25–27) | 4472–4582, 4720–4787, 4789–5321, Figs 4877–5223 | READ-FULL | AdvGLUE tables reflowed but row/column labels present; Acc-drop metric defined at L4576–4580. |
| §6 Dialogue (§6.1 setups, §6.2 results, Table 6) | 1225–1231, 1337–1379, 1469–1585, 1596–1598 | READ-FULL | Table 6 GPT-4 scores reflowed; model/turn labels recoverable. |
| Appendix E (E.1 MT-Bench intro, E.2 judge metrics, Table 19 full dialogue table, E.3 case studies) | 5103–5115, 5224–5330, 5331–5883, 5884–6060 | READ-FULL | Long verbatim model outputs (good/bad cases) occupy L5896–6060; read in full. |
| §7 Long-context (§7.1 setups, §7.2 results) | 1600–1622, 1634 | READ-FULL | One sentence splits across the page break (L1634 continues §7.2). |
| Appendix F (F.1 datasets, F.2 metrics, Figs 28–33, F.3 LongEval details, F.4 multi-doc QA) | 6062–6505, Figs 6105–6762, 6681–6791 | READ-FULL | Figure panels contribute axis ticks only; findings carried by F.3/F.4 prose. |
| §8 Limitations | 1624–1632 | READ-FULL | Paper's own caveats, quoted in summary below. |
| Acknowledgement | 1639–1650 | READ-FULL | Funding: NSFC grants, Tsinghua EE Xilinx AI Research Fund, BNRist. |
| Impact Statement | 1661–1667 | READ-FULL | Carbon-footprint framing; no additional technical claims. |
| References | 1678–1930 | READ-FULL | Includes duplicate Zheng et al. entries cited as 2023a and 2023b (L1889–1901). |

No UNREADABLE rows. Extraction artefacts noted where they occur (all interpretable from context): "OOPT" for OPT at L129; "(¡= 4-bit)" for "≤ 4-bit" at L1953; figure-axis numerals scattered as bare lines throughout figure regions.

## Upgraded summary

- **Bibliography entry:** [9] S. Li, X. Ning, L. Wang, T. Liu, X. Shi, S. Yan, G. Dai, H. Yang, and Y. Wang, "Evaluating quantized large language models," in Proc. International Conference on Machine Learning (ICML), 2024. arXiv:2402.18158.
- **Repo PDF:** `Sources/[09] Li et al. - Evaluating quantized large language models.pdf`
- **Publication status:** Proceedings of the 41st International Conference on Machine Learning (ICML 2024), Vienna, PMLR 235 — venue footer visible in the extracted PDF text (L73–75).
- **Problem & context:** Prior quantization evaluations concentrated on language modelling and zero-shot understanding/emergent tasks, leaving dialogue, long-context processing and trustworthiness essentially untested, and gave practitioners no cross-model guidance on which tensors to quantize at which bit-widths (L121–125, L138–151). This paper positions itself as the comprehensive four-dimension audit: task breadth, model-family breadth, tensor type (Weight vs Activation vs KV Cache), and SOTA-method applicability (AWQ, SmoothQuant) (L138–153). Code released at thu-nics/qllm-eval (L25–26).
- **Approach:** PTQ applied separately to Weights, Activations and KV Cache over 11 model families (OPT, LLaMA2, Falcon, Bloomz, Mistral, ChatGLM, Vicuna, LongChat, StableLM, Gemma, Mamba), 125M–180B parameters, across five task types: basic NLP, emergent ability, trustworthiness, dialogue, long-context (L16–24). Benchmark suite (Table 2): basic NLP on CHID/Winogrande/RACE/LAMBADA/SIQA/PIQA; emergent abilities on MMLU (14,079 test questions, L211), CEval, GSM8K, StrategyQA, ARC, HellaSwag; trustworthiness on ETHICS (15,160, L257), TruthfulQA (817, L263) and AdvGLUE (738, L269); dialogue on MT-bench (80 two-turn questions, L277) scored by GPT-4-0613 single-answer grading on a 1–10 scale (L1228–1231); long-context on LongEval key-value retrieval up to 16K tokens (3,000 samples, L287, L1602) and multi-document QA up to 6K tokens (700 samples, L293, L1604–1605). Quantization settings: uniform formats — asymmetric group-wise weight quantization with group size = attention head dimension (128 for Mistral/LLaMA2/Vicuna/LongChat/ChatGLM; 64 for Falcon; L1978–1985), symmetric per-token activation quantization (dynamic scaling; static scaling badly hurts via down-projection outliers — Wikitext PPL 11.71 FP16 → 12.51 dynamic vs 40.78 static for W4A8 LLaMA2-7B, hybrid static-except-down-proj 12.63; L3300–3347), asymmetric group-wise KV Cache quantization (L1946–1950); bit-width sweep W2–W8, W8A8–W4A4, KV2–KV8 around an RTN baseline, plus AWQ (weight-only) and SmoothQuant (weight-activation) as SOTA methods (L151–153, L1019–1021, L3075–3086). Evaluation runs on OpenCompass-integrated datasets with PPL-mode (multiple-choice perplexity) or Gen-mode scoring (L1994, L2029–2075).
- **Key findings:** Capability side — most LLMs hold within ~2% of FP16 at W4, W4A8 and KV4, and degradation concentrates below those points: noticeable loss at W3/KV3 (small models), collapse for most models at W2/KV2; W4A8 is the frontier for weight-activation quantization and W4A4 causes complete performance loss in most LLMs (L479–487). Larger models tolerate Weight-only and KV Cache quantization better but Activation quantization worse, tracking kurtosis trends (activation kurtosis >1000 vs ~10 for weights/KV; e.g., OPT-6.7B activation kurtosis 1562.67 and OPT-66B 4945.32 vs LLaMA2-70B 1279.15; L88–89, L431–478, L801–804, L2891–2893). Model rankings are order-preserving above W4/W4A8/KV4 (Spearman correlations typically >0.9; L95–96, L2920–2923), but MoE scaling does not buy tolerance: Mixtral-8x7B matches LLaMA2-70B at FP16 yet quantizes like ~7B peers (L97–98, L727–735). Emergent abilities: mathematical multi-step reasoning and self-calibration are the least tolerant; their failures are dominated by incorrect logic (~50%) then calculation errors (~20%) (L1026–1035, L1094–1096, L1173). Dialogue: KV Cache quant is better tolerated than weight quant; W8/W8A8/KV4 keep GPT-4 score within ~2%, while W4 costs >0.3 score on LLaMA2 chat models; failures progress from sentence-level repetition (W3/KV3) to token-level repetition/randomness (W2/KV2/W4A4); second-turn answers are more fragile (LLaMA2-13B KV8→KV3 loses 1.19 turn-2 vs 0.31 turn-1); SmoothQuant partially recovers W4A4 dialogue (L104–105, L1341–1362, L1369–1372, L1347–1353). Long-context: texts ≥4K suffer markedly more under Weight-only and KV Cache quantization (no such length penalty for weight-activation), and at length most LLMs are more KV-sensitive than weight-sensitive; combined W+KV shows it starkly — W8KV4 is near-lossless on LAMBADA/MT-bench but drops >10% on 16K LongEval (Vicuna-7B 57.80→37.00) (L106–108, L1613–1622, L1634, L2486–2493, L2873, L2885). Trustworthiness/safety side — hallucination (TruthfulQA) and adversarial robustness (AdvGLUE) behave like the basic-NLP tasks; quantized models do not inherit the FP16 inverse scaling law on TruthfulQA (they simply decline), and adversarial Acc-drop shows no clear relationship with bit-width (L1207–1210, L4581–4582, L4773–4787). The distinctive ethics result: for small (<7B) models, W3 breaks the FP16 refusal pattern and answers ethical questions the full-precision model declined, so measured accuracy paradoxically rises, whereas KV Cache quantization pushes small models to refuse more items and lose accuracy; AWQ keeps outputs closest to the FP16 refusal behaviour at a slight accuracy cost; ≥7B models just decline monotonically (L1177–1184, L1217–1224, L4720–4732). Table 1 condenses per-task bit-width recommendations calibrated to within 2% accuracy loss on the evaluated tasks, with the authors' explicit warning that they may not generalize to other LLMs or tasks (L110–119, L157–158).
- **Cited in the report at:** Chapter 1 > 1.1 Background and Motivation + Chapter 2 > 2.2
- **Limitations & caveats:** The paper's own §8: post-training quantization only (no QAT); no ablations of hyperparameters such as group size; incomplete coverage of newly released small models (1B–3B); bit-width recommendations are specific to the evaluated LLMs/tasks and might not transfer (L1624–1632). Measurement caveats the authors themselves surface: multiple-choice PPL-mode scoring can reward guessing on high-uncertainty items — Mistral-7B gains 3.5% on RACE after W8A8 (L2991–3003) — and StrategyQA yes/no keyword matching credits a correct answer to a W2 model emitting gibberish containing "No" (L3517–3531); dialogue scoring rests on 80 MT-bench questions judged by a single GPT-4-0613 judge (L277, L1228–1231). Scope note for safety work: its "trustworthiness" axis is ethics/hallucination/adversarial robustness — it does not evaluate open-ended harmfulness, jailbreaks or over-refusal suites (L4473–4483).
- **Relevance to this FYP:** Anchors the FYP's premise that 4-bit weight-only sits in this paper's near-lossless regime (within ~2%) with degradation concentrating below it (L479–487, L110–119), consistent with using NF4 4-bit as the main study condition plus an INT8 point. Its ethics finding — quantization flipping refusal-like behaviour in small models in either direction depending on which tensor is quantized, while capability metrics move little at 4-bit (L1177–1184) — is direct prior evidence that low-bit loading can shift refusal behaviour independently of capability, i.e., exactly the alignment-shift-vs-degradation confound the FYP's HarmBench/XSTest/MMLU design is built to separate. Its documented measurement artefacts (PPL-mode guessing gains, keyword-matched scoring crediting gibberish, generation-format failures such as repetition) parallel the FYP's scorer-validity concerns (judge-primary ASR, regex over-counting, truncation artefacts) and support multi-scorer cross-checking. Its explicit non-generalization warning on recommended bit-widths (L118–119) cautions against treating any blanket "4-bit is safe" claim as universal — supporting the FYP's per-pair, per-benchmark analysis rather than global assertions.

## Discrepancy check vs current summary (per bullet CONFIRMED / CORRECTED + anchor / UNVERIFIABLE)

Current summary = `PAPER_SUMMARIES.md` lines 80–87 ("## [9]" block).

- **Bibliography entry** (L81): author names, initials, order and title CONFIRMED against the title page (extract L1–4) and ICML footer (L73–75). The trailing identifier "arXiv:2402.18158" is UNVERIFIABLE from this extract — the string does not appear anywhere in ref_9.txt (it is an external identifier; resolve against arXiv/Crossref/OpenAlex before relying on it, per repo rule D61).
- **Repo PDF** (L82): CONFIRMED — file exists at `Sources/[09] Li et al. - Evaluating quantized large language models.pdf`, and the extract's title/authors/venue match ref [9].
- **Publication status** (L83): CONFIRMED — "Proceedings of the 41 st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024" appears verbatim at extract L73–75.
- **Problem & context** (L84): CONFIRMED — existing evaluations mainly cover zero-shot understanding and language modelling, leaving dialogue, long-context and trustworthiness unknown (L138–144), and prior work covered only three emergent abilities without trustworthiness/dialogue/long-context (L69–72, L121–125).
- **Approach** (L85): CONFIRMED — three tensor targets, eleven named families, 125M–180B, five task types, AWQ/SmoothQuant comparisons all present (L16–24, L129–136, L151–153); MT-bench scored by GPT-4-0613 single-answer grading 1–10 (L1228–1231); trustworthiness = ethics + hallucination + adversarial robustness (L1205–1210). No corrections needed.
- **Key findings** (L86):
  - Rankings consistent above W4/W4A8/KV4 — CONFIRMED (Table 1 L95–96; §3.3 L711–713; Spearman typically >0.9, L2920–2923).
  - Size-dependent tolerance (larger ⇒ better weight/KV tolerance, worse activation tolerance) — CONFIRMED (L88–89, L431–447, L471–478).
  - Table 1 per-task bit-width recommendations within 2% accuracy loss — CONFIRMED (L110–119).
  - Ethics claim ("W3 models abandon the FP16 refusal behaviour … so measured accuracy paradoxically rises") — CORRECTED (scope): the paper reports this only for small models <7B; for ≥7B "the lower the bit-width, the lower the performance" (L1182–1184); additionally, AWQ restores refusal-like behaviour with a slight accuracy decrease (L1222–1224, L4730–4732), and KV Cache quantization moves small models the opposite way (refuse more, lose accuracy; L1179–1184). The current bullet is accurate for the small-model case it describes but omits the size qualifier.
  - Longer inputs suffer more under weight and KV-cache quantization — CONFIRMED (L106–108, L1613–1622); precision worth adding: weight-activation quantization shows no extra long-text penalty (L1619–1622).
  - Dialogue failure ladder sentence-level repetition → token-level repetition → token-level randomness — CONFIRMED (L104–105, L1347–1353; case studies L5950–6003).
- **Cited in the report at** (L87): carried forward unchanged into this deep-read (same locations). Report-side presence of the citations was not re-checked here (out of scope for a source-extract read).

No contradictions found between the current summary and the paper beyond the ethics-scope nuance above.

## Self-test

- **Total extract lines:** `wc -l docs/audits/2026-08-25_interim_progress_citation_check/tmp/extract/ref_9.txt` → **6794** (newline count; final line has no trailing newline). The read tool reports EOF at **line 6795**. Both counts refer to the same file; the difference is solely the missing trailing newline.
- **Read-to-end statement:** the extract was read sequentially in four passes with the read tool — offsets 1–2000 (returned through L1945), 1946–3545, 3546–5145, and 5146–6795 (EOF marker returned). Every line of the file was displayed exactly once across these passes, appendices included; nothing was skipped.
- **Grep proof for every quoted number/string** (each pattern run against `tmp/extract/ref_9.txt`; matched line(s) listed):

| Quoted item | Pattern grep-matched at extract line(s) |
|---|---|
| 125M to 180B | "180B": L20, L132 |
| PMLR 235, 2024 (Vienna) | "PMLR 235": L74 |
| code URL thu-nics/qllm-eval | L26 |
| MMLU 14,079 test questions | "14079": L211, L249 |
| ETHICS 15,160 | "15160": L257 |
| TruthfulQA 817 | "817": L263 |
| AdvGLUE 738 | "738": L269 |
| MT-bench 80 questions | "80": L277 |
| LongEval 3,000 samples | "3000": L287 |
| Multi-Doc QA 700 samples | "700": L293 |
| within 2% accuracy loss | "Within 2% accuracy loss": L118 |
| activation kurtosis 1562.67 (OPT-6.7B) | L801, L2891 |
| 4945.32 (OPT-66B) | L802, L2893 |
| 1279.15 (LLaMA2-70B) | L804, L2891 |
| incorrect logic ~50% | "around 50%": L1095 |
| calculation error ~20% | "at around 20%": L1173 |
| W4 GPT-4 score loss "> 0.3" | L1346 |
| turn-2 drop "by 1.19" | L1360 |
| turn-1 drop "0.31" | L1362 |
| Mistral-7B RACE "+3.5%" after W8A8 | "by 3.5% after": L2995 |
| W8KV4 LongEval ">10%" loss | "over 10%.": L2493 |
| Vicuna-7B LongEval FP16 57.80 | "57.80": L2873 |
| Vicuna-7B LongEval W8KV4 37.00 | "37.00": L2885 |
| Wikitext PPL 11.71 (FP16) | L3333 |
| 12.51 (dynamic) | L3337 |
| 40.78 (static) | L3341 |
| 12.63 (static w.o. down proj) | L3345 |
| LAMBADA LLaMA2-7B FP16 73.32 | L1058, L2841 |
| LLaMA2-13B-chat MT-bench FP16 5.72/5.05 | L1509–1510, L5411–5412 |
| Mixtral-8x7B-instruct FP16 7.89/6.55 | L5655, L5656 |
| memory-budget example "27GB memory" | L723 |
| Spearman "more than 0.9, typically" | L2922 |
| AWQ beats RTN "more than 3%" (GSM8K W3) | L4412 |
| long-context W3 residual loss "exceeding 20%" | L6786 |
| group size is 128 | L1981 |
| GPT-4-0613 single-answer grading | L1229 |
| small-model gap "1B to 3B" (§8 limitations) | L1629 |
| W4KV4 near-lossless (<2%) on LAMBADA/MT-bench | L2491 |
| garble sample "OOPT" | L129 |
| garble sample "(¡= 4-bit)" | L1953 |

All quoted numbers above were located by grep in the same extract used for the full read; no number in this document lacks a listed proof.
