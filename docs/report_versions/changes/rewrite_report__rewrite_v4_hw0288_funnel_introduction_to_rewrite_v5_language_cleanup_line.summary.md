## Rewritten full report (plain-academic line): Rewrite v4 (HW0288 funnel introduction) → Rewrite v5 (language-cleanup line)

*From:* `docs/rewritten_fyp_report_v4.docx`  
*To:* `docs/rewritten_fyp_report_v5.docx`  
*Decisions:* D66, D67, D68, D69, D75  

**Paragraph/body blocks:** +16 added · −16 removed · ~132 edited · ⟳0 citation-renumber-only · tables: +0 −0 ~2 · front matter blocks 18→18

*Front matter (cover/abstract/TOC/lists):* +0 −0 ~34 — mostly static-TOC rebuilds; non-TOC items below.
  - ❌ Post-training quantization lowers the precision of a model's weights to reduce its memory footprint…
  - ❌ Figure 5.1. HarmBench classifier ASR versus the refusal-regex proxy 31
  - ❌ Figure 5.2. Qwen3-1.7B greedy result and five-seed HarmBench sensitivity 35
  - ❌ Table 5.1. Per-pair HarmBench deltas under three judges 32
  - ❌ Table 5.2. HarmBench classifier deltas at 128 and 512 tokens 34
  - ❌ Table 5.3. Five-seed HarmBench sensitivity 34
  - ❌ Table 6.1. Possible explanations for the HarmBench result 37
  - ❌ Table 6.2. Comparison with published quantization results 38
  - ❌ Table A.1. Primary run controls 49
  - ❌ Table B.1. Capability effects and minimum detectable effects 50

### Per-chapter delta (body)

| Chapter | added | removed | edited | renumber-only |
|---|---|---|---|---|
| Appendix C: Human-Validation Boundaries | 0 | 0 | 2 | 0 |
| Chapter 1: Introduction | 0 | 0 | 8 | 0 |
| Chapter 2: Literature Review | 1 | 1 | 21 | 0 |
| Chapter 3: Methodology and Experimental Setup | 1 | 1 | 12 | 0 |
| Chapter 4: Results | 8 | 8 | 23 | 0 |
| Chapter 5: Scorer Validation and Robustness | 5 | 5 | 11 | 0 |
| Chapter 6: Discussion | 1 | 1 | 20 | 0 |
| Chapter 7: Conclusion and Future Work | 0 | 0 | 1 | 0 |

### Removed paragraphs (16 body; 0 front matter)

<details><summary>show</summary>

- `f0c9fa40` (76w) Chapter 2: Literature Review § 2.7 Implications for Experimental Design — “A non-significant result is informative only when the experiment can detect effects of pr…”
- `9652cc28` (65w) Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis — “Interpretation labels combine direction and capability context. Alignment degradation mea…”
- `f05446c2` (44w) Chapter 4: Results § 4.1 Results at a Glance — “Applying multiple-comparison correction clarifies that the primary statistically defensib…”
- `3ae8151b` (58w) Chapter 4: Results § 4.2 Harmful Compliance (RQ1) — “Llama-3.2-3B decreases from 0.100 to 0.060. Its individually significant ΔASR is −0.040, …”
- `e396b821` (51w) Chapter 4: Results § 4.3 Over-Refusal on Benign Prompts (RQ2) — “No pair shows a statistically significant increase in over-refusal, and four pairs are no…”
- `b8d052c7` (64w) Chapter 4: Results § 4.3 Over-Refusal on Benign Prompts (RQ2) — “The primary family retains the regex result because it was the pre-specified outcome. By …”
- `cea2c399` (45w) Chapter 4: Results § 4.4 Capability (RQ3) — “Mistral loses 0.020 on MMLU but gains 0.0094 on ARC, while Phi loses 0.0267 on MMLU and 0…”
- `237c1278` (88w) Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4) — “The interpretation labels are descriptive summaries of the combined changes. Qwen3-1.7B i…”
- `76b7a267` (46w) Chapter 4: Results § 4.7 Research-Question Synthesis — “RQ1 asks whether NF4 increases harmful compliance. The point estimates are 0.000, +0.040,…”
- `7feb3ee5` (43w) Chapter 4: Results § 4.7 Research-Question Synthesis — “RQ3 asks whether capability falls. Qwen3-1.7B MMLU and Llama ARC survive correction. Qwen…”
- `81c5af19` (47w) Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers — “The human audit sample purposefully oversamples scorer disagreements to maximise diagnost…”
- `b29248f9` (43w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “One Qwen3-4B seed and one Phi seed are individually significant, in opposite directions, …”
- `0b6c9392` (58w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “The reported null finding is supported along four partly independent dimensions: no HarmB…”
- `737334db` (32w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “The empirical evidence demonstrates the absence of a statistically significant increase i…”
- `041ea2ef` (28w) Chapter 5: Scorer Validation and Robustness § 5.8 Robustness Synthesis — “Systematic analysis of scorer divergence provides deeper methodological insights than unc…”
- `b214d2ad` (31w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “A more accurate characterisation of low-bit quantization is prompt-level decision boundar…”

</details>

### Added paragraphs (16)

<details><summary>show</summary>

- `d2834c81` (79w) Chapter 2: Literature Review § 2.7 Implications for Experimental Design — “A non-significant result is informative only when the experiment can detect effects of pr…”
- `8de7c3c9` (82w) Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis — “Interpretation labels combine the direction of the safety change with the capability cont…”
- `ba0cf0c3` (45w) Chapter 4: Results § 4.1 Results at a Glance — “After multiple-comparison correction, the effects that remain statistically significant c…”
- `a08e89d4` (75w) Chapter 4: Results § 4.2 Harmful Compliance (RQ1) — “Llama-3.2-3B decreases from 0.100 to 0.060. The estimated change in harmful compliance is…”
- `1054430d` (61w) Chapter 4: Results § 4.3 Over-Refusal on Benign Prompts (RQ2) — “No pair shows a statistically significant increase in over-refusal. Four pairs show no st…”
- `d2ebd9b4` (66w) Chapter 4: Results § 4.3 Over-Refusal on Benign Prompts (RQ2) — “The primary analysis retains the regex score because it was pre-specified. The validation…”
- `7947d39e` (48w) Chapter 4: Results § 4.4 Capability (RQ3) — “Mistral loses 0.020 on MMLU but gains 0.0094 on ARC, while Phi loses 0.0267 on MMLU and 0…”
- `cf64abba` (117w) Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4) — “The interpretation labels summarise the observed combination of safety and capability cha…”
- `cd3b1c2e` (56w) Chapter 4: Results § 4.7 Research-Question Synthesis — “RQ1 asks whether NF4 increases harmful compliance. The estimated changes are 0.000, +0.04…”
- `194d13ba` (60w) Chapter 4: Results § 4.7 Research-Question Synthesis — “RQ3 asks whether capability falls. Qwen3-1.7B MMLU and Llama ARC remain statistically sig…”
- `054b528c` (52w) Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers — “The human-labelled sample deliberately oversamples the items on which the two scorers dis…”
- `b287f2e0` (43w) Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity — “One Qwen3-4B seed and one Phi seed are individually significant, in opposite directions, …”
- `53ce624d` (59w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “Four partly independent checks support the null result: no HarmBench comparison remains s…”
- `dc4e8c3d` (29w) Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null — “Within the detection limits of the experimental design, the study finds no statistically …”
- `dca7fad2` (31w) Chapter 5: Scorer Validation and Robustness § 5.8 Robustness Synthesis — “Analysing why scorers disagree is more informative than averaging their scores. When eval…”
- `46af4b45` (37w) Chapter 6: Discussion § 6.1 Answer to the Central Question — “A more accurate description of low-bit quantization in this study is instability in indiv…”

</details>

### Edited paragraphs (132; change window shown)

<details><summary>show</summary>

- (front matter): “…ultiple-comparison correction. Three results remained s[+tatistically s][−reveal][+show]…”
- (front matter): “2.4 Refusal, Harmfulness, and Threat Models [−8][+9]”
- (front matter): “Chapter 5: Scorer Validation and Robustness [−29][+30]”
- (front matter): “5.1 HarmBench Classifier versus Refusal Proxy [−29][+30]”
- (front matter): “5.2 Independent HarmBench Judges 3[−0][+1]”
- (front matter): “5.3 Human Validation of Harmful-Compliance Scorers 3[−1][+2]”
- (front matter): “5.4 XSTest Scorer Sensitivity and Human Validation 3[−1][+2]”
- (front matter): “5.5 Generation-Length Robustness 3[−2][+3]”
- (front matter): “5.6 Multi-Seed Decoding Sensitivity 3[−3][+4]”
- (front matter): “5.7 Power and the Meaning of the Null 3[−4][+5]”
- (front matter): “5.8 Robustness Synthesis 3[−5][+6]”
- (front matter): “Chapter 6: Discussion 3[−6][+7]”
- (front matter): “6.1 Answer to the Central Question 3[−6][+7]”
- (front matter): “6.2 Relationship to Prior Work 3[−6][+7]”
- (front matter): “6.3 Capability Effects and Deployment [−39][+40]”
- (front matter): “6.4 Validity of the Scoring Methods 4[−0][+1]”
- (front matter): “6.5 Threats to Validity 4[−0][+1]”
- (front matter): “6.6 Practical Implications 4[−2][+3]”
- (front matter): “Chapter 7: Conclusion and Future Work 4[−3][+4]”
- (front matter): “7.1 Conclusions 4[−3][+4]”
- (front matter): “7.2 Prioritised Future Work 4[−3][+4]”
- (front matter): “References 4[−5][+6]”
- (front matter): “Appendix A: Reproducibility Summary 4[−8][+9]”
- (front matter): “Appendix B: Minimum Detectable Capability Effects [−49][+50]”
- (front matter): “Appendix C: Human-Validation Boundaries 5[−0][+1]”
- (front matter): “…rmBench classifier ASR versus the refusal-regex proxy 3[−0][+1]”
- (front matter): “….7B greedy result and five-seed HarmBench sensitivity 3[−4][+5]”
- (front matter): “…ble 5.1. Per-pair HarmBench deltas under three judges 3[−1][+2]”
- (front matter): “….2. HarmBench classifier deltas at 128 and 512 tokens 3[−3][+4]”
- (front matter): “Table 5.3. Five-seed HarmBench sensitivity 3[−3][+4]”
- (front matter): “…e 6.1. Possible explanations for the HarmBench result 3[−6][+7]”
- (front matter): “…e 6.2. Comparison with published quantization results 3[−7][+8]”
- (front matter): “Table A.1. Primary run controls 4[−8][+9]”
- (front matter): “…B.1. Capability effects and minimum detectable effects [−49][+50]”
- Chapter 1: Introduction § 1.1 Background and Motivation: “…ntary ways to measure these behaviours [11], [12]. The [+d][−n][+fficul][−erpretive prob…”
- Chapter 1: Introduction § 1.1 Background and Motivation: “…, Benjamini–Hochberg false-discovery-rate control, and [−power-oriented ][−,][−a][+i][+f…”
- Chapter 1: Introduction § 1.2 Problem Statement: “The[+ central][+one of ][+at][−ve][+on][+asu][+ment a][−y operati][−al][+e]: a change in …”
- Chapter 1: Introduction § 1.2 Problem Statement: “…the same prompts under a single controlled precision co[+mpariso][−trast], and treats th…”
- Chapter 1: Introduction § 1.2 Problem Statement: “…n the direction of decreased harmful compliance, while [+the estimated MMLU change is a …”
- Chapter 1: Introduction § 1.4 Scope: “…enge. These multiple-choice instruments provide useful [−a][+refere][−h][+e p][−r][+int]…”
- Chapter 1: Introduction § 1.5 Contributions: “…s bootstrap confidence intervals, exact McNemar tests, [+correction for ][−icity][+e][+m…”
- Chapter 1: Introduction § 1.5 Contributions: “…derived scorer classifications in separate, verifiable [−audit ][+ file]s.”
- Chapter 2: Literature Review: “…iment must control: the compression method, the safety [−c][+behavi][+ur bei][+g mea][−t…”
- Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation: “…s [20]. Neither represents all useful behaviour. Their [−va][+ro][−u][−di][+expl][−g][+a…”
- Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation: “…tched quantization study benefits from fewer outcomes w[+hose scoring cr][−h][+eria are …”
- Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation: “…es, equivocates, or supplies partial information. That [+extr][−dd][+ uncerta][+n][+y ][…”
- Chapter 2: Literature Review § 2.1 Trustworthiness and Safety Evaluation: “…that the effect is task-specific. The report therefore [−avoids colla][+kee][−ing][+sepa…”
- Chapter 2: Literature Review § 2.2 Quantization Methods and Capability Effects: “… pruning, can remove many weights in one shot [23]. The[−refore, the][+re therefo][− int…”
- Chapter 2: Literature Review § 2.2 Quantization Methods and Capability Effects: “…ety and capability changed smoothly as precision fell, [+beh][− graded fp16–INT8–NF4 c][…”
- Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour: “…evel of harm in a response [29]. These are related but [−non-][+d][−d][+ffer][+ ways of …”
- Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour: “…needed precisely because worst-case vulnerability and t[+he t][−treatment ][+ of ordinar…”
- Chapter 2: Literature Review § 2.3 Quantization and Safety Behaviour: “…prior study most similar to this experiment because it [−observe][+find][−bro][+th][−d][…”
- Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models: “…es everything. XSTest provides a smaller human-curated [−diagno][+te][−ic][+ targeted at…”
- Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models: “…al [41]. It is useful as an independent moderator, but [−i][−s][+he][−con][+it applie][−…”
- Chapter 2: Literature Review § 2.4 Refusal, Harmfulness, and Threat Models: “…In a DirectRequest test, the prompt is fixed and the co[+mpariso][−trast] focuses on the…”
- Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation: “…e making model-specific errors that distort a paired co[+mpariso][−trast].”
- Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation: “…valuable for reproducibility, a semantic judge tests co[−nstruct co][+ of the behaviour …”
- Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation: “…[41]. Their openness permits rerunning the scorer, yet [−construct ][+ between what the …”
- Chapter 2: Literature Review § 2.5 Automatic Evaluators and Human Validation: “… and discloses a pre-registration deviation. These two [+v][−u][+li][+at][−t][+on exerci…”
- Chapter 2: Literature Review § 2.6 Synthesis and Research Gap: “…ps with a five-pair matched design, NF4 as the primary [−i][+qua][−erve][+izatio][+ cond…”
- Chapter 2: Literature Review § 2.6 Synthesis and Research Gap: “…eparated from capability, scorer error, truncation, or [+the nu][−ul][+ber of comparison…”
- Chapter 2: Literature Review § 2.7 Implications for Experimental Design: “…formatting, decoding controls, and generation budgets. [−Crucially, a][+A][+therefore ][…”
- Chapter 2: Literature Review § 2.7 Implications for Experimental Design: “…al design is equally important. Twenty model–outcome co[+mpariso][−trast][−multiplicity …”
- Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design: “…sion selected by the TC1 CUDA environment, whereas the [−tre][+qu][−tme][+ized] member i…”
- Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design: “The primary [+quan][−re][+iz][−me][+io][+ condi][+ion] is NF4 four-bit quantiza…”
- Chapter 3: Methodology and Experimental Setup § 3.1 Matched-Pair Experimental Design: “…cale are observed grouping variables rather than random[+ly ass][−s][+gn][+condi][−reatm…”
- Chapter 3: Methodology and Experimental Setup § 3.3 Benchmarks and Outcomes: “…ce prompts, and ARC-Challenge provides a larger second [+c][+pability be][−o][+ma][+k] w…”
- Chapter 3: Methodology and Experimental Setup § 3.3 Benchmarks and Outcomes: “…esholds, which is why a similar absolute delta can be s[+tatistically s][−a][+yet i][−d …”
- Chapter 3: Methodology and Experimental Setup § 3.4 Scoring and Validation: “…riginal analysis while testing whether its conclusion s[−urv][+t][−v][+ll holds wh][−s][…”
- Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis: “…primary NF4 set of corrected comparisons contains 20 co[+mpariso][−trast][−n][+rrec][+s …”
- Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis: “…tion used in the scorer sensitivity analysis [55], and [+the behaviour ][+ label][−co][+…”
- Chapter 3: Methodology and Experimental Setup § 3.5 Statistical Analysis: “… not remove measurement error. Exactly three primary co[+mparisons remained statisticall…”
- Chapter 3: Methodology and Experimental Setup § 3.6 Implementation and Execution: “…ions and summaries are treated as immutable run files. [−R][+Outputs of r][− output][+ p…”
- Chapter 3: Methodology and Experimental Setup § 3.7 Reproducibility Controls: “…not prove that a citation entails a sentence or that a [+s][+ri][+g method mea][+ures wh…”
- Chapter 3: Methodology and Experimental Setup § 3.8 Analysis Workflow and Quality Control: “Multipl[+e-compar][−city][+son] correction is performed …”
- Chapter 4: Results § 4.1 Results at a Glance: “The primary [+analy][−tat][−tical][−family en][−mpass][+v][+r]s 20 NF4 comparisons acro…”
- Chapter 4: Results § 4.1 Results at a Glance: “… sizes and statistical uncertainty. Exact p-values and [−correc][+adjus][−cont][−a][+e][…”
- Chapter 4: Results § 4.2 Harmful Compliance (RQ1): “No[+ne of the five][+mparisons remains statistically significa][+ afte][−ast survives mul…”
- Chapter 4: Results § 4.2 Harmful Compliance (RQ1): “…eline ASR is 0.585, more than eight times Phi's 0.070. [−P][+The p][+iff][−lta][+rence][…”
- Chapter 4: Results § 4.2 Harmful Compliance (RQ1): “Qwen3-4B [−i][+show][−point ][− for][+d][−d][+ in][+, at 0.040][−is ][+pa][−h][+t][−ef][+…”
- Chapter 4: Results § 4.2 Harmful Compliance (RQ1): “… from safe to harmful, and that imbalance produces the [+decrease that ][−nd][+s stat][−…”
- Chapter 4: Results § 4.3 Over-Refusal on Benign Prompts (RQ2): “…ts in the set of 250 prompts, which is large enough to [+remain ][−u][+tatistically sign…”
- Chapter 4: Results § 4.4 Capability (RQ3): “…r Qwen3-1.7B, MMLU accuracy falls from 0.643 to 0.553, [−representing ][−lo][+decrea][−s…”
- Chapter 4: Results § 4.4 Capability (RQ3): “Llama-3.2-3B [+s][−a][+ow][+de][−ont][+e][−t][+e][+remain][−u][+ statistically significan…”
- Chapter 4: Results § 4.4 Capability (RQ3): “…while ARC falls from 0.876 to 0.860. The ARC result is [+stat][−nd][+st][−vidu][+c][+ be…”
- Chapter 4: Results § 4.4 Capability (RQ3): “… similar in magnitude but estimated more precisely, so [+the ][+re][+lt ][−v][+ema][−v][…”
- Chapter 4: Results § 4.4 Capability (RQ3): “…0% power thresholds, whereas several smaller observed d[+iff][−lta][+rence][+:][−rather …”
- Chapter 4: Results § 4.4 Capability (RQ3): “…ive pairs in the joint space these results define. The [−MMLU point ][+d MMLU change is …”
- Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4): “…whereas the 4B model changes by −0.0033. However, that [+differen][−ontrast][+e][−suppor…”
- Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4): “…s different practical meaning at those baselines. This [−he][+varia][−e][+ion ac][−g][+s…”
- Chapter 4: Results § 4.5 Model Scale and Family Differences (RQ4): “[−C][+Diffe][+ent m][−ss-][+del ][−y][+ies produce][+s in different][+,][+which c][−lso][…”
- Chapter 4: Results § 4.6 INT8 Precision Point: “The INT8 comparison asks whether the observed pattern [−i][+change][−ed][+ually][−follow]…”
- Chapter 4: Results § 4.6 INT8 Precision Point: “…upports practical similarity for the INT8 capability co[+mpariso][−trast]s within the st…”
- Chapter 4: Results § 4.6 INT8 Precision Point: “For deployment, the[− evidenc][+s][−fav][+results supp][−u][−s][+t] a staged decision. IN…”
- Chapter 4: Results § 4.6 INT8 Precision Point: “… in shape or in direction. Together they argue against [−o][+a si][+gl][−caused by][+of]…”
- Chapter 4: Results § 4.7 Research-Question Synthesis: “…benign over-refusal. The pre-specified regex finds no s[+tatistically s][+ statistically…”
- Chapter 4: Results § 4.7 Research-Question Synthesis: “…y while NF4 contains the largest losses. Overall, the e[−v][+ffect of lower prec][+sion …”
- Chapter 4: Results § 4.7 Research-Question Synthesis: “RQ5 [−inve][+a][−tigate][+k][−remain r][+h][−bust][+ld][−fav][+supp][−u][+t]s the HarmBen…”
- Chapter 5: Scorer Validation and Robustness § 5.1 HarmBench Classifier versus Refusal Proxy: “…ts non-refusal as compliance. However, that is not the [−c][+behavi][−nst][+u][−uct] Har…”
- Chapter 5: Scorer Validation and Robustness § 5.1 HarmBench Classifier versus Refusal Proxy: “…change response style, length, or refusal phrasing, so [+the ][+'s][−become treatment-][…”
- Chapter 5: Scorer Validation and Robustness § 5.2 Independent HarmBench Judges: “…on Phi, where Llama Guard's broader “unsafe response” c[+riteri][−struct] differs most f…”
- Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers: “… over-flags 101. The classifier is not perfect, but it [−i][+agree][+more ][−r][+ly][+wi…”
- Chapter 5: Scorer Validation and Robustness § 5.3 Human Validation of Harmful-Compliance Scorers: “…er's recall is 0.757 and its precision is 0.609 on the [+hum][−u][+n-labelle][−it] sampl…”
- Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation: “Human[− ground-truth] annotation clarifies thi…”
- Chapter 5: Scorer Validation and Robustness § 5.4 XSTest Scorer Sensitivity and Human Validation: “… by the validated strict semantic judge, exactly two co[+mparisons remain statistically …”
- Chapter 5: Scorer Validation and Robustness § 5.5 Generation-Length Robustness: “…512 tokens, the same pair is exactly flat. The earlier [+apparent increa][−ignal][+e][−g…”
- Chapter 5: Scorer Validation and Robustness § 5.6 Multi-Seed Decoding Sensitivity: “…ustrates the relationship between the Qwen3-1.7B greedy[− point][−demon][+which i][−trat…”
- Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null: “…tematic safety degradations; smaller localised effects [−rem][+f][−in][+ll][−i][+he stud…”
- Chapter 5: Scorer Validation and Robustness § 5.7 Power and the Meaning of the Null: “Capability evaluations [−poss][+diff][−ss][+r][−vary][−g][−d][+beca][+s][−to][+MMLU and A…”
- Chapter 6: Discussion § 6.1 Answer to the Central Question: “The empirical findings [−under][−c][+h][−re][+w][−t][+w][−e necessit][− of jointly evalua…”
- Chapter 6: Discussion § 6.1 Answer to the Central Question: “…ed \|\| Generation truncation \| 60.3% truncation at 128; [−signal][+the effect] disappe…”
- Chapter 6: Discussion § 6.2 Relationship to Prior Work: “…is closest to Decoding Compressed Trust, which reports [−broad preserv][+th][−ion of][+ …”
- Chapter 6: Discussion § 6.2 Relationship to Prior Work: “…on \| ASR −0.040 to +0.040 (−4.0 to +4.0 points), none s[−u][+tatistically significant a…”
- Chapter 6: Discussion § 6.2 Relationship to Prior Work: “…can therefore fail in both directions depending on the [+behaviour being s][−nst][−uct][…”
- Chapter 6: Discussion § 6.2 Relationship to Prior Work: “…xperiment is English-only [28]. Their human evaluation [−reveal][+show][−r][+id][−v][+nt…”
- Chapter 6: Discussion § 6.3 Capability Effects and Deployment: “…provides a useful intermediate point. Its capability co[+mpariso][−trast]s are close to …”
- Chapter 6: Discussion § 6.4 Validity of the Scoring Methods: “…pshot was not recorded, Llama Guard is open-weight but [−us][+appli][−c][+definiti][−str…”
- Chapter 6: Discussion § 6.4 Validity of the Scoring Methods: “…bservation of an abstract property. When the estimated [−treatment ][+ of quantization] …”
- Chapter 6: Discussion § 6.4 Validity of the Scoring Methods: “…ts bias depends on how its text patterns relate to the [+behaviour being ][+u][−s][−ruct…”
- Chapter 6: Discussion § 6.4 Validity of the Scoring Methods: “… while reporting the validated semantic judge reflects [+it][−up][+ clos][−io][+ ag][− m…”
- …and 12 more (see .full.json)

</details>

*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings of this file (regenerate via `make report-versions`).*
