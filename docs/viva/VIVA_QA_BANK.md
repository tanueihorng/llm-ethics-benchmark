# Viva Q&A Bank — Quantization, Safety and Capability in Small LLMs
**Oral defence, NTU CCDS FYP · 4/7/8 Dec 2026 · 20 min talk + 10 min Q&A**
Every number below was re-verified against `results_512/analysis/` and the source tree on 2026-08-10.

---

## How to use this

Read the cheat table last thing before you walk in — those numbers are the only ones you must recall cold. Everything else is a *shape* of answer: direct answer first, evidence second, concession third. Speak the answers aloud at least twice; they are written to be spoken in 20–60 seconds, not read. Where a section says **Concede**, say it *before* the examiner finds it — a volunteered limit reads as command of the work, a discovered one reads as a gap. If you genuinely do not know, say so and say what you would do; that is a scoring answer, not a losing one.

---

## The six answers you must know cold

### 1. "In one sentence — what did you actually contribute?"

Two contributions, and I'd put the methodological one first. Methodologically, I show that *how* you measure a quantized model's safety decides what you find: the generation budget and the refusal scorer each flipped my own headline, and I have the artefacts for both. Empirically, across five matched pairs in four families, four-bit NF4 does not produce a statistically robust increase in harmful compliance under direct requests, but it does produce real capability loss — the opposite of the usual worry. Concretely, my correction family is twenty contrasts, five pairs times four metrics. Five are nominally significant, three survive Benjamini-Hochberg, and none of the survivors is a harmful-compliance contrast: two are capability losses and the third is an over-refusal *decrease*. It's an evaluation contribution, not a new quantization method.

**Point at:** `results_512/analysis/multiple_comparisons.json` — `family_size` 20, `n_uncorrected_significant` 5, `n_bh_significant_q05` 3, and the three survivors listed by name.

**If they push — "what were the twenty?":** Five pairs × four primary metrics: HarmBench ASR under the classifier, MMLU, ARC-Challenge, and XSTest over-refusal. One NF4-versus-fp16 contrast per cell. It's in the file's own `description` field.

---

### 2. "When you report an attack success rate of 0.255, what exactly has been measured?"

It's the fraction of two hundred HarmBench prompts — fifty-one of two hundred for that model — where the response was judged a non-minimal, unambiguous instance of the requested harmful behaviour. That's HarmBench's own four-rule rubric applied by their fine-tuned Llama-2-13B classifier, my primary scorer; the prompt is pinned verbatim in the judge module. Three things it is not. It is not real-world harm: the rubric has no functionality condition anywhere in its four rules, so a fabricated recipe or non-working exploit code still counts, and I wrote that inference explicitly into my labelling protocol. It is not robustness to attack, because every prompt is presented directly. And the absolute level isn't comparable across papers, because it depends on the prompt set, the rubric and the generation budget. So: harmful compliance under direct requests, at five hundred and twelve tokens.

**Point at:** `results_512/qwen_2b_base/harmbench/summary.judge.harmbench_cls.json` (51/200, zero parse errors); the pinned rubric in `ethical_benchmark/judges/validation.py`; `docs/HUMAN_LABEL_PROTOCOL.md` for the no-functionality-condition reading.

**Concede:** The "no functionality condition" is *my* reading of their rubric, not a sentence HarmBench writes. Say it that way — it's in my protocol document, not theirs.

---

### 3. "Two hundred prompts, two models, a binary label. What test did you run, and why that one?"

McNemar's exact test on the discordant prompts. Both members see the identical prompts in the same order, so the design is matched, and only the prompts where the two models disagree carry information. For Llama that's ten prompts — one flipping one way and nine the other. I use the exact binomial version rather than chi-square because discordant counts get as low as two or three on XSTest, where the asymptotic approximation isn't trustworthy. The function is `mcnemar_exact_test` in `compare_quant_pairs.py`; it computes the doubled lower binomial tail directly, no SciPy. Alongside it I report a paired bootstrap interval, resampling prompt IDs jointly for both members, two thousand resamples, seed forty-two, percentile bounds. The test is the inference; the interval shows magnitude.

**Point at:** `ethical_benchmark/analysis/compare_quant_pairs.py` — `mcnemar_exact_test`, `compute_paired_bootstrap_ci`. Discordant counts live in `multiple_comparisons.json`.

**Concede:** At n=200 on HarmBench the bootstrap bounds land on multiples of 0.005, so the interval is granular. Where the two ever disagree, the exact McNemar governs.

---

### 4. "Explain this scorer business in plain words — why does it matter that two scorers disagreed?"

I started by detecting refusals with a regular expression looking for canonical refusal phrases. Then I scored the same responses with HarmBench's own trained classifier and they disagreed badly — agreement from 0.25 to 0.84 depending on the model, on a scale where one is perfect. To settle which was right I labelled two hundred responses myself against the HarmBench rubric, blind to what either scorer had said. The classifier matched me at 0.59; my regex matched at 0.11, and it produced a hundred and one false harmful calls to the classifier's eighteen. I'm the annotator, and the sample deliberately over-weights the disputed cases, so those are contested-case contrasts, not population figures — a second annotator is my stated open work. But the ordering is the point, and it changed which model looked least safe.

**Point at:** `results_512/analysis/judge_agreement.json` (per-alias κ 0.254 Mistral-4bit → 0.836 Llama-4bit); `human_validation.json` (κ 0.5908 vs 0.1095; 18 over-flags vs 101).

**If they push — "who was the human?":** Me. The tool hid both scorers' labels, so it was label-blind, but items were not presented in randomised order and there is no inter-rater κ.

---

### 5. "You had a significant safety regression at 128 tokens. You changed your primary configuration and it went away. Why is that not moving the goalposts?"

Two things push back. First, the plus 0.055 was never a multiplicity-robust finding — my own 128-token artefact records McNemar p of 0.027 with BH-FDR false — so promoting 512 did not kill a surviving result. Second, the re-base did not only work in my favour. Qwen-4B's ASR increase got *bigger*, from plus 0.025 at p 0.18 to plus 0.040 at p 0.096, and Phi's over-refusal effect grew until it became an FDR survivor. It also removed a Llama INT8 increase that had been significant under both judges. The justification is mechanical, not rhetorical: a direct prefix test shows 60.3 percent of 128-token responses were provably cut off mid-generation, and 512 is HarmBench's own reference budget.

**Point at:** `results_512/analysis/genlen_robustness.json` — `asr_128_classifier` (BH false), `prefix_truncation_128` (60.3%), `capability_128_vs_512`. Decisions D39 (512 as robustness check) and D41 (promotion) in `docs/PROJECT_LOG.md`.

**Concede:** 512 started as a robustness check under D39 and was promoted to primary under D41 *after* I had seen its numbers. The defence is the truncation evidence, not blindness.

---

### 6. "Your headline is a null. State plainly what effect sizes you cannot rule out."

At two hundred paired prompts and the flip rates I observe, the smallest ASR change I could detect at eighty percent power runs from 0.044 to 0.086 across pairs, with a representative value of 0.059. So a real increase in harmful compliance of a few percentage points would very likely have been invisible to me. The bound I'd actually stand behind is the interval rather than the power number: the widest upper edge anywhere is plus 0.080, on Qwen3-4B, so an increase of up to about eight points on that pair is not excluded by my data. That's why I always say *no detectable change at n equals two hundred*, never *no effect*.

**Point at:** the `power_analysis` block of `multiple_comparisons.json`; the multiple-comparison section and limitations chapter of the report; item 2 of `docs/RESULTS_CARD.md`.

**If they push — "what would fix it?":** Sample size. Quadrupling n roughly halves the detectable effect; to reach a 0.02 floor I'd need around eighteen hundred paired prompts.

**Concede:** Post-hoc power is in that block too, but I don't lean on it — Card et al., whom I cite, warn against exactly that inference.

---

## Statistics

### **"Who chose twenty tests for your family, and was that before or after you saw the results?"**

Twenty is five pairs times four primary metrics, one NF4-versus-fp16 contrast per cell, and that's in the file's own description field. On timing I'll be straight: the family was written down and computed *after* the primary results existed. The script's docstring says why it was added — an earlier draft acknowledged multiplicity in prose and never reported a corrected view. The only family whose composition I locked before computing it is the validation-informed parallel one, and that lock is a dated file which also says plainly that it isn't blind. What I can offer instead of pre-registration is insensitivity: the same three contrasts survive under Bonferroni over the same twenty, at 0.012, 0.016 and 0.037, and under BH they still survive if I inflate the family to thirty or sixty.

**Point at:** docstring of `scripts/multiple_comparisons.py`; `docs/VALIDATION_INFORMED_FAMILY_NOTE.md` (composition locked 2026-07-19, self-declared not blind).

**If they push — "fold in the INT8 contrasts too":** There are twenty of those, not ten — INT8 is present on all four axes for all five pairs. Folding them in doubles the q-values to about 0.016, 0.016 and 0.024. All three still survive.

**Concede:** *Bonferroni* at thirty or sixty would drop the Phi contrast; only BH survives the inflation. No ASR contrast is affected either way.

---

### **"Why Benjamini-Hochberg — the permissive correction — when correcting deletes findings and your headline is a null?"**

Two answers. First, I'm not leaning on BH's permissiveness. I checked Bonferroni over the same twenty p-values and the same three contrasts survive, at 0.012, 0.016 and 0.037, so nothing in my conclusion needs FDR rather than family-wise control. Second, on the conflict of interest, the test is whether the null *needs* the correction, and it doesn't. Before any correction, four of the five ASR contrasts are non-significant and the fifth is a decrease. The correction takes away findings I'd have liked to have — the borderline capability and ASR effects — not findings that would have embarrassed me.

**Point at:** the twenty p-values in `multiple_comparisons.json`; the arithmetic is one multiplication each.

**Concede:** Two honest limits. BH controls the expected *proportion* of false discoveries among my three survivors — about 0.15 of one in expectation — not the probability that any individual survivor is real. And no correction repairs a low-powered design, which is why the MDE analysis sits in the same section.

---

### **"Correct within each pair instead — four tests — and your Llama ASR decrease survives. So your headline depends on a family you chose."**

You're right, and I'll give you the number: with a per-pair family of four, Llama's ASR q-value is about 0.043, so it would survive. I chose the wider family because I test five pairs across four benchmarks and then report the most extreme cells, so the multiplicity is across the matrix, not within one pair. But look at which direction the family choice moves. The contrast the narrow family rescues is a *decrease* in harmful compliance. No family I ran, and none of the three judges, produces a significant increase anywhere. The family-invariant claim is that no pair shows a statistically significant *increase* in harmful compliance under NF4 at the reference budget.

**Point at:** Llama's four p-values — ARC 0.0008, ASR 0.0215, MMLU 0.0801, XSTest 0.2891.

**If they push — "your own file labels Mistral alignment_degradation with a +0.060 delta":** That's `pair_interpretations.json`, the v2-regex proxy — its sibling in the tree is literally named `pair_interpretations.SUPERSEDED.md`. The classifier puts Mistral at −0.020, non-significant.

**Concede:** My headline should read "no ASR *increase* survives", not "zero ASR contrasts survive". That's a wording fix I'd make.

---

### **"You report both bootstrap intervals and McNemar p-values. Where do they disagree, and which governs?"**

They mostly agree, and the cases where the convention bites are all knife-edge. Qwen3-4B ASR is plus 0.040 with a bootstrap lower bound landing exactly on zero, and McNemar gives 0.096. I treat the interval as closed, so a bound sitting on zero counts as containing zero. That's documented in a comment inside `compute_paired_bootstrap_ci`, and it's what makes the interval and the test agree rather than disagree; the report footnotes that cell and calls Qwen-4B directional rather than confirmed. The same convention decides Qwen3-1.7B's over-refusal — interval minus 0.048 to exactly zero, McNemar 0.109. Where the two could genuinely diverge, the exact McNemar governs, because it uses the pairing directly.

**Point at:** the comment in `compute_paired_bootstrap_ci`; the Table 6.2 caption footnote in the report builder.

**Concede:** The closed convention is my own decision, documented in code rather than drawn from a standard. An exact conditional interval would be the more principled companion to an exact test.

---

### **"Your limitations chapter says the evidence 'rules out any harmful-compliance increase larger than that detectable effect'. Is that sentence correct?"**

No, not as written, and I'd correct it. Failing to detect an effect at eighty percent power doesn't rule out effects above the detection floor — it means that if such an effect existed I'd have caught it four times in five, so one time in five I'd miss it. Power is a statement about the design, not an exclusion drawn from the data. The defensible version is the interval one: my ninety-five percent upper bounds on the NF4 ASR change are plus 0.055 for Qwen3-1.7B, plus 0.080 for Qwen3-4B and plus 0.055 for Phi. So the data are compatible with an increase of up to about eight points on the worst pair, and that's the sentence I should have written.

**Point at:** the sample-size limitation entry in Chapter 8 of both report builders; the CI upper bounds in `genlen_robustness.json`.

**Concede:** There are two instances of that over-strong phrasing in the v5 builder, not one. It's a two-line edit and a rebuild; no number in the study changes.

---

### **"You ran a formal equivalence test for INT8 on capability, but not on safety, where your headline lives. Isn't that convenient?"**

It's an asymmetry and I won't pretend otherwise. I added the equivalence test on capability because non-significance was being read as *no change*, and there I could support a real margin: all ten INT8 capability contrasts are equivalent to full precision within five percentage points, with the widest ninety-percent interval closing at 0.038. I didn't run it on safety, and the honest reason is that it wouldn't have been flattering. Recomputing from the committed discordant counts, using the same standard error the equivalence script uses, the tightest margin all five pairs would pass is about 0.075, driven by Qwen3-4B. That's too wide to be a useful deployment guarantee, so I reported the detection floor and the intervals instead. I still think that bound belongs in the document.

**Point at:** `results_512/analysis/capability_equivalence.json` — TOST, α 0.05, margin 0.05, 10 of 10 equivalent, `tightest_common_margin_DATA_DERIVED` 0.0378.

**Concede:** The 0.075 isn't a committed artefact yet. And the artefact labels the whole equivalence analysis post hoc — the margin was fixed before the test ran, but after the INT8 results were known.

---

### **"Your multi-seed arm runs at temperature 0.7 while your headline is greedy — two things at once. And one Qwen3-4B seed shows a significant five-point increase."**

On the design point: greedy decoding is deterministic, so at the headline setting there's no seed variance to measure — rerunning gives identical output. The arm therefore asks a different question, which is whether the null survives when I decode stochastically at all: temperature nought point seven, five seeds, five pairs, same classifier. On Qwen3-4B, yes — one seed in five reaches significance at plus 0.050, p 0.013; the seed mean is plus 0.029, and every seed sits below that pair's detection floor of about 0.059. I report that as a directional non-null rather than smoothing it. Across the whole arm, four seeds anywhere reach individual significance, and they split one increase against three decreases.

**Point at:** `results_512/analysis/sensitivity_multiseed.json` — qwen_4b seed2 +0.050 p 0.0129; llama seeds 4 and 5 significant decreases; phi seed5 −0.055.

**Concede:** Five seeds is a thin basis for a range; the arm confounds seed with temperature so it isn't a clean seed-only test; and greedy sits *outside* the seed range for Mistral and Phi, opposite in sign to the seed mean in both.

---

### **"Your single significant ASR result rests on ten discordant prompts, one against nine. How many would have to fall differently before it disappears?"**

One. The counts are one flip harmful-ward against nine safe-ward, exact p 0.021; move a single prompt to the other side and it's two against eight, p 0.109 — gone. So I don't build anything on that test alone, and it doesn't survive the correction either: its q-value is 0.107, so I carry it as directional and uncorrected. What gives it some weight is that it reproduces on an independent instrument — the gpt-4o second judge, scoring the same saved generations, gives minus 0.035 with zero flips one way and seven the other, p 0.016 — and in the multi-seed arm all five Llama seeds are negative, two individually significant.

**Point at:** `multiple_comparisons.json` (b=1, c=9, p 0.0215, q 0.1074); `genlen_robustness.json` `asr_512_gpt4o`.

**Concede:** It is *not* judge-invariant. LlamaGuard gives minus 0.03 at p 0.146 — same sign, not significant. And the design is correlational, so I can't rule out a genuine improvement; the pair's label is *capability collapse masquerading as safety*, not *alignment improvement*.

---

## Safety-ML

### **"You ran HarmBench with no jailbreak. Isn't a safety conclusion from direct requests alone uninformative about deployment?"**

It bounds the claim rather than voiding it. What I claim is that NF4 quantization does not raise harmful compliance *under direct requests*. That's HarmBench's DirectRequest baseline, the weakest threat model in their own taxonomy, and I state that boundary in scope, in methodology, and as the first of six external-validity bounds in the discussion. Note also that "standard" in my config is HarmBench's *behaviour category*, not an absence of attack. It matters because most prior work finding quantization-safety effects finds them attack-dependent: HarmLevelBench reports the direction varies by attack, and Egashira shows an NF4 model can be made harmful adversarially. So my null is a null about the no-attack baseline.

**Point at:** report §3.5 and the six external-validity bounds in §7.2; `ethical_benchmark/benchmarks/harmbench.py` (`config_name` default "standard").

**If they push:** Attack-augmented evaluation is the first item in my future work — the harness would need an attack-augmentation layer at prompt-construction time, not a new benchmark.

---

### **"Your only significant safety result is minus 0.040, but your classifier disagrees with a human on 27 of 200 items. Your measurement error exceeds your effect."**

Because it isn't a difference between two independently noisy rates. It's a paired within-item contrast: same classifier, same prompts, both members of the pair. That minus 0.040 is ten discordant prompts out of two hundred, nine moving safe-ward and one harmful-ward. Any scorer bias constant across the pair cancels in the difference. I'd also push back on the premise: those twenty-seven disagreements come from a deliberately disagreement-enriched gold set that over-weights contested items roughly threefold, so it is not a population error rate.

**Point at:** `human_validation.json` (18 over-flags + 9 missed = 27); `docs/HUMAN_LABEL_PROTOCOL.md` on the ~3× enrichment.

**Concede:** What does *not* cancel is classifier error that depends on which member produced the text, and I can't exclude that — see the differential-misclassification question below. It's also why I don't lead with this number: it doesn't survive BH.

---

### **"Your over-refusal regex caught two of sixty-three genuine refusals. What does a reported over-refusal rate of 0.05 actually measure?"**

It measures whether the response contains a canonical refusal template phrase. That's phrase matching, not refusal. Against my human gold set the regex recovered two of sixty-three full refusals, strict kappa minus 0.006, while the three-class judge reached 0.485 strict and 0.662 broad. Concretely, across my ten fp16 and NF4 aliases the regex reports over-refusal anywhere from 0.004 to 0.128, median 0.030, while the judge's strict rate runs 0.124 to 0.232, median 0.156. So the regex understates refusal, and by a factor that varies by model — about five-fold at the median, but only 1.7-fold on Phi.

**Point at:** `xstest_human_validation.json`; per-alias rates in `xstest_judge_agreement.json`.

**Concede:** The report's limitation says only that the parser "may under-count nuanced refusals" — that's weaker than what my own gold set shows. I'd strengthen it.

---

### **"Kappa 0.59 against humans — so that's the classifier's accuracy against ground truth?"**

No, and I'd resist that reading of my own number, for two reasons. First, the sample is disagreement-enriched by design: a hundred and twenty of the two hundred items are classifier-versus-regex disagreements, against a base disagreement rate near eighteen percent — roughly a threefold over-weighting. My protocol document says in terms that these are sample statistics, not population estimates, and that both scorers' population kappas would sit above their sample values. Second, it's one annotator, and that annotator is me, working from a rubric aligned to the classifier's own rules. What the number honestly supports is the *contrast* on contested cases: eighteen false-harmful calls from the classifier against a hundred and one from the regex.

**Point at:** `docs/HUMAN_LABEL_PROTOCOL.md`; `human_validation.json`.

**Concede:** The tool hid both scorers' labels, so it was label-blind — but items were **not** presented in randomised order, and there is no inter-rater kappa. A second annotator is open work (T30b).

---

### **"You claim no bit-width gradient in safety. But NF4 versus LLM.int8 are different algorithms. Have you tested bit-width at all?"**

No, and the claim should be read as a *method* claim, not a bit-width claim. INT8 here is LLM.int8, a mixed-precision method with a separate high-precision path for outlier features. It's a different algorithm, not NF4 at eight bits, so my fp16-to-INT8-to-NF4 ladder moves width and algorithm together. The defensible statement is the one in my precision-point section: across the two bitsandbytes methods I tested, I find no graded precision effect, descriptively, on two points. A clean bit-width claim needs one scheme at several widths. What *is* robust across both methods and both budgets is the capability side: all ten INT8 capability contrasts are statistically equivalent to full precision within a five-point margin and none is individually significant, while every significant capability loss is NF4-only.

**Point at:** the INT8 section of the report (states the algorithm distinction explicitly); `capability_equivalence.json`; `precision_sweep.json`.

**If they push:** At 128 tokens Llama's INT8 ASR delta was +0.040, significant under both judges; at 512 it is +0.005, p 1.0. Another budget artefact.

---

### **"Five pairs. What population of models does your null speak for?"**

Formally, these five pairs at this configuration and nothing wider. With five pairs there's no population to estimate a family-level effect from, and I run no interaction test, so I make no causal claim about family identity. What makes it more than five anecdotes is the spread it survives: four families, one-point-seven to seven-point-two billion parameters, and baseline ASRs from 0.070 on Phi to 0.585 on Mistral. Across that whole range no pair shows a significant harmful-compliance increase, while MMLU falls in all five pairs and ARC in four of five. That's a consistency argument, not an inference to the population of small models.

**Point at:** §7.5 of the report (no causal family claim); baseline ASRs in `genlen_robustness.json`.

**Concede:** Weakest at the low-ASR end — Phi starts at 0.070, so there's very little room downward — and per-pair MDE runs 0.044 to 0.086. Mistral's ARC actually moves *up* 0.9 points; say "four of five", not "all five".

---

### **"Your Qwen-1.7B delta is exactly zero. Doesn't a flat aggregate hide a change in *which* harms the model complies with?"**

It hides a genuine reshuffle, and I checked exactly that rather than leaving it implicit. Broken down by HarmBench semantic category, that pair moves in both directions: misinformation up about eighteen points, harassment up about eleven, while cybercrime and intrusion is down about eighteen and chemical-biological down about fourteen. They roughly cancel. At the prompt level it's high churn too — sixteen prompts flipping harmful-ward and sixteen safe-ward. But I report those category rates as descriptive only, because the cells hold nineteen to fifty-eight prompts, so movements of that size sit inside sampling noise and none is individually tested. So the honest statement isn't "nothing moved" — it's "no net change, and no single category driving an increase."

**Point at:** `results_512/analysis/harmbench_category_breakdown.json`, generated by its own script from the immutable raw records; the category-breakdown appendix (I.2).

---

## Generalist

### **"Explain quantization to me — I don't work in your area."**

A language model is a few billion numbers, and by default each is stored in sixteen bits. Quantization stores them in fewer bits — four, in my main condition — so the weights take roughly a quarter of the memory and the model fits on hardware someone could actually afford, a laptop or a single cheap GPU instead of a data-centre one. That four-times figure is on the weights only: the activations and the key-value cache still run in sixteen bits. The catch is that you're discarding precision in every one of those numbers, so the outputs shift. My whole project is one question about that shift: does it touch the model's willingness to refuse harmful requests, or only make it less competent? Those are very different deployment problems.

**Concede:** I never benchmarked the actual memory or speed saving, so I can only speak to the cost side.

---

### **"You keep saying NF4. What does it do that plain rounding to four bits doesn't?"**

NF4 is four-bit NormalFloat, from the QLoRA paper, and it does two things naive rounding doesn't. First, the sixteen levels aren't evenly spaced — they sit on the quantiles of a normal distribution, because trained weights are roughly normally distributed, so the resolution is where the weights actually are. Second, it quantizes blockwise: each block carries its own scale from its largest absolute value, so one outlier can't wreck a whole tensor, and double quantization compresses those scale constants too. At run time the weights are dequantized back to the compute dtype for each matrix multiply. So it's weight-only, applied at load time. That's set in one place in my code.

**Point at:** `_build_bnb_4bit_config` in `ethical_benchmark/models/loader.py` — `load_in_4bit`, `bnb_4bit_quant_type='nf4'`, `bnb_4bit_use_double_quant=True`.

**Concede:** I did not implement or re-verify the kernel. bitsandbytes is the artefact under test.

**If they push — "why not GPTQ or AWQ?":** They need a calibration corpus, which would become a confound between the two members of a pair. My external-validity section says method families may behave differently.

---

### **"Hand you a new model this afternoon — walk me through getting it into your study."**

No pipeline code changes — that was the design goal. I add two entries to the YAML config, the fp16 one and the four-bit one, sharing the same model identifier and the same pair id; the schema refuses to load unless the pair has both members and its benchmark list resolves. Then I pre-cache the weights on the cluster head node, generate the batch scripts, and run a five-sample smoke job first to confirm the model loads and uses its native chat template rather than the raw-prompt fallback. Then the full matrix, submitted pair by pair with direct sbatch, because our cluster gives me one GPU at a time and head-node Python is forbidden. Then the judge scoring pass and the analysis scripts.

**Point at:** the pair-completeness rule in `ethical_benchmark/quant/config_schema.py`; `slurm/jobs_tc1_smoke/`; the dashboard form that emits config + sbatch into `configs/generated/`.

**Concede:** A new *benchmark* is genuinely more work — that's a new plugin class.

---

### **"Be blunt — you found nothing. Isn't a null result a failed FYP?"**

I'd push back on both halves. It isn't nothing: the capability losses are real and survive correction — Qwen-1.7B drops nine points of MMLU — and the generation-length result overturned a significant safety finding I had myself reported, which is a positive result about how the field measures this. And I'm careful about what kind of null it is. With two hundred paired prompts, the smallest ASR change I could detect at eighty percent power is about six points, ranging 0.044 to 0.086 depending on how many prompts actually flip. Anything smaller is invisible to me, and that sentence is in the report and in the analysis file itself. So my claim is not "quantization is safe" — it's "nothing at the scale I could see showed up, and here is exactly how big my blind spot is."

**If they push — "so fix it":** More prompts. Quadruple n, halve the detectable effect.

**Concede:** Part of the MMLU gap is answer-format parsing, which the report discloses; I ran a strict-parser sidecar as a sensitivity check.

---

### **"Who actually changes what they do because of your result?"**

Anyone shipping a small model on constrained hardware — a phone, a laptop, one cheap GPU. They quantize because they have no choice, and the worry they carry is that four-bit makes the model likelier to help with harmful requests. What I can tell them is that across five pairs in four families, all under about seven billion parameters, for direct harmful requests at the benchmark's own reference generation budget, that increase is not what shows up. What shows up reliably is capability loss. So the advice inverts: budget for a capability regression test, don't assume a safety regression, evaluate the exact quantized artefact at your real generation budget rather than the full-precision release, and don't trust a keyword-matching refusal detector to tell you either way, because mine was wrong.

**Point at:** the deployment-guidance section of the report (§7.6).

**Concede:** I won't say quantized models are safe to deploy — I ran no jailbreaks, so I've said nothing about an adversary. And the advice doesn't extend above about seven billion parameters.

---

### **"If you started again tomorrow, what would you do differently?"**

Three things, in the order of what they cost me. First, I'd fix the generation budget to the benchmark's own reference value on day one, instead of discovering at the end that my shorter budget had cut off sixty percent of generations mid-sentence — that's a direct prefix test, not an estimate — and manufactured my headline. Correcting it meant re-running fifteen model variants and three thousand judge classifications. Second, I'd validate the scorer *before* the experiment, not after; I wrote a regex, trusted it, and eventually had to demote my own instrument. Third, I'd power the study properly: decide what effect size matters, then pick the sample size, instead of running two hundred prompts and computing the detectable effect afterwards.

**Concede:** I only half-applied that third one — I report the detectable effect honestly, but I never re-ran larger, so the study is still under-powered by my own standard.

---

## Engineering

### **"What code in this repository did you write, and what did you simply call?"**

Off the shelf: the models, the four benchmark datasets, HuggingFace transformers and bitsandbytes for loading, and the three judge models. Mine is everything that turns those into a matched-pair experiment — about nine and a half thousand lines under `ethical_benchmark`: a benchmark-plugin framework so each benchmark owns its loading and scoring, a config schema that refuses to run if a pair is missing a member, a matrix runner that loads each model once and frees GPU memory between models, the SLURM generation, the judge layer that writes redacted score sidecars, and the statistics layer. That last one I wrote by hand — the paired bootstrap, Benjamini-Hochberg, the McNemar detectable-effect using the Connor normal approximation, and the power calculation. Plus about eight thousand lines of tests and eighteen thousand lines of analysis scripts.

**Point at:** `git ls-files ethical_benchmark | xargs wc -l` → 9,505; `benjamini_hochberg`, `mde_mcnemar`, `power_mcnemar` in `scripts/multiple_comparisons.py`; `compute_bootstrap_ci` in `ethical_benchmark/metrics/aggregate.py`.

**If they push — "did you use statsmodels?":** No. There's no statsmodels in the repo at all. SciPy appears once, in the refusal-margin analysis.

**Concede:** I did not implement NF4. Quantization is one config object built in `models/loader.py` and applied at load time.

---

### **"I ran `make analyze` and got a different Mistral ASR delta than your report. Which is wrong?"**

Neither is wrong — they're two different scorers, and the report uses the one I validated. `make analyze` runs the legacy v2 refusal regex, and by default it reads `results/`, the retired 128-token tree, so you're seeing a difference in budget *as well as* scorer. On Mistral the regex gives a positive delta either way — plus 0.055 at 128, plus 0.060 at 512 — and labels the pair *alignment degradation*; the classifier gives minus 0.02 and labels it *alignment improvement*. Three of the five pair labels flip. That's not a bug I hid, it's the finding: the regex over-counts harmful compliance, kappa against the classifier runs 0.25 on Mistral to 0.84 on Llama, and against human labels the classifier gets 0.59 where the regex gets 0.11.

**Point at:** `Makefile` line 3, `RESULTS_DIR ?= results`; `results/analysis/pair_interpretations.json` vs `results_512/`; the file named `pair_interpretations.SUPERSEDED.md` sitting in the tree saying exactly this.

---

### **"You call your raw outputs immutable. What stops you quietly editing a number?"**

Three layers. First, every `raw.jsonl` and `summary.json` is SHA-256 hashed into a manifest — three hundred and forty entries across the 128 tree, the 512 tree and the multi-seed arm — and `make agent-check` recomputes them and fails on a changed hash or on a raw file present but not listed. Second, the runner refuses to delete: a guard function raises if a restart would unlink a raw file inside the protected trees, unless you set an explicit override variable. Third, `verify_report_claims` re-derives ninety-three locked numbers from the artefacts and fails if the document and the evidence disagree.

**Point at:** `results/raw_artifact_manifest.sha256` (340 lines, tracked); `_guard_protected_delete` in `run_quant_benchmark.py`; `make verify-claims`, which runs the claim registry, the report-number checks and the surface checks (run it in the room rather than quoting a total — the counts move with every commit).

**Concede:** Two honest limits. The check tolerates *absence* by design, so on a clean clone it passes vacuously. And the manifest is a file I can rewrite — this is tamper-evidence, and the real audit trail is the git history of that manifest.

---

### **"You changed the HarmBench scorer after the experiments ran. How did that not contaminate the results?"**

By never writing back. Every rescoring reads `raw.jsonl` and writes a *new* derived sidecar beside it: `scores.v2.jsonl` for the corrected regex, `scores.judge.harmbench_cls.jsonl` for the official classifier, plus the API-judge and LlamaGuard cross-checks — fifteen aliases, three thousand classifications, zero parse errors. The sidecars carry prompt IDs and booleans only, and there's an `_assert_redacted` function that blocks a text field from being written, with a test proving it fires. Be precise with me on the diagnostics CSV: it *does* have a `response_preview` column, and every one of its two thousand rows is the literal string REDACTED-underscore-BENCHMARK-underscore-OUTPUT. So anyone can re-audit my scoring decisions without a GPU and without re-exposing harmful generations.

**Point at:** `_assert_redacted` in `ethical_benchmark/judges/validation.py`; `tests/test_judge_validation.py::test_assert_redacted_blocks_text_fields`, `..._does_not_mutate_raw`, `..._idempotent`; the redaction patterns in `configs/artifact_policy.yaml`.

**Concede:** Two HarmBench scorer corrections, not three. The `parser_strict` sidecars are a capability answer-parsing sensitivity under `mmlu/` and `arc/`; there are none under `harmbench/`.

---

### **"You have several hundred tests. What do they prove — and what don't they touch?"**

They prove the plumbing, not the physics. Concretely: the config schema rejects bad configs, the SLURM generator emits the right bootstrap block and rejects an ambiguous walltime, the matrix runner loads each model exactly once, a generator returning fewer responses than prompts raises instead of silently dropping them, the judge layer never writes text fields, and ninety-three report numbers match the artefacts. What they do not touch: no test runs on a GPU, none loads a real model, and none validates the NF4 numerics — the model stack is monkeypatched with a dummy loader so the suite runs on my laptop in seconds. So the tests can tell you my pipeline is correct *given* the generations; they cannot tell you the generations are correct.

**Point at:** `pytest --collect-only` → 560; `tests/test_slurm_helpers.py::test_slurm_time_rejects_bare_minutes`; the short-generator `RuntimeError` case in `tests/test_quant_smoke.py`.

**If they push — "do your gates prove they'd catch a regression?":** For the locks I added in the R2 hardening pass, yes — `tests/test_r2_locks_must_fire.py` deliberately regresses each and asserts the gate fires. That's four must-fire tests plus a clean-state test, not blanket coverage of all ninety-three.

---

### **"Your jobs ran on a cluster you can't run code on. How do you know they used the weights and prompt format you claim?"**

Each generation record carries its own provenance, and I want to be precise about what that does and doesn't prove. Every row in `raw.jsonl` records the model id, pair id, seed, generation config, timestamp, and a `chat_templated` flag — the generator exposes `prompt_was_templated`, so the artefact proves the native chat template was applied rather than the silent raw-prompt fallback, instead of me trusting a log line. The sbatch files are generated, not hand-written, and all ten export the three HuggingFace offline flags, so a compute node cannot redownload a different snapshot. On resume the runner de-duplicates by prompt id.

**Concede:** It does **not** prove the weight revision. The raw record has no revision field, and I only pinned commit SHAs in `configs/tc1_512.yaml` on the ninth of July, whereas the 512 runs are dated the twenty-ninth of June. So the pins secure a future re-run; what this study actually rests on is the offline TC1 cache.

---

### **"Let me clone your repo fresh and run that command. Does it work?"**

Partly, and I should be honest about which part. On a fresh clone, `judge_agreement.py` runs and reproduces the judge table byte-identically — I tested that. But `multiple_comparisons.py` fails with a file-not-found on the MMLU raw file, because the raw generation files are deliberately gitignored: they contain model responses to harmful prompts and I chose not to publish those. So ten of the twenty contrasts in the BH family — the five ASR and the five over-refusal rows — regenerate cold from committed redacted sidecars; the ten MMLU and ARC rows need the local raw tree.

**Concede:** The README sentence saying all inputs are committed is over-stated for that script. The fix is a redacted per-prompt correctness sidecar for MMLU and ARC — the same pattern I already use for XSTest.

---

### **"Reproduce this from scratch — not your CSVs, the experiment. What breaks?"**

Four things, and preparing for today I found a real defect in my own instructions. One, hardware: a V100-class GPU on SLURM, and Mistral-7B is gated on HuggingFace so you need an accepted licence and a token. Measured from the record timestamps the full ten-alias sweep is about fourteen GPU-hours — twenty minutes to under four hours per model, well inside the six-hour cap. Two, environment, and this is the defect: my README tells you to install with `constraints-tested.txt`, and that command fails with a resolution error, because the constraints file records my local *analysis* interpreter at transformers 4.32.1 while requirements floors it at 4.41. Bitsandbytes isn't constrained at all. Three, no pip freeze and no CUDA pin, so I expect near-identical, not bit-identical, generations. What does reproduce exactly today is the analysis layer, seeded at forty-two.

**Concede:** the constraints defect is real and is tracked as T60 — say so plainly rather than talking around it. (Both stale rows this answer originally cited in `docs/REPRODUCIBILITY.md`, the multi-seed pair count and the manifest count, were corrected on 2026-08-11; the install-resolution defect is the one that remains open.)

---

## Hostile

### **"Your motivation is memory and latency. Show me where you measured either. One number."**

Nowhere. I never measured memory or latency — there is no timing or memory instrumentation anywhere in the package, and the summary files record only the generation config, the device and the seed. So the benefit side of quantization is taken from the literature, not from my own measurement. I should be precise about what my title means: the trade-off I study is safety against capability, not accuracy against efficiency. And what I find is that at four bits, under direct requests, I detect no significant increase in harmful compliance in any of the five pairs — capability is what moves. That makes the deployer's decision look one-sided, but I have not weighed the size of the prize.

**Concede:** §7.2 lists six external-validity bounds and efficiency isn't one of them. It should be named as an explicit seventh bound.

---

### **"Prove 512 isn't doing the same thing 128 did. What fraction of your 512-token responses hit the ceiling?"**

I cannot give you that number, and that is the honest answer. My truncation proof at 128 is a direct prefix test — under greedy decoding a cut-off 128-token response must be a proper prefix of its 512-token counterpart, and 60.3 percent of them were, with Mistral's baseline at 93.5. To run the same test at 512 I would need a 1024-token arm, and I don't have one. I also never logged a finish reason, so I can't count ceiling hits after the fact, and the 62 percent figure in my report is responses exceeding the *128* ceiling, not the 512 one. What worries me is Mistral, median about two thousand characters, and Phi at about twenty-five hundred.

**Point at:** `prefix_truncation_128` in `genlen_robustness.json`; `scripts/genlen_512_analysis.py`, `prefix_truncation_block`.

**Concede:** 512 is defensible as HarmBench's own reference budget; it is *not* demonstrated to be non-truncating. There's also a 9.2 percent bucket where the two budgets diverge outright rather than nesting.

---

### **"You label pairs off deltas of two and four points that your own power analysis says you can't detect. Are you not naming noise?"**

For four of the five pairs, yes — those labels are descriptive, not inferential, and my results card marks them *directional* for exactly that reason. The classifier function uses fixed a-priori tolerances, 0.02 on harm and 0.03 on capability, and my per-pair minimum detectable effect runs from 0.044 to 0.086, so a label can move on noise. It demonstrably did: Phi's ΔASR of 0.09 minus 0.07 evaluated to 0.019999999999999997 in float64 and mislabelled the pair *robust preservation* until I rounded to six decimal places — the code comment records it. So the labels answer the *shape* of my research question; the inferential claim is the BH-FDR table alone.

**Point at:** `classify_pair_change` in `compare_quant_pairs.py` and its rounding comment; `docs/RESULTS_CARD.md` rows marked "(dir.)".

**Concede:** Only Qwen-1.7B's label is backed by a surviving contrast. If I redesigned it, the label would be gated on significance, not on a point estimate.

---

### **"Your delta is only unbiased if classifier error is the same for fp16 and four-bit outputs. Did you check?"**

No, I did not, and I cannot rule out differential misclassification. I scored the two-hundred-item gold set pooled, so I have no per-condition precision. What I can offer is indirect. Cross-judge agreement *is* computed per alias, and for four of five pairs the classifier and gpt-4o agree about equally well on the full-precision and the four-bit member — 0.785 against 0.800 for Qwen-1.7B, 0.942 against 0.954 for Llama — which is at least consistent with stable behaviour across conditions. Phi is the exception, 0.678 against 0.826.

**Point at:** per-alias kappas in `judge_pairwise_agreement.json`.

**Concede:** On my one significant result, gpt-4o reproduces the Llama decrease at minus 0.035, but LlamaGuard gives minus 0.03 and non-significant — so it is not judge-invariant. And the real test is cheap: my per-item sheet is on my machine, gitignored because it holds raw harmful text, so stratifying it by condition is a re-analysis, not a re-run. I should have done it.

---

### **"Your most robust result is nine points of MMLU. Break it down by subject. How much is one subject?"**

Twenty-seven items out of three hundred, and twenty-three of those twenty-seven sit in two subjects: high-school macroeconomics loses fifteen and clinical knowledge loses eight. World history loses one, college biology two, business ethics three, and human aging actually gains two. So the loss is real and survives correction at q equals 0.008, but I have not shown it is uniform across knowledge domains, and per-subject n is only twenty-five to ninety-four, so I won't build a story about macroeconomics specifically.

**Point at:** `results_512/analysis/mmlu_subject_breakdown.json`; the six subjects are declared in `configs/tc1.yaml` before the run.

**Concede:** My second capability anchor disagrees in magnitude for that pair — ARC, on eleven hundred and seventy-two items, moves minus nine-tenths of a point at p 0.46, not significant. I'd state it as a capability loss on one anchor, concentrated in some subjects, rather than a general capability collapse.

---

### **"You demoted the regex for HarmBench but kept it for XSTest — and then dismissed the one surviving over-refusal result as a regex artefact. That's choosing your instrument by its answer."**

The direction of that dismissal runs against me, not for me. The surviving over-refusal result is a *decrease* — Phi refusing fewer benign prompts under quantization — which is a flattering result, and I'm the one arguing it's probably a measurement artefact. On the instrument: XSTest ships no released classifier, so I kept the deterministic regex as the pre-specified scorer for reproducibility, then tested it rather than trusted it. The gold set is damning for it: the regex flags seven refusals where the annotator marks sixty-three full refusals, recall of two in sixty-three, kappa essentially zero, against the judge's 0.485. Rather than swap scorers, I ran a parallel judge-strict family alongside the registered one and report both.

**Point at:** `xstest_human_validation.json`; `multiple_comparisons_judge_strict.json` — 2 survivors, both capability.

**Concede — say these first:** That gold set carries a disclosed protocol deviation. The blindness exclusion ran prompt-wide across five prompt ids in all ten aliases, rather than the ten pre-specified alias-prompt items, so under my own pre-registration that's Outcome X and the clean confirmatory badge is withdrawn. Footprint at most fourteen of two hundred items; on the intersection the kappa gap is unchanged at 0.51. And the judge-strict family is composition-locked but **not blind** — the note says so itself, because those p-values were already known.

---

## Questions with no good answer in the documents

These five came out of my own adversarial examiner panel and were recorded as gaps. Answer each by conceding cleanly, then naming the fix. Do not improvise past the concession.

### G1. "Both Qwen pairs ran with thinking mode off — and you turned it off because it would eat the very token budget you say is decisive. Isn't your contribution eating itself?"

That's the sharpest form of the limitation and I'll take it. Yes — `enable_thinking` is false for both Qwen pairs, it's a token-budget necessity at 512, and it's the same knob I argue is decisive. So two of my five pairs were evaluated in a configuration Qwen3 doesn't ship as its default, and quantization's interaction with thinking-mode generation is completely untested. The other three pairs — Llama, Mistral and Phi — are unaffected; their chat templates ignore the flag. My discussion states the bound in those words, but it does not state the circularity, and it should. What follows is that my null covers Qwen3 in non-thinking mode, not Qwen3 as shipped. The fix is a factorial: rerun both Qwen pairs with thinking on at a larger budget, say two thousand and forty-eight tokens, and treat mode as a second factor beside precision. Which way it would move ASR, I don't know.

**Point at:** `enable_thinking=False` in `ethical_benchmark/models/generation.py` (with a TypeError retry); Table 3.3 decoding controls, marked Critical; the sixth external-validity bound in §7.2, which cites the Qwen3 technical report.

**Say "three families / three pairs unaffected", not "two" — the study has four families and Qwen is one of them.**

---

### G2. "Your abstract says a refusal-margin probe 'independently supports' your conclusion. Your body says it's near-chance in the one pair that moved. Which do you stand behind?"

The body. "Independently supports" was an over-claim, my own adversarial audit flagged it, and the current build says the probe "is consistent with" the capability-driven reading — that's the strength I earned. On substance: the probe measures the refusal margin at the first generated token. It passes its validation gate for four of my five pairs, area under the curve 0.86 to 0.90, but only 0.69 for Mistral, which opens with a refusal-shaped preamble and then complies. For Qwen-1.7B, the pair my old headline hung on, the first-token margin shifts strongly, but the whole-sequence *minimum* margin moves the other way with an interval crossing zero, and the flip asymmetry the probe was keyed to — sixteen harmful-ward against five — is itself a short-budget artefact: at the reference budget it's sixteen and sixteen. So it's corroborating colour. Nothing in my conclusions depends on it.

**Point at:** `results_512/analysis/refusal_margin.json` — gate AUCs 0.864/0.871/0.871/0.902 with Mistral 0.692; qwen_2b sequence-minimum Cohen's dz −0.046 with CI spanning zero; pooled AUC 0.763 but within-pair 0.54 on Mistral and 0.61 on Qwen-1.7B.

---

### G3. "Hong et al. is your nearest analogue, but they evaluate GPTQ and AWQ, which your own limitations chapter says may differ. How much corroboration is that?"

Less than a casual reading suggests, and I'd rather say so than defend it. Hong et al. report that four-bit compression broadly preserves trustworthiness, which is directionally the same picture. But their models are seven billion and above, their construct is eight trust dimensions rather than harmful compliance on HarmBench, and their compression paths are not bitsandbytes NF4 applied on the fly. My own external-validity section says method families may differ, so I can't lean on them as corroboration while saying methods aren't comparable. The phrase I use in the report is the one I can defend: consistent in direction, not independent replication.

**Point at:** the positioning table (Table 2.1) — the Hong row reads "7B and above", safety "yes (8 trust dimensions)", capability anchor **yes**, matched weights "not reported", significance test "not reported"; §7.7 already says "consistent with that finding and extends it".

**Do not claim a capability anchor as your addition — Hong has one.** What you add is matched weights, a significance test, and the scorer validation.

---

### G4. "Regenerate the headline in front of me. I want the BH result out of the artefacts, not the document."

Two commands from the repo root, no GPU. First, `python scripts/judge_agreement.py --results-dir results_512` — that reads the redacted judge sidecars and the regex sidecars and rebuilds the per-pair HarmBench deltas, the paired-bootstrap CIs at seed forty-two, and the McNemar discordant counts. Then `python scripts/multiple_comparisons.py --results-dir results_512` — it reuses those HarmBench McNemar results and computes the exact test fresh for MMLU, ARC and XSTest from the per-prompt outcomes, applies Benjamini-Hochberg across all twenty contrasts, and prints family size twenty, five uncorrected significant, three surviving, with a representative ASR minimum detectable effect of 0.0594. Then `make verify-claims` re-checks the document against those files — the claim registry, ninety-three locked report numbers, and three hundred and forty-two surface checks.

**Point at:** both commands are in the README; I re-ran the chain this week and the JSON came back byte-identical, with `verify_report_claims` returning 93 pass / 0 fail.

**Concede — this is the actual gap:** the chain works, but it is nowhere written down *as a chain* in the report. That's a documentation defect, and the fix is a "regenerating the results" appendix. Also be precise: `multiple_comparisons.py` does **not** recompute the five HarmBench p-values — it reuses the committed McNemar results from `judge_agreement`, by design, so the primary scorer is computed in exactly one place. And on a fresh clone only ten of the twenty rows regenerate (see the clone question in Engineering).

---

### G5. "State plainly what effect sizes your study cannot rule out — and why that's disclosed rather than buried."

At two hundred paired prompts and the flip rates I observe, my minimum detectable ASR change at eighty percent power runs 0.044 to 0.086 across pairs, representative 0.059. Anything smaller than that is invisible to my design. The bound I'd actually stand behind is the interval, not the power number: my widest ninety-five percent upper edge anywhere is plus 0.080, on Qwen3-4B, so an increase of up to about eight points on that pair is not excluded. It's disclosed because a null with no stated floor is not a result, it's a failure to look — so the floor sits in the analysis artefact's own note field, in the multiple-comparison section, in the limitations chapter, and on the results card. If I could rerun it, the fix is sample size, not statistics: quadrupling n roughly halves the floor, and around eighteen hundred paired prompts would get me to 0.02.

**Point at:** the `power_analysis` block of `multiple_comparisons.json`, including its own note that "absence of significance is bounded by power, not proof of a zero effect".

**Concede:** One sentence in Chapter 8 currently says the evidence "rules out" increases above the detectable effect. That's wrong and I'd correct it to the interval statement — power is a property of the design, not an exclusion drawn from the data.

---

## Numbers I must not get wrong

| Thing | Value |
|---|---|
| **ASR Δ, Qwen3-1.7B** | +0.000, CI [−0.055, +0.055], p 1.0000, n.s. (base 0.255) |
| **ASR Δ, Qwen3-4B** | +0.040, CI [0.000, +0.080], p 0.0963, n.s. (base 0.115) |
| **ASR Δ, Llama-3.2-3B** | **−0.040**, CI [−0.075, −0.010], p 0.0215, **sig — a decrease** (base 0.100) |
| **ASR Δ, Mistral-7B** | −0.020, CI [−0.080, +0.040], p 0.6271, n.s. (base 0.585) |
| **ASR Δ, Phi-4-mini** | +0.020, CI [−0.015, +0.055], p 0.4240, n.s. (base 0.070) |
| **Llama discordant counts** | b=1, c=9. Move **one** prompt → p 0.109, gone |
| **BH family** | 20 contrasts (5 pairs × 4 metrics), α 0.05; 5 uncorrected sig; **3 survive**; **0 ASR** |
| **Survivor 1** | Qwen-1.7B MMLU −0.090, p 0.00058, q 0.0080 |
| **Survivor 2** | Llama ARC −0.0316, p 0.00080, q 0.0080 |
| **Survivor 3** | Phi XSTest over-refusal −0.048, p 0.00183, q 0.0122 (a **decrease**) |
| **Bonferroni ×20** | 0.012 / 0.016 / 0.037 — all three still survive |
| **Judge-strict parallel family** | 4 uncorrected sig, **2 survive**, both capability |
| **MDE (ASR, 80%)** | per-pair 0.0443 → 0.0864; representative **0.0594**; median discordant rate 0.09 |
| **Truncation at 128** | **60.3%** provably truncated (1206/2000); 30.5% natural stop; 9.2% mismatch; Mistral base 93.5% |
| **Response length** | median 567 → 1675 chars; 62% of 512 responses exceed the 128 ceiling (1240/2000) |
| **Regex vs classifier κ** | 0.254 (Mistral-4bit) → 0.836 (Llama-4bit) |
| **HarmBench human gold (n=200)** | classifier κ **0.5908** vs regex κ **0.1095**; over-flags 18 vs 101; human harmful rate 0.185 |
| **XSTest human gold (n=200)** | judge strict κ **0.4848**, broad 0.6617; regex strict κ **−0.0063**; regex recall **2/63** full refusals; 63 full / 34 partial / 103 compliance |
| **Second judge (gpt-4o)** | cross-judge κ **0.678 → 0.954** over 10 aliases; Llama Δ −0.035, p 0.0156 |
| **Third judge (LlamaGuard)** | κ **0.362 → 0.918**; Llama Δ −0.03, **p 0.146 n.s.**; every per-pair Δ n.s. |
| **XSTest rates (10 aliases)** | regex OR 0.004 → 0.128, median 0.030; judge strict 0.124 → 0.232, median 0.156 |
| **INT8 capability (TOST)** | 10/10 equivalent at the pre-fixed 0.05 margin; tightest data-derived margin 0.0378 |
| **INT8 safety (uncommitted)** | tightest margin all 5 pairs would pass ≈ 0.075, driven by Qwen3-4B |
| **Multi-seed (T=0.7, 5 seeds)** | 4 individually sig seeds total: 1 increase (Qwen-4B seed2 +0.050, p 0.0129), 3 decreases |
| **Qwen-1.7B categories** | misinfo +0.176 (n 34), harassment +0.105 (n 19), cyber −0.175 (n 40), chem-bio −0.143 (n 28); cells 19–58 |
| **Qwen-1.7B MMLU** | −27 of 300; macroeconomics −15 (n 94), clinical −8 (n 48); human aging **+2** |
| **Refusal-margin gate AUC** | 0.864 / 0.871 / 0.871 / 0.902, Mistral **0.692**; pooled flip AUC 0.763 |
| **Scale** | 5 pairs, 4 families, 1.7–7.2B; n=200 HarmBench, 250 XSTest, 300 MMLU, 1172 ARC |
| **Repo** | 9,505 lines in `ethical_benchmark`; 340 manifest hashes; 3,000 judge classifications, 0 parse errors |
| **Repo counts that move** | Test, claim-lock and surface-check totals change with every commit — do **not** memorise them. If an examiner asks, say "several hundred, and I can run them right now", then run `make agent-check` and read the live figures off the screen. That is a stronger answer than a memorised total, and it cannot be stale. |