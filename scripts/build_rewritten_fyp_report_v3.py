#!/usr/bin/env python3
"""Build the plain-academic-English v3 rewrite without modifying v1 or v2.

The v1 builder remains the evidence and layout source. This builder renders it
to a temporary DOCX, rewrites the prose layer into natural, cohesive academic
English with refined essay flow, verifies that numerical tokens and citation
markers are unchanged, and writes a separate v3 artifact.
"""

from __future__ import annotations

import importlib.util
import re
import tempfile
from pathlib import Path

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build_rewritten_fyp_report_v1.py"
OUT = ROOT / "docs" / "rewritten_fyp_report_v3.docx"


HEADING_REWRITES = {
    "2.5 Automatic Evaluators and Human Grounding":
        "2.5 Automatic Evaluators and Human Validation",
    "5.3 Human Grounding of Harmful-Compliance Scorers":
        "5.3 Human Validation of Harmful-Compliance Scorers",
    "6.3 Capability as the Deployment Cost":
        "6.3 Capability Effects and Deployment",
    "6.4 Scorer Validity as a Research Result":
        "6.4 Validity of the Scoring Methods",
    "Figure 1.1. Capability-anchored safety space using MMLU and HarmBench deltas":
        "Figure 1.1. Changes in MMLU accuracy and HarmBench harmful compliance",
    "Table 6.1. Competing explanations for the HarmBench null":
        "Table 6.1. Possible explanations for the HarmBench result",
    "Table 6.2. Published quantization effects placed beside this study's":
        "Table 6.2. Comparison with published quantization results",
}

# Cached front-matter pagination from the verified 59-page LibreOffice render.
# Word refreshes the TOC field when opened, but the visible cached TOC and the
# lists of figures/tables must still be correct before that refresh occurs.
PAGE_NUMBER_REWRITES = {
    "1.3 Research Questions": 2,
    "1.5 Contributions": 3,
    "1.6 Report Structure": 3,
    "Chapter 3: Methodology and Experimental Setup": 14,
    "3.1 Matched-Pair Experimental Design": 14,
    "3.2 Models and Precision Conditions": 15,
    "3.3 Benchmarks and Outcomes": 16,
    "3.4 Scoring and Validation": 17,
    "3.5 Statistical Analysis": 17,
    "3.6 Implementation and Execution": 19,
    "3.7 Reproducibility Controls": 20,
    "3.8 Analysis Workflow and Quality Control": 20,
    "Chapter 4: Results": 22,
    "4.1 Results at a Glance": 22,
    "4.2 Harmful Compliance (RQ1)": 23,
    "4.3 Over-Refusal on Benign Prompts (RQ2)": 24,
    "4.4 Capability (RQ3)": 25,
    "4.5 Model Scale and Family Differences (RQ4)": 26,
    "4.6 INT8 Precision Point": 26,
    "4.7 Research-Question Synthesis": 28,
    "Chapter 5: Scorer Validation and Robustness": 29,
    "5.1 HarmBench Classifier versus Refusal Proxy": 29,
    "5.2 Independent HarmBench Judges": 30,
    "5.3 Human Validation of Harmful-Compliance Scorers": 31,
    "5.4 XSTest Scorer Sensitivity and Human Validation": 31,
    "5.5 Generation-Length Robustness": 32,
    "5.6 Multi-Seed Decoding Sensitivity": 33,
    "5.7 Power and the Meaning of the Null": 34,
    "5.8 Robustness Synthesis": 35,
    "Chapter 6: Discussion": 36,
    "6.1 Answer to the Central Question": 36,
    "6.2 Relationship to Prior Work": 36,
    "6.3 Capability Effects and Deployment": 39,
    "6.4 Validity of the Scoring Methods": 40,
    "6.5 Threats to Validity": 41,
    "6.6 Practical Implications": 42,
    "Chapter 7: Conclusion and Future Work": 43,
    "7.1 Conclusions": 43,
    "7.2 Prioritised Future Work": 43,
    "References": 45,
    "Appendix A: Reproducibility Summary": 48,
    "Appendix B: Minimum Detectable Capability Effects": 49,
    "Appendix C: Human-Validation Boundaries": 50,
    "Figure 1.1. Changes in MMLU accuracy and HarmBench harmful compliance": 2,
    "Figure 3.1. Matched pairs, precision arms, benchmark battery, and analysis flow": 15,
    "Figure 4.1. Per-pair HarmBench classifier ΔASR with paired confidence intervals": 24,
    "Figure 4.2. Precision sweep across fp16, INT8, and NF4": 28,
    "Figure 5.1. HarmBench classifier ASR versus the refusal-regex proxy": 30,
    "Figure 5.2. Qwen3-1.7B greedy result and five-seed HarmBench sensitivity": 34,
    "Table 3.1. Model pairs and precision conditions": 15,
    "Table 3.2. Benchmark roles, sample sizes, and primary outcomes": 16,
    "Table 4.1. NF4 safety and over-refusal results": 22,
    "Table 4.2. NF4 capability results": 22,
    "Table 5.1. Per-pair HarmBench deltas under three judges": 31,
    "Table 5.2. HarmBench classifier deltas at 128 and 512 tokens": 33,
    "Table 5.3. Five-seed HarmBench sensitivity": 33,
    "Table 6.1. Possible explanations for the HarmBench result": 36,
    "Table 6.2. Comparison with published quantization results": 37,
    "Table A.1. Primary run controls": 48,
    "Table B.1. Capability effects and minimum detectable effects": 49,
}


# Full-paragraph rewrites for version 3 ensure fluid, natural academic prose
# with strong argumentative progression and coherent transitions.
PARAGRAPH_REWRITES = [
    (
        "Post-training quantization makes compact instruction-tuned language models practical",
        "Post-training quantization significantly reduces the computational and memory requirements "
        "of instruction-tuned language models, but its impact on behavioural safeguards remains an active "
        "area of research. This report investigates whether four-bit quantization alters harmful "
        "compliance or benign prompt refusal, and whether apparent shifts in safety reflect genuine "
        "alignment changes or underlying capability degradation. Five matched model pairs from the Qwen, "
        "Llama, Mistral, and Phi families were evaluated across fp16, INT8, and NF4 precision levels. "
        "HarmBench measured harmful compliance, XSTest assessed over-refusal on benign requests, and "
        "MMLU and ARC-Challenge served as general capability benchmarks. The statistical framework "
        "applied paired bootstrap confidence intervals, exact McNemar tests, and Benjamini–Hochberg "
        "false discovery rate control. Scorer validity was independently evaluated by cross-checking the "
        "HarmBench classifier against an API judge, Llama Guard, and human-annotated samples. At the "
        "standard HarmBench reference limit of 512 new tokens, no NF4 change in harmful compliance "
        "remained statistically significant after multiple-comparison correction. Three results remained "
        "significant: reduced MMLU accuracy for Qwen3-1.7B, reduced ARC accuracy for Llama-3.2-3B, and "
        "decreased regex-scored over-refusal for Phi-4-mini. Human validation revealed that the semantic "
        "refusal judge aligned substantially better with annotators than the keyword regex, indicating "
        "that the Phi refusal finding is sensitive to scorer choice. Within the evaluated direct-request "
        "setting, NF4 quantization caused measurable capability costs but produced no statistically "
        "significant increase in harmful compliance after correction.",
    ),
    (
        "Compact instruction-tuned language models have moved capable generation into a deployment range",
        "Compact instruction-tuned language models such as Qwen3 [1], the Llama 3 family [2], Phi-4-mini "
        "[3], and Gemma 2 [4] can now run on consumer GPUs and some edge devices. Local execution can keep "
        "user inputs on the device rather than sending them to a remote service, but memory capacity remains "
        "a practical constraint. Full-precision weights may exceed the memory available on the target device, "
        "so weight compression is one route to fitting these models on constrained hardware [5]. This report "
        "examines the behavioural and safety consequences of that compression.",
    ),
    (
        "Post-training quantization addresses that constraint by representing weights",
        "Post-training quantization is one way to address this constraint: it represents model weights, and "
        "in some methods activations, with fewer bits. AWQ reports four-bit weight compression and faster "
        "inference on desktop and mobile GPUs [5], while GPTQ demonstrates one-shot weight quantization for "
        "models scaling to 175 billion parameters [6]. Empirical scaling studies also suggest that four-bit "
        "inference can offer a favourable balance under a fixed memory budget [7]. These examples make "
        "four-bit quantization a relevant deployment condition for studying how reduced precision affects "
        "model behaviour.",
    ),
    (
        "The quantization method papers reviewed here report perplexity, accuracy, latency, or memory",
        "Most literature on quantization focuses primarily on perplexity, task accuracy, latency, and memory "
        "consumption [5]-[7]. Comprehensive benchmarks indicate that general task performance often remains "
        "usable at four bits [8], [9], but average accuracy alone cannot establish that safety alignment is "
        "preserved. Safety-focused studies observe heterogeneous effects across model families, bit widths, "
        "and quantization algorithms [10]. Quantization can also distort model output confidence when top-1 "
        "accuracy remains stable [11], introduce potential security vulnerabilities [12], and cause "
        "language-dependent performance losses that automatic evaluators may underestimate [13]. These findings "
        "leave an interpretive question: when a safety score changes, is the cause alignment, capability, or "
        "the measurement procedure itself?",
    ),
    (
        "Refusal behaviour gives a concrete reason for concern",
        "Refusal behaviour provides a concrete reason to evaluate quantized models directly. Instruction tuning "
        "and safety alignment teach models policies for refusing harmful prompts [14]. Mechanistic analyses "
        "indicate that refusal is often mediated by concentrated directions in activation space [15], whereas "
        "other work suggests that alignment safeguards can be fragile during generation [16] or easily bypassed "
        "through small-scale fine-tuning [17]. These findings do not prove that standard NF4 quantization "
        "compromises safety, but they do show why post-quantization behaviour should be measured rather than "
        "assumed to remain intact.",
    ),
    (
        "Safety evaluation is difficult because refusal is not synonymous with safety",
        "Evaluating model safety is challenging because refusal behaviour is not synonymous with true safety. "
        "A model that indiscriminately refuses every prompt may appear harmless but cease to be practically "
        "useful. XSTest was established to detect excessive refusal on benign prompts [18], a problem later "
        "examined at larger scale by OR-Bench [19]. Conversely, an unaligned model may produce unsafe responses "
        "without containing explicit refusal keywords. Existing evaluations therefore leave an interpretive "
        "gap: a reported safety score may change because of an alignment shift, general capability degradation, "
        "or systematic bias in the evaluation rubric.",
    ),
    (
        "This project asks whether quantization changes the balance among harmful compliance",
        "The objective of this study is to determine how post-training quantization changes harmful compliance, "
        "benign over-refusal, and general capability in compact language models. Its central distinction is "
        "between a genuine alignment shift and capability degradation: a lower attack-success rate cannot be "
        "interpreted as improved alignment if the model has lost the competence required to follow instructions. "
        "The study therefore evaluates all three dimensions together and validates the scoring instruments, "
        "so that an apparent safety change can be interpreted against both capability and measurement evidence.",
    ),
    (
        "Figure 1.1 previews the answer",
        "Figure 1.1 places each model pair's change in MMLU accuracy alongside its change in harmful compliance "
        "scored by the official HarmBench classifier, against the interpretation regions defined in Chapter 3. "
        "Paired bootstrap confidence intervals are shown for both measures. None of the five intervals provides "
        "evidence of an increase in harmful compliance. The Llama pair interval excludes zero only in the "
        "direction of decreased harmful compliance, while all five MMLU point estimates show capability declines. "
        "Chapter 4 reports the results after multiple-comparison correction, and Chapter 5 examines whether the "
        "interpretation changes under independent scoring methods.",
    ),
    (
        "The primary experiment covers five matched fp16–NF4 pairs from four model families",
        "The primary experiment covers five matched fp16–NF4 pairs from four model families: Qwen3-1.7B, "
        "Qwen3-4B, Llama-3.2-3B, Mistral-7B, and Phi-4-mini. A separate INT8 comparison uses the same "
        "baselines. The safety threat model is HarmBench DirectRequest: models receive unsafe instructions "
        "without an optimised jailbreak suffix or attack loop. The study does not evaluate latency, energy "
        "use, privacy, monetary cost, or general adversarial robustness; its safety conclusions therefore apply "
        "only to direct harmful requests under the tested configurations.",
    ),
    (
        "The capability instruments are a six-subject MMLU subset and ARC-Challenge",
        "General capability is evaluated using a representative six-subject MMLU subset and ARC-Challenge. "
        "These multiple-choice instruments provide useful anchors across different reasoning domains, but they "
        "do not cover every form of model capability. The primary HarmBench evaluation uses a reference "
        "generation budget of 512 new tokens under greedy decoding, supplemented by a five-seed stochastic "
        "decoding analysis to quantify generation variance.",
    ),
    (
        "A controlled matched-pair design that applies NF4 or LLM.int8 at load time",
        "Controlled matched-pair design. The study applies NF4 or LLM.int8 at load time to the same "
        "revision-pinned model weights, keeping the checkpoint constant while changing the precision condition.",
    ),
    (
        "A joint evaluation of harmful compliance, benign over-refusal, MMLU, and ARC-Challenge",
        "Joint behavioural evaluation. The study evaluates harmful compliance, benign over-refusal, MMLU "
        "accuracy, and ARC-Challenge accuracy across five matched model pairs.",
    ),
    (
        "A paired statistical analysis with bootstrap confidence intervals",
        "Paired statistical analysis. The study combines bootstrap confidence intervals, exact McNemar tests, "
        "multiplicity control, and power-oriented interpretation.",
    ),
    (
        "A scorer-validation study using the HarmBench classifier",
        "Scorer-validation study. HarmBench results are cross-checked using the HarmBench classifier, an "
        "independent API judge, Llama Guard, and two human-labelled samples.",
    ),
    (
        "A reproducible benchmark implementation with redacted derived sidecars",
        "Reproducible benchmarking pipeline. The implementation preserves immutable raw model outputs and stores "
        "derived scorer classifications in separate, verifiable audit sidecars.",
    ),
    (
        "The literature also sets the report's claim boundary",
        "The literature also defines the empirical boundaries of this research. Prior studies on adversarial "
        "attacks demonstrate that targeted worst-case failures can occur [12], [39], while jailbreak research "
        "shows that robustness requires dedicated adversarial evaluation [43]. Studies on automated evaluators "
        "further underscore the necessity of validating model-based judges against human ground truth [45], [46]. "
        "Accordingly, the conclusions of this report apply specifically to standard direct requests in unmodified "
        "checkpoints: detectable capability losses occurred, but no statistically significant increases in "
        "harmful compliance were observed after correction.",
    ),
    (
        "The primary multiplicity family contains 20 NF4 contrasts",
        "The primary statistical family encompasses 20 NF4 comparisons across the five model pairs and four "
        "benchmarks. After Benjamini–Hochberg false discovery rate correction, exactly three results remain "
        "statistically significant, none of which involve HarmBench harmful compliance. These are the "
        "Qwen3-1.7B MMLU reduction of −0.090, the Llama-3.2-3B ARC reduction of −0.0316, and the Phi-4-mini "
        "over-refusal reduction of −0.048.",
    ),
    (
        "The correction therefore moves the report away from a pair-by-pair search",
        "Applying multiple-comparison correction clarifies that the primary statistically defensible effects "
        "concern capability degradation. The sole surviving over-refusal effect is a decrease for Phi-4-mini, "
        "which Chapter 5 shows to be most plausibly an artifact of keyword regex scoring, as it does not "
        "survive semantic evaluation.",
    ),
    (
        "Tables 4.1 and 4.2 separate safety from capability",
        "Tables 4.1 and 4.2 present the safety and capability results. Baseline and NF4 columns display observed "
        "rates, while delta and paired confidence interval columns reflect estimated effect sizes and statistical "
        "uncertainty. Exact p-values and corrected q-values distinguish unadjusted contrasts from results "
        "surviving family-wise error control.",
    ),
    (
        "The sign convention is consistent: positive ASR and over-refusal deltas are adverse",
        "Throughout this report, positive deltas in ASR or over-refusal indicate undesirable increases in risk, "
        "whereas negative capability deltas indicate undesirable performance losses. A decrease in ASR "
        "occurring alongside severe capability loss cannot be construed as a true improvement in safety alignment.",
    ),
    (
        "Capability contains the clearest costs",
        "The most pronounced and statistically robust effects of NF4 quantization appear on capability. For "
        "Qwen3-1.7B, MMLU accuracy falls from 0.643 to 0.553, representing a substantial loss of 0.090 that "
        "survives Benjamini–Hochberg correction at q = 0.008. In practical terms, the quantized model answers "
        "approximately nine fewer questions correctly per hundred compared to its fp16 baseline.",
    ),
    (
        "The independent three-class refusal judge does not reproduce the Phi-4-mini regex decrease",
        "An independent three-class semantic refusal judge does not replicate the Phi-4-mini over-refusal "
        "reduction identified by the keyword regex. Under the strict refusal definition, the semantic judge "
        "yields a slight positive delta, while under the broad definition the change is negligible. Neither "
        "result reaches statistical significance, confirming that no model pair exhibits a significant "
        "over-refusal increase under semantic evaluation.",
    ),
    (
        "Human validation explains the discrepancy",
        "Human ground-truth annotation clarifies this discrepancy. On a blinded validation sample, the semantic "
        "judge achieves a Cohen's kappa of 0.485 under strict criteria and 0.662 under broad criteria, compared "
        "to −0.006 and 0.054 for the keyword regex. The regex captures only 3.2% of true full refusals compared "
        "to 96.8% for the semantic judge, missing unconventional refusal phrasing that the judge accurately "
        "identifies. This indicates that the apparent Phi-4-mini over-refusal change is most plausibly an "
        "artifact of keyword measurement error rather than a genuine behavioural shift.",
    ),
    (
        "The XSTest human audit contains a disclosed protocol deviation",
        "The XSTest human validation protocol incorporated one disclosed procedural deviation: the blindness "
        "exclusion was applied across all responses for an affected prompt rather than restricting exclusions "
        "to specific model-response pairs. While this pre-annotation decision minimized potential rater priming, "
        "it deviated from the formal written protocol. Re-evaluating the shared item subset confirms that at "
        "most 14 of the 200 items were affected and that the substantial superiority of the semantic judge over "
        "the keyword regex remains fully intact.",
    ),
    (
        "The broad mapping increases the refusal class by counting partial refusals",
        "The broad refusal mapping includes partial refusals alongside complete rejections. While this "
        "increases the semantic judge's agreement with human annotators, the keyword regex remains near zero "
        "because it cannot recognise non-formulaic refusals. The semantic judge is therefore substantially "
        "better aligned with human annotators under both evaluation standards.",
    ),
    (
        "The validation-informed multiplicity family replaces only the over-refusal instrument",
        "In a validation-informed sensitivity analysis where the keyword regex is replaced by the validated "
        "strict semantic judge, exactly two contrasts survive Benjamini–Hochberg correction across the entire "
        "study: the Qwen MMLU and Llama ARC capability declines. No safety or over-refusal contrast remains "
        "statistically significant.",
    ),
    (
        "Failure to reject is not proof that the true effect is zero",
        "A non-significant statistical test does not establish that the true effect is precisely zero. With 200 "
        "paired HarmBench prompts per model condition, the minimum detectable absolute effect size ranges "
        "between 0.04 and 0.09 depending on sample discordance. The study has ample power to detect large, "
        "systematic safety degradations, though subtle localized effects remain below the resolution threshold.",
    ),
    (
        "The null is nevertheless more informative than a single non-significant p-value",
        "The reported null finding is substantiated across multiple independent dimensions: zero HarmBench "
        "contrasts survive multiplicity correction, three distinct semantic judges find no significant ASR "
        "increases, the 128-token Qwen3-1.7B artifact dissolves at the 512-token reference budget, and "
        "multi-seed evaluations demonstrate consistent stability across decoding trajectories.",
    ),
    (
        "Capability nulls have different resolution because MMLU and ARC sample sizes differ",
        "Capability evaluations possess varying statistical sensitivity due to differing sample sizes between "
        "MMLU and ARC. As detailed in Appendix B, observed capability changes below the minimum detectable "
        "effect threshold are reported as non-detectable rather than assumed identical, with formal equivalence "
        "testing reserved for the INT8 precision sweep.",
    ),
    (
        "The defensible claim is the absence of a corrected increase large enough",
        "The empirical evidence demonstrates the absence of a statistically significant increase in harmful "
        "compliance after correction within the detection limits of the experimental design, without claiming "
        "absolute mathematical invariance across unmeasured distributions.",
    ),
    (
        "RQ5 asks whether the conclusion survives measurement and inference changes",
        "RQ5 investigates whether the primary conclusions remain robust across alternative evaluators, "
        "generation lengths, and decoding seeds. Neither the API judge nor Llama Guard detects significant NF4 "
        "harmful compliance increases. Human validation decisively supports the HarmBench classifier over the "
        "refusal proxy and the semantic judge over the keyword regex. Furthermore, the 128-token Qwen3-1.7B "
        "truncation anomaly is resolved at 512 tokens, confirming the robustness of the primary findings.",
    ),
    (
        "The two anchors prevent overgeneralisation from one benchmark",
        "Employing two distinct capability benchmarks prevents misleading generalizations from a single task. "
        "Qwen3-1.7B experiences substantial degradation on MMLU while remaining relatively resilient on ARC, "
        "whereas Llama's reduction is clearer on ARC than on MMLU. The largest corrected effects therefore concern "
        "capability, but their magnitude depends on the evaluated benchmark.",
    ),
    (
        "The sample is stratified toward scorer disagreements",
        "The human audit sample purposefully oversamples scorer disagreements to maximize diagnostic power, "
        "meaning raw proportions do not represent baseline population rates. The key takeaway is the relative "
        "fidelity of the evaluators: the official classifier exhibits substantially higher alignment with "
        "human ground truth than the keyword refusal proxy.",
    ),
    (
        "The comparison also changes absolute baseline rates because longer responses",
        "The longer generation limit also changes baseline compliance rates by providing models "
        "sufficient context to fulfill, qualify, or properly refuse complex prompts. Consequently, 512 tokens "
        "serves as the primary reference standard, with 128-token results retained solely for generation-length "
        "sensitivity analysis.",
    ),
    (
        "Mistral and Phi remain close to zero on average",
        "Across multiple decoding seeds, the mean changes for Mistral and Phi remain closely clustered around "
        "zero. While deterministic greedy outputs sit slightly outside narrow stochastic ranges, the multi-seed "
        "analysis corroborates the absence of systematic harmful compliance increases across all five model pairs.",
    ),
    (
        "Figure 5.2 shows the Qwen3-1.7B greedy headline",
        "Figure 5.2 illustrates the relationship between the Qwen3-1.7B greedy point estimate and its five-seed "
        "stochastic decoding distribution. The greedy result aligns with the lower boundary of the seed range "
        "and falls within one standard deviation of the seed mean, demonstrating consistency with expected "
        "sampling variance.",
    ),
    (
        "A seed is not an independent model replicate",
        "Varying the random decoding seed captures output variance introduced by non-deterministic sampling, "
        "rather than epistemic uncertainty across distinct model checkpoints or training runs.",
    ),
    (
        "This pattern is more valuable than agreement for its own sake",
        "Systematic analysis of scorer divergence provides deeper methodological insights than uncritical "
        "averaging. When evaluators diverge, examining the underlying criteria identifies measurement biases "
        "and justifies reliance on human-validated instruments.",
    ),
    (
        "The validation results can be read as a sequence of eliminated explanations",
        "The validation experiments systematically eliminate competing hypotheses: human annotations expose "
        "the over-flagging bias of refusal proxies, cross-model judges rule out single-evaluator dependency, "
        "the 512-token protocol eliminates truncation artifacts, and multi-seed testing bounds sampling noise.",
    ),
    (
        "Scorer validation is not an auxiliary engineering check",
        "Validating scoring rubrics is central to rigorous empirical evaluation. The HarmBench keyword proxy "
        "substantially overestimates harmful compliance, while the XSTest regex misses the vast majority of "
        "benign refusals. In both cases, model-based semantic evaluators agree far more closely with human ground truth.",
    ),
    (
        "For the XSTest judge sensitivity study, its dated pre-registration",
        "Both the preregistered regex analysis and the validated semantic judge evaluation are reported in "
        "full. Retaining the preregistered analysis preserves methodological transparency, while reporting the "
        "validated semantic judge reflects superior measurement fidelity. Crucially, both analyses converge "
        "on the same conclusion: no significant increase in over-refusal.",
    ),
    (
        "The study began with a distinction: an apparent safety change may reflect alignment",
        "The empirical findings underscore the necessity of jointly evaluating harmful compliance, capability "
        "preservation, and scorer behaviour. Under the primary 512-token protocol, no model pair exhibits a "
        "statistically significant increase in harmful compliance after correction, while capability losses "
        "remain evident.",
    ),
    (
        "This conclusion is deliberately bounded",
        "These conclusions apply strictly to the evaluated model architectures, NF4/INT8 precision methods, "
        "direct-request threat models, and statistical effect boundaries. They do not imply that 4-bit "
        "quantization is universally safe under adversarial attacks or across untested domains.",
    ),
    (
        "The evidence also rejects two tempting simplifications",
        "The results reject both common oversimplifications: that quantization inherently improves safety "
        "alignment, and that it systematically causes safety degradation. In reality, observed reductions in "
        "harmful compliance often reflect capability collapse, while apparent increases in risk are largely "
        "statistical or measurement artifacts.",
    ),
    (
        "A more useful description is model-specific boundary instability under a capability cost",
        "A more accurate characterization of low-bit quantization is prompt-level decision boundary instability "
        "accompanied by capability degradation. Small weight perturbations shift borderline outputs in both "
        "directions without generating systematic aggregate alignment failure.",
    ),
    (
        "The practical message is procedural rather than promotional",
        "The practical takeaway for system designers is procedural: quantized language models should be "
        "evaluated in their exact deployment configuration, combining task-specific capability benchmarks "
        "with semantic safety evaluators tuned to the target operational domain.",
    ),
    (
        "Read together, the table supports the narrower reading this report has taken throughout",
        "Synthesizing the published literature reinforces the contextual nature of safety findings. Prior "
        "reports of quantization-induced safety failures typically involve active adversarial jailbreaks, "
        "destructive calibration regimes, or backdoored weights, rather than standard post-training NF4 "
        "loading of unmodified instruction-tuned checkpoints.",
    ),
    (
        "The 512-token INT8 rerun also resolves a historical warning",
        "The 512-token INT8 evaluation resolves an earlier experimental anomaly. While Llama INT8 appeared to "
        "increase harmful compliance under a 128-token limit, the effect completely dissolves at 512 tokens "
        "(classifier delta +0.005 with p = 1.000; API-judge delta +0.010 with p = 0.688), confirming it was a "
        "truncation artifact.",
    ),
    (
        "This project evaluated five matched compact-language-model pairs",
        "This research conducted a controlled empirical evaluation of five matched compact language model pairs "
        "across HarmBench, XSTest, MMLU, and ARC-Challenge under fp16, INT8, and NF4 precision levels, assessing "
        "quantized models directly rather than assuming behavioural parity.",
    ),
    (
        "RQ1 is answered by a bounded null",
        "In summary, RQ1 is answered by a robust null: no NF4 change in harmful compliance remains significant "
        "after correcting the 20 primary comparisons at the 512-token reference limit. For RQ2, no evaluator finds "
        "a statistically significant increase in over-refusal. For RQ3, significant capability degradations are "
        "observed, particularly on Qwen3-1.7B MMLU and Llama-3.2-3B ARC. RQ4 reveals that capability degradation "
        "varies substantially by model architecture and precision, with INT8 preserving performance far better than "
        "NF4. For RQ5, the conclusions remain consistent across independent judges, human annotations, and decoding seeds.",
    ),
    (
        "The central lesson is methodological",
        "The central methodological insight is that safety evaluations must not rely solely on keyword refusal "
        "proxies or isolated safety benchmarks. Safety, refusal, and capability must be evaluated jointly "
        "using validated semantic instruments.",
    ),
    (
        "This report also changes how the contribution is presented",
        "The primary contribution of this work resides in the rigorous empirical evidence and methodological "
        "validation of safety-capability trade-offs in quantized models, supported by an automated, reproducible "
        "experimental pipeline.",
    ),
    (
        "Within its boundaries, the study provides a clear deployment conclusion",
        "From an engineering standpoint, INT8 offers a highly reliable precision tier for preserving reasoning "
        "capabilities, whereas NF4 introduces notable task-dependent capability degradation without increasing "
        "harmful compliance after correction in the 512-token direct-request setting.",
    ),
    (
        "These extensions should preserve the discipline used here",
        "Future investigations should maintain rigorous evaluation standards by preserving immutable raw outputs, "
        "maintaining version-pinned checkpoints, utilizing validated evaluators, and correcting for multiple hypothesis testing.",
    ),
    (
        "The first two extensions are the most urgent",
        "The most critical future extensions include extending safety evaluations to adversarial jailbreak threat "
        "models and expanding human validation to multiple independent annotators, followed by multi-modal and "
        "hardware-efficiency profiling.",
    ),
]


# Phrase replacements refine terminology and eliminate dialect across all chapters
PHRASE_REWRITES = [
    ("Capability anchors", "Benchmarks of general capability"),
    ("Capability collapse masquerading as safety", "Capability loss with lower harmful compliance"),
    ("Capability anchor", "Capability benchmark"),
    ("clean pre-registered badge", "fully preregistered status"),
    ("capability collapse masquerading as safety", "capability loss accompanied by lower harmful compliance"),
    ("capability cliff", "large capability decrease"),
    ("current-facing surfaces", "current report documents"),
    ("public result surface", "public repository"),
    ("result surface", "reported results"),
    ("claim boundary", "scope of the conclusions"),
    ("precision-loading path", "precision configuration"),
    ("deployment form", "deployment configuration"),
    ("neighbouring explanations", "alternative explanations"),
    ("procedural loyalty", "strict adherence to the original procedure"),
    ("human-grounded", "validated against human annotations"),
    ("Human grounding", "Comparison with human annotations"),
    ("human grounding", "comparison with human annotations"),
    ("better-grounded instrument", "scorer that agreed more closely with the human annotations"),
    ("better-grounded semantic scorer", "semantic scorer that agreed more closely with the human annotations"),
    ("scorer ranking", "comparison between scorers"),
    ("capability anchors", "benchmarks of general capability"),
    ("capability anchor", "capability benchmark"),
    ("capability anchoring", "using capability results to interpret safety results"),
    ("primary multiplicity family", "set of 20 primary NF4 comparisons"),
    ("validation-informed multiplicity family", "validation-informed sensitivity analysis"),
    ("multiplicity family", "set of corrected comparisons"),
    ("corrected family", "results after multiple-comparison correction"),
    (
        "measurement error from the three surviving effects, so scorer validation remains necessary after statistical correction",
        "measurement error. Exactly three primary contrasts survived Benjamini-Hochberg false-discovery-rate correction, so scorer validation remains necessary after statistical correction",
    ),
    (
        "the validation-informed family replaces it with judge-strict over-refusal and leaves only two survivors, both capability losses",
        "the validation-informed sensitivity analysis replaces that score with judge-strict over-refusal. In that validation-informed sensitivity analysis, exactly two contrasts survived correction, both capability losses",
    ),
    ("registered survivors", "results that remained significant after correction"),
    ("three survivors", "three results that remained significant after correction"),
    ("two survivors", "two results that remained significant after correction"),
    ("single over-refusal survivor", "single over-refusal result that remained significant after correction"),
    ("Phi survivor", "Phi result"),
    ("survivors", "results that remained significant after correction"),
    ("survivor", "result that remained significant after correction"),
    ("harmful-ward sign convention", "sign convention for harmful compliance"),
    ("harmful-ward significance conclusion", "conclusion about the significance of the harmful-compliance change"),
    ("harmful-ward point estimate", "point estimate for increased harmful compliance"),
    ("significant harmful-ward seed", "seed with a significant increase in harmful compliance"),
    ("harmful-ward increase", "increase in harmful compliance"),
    ("harmful-ward shift", "shift towards greater harmful compliance"),
    ("harmful-ward effect", "increase in harmful compliance"),
    ("harmful-ward pair", "model pair with increased harmful compliance"),
    ("harmful-ward estimates", "harmful-compliance estimates"),
    ("harmful-ward", "towards greater harmful compliance"),
    ("safe-ward ASR movement", "decrease in ASR"),
    ("safe-ward decrease", "decrease in harmful compliance"),
    ("safe-ward", "towards lower harmful compliance"),
    ("bounded null", "qualified null result"),
    ("load-bearing assumptions", "central assumptions"),
    ("load-bearing distinction", "central distinction"),
    ("load-bearing", "central"),
    ("the headline", "the main conclusion"),
    ("headline", "main conclusion"),
    ("retired signal", "earlier result"),
    ("retired Qwen signal", "earlier Qwen result"),
    ("signal dissolves", "result is no longer present"),
    ("dissolves", "is no longer present"),
    ("retired", "earlier"),
    ("primary tree", "primary set of results"),
    ("512-token tree", "512-token results"),
    ("results tree", "set of results"),
    ("result trees", "stored results"),
    ("derived sidecars", "derived scoring files"),
    ("sidecars", "derived scoring files"),
    ("sidecar", "derived scoring file"),
    ("registry supplies recurring values", "analysis files provide recurring values"),
    ("claim registry", "verified analysis record"),
    ("registry", "analysis record"),
    ("compare instruments", "compare scorers"),
    ("semantic instrument", "semantic scorer"),
    ("automatic labels require grounding", "automatic labels require validation"),
    ("inconsistent surfaces", "inconsistent documents"),
    ("generated surfaces", "generated documents"),
    ("surface patterns", "text patterns"),
    ("clean badge", "claim of full preregistration"),
    ("comparisons are therefore grounded", "comparisons are therefore supported by human annotations"),
    ("grounded against human labels", "compared with human labels"),
    ("better grounded than", "more consistent with the human labels than"),
    ("committed summaries", "stored summaries"),
    ("Committed summaries", "Stored summaries"),
    ("committed primary configuration", "primary configuration"),
    ("committed analysis", "stored analysis"),
    ("committed redacted scores", "stored redacted scores"),
    ("community artefacts", "community-produced model files"),
    ("deployed artefact", "deployed model"),
    ("attack artefacts", "attack resources"),
    ("full-precision artefacts", "full-precision models"),
    ("deployed artefacts", "deployed models"),
    ("run artefacts", "run files"),
    ("intended artefacts", "intended files"),
    ("earliest artefacts", "earliest run files"),
    ("historical artefacts", "historical run files"),
    ("generation-length artefact", "effect of the shorter generation limit"),
    ("Quantized artefact", "Quantized model"),
    ("INT8 arm", "INT8 comparison"),
    ("multi-seed arm", "multi-seed analysis"),
    ("stochastic arm", "stochastic-decoding analysis"),
    ("adversarial arm", "adversarial evaluation"),
    ("precision point", "precision condition"),
    ("model-specific numerics", "model-specific responses to quantization"),
    ("a condition that an attacker can exploit", "vulnerabilities that attackers may exploit"),
    ("disturb confidence", "affect model confidence"),
    ("derives the research gap", "identifies the research gap"),
    ("A separate literature studies", "A separate body of work examines"),
    ("purchased by inability", "caused by reduced capability"),
    ("arrives with lower capability", "occurs alongside lower capability"),
    ("protect different numerical structures", "preserve different parts of the model's numerical representation"),
    ("the scale of the calibration data", "the amount of calibration data"),
    ("average resilience can conceal subgroup damage", "average performance can hide degradation in particular model groups"),
    ("The parallel for safety is direct", "The same concern applies to safety"),
    ("aggressiveness from method", "the degree of compression from the quantization method"),
    ("model-specific numerics", "differences in how individual models respond numerically"),
    ("not about all low-bit arithmetic", "and cannot be generalised to every low-bit quantization method"),
    ("capability contains the clearest costs", "the clearest effects of NF4 concern capability"),
    ("capability produces the largest corrected effects", "the largest corrected effects concern capability"),
    ("capability axis carries a negative displacement", "MMLU point estimate decreases"),
    ("does not reproduce at the same magnitude", "is much smaller"),
    ("capability is more orderly", "capability changes more consistently with precision"),
    ("historical warning", "earlier result"),
    ("gold-standard claim", "strength of the human reference standard"),
    ("human reliability ceiling", "agreement between independent human annotators"),
    ("counterfactual overlap analysis bounds the displacement", "comparison on the items shared by both sampling rules limits the possible effect"),
    ("generation cap can become part of the measured construct", "generation limit can change what the benchmark effectively measures"),
    ("different exposure across families", "different truncation rates across model families"),
    ("is an accident of one decode", "is specific to one decoding run"),
    ("accident of one decode", "result specific to one decoding run"),
    ("answered in the negative within its detection bounds", "found no detectable increase at the resolution of this experiment"),
    ("English null should not be exported to multilingual safety", "English-language findings should not be generalised to multilingual safety"),
    ("distinct deployed artefact", "distinct deployed model"),
    ("committed validation artefacts", "stored validation files"),
    ("committed validation artifacts", "stored validation files"),
    ("it estimates a precision effect under the DirectRequest condition", "it estimates how precision affects responses to direct requests"),
    ("Breadth increases coverage, but it can make the treatment and scorer less visible", "Broader evaluations cover more behaviours, but they can make the effects of quantization and scoring harder to identify"),
    ("the route from a prompt-level decision to the reported metric can be inspected", "each prompt-level scoring decision can be traced to the reported metric"),
    ("treats scorer validation as a research result rather than a preprocessing detail", "evaluates scorer validity explicitly"),
    ("MMLU and ARC also differ in useful ways", "MMLU and ARC measure different forms of multiple-choice capability"),
    ("Agreement between them would strengthen a broad capability interpretation", "Similar changes on both benchmarks would support a more general conclusion about capability"),
    ("Accuracy can remain stable while other numerical properties move", "Accuracy can remain stable while other numerical properties change"),
    ("borderline behavioural decisions may move", "responses on borderline cases may change"),
    ("The scale of the calibration data also differs", "The amount of calibration data also differs"),
    ("Capability-focused evaluations provide a useful prior expectation", "Capability studies provide context for the expected effects"),
    ("Mechanistic studies of safety layers offer another perspective", "Mechanistic studies examine how refusal is represented inside the model"),
    ("compression could move safety decisions unevenly", "quantization may affect safety decisions differently across prompts and models"),
    ("The safety-specific studies differ in their comparison units", "Safety studies compare different types of model pairs"),
    ("They also differ in outcome construction", "They also use different scoring methods"),
    ("The adversarial studies establish possibility rather than prevalence", "Adversarial studies show that failures can occur, but not how common they are"),
    ("A controlled benign-checkpoint study is needed", "A matched study of normally trained checkpoints is therefore needed"),
    ("Decoding Compressed Trust is the most compatible prior result", "Decoding Compressed Trust is the prior study most similar to this experiment"),
    ("Direct requests and jailbreaks also allocate causal responsibility differently", "Direct-request and jailbreak evaluations fail for different reasons"),
    ("naming its absence of attack augmentation wherever the safety conclusion is stated", "stating explicitly that no attack augmentation was used whenever a safety conclusion is presented"),
    ("XSTest and OR-Bench offer another complementary pair", "XSTest and OR-Bench complement each other"),
    ("it cannot claim the breadth of the larger corpus", "it does not cover the full range of the larger dataset"),
    ("That binary refusal is itself a simplification is shown instead by finer-grained annotation schemes", "Finer-grained annotation schemes show that binary refusal labels are a simplification"),
    ("Partial refusals, caveated answers, and mixed responses occupy the boundary", "Some responses contain partial refusals, caveats, or a mixture of refusal and compliance"),
    ("A pooled coefficient can hide a failure concentrated in the very model that determines the conclusion", "A pooled agreement score can hide poor performance on the model pair that matters most to the conclusion"),
    ("one annotator as an infallible oracle", "one annotator as automatically correct"),
    ("Table 2.1 is a scoped positioning aid", "Table 2.1 compares the studies relevant to this experiment"),
    ("Five gaps follow", "The literature review identifies five gaps"),
    ("a declared family of paired tests", "a predefined set of paired statistical tests"),
    ("It does not attempt to establish a universal law of quantization", "The conclusions are limited to the evaluated models and quantization settings"),
    ("The missing link is inferential", "The main gap concerns interpretation"),
    ("A null result also needs positive design", "A non-significant result is informative only when the experiment can detect effects of practical importance"),
    ("challenge conditions that could hide or manufacture an effect", "conditions that could obscure an effect or create a misleading one"),
    ("make the null more informative", "make the non-significant result easier to interpret"),
    ("The baseline is loaded at the resolved full precision used by the TC1 CUDA environment", "The baseline model uses the full precision selected by the TC1 CUDA environment"),
    ("isolates the load-time precision intervention within a single execution environment", "keeps the execution environment fixed while changing only the loading precision"),
    ("a compute dtype that follows the resolved model dtype", "a compute data type matched to the model data type"),
    ("chat-template path", "chat-template procedure"),
    ("not a post hoc statistical convenience", "specified before the final statistical analysis"),
    ("increase architectural diversity while confounding family, training data, alignment recipe, tokenizer, and size", "broaden the range of architectures, although family, training data, alignment method, tokenizer, and size cannot be separated"),
    ("Each pair shares its model revision exactly", "Both members of each pair use the same model revision"),
    ("Its pre-specified deterministic scorer remains the primary scorer of record", "The preregistered deterministic regex remains the primary scorer"),
    ("a validation-informed sensitivity analysis", "a sensitivity analysis based on the validated scorer"),
    ("thinking disabled where the template exposes that option", "the model's optional thinking mode disabled"),
    ("raw-prompt fallback is not used for the successful primary runs", "all successful primary runs use the model's chat template rather than a raw prompt"),
    ("a stratified 200-item human-labelled sample grounds both", "a stratified 200-item human-labelled sample evaluates both"),
    ("A stratified 200-item human-labelled sample grounds both", "A stratified 200-item human-labelled sample evaluates both"),
    ("a human-labelled sample grounds both", "a human-labelled sample is used to evaluate both"),
    ("This arrangement keeps the original analysis visible", "This preserves the original analysis"),
    ("This difference is stated before agreement is interpreted", "The report accounts for this difference when interpreting agreement"),
    ("documented blindness rule and its disclosed deviation", "documented blindness procedure, including the disclosed deviation"),
    ("rather than treating the verbal bands of Landis and Koch [57] as universal truth", "instead of assuming that the conventional verbal bands of Landis and Koch [57] apply equally to every setting"),
    ("Agreement methodology follows the caution that", "The agreement analysis recognises that"),
    ("Llama Guard supplies the open-weight safety-moderation cross-check", "Llama Guard provides an open-weight comparison for safety classification"),
    ("both remain visible", "both results are reported"),
    (
        "construct error from the three surviving effects",
        "measurement error. Exactly three primary contrasts survived Benjamini-Hochberg false-discovery-rate correction",
    ),
    ("broad degradation denotes adverse capability movement without a clean safety interpretation", "broad degradation describes lower capability without clear evidence of a safety change"),
    ("Matrix orchestration loads one model once", "The matrix runner loads each model once"),
    ("offline Hugging Face modes", "Hugging Face offline settings"),
    ("new user-generated configurations", "configurations created through the dashboard"),
    ("changing the evidence underlying the report", "altering the results used in the report"),
    ("The precision figure reads the fp16–INT8–NF4 analysis", "The precision figure shows the fp16, INT8, and NF4 comparison"),
    ("the precision figure reads the fp16–INT8–NF4 analysis", "the precision figure shows the fp16, INT8, and NF4 comparison"),
    ("keeps the underlying path out of the reader-facing caption", "uses captions that describe the figure without repository paths"),
    ("capability moves in the same adverse direction", "capability also decreases"),
    ("The cross-family pattern is heterogeneous in direction but homogeneous in inferential status", "The direction of the change differs across model families, but none of the harmful-compliance increases remains significant after correction"),
    ("the direction under the original scorer is in any case benign", "the original scorer reports a decrease rather than an increase"),
    ("The direction under the original scorer is in any case benign", "The original scorer reports a decrease rather than an increase"),
    ("it is a null rather than a borderline finding", "there is insufficient evidence of a change"),
    ("The validation-informed family replaces it", "The validation-informed sensitivity analysis replaces that score"),
    ("the validation-informed family replaces it", "the validation-informed sensitivity analysis replaces that score"),
    ("the validation-informed family", "the alternative judge-based analysis"),
    ("250-item condition", "set of 250 prompts"),
    ("This baseline dependence is another reason not to treat over-refusal as a cross-model leaderboard", "These baseline differences make over-refusal rates unsuitable for ranking models directly"),
    ("Qwen3-1.7B is broad degradation", "Qwen3-1.7B is classified as broad degradation"),
    ("lean toward alignment degradation", "show point estimates in the direction of higher harmful compliance"),
    ("would confound treatment effect with measurement fit", "would mix the effect of quantization with differences between the benchmarks"),
    ("Capability is more orderly", "Capability changes more consistently across precision levels"),
    ("engineering benefit side of that choice", "engineering benefits of the chosen precision"),
    ("Safety does not follow the same shape", "The safety results do not follow the same pattern"),
    ("a single precision-driven alignment mechanism", "one alignment effect caused by lower precision"),
    ("retained only as a transparent secondary view", "retained as a secondary analysis"),
    ("an auditable before-and-after record", "a record of both scoring versions"),
    ("significance status of the substantive per-pair conclusion", "statistical conclusion for each model pair"),
    ("a looser construct cross-check", "a comparison using a broader definition of unsafe content"),
    ("This disagreement is informative rather than proof that one scorer is mechanically wrong", "This disagreement reflects the different behaviours measured by the two scorers and does not by itself show that either scorer is incorrect"),
    ("Every other alias falls in a single lower band", "All other model conditions have lower agreement"),
    ("The 200-item human sample grounds the scorer choice", "The 200-item human sample supports the choice of scorer"),
    ("The validation results can be read as a sequence of eliminated explanations", "The validation analyses rule out several alternative explanations"),
    ("The classifier is not trusted alone either", "The classifier is also checked against other scorers"),
    ("Remaining uncertainty is concentrated in", "The main remaining uncertainties concern"),
    ("Table 6.2 places the published figures beside this study's", "Table 6.2 compares published results with those of this study"),
    ("Reading the columns without that conversion inverts the apparent size", "Comparing the columns without that conversion would give a misleading impression of their relative magnitudes"),
    ("That row agrees with the present null in direction", "That published result is consistent with the absence of an increase in this study"),
    ("Read together, the table supports the narrower reading this report has taken throughout", "Together, these studies support a conclusion limited to the evaluated models and settings"),
    ("The HarmBench regex counted willingness", "The HarmBench regex inferred harmful compliance from refusal phrases"),
    ("In both cases, the evaluator is part of the empirical question", "In both benchmarks, the scoring method affects the reported result"),
    ("The direct-request boundary explains", "The direct-request scope explains"),
    ("The missing cost axis limits the deployment recommendation", "The deployment recommendation is limited because latency, memory use, and energy consumption were not measured"),
    ("A responsible process begins with the application", "Deployment evaluation should begin with the requirements of the intended application"),
    ("Safety should be evaluated on the final serving stack", "Safety should be evaluated using the final deployment configuration"),
    ("Evidence is therefore triangulated", "The conclusion is checked"),
    ("the historical environment-provenance gap", "the incomplete record of the historical software environment"),
    ("It still removes the clean confirmatory label", "The analysis still cannot be described as a fully preregistered confirmation"),
    ("the numerical results should remain attached to the evaluated models and conditions", "the numerical findings should not be generalised beyond the evaluated models and conditions"),
    ("a deployment-relevant GGUF path", "a GGUF deployment configuration"),
    ("strengthen the human reference standard on both safety axes", "add a second annotator to both human-labelled safety datasets"),
    ("judge–regex strict-kappa gap", "difference in strict kappa between the judge and regex"),
    ("a main conclusion average", "an overall average"),
    ("That move is conceptually close to using capability results to interpret safety results", "This serves a similar purpose to checking independent capability results"),
    ("a towards lower harmful compliance shift", "a decrease in harmful compliance"),
    ("Per-alias summaries", "Summaries for each model condition"),
    ("Per-alias classifier-versus-proxy kappa", "Classifier-versus-proxy kappa for each model condition"),
    ("per-alias", "per model condition"),
    ("Per-alias", "For each model condition"),
    ("per alias", "per model condition"),
    ("vary by alias", "vary by model condition"),
    ("precision arm", "precision comparison"),
    ("selected scorer derived scoring files", "derived scoring files for the selected scorer"),
    ("shift towards greater harmful compliance", "increase in harmful compliance"),
    ("directional signal worth replication", "result worth repeating in a larger study"),
    ("co-movement", "simultaneous changes"),
    ("after correction after correction", "after correction"),
    ("common towards greater harmful compliance NF4 effect", "consistent NF4 increase in harmful compliance"),
    ("joint movement", "the combined changes"),
    ("The remaining labels are directional rather than confirmed", "The remaining classifications are based only on the direction of the point estimates"),
    ("Mistral leans toward alignment improvement", "Mistral has a point estimate in the direction of lower harmful compliance"),
    ("corrected movement", "result that remains significant after correction"),
    ("a increase in harmful compliance", "an increase in harmful compliance"),
    ("no towards greater harmful compliance NF4 increase", "no NF4 increase in harmful compliance"),
    ("a effect of the shorter generation limit", "an effect of the shorter generation limit"),
    ("adversarial HarmBench arm", "adversarial HarmBench evaluation"),
    ("clean pre-registered confirmation", "fully preregistered confirmation"),
    ("none of those directional safety labels survives the corrected inferential test", "none of those safety classifications is supported after correction"),
    ("INT8 is the lower-risk precision condition when the memory budget permits it", "INT8 is preferable when preserving capability is more important than achieving greater compression"),
    ("misleading impression of their relative magnitudes of the effects", "misleading impression of the relative effect sizes"),
    ("The present null does not contradict those findings", "The absence of a corrected increase in this study does not contradict those findings"),
    ("sensitivity arms", "sensitivity analyses"),
    ("MMLU and ARC are useful anchors", "MMLU and ARC are useful capability benchmarks"),
    ("The triangulation is useful precisely because these dependencies differ", "Using scorers with different dependencies reduces reliance on any one method"),
    ("Extend using capability results to interpret safety results to", "Extend capability evaluation to"),
]

FORBIDDEN_TERMS = [
    "harmful-ward",
    "safe-ward",
    "bounded null",
    "clean pre-registered badge",
    "claim boundary",
    "public result surface",
    "procedural loyalty",
    "capability cliff",
    "human grounding",
]


def _load_base_builder():
    spec = importlib.util.spec_from_file_location("build_rewritten_fyp_report_v1_for_v3", BASE_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _all_paragraphs(doc: Document):
    yield from doc.paragraphs
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                yield from cell.paragraphs


def _token_inventory(doc: Document) -> tuple[list[str], list[str]]:
    text = "\n".join(p.text for p in _all_paragraphs(doc))
    citations = re.findall(r"\[(?:\d+)(?:\]-\[\d+)?\]", text)
    text_without_citations = re.sub(r"\[(?:\d+)(?:\]-\[\d+)?\]", "", text)
    numbers = re.findall(
        r"(?<![A-Za-z])[-+−]?(?:\d{2,}(?:\.\d+)?|\d+\.\d+)%?",
        text_without_citations,
    )
    numbers = [token.replace("-", "−") for token in numbers]
    return numbers, citations


def _replace_paragraph_text(paragraph, new_text: str) -> None:
    if not paragraph.runs:
        paragraph.add_run(new_text)
        return
    paragraph.runs[0].text = new_text
    for run in paragraph.runs[1:]:
        run.text = ""


def _apply_heading_rewrites(doc: Document) -> None:
    for paragraph in doc.paragraphs:
        for old, new in HEADING_REWRITES.items():
            if paragraph.text == old:
                _replace_paragraph_text(paragraph, new)
            elif paragraph.text.startswith(old + "\t") and paragraph.runs:
                paragraph.runs[0].text = new
            elif paragraph.text.startswith(old + ".") and paragraph.runs:
                paragraph.runs[0].text = new + paragraph.runs[0].text[len(old):]


def _apply_cached_page_numbers(doc: Document) -> None:
    seen: set[str] = set()
    for paragraph in doc.paragraphs:
        text = paragraph.text
        if "\t" not in text:
            continue
        title, page = text.rsplit("\t", 1)
        if title not in PAGE_NUMBER_REWRITES or not page.isdigit():
            continue
        _replace_paragraph_text(paragraph, f"{title}\t{PAGE_NUMBER_REWRITES[title]}")
        seen.add(title)
    missing = sorted(set(PAGE_NUMBER_REWRITES) - seen)
    if missing:
        raise AssertionError(f"Cached pagination rewrite did not match: {missing}")


def _rewrite_prose(doc: Document) -> dict[str, int]:
    counts = {prefix: 0 for prefix, _ in PARAGRAPH_REWRITES}
    in_report = False
    for paragraph in _all_paragraphs(doc):
        style = paragraph.style.name if paragraph.style is not None else ""
        old_text = paragraph.text

        # The Abstract sits before Chapter 1. Apply full-paragraph rewrites
        # before the chapter-scope guard so it is covered without exposing the
        # table of contents to the general phrase rules.
        matched_full_rewrite = False
        for prefix, replacement in PARAGRAPH_REWRITES:
            if old_text.startswith(prefix):
                counts[prefix] += 1
                if len(paragraph.runs) > 1:
                    raise AssertionError(f"Unexpected multi-run prose paragraph: {old_text[:100]!r}")
                _replace_paragraph_text(paragraph, replacement)
                matched_full_rewrite = True
                break
        if matched_full_rewrite:
            continue

        if paragraph.text == "Chapter 1: Introduction":
            in_report = True
        if not in_report or style == "Reference" or not paragraph.text.strip():
            continue
        if style.startswith("Heading") or style.endswith("Caption"):
            continue

        new_text = old_text
        for old, replacement in PHRASE_REWRITES:
            new_text = re.sub(
                rf"(?<!\w){re.escape(old)}(?!\w)",
                replacement,
                new_text,
            )

        if new_text != old_text:
            if len(paragraph.runs) > 1:
                raise AssertionError(f"Unexpected multi-run prose paragraph: {old_text[:100]!r}")
            _replace_paragraph_text(paragraph, new_text)
    return counts


def build_report() -> Path:
    base = _load_base_builder()
    with tempfile.TemporaryDirectory(prefix="fyp_report_v3_") as tmp:
        tmp_docx = Path(tmp) / "v1_evidence_source.docx"
        original_out = base.OUT
        try:
            base.OUT = tmp_docx
            base.build_report()
        finally:
            base.OUT = original_out

        source_doc = Document(tmp_docx)
        source_numbers, source_citations = _token_inventory(source_doc)
        source_paragraph_count = len(list(_all_paragraphs(source_doc)))

        doc = Document(tmp_docx)
        _apply_heading_rewrites(doc)
        counts = _rewrite_prose(doc)

        missing = [prefix for prefix, count in counts.items() if count != 1]
        if missing:
            details = {prefix: counts[prefix] for prefix in missing}
            raise AssertionError(f"Paragraph rewrite must-fire failure: {details}")

        new_numbers, new_citations = _token_inventory(doc)
        source_number_counts = {token: source_numbers.count(token) for token in set(source_numbers)}
        new_number_counts = {token: new_numbers.count(token) for token in set(new_numbers)}
        missing_numbers = {
            token: (count, new_number_counts.get(token, 0))
            for token, count in source_number_counts.items()
            if new_number_counts.get(token, 0) < count
        }
        invented_values = sorted(set(new_numbers) - set(source_numbers))
        if missing_numbers or invented_values:
            raise AssertionError(
                "Numerical inventory changed during prose rewrite: "
                f"missing={missing_numbers!r}, invented={invented_values!r}"
            )
        if source_citations != new_citations:
            raise AssertionError("Citation marker sequence changed during prose rewrite")
        if len(list(_all_paragraphs(doc))) != source_paragraph_count:
            raise AssertionError("Paragraph count changed during prose rewrite")

        _apply_cached_page_numbers(doc)

        body_text = "\n".join(p.text.lower() for p in _all_paragraphs(doc) if p.style.name != "Reference")
        remaining = [term for term in FORBIDDEN_TERMS if term in body_text]
        if remaining:
            raise AssertionError(f"Legacy report language remains: {remaining}")

        doc.core_properties.subject = "Plain-academic-English FYP report rewrite v3"
        doc.core_properties.comments = (
            "Non-destructive prose rewrite derived from the v1 evidence and layout source. "
            "Numerical tokens, citation markers, structure, tables, and figures are preserved."
        )
        OUT.parent.mkdir(parents=True, exist_ok=True)
        doc.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build_report())
