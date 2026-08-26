#!/usr/bin/env python3
"""Build the v5 language-cleanup rewrite on top of the verified v4 line.

Non-destructive contract, same as v3 -> v4: the v4 report builder runs first
into a temporary DOCX, then this builder applies a table of exact-substring
prose corrections and writes a separate v5 artifact. Nothing upstream changes.

Purpose (user-requested language pass, 2026-08-25): replace compressed
statistical shorthand ("survives correction", "survivors", "contrast",
"corrected family", "multiplicity", "unresolved", "construct", "point
estimate", "inferential threshold", "heterogeneity", "treatment effect",
interpretation-label jargon) with conventional academic English, and split
sentences that pack a result, its statistics, and its interpretation together.

Hard guarantees enforced at build time:
  * every correction matches exactly once (a silent no-op fails the build);
  * the numerical-token multiset never loses a value and never invents one;
  * the bracketed-citation sequence is byte-identical to v4's;
  * the paragraph count is unchanged;
  * no em dash is introduced and no banned shorthand term survives in the body.

Like v2/v3/v4, this builder has no REFERENCES of its own: the printed
bibliography is inherited from scripts/build_rewritten_fyp_report_v1.py through
the v3 -> v4 chain, and that inheritance is asserted against the built document.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER = ROOT / "scripts" / "build_rewritten_fyp_report_v4.py"
V1_BUILDER = ROOT / "scripts" / "build_rewritten_fyp_report_v1.py"
OUT = ROOT / "docs" / "rewritten_fyp_report_v5.docx"

CITATION = re.compile(r"\[(\d+)\]")
CITATION_RANGE = re.compile(r"\[(\d+)\]\s*[\u2013\u2014-]\s*\[(\d+)\]")

# ---------------------------------------------------------------------------
# Language corrections.
#
# Each entry is (old substring, new substring) over the built v4 text. Every
# entry must match exactly once in the whole document (body + table cells);
# anything else fails the build. Numbers, citation markers, and the claims they
# carry are reproduced verbatim. British spelling and curly double quotes are
# kept; no em dashes are introduced.
#
# Reviewed and deliberately NOT changed (kept technical or locked):
#   * RQ5's own wording ("substantive conclusion remain credible") -- research
#     questions are preserved verbatim.
#   * "Preservation under direct requests is not evidence ..." -- a locked
#     scope sentence (tests/test_report_scope_invariants.py).
#   * "A result obtained with one method is evidence about that intervention"
#     -- the named-intervention scope rule; "intervention" is load-bearing.
#   * Section 5.4 heading "XSTest Scorer Sensitivity and Human Validation",
#     "sensitivity analysis", "audit trail", "equivalence testing", "statistical
#     power", "confidence interval", "null hypothesis", "BH-FDR" after its
#     abbreviation-table definition -- necessary technical terminology.
# ---------------------------------------------------------------------------
LANGUAGE_REWRITES: list[tuple[str, str]] = [
    # -- Abstract ----------------------------------------------------------
    (
        "Human validation revealed that the semantic refusal judge aligned substantially "
        "better with annotators than the keyword regex, indicating that the Phi refusal "
        "finding is sensitive to scorer choice.",
        "Human validation showed that the semantic refusal judge agreed much more closely "
        "with the human annotations than the keyword regex did. The Phi-4-mini over-refusal "
        "result therefore depends on the scoring method and is not reproduced by the "
        "semantic judge.",
    ),
    (
        "Within the evaluated direct-request setting, NF4 quantization caused measurable "
        "capability costs but produced no statistically significant increase",
        "Within the evaluated direct-request setting, NF4 quantization produced measurable "
        "capability decreases for specific model and benchmark combinations but no "
        "statistically significant increase",
    ),
    (
        "Three results remained significant: reduced MMLU accuracy",
        "Three results remained statistically significant: reduced MMLU accuracy",
    ),
    # -- Chapter 1 ---------------------------------------------------------
    (
        "The interpretive problem is not only to detect a change, but to determine what "
        "the change means.",
        "The difficulty is not only to detect a change, but to determine what the change "
        "means.",
    ),
    (
        "This creates a practical methodological gap for developers who need evidence that "
        "is reproducible and interpretable before deployment.",
        "This leaves developers without evidence that is reproducible and interpretable at "
        "the point where deployment decisions are made.",
    ),
    (
        "false-discovery-rate control, and power-oriented interpretation, and scorer "
        "validity is examined",
        "false-discovery-rate control, and interpretation informed by statistical power. "
        "Scorer validity is examined",
    ),
    (
        "The problem addressed by this study is interpretive rather than merely operational:",
        "The central problem addressed by this study is one of interpretation rather than "
        "measurement alone:",
    ),
    (
        "under a single controlled precision contrast,",
        "under a single controlled precision comparison,",
    ),
    (
        "while all five MMLU point estimates show capability declines.",
        "while the estimated MMLU change is a decline for all five pairs.",
    ),
    (
        "provide useful anchors across different reasoning domains",
        "provide useful reference points across different reasoning domains",
    ),
    (
        "exact McNemar tests, multiplicity control, and power-oriented interpretation.",
        "exact McNemar tests, correction for multiple comparisons, and interpretation "
        "informed by statistical power.",
    ),
    (
        "stores derived scorer classifications in separate, verifiable audit sidecars.",
        "stores derived scorer classifications in separate, verifiable sidecar files.",
    ),
    # -- Chapter 2 ---------------------------------------------------------
    (
        "the compression method, the safety construct, the capability benchmark",
        "the compression method, the safety behaviour being measured, the capability "
        "benchmark",
    ),
    (
        "Their value here is diagnostic: if a lower harmful-compliance score occurs",
        "Their role here is explanatory: if a lower harmful-compliance score occurs",
    ),
    (
        "benefits from fewer outcomes with explicit constructs,",
        "benefits from fewer outcomes whose scoring criteria are defined explicitly,",
    ),
    (
        "That additional measurement uncertainty is one reason",
        "That extra uncertainty in measurement is one reason",
    ),
    (
        "avoids collapsing the two benchmarks into one undocumented capability index.",
        "keeps the two benchmarks separate rather than combining them into a single "
        "capability index.",
    ),
    (
        "Therefore, these methods are not interchangeable interventions.",
        "These methods are therefore not interchangeable.",
    ),
    (
        "a graded fp16–INT8–NF4 curve would be plausible.",
        "behaviour would be expected to move gradually across fp16, INT8, and NF4.",
    ),
    (
        "These are related but non-identical constructs.",
        "These are related but different ways of defining safety.",
    ),
    (
        "worst-case vulnerability and typical treatment effect are different questions.",
        "worst-case vulnerability and the typical effect of ordinary quantization are "
        "different questions.",
    ),
    (
        "because it observes broad four-bit preservation [31].",
        "because it finds that four-bit compression broadly preserves trustworthiness [31].",
    ),
    (
        "XSTest provides a smaller human-curated diagnostic set, while OR-Bench supplies "
        "breadth.",
        "XSTest provides a smaller human-curated test set targeted at exaggerated safety, "
        "while OR-Bench supplies breadth.",
    ),
    (
        "its policy construct is not identical to HarmBench's behaviour-specific criterion",
        "the policy it applies is not identical to HarmBench's behaviour-specific criterion",
    ),
    (
        "making model-specific errors that distort a paired contrast.",
        "making model-specific errors that distort a paired comparison.",
    ),
    (
        "a semantic judge tests construct coverage, and a human-labelled sample",
        "a semantic judge tests coverage of the behaviour itself, and a human-labelled "
        "sample",
    ),
    (
        "yet construct alignment still matters.",
        "yet alignment between what the scorer measures and what the benchmark defines "
        "still matters.",
    ),
    (
        "These two audits address different constructs, so their agreement coefficients",
        "These two validation exercises evaluate different behaviours and scoring criteria, "
        "so their agreement coefficients",
    ),
    (
        "with a five-pair matched design, NF4 as the primary intervention, INT8 as an "
        "additional precision condition,",
        "with a five-pair matched design, NF4 as the primary quantization condition, INT8 "
        "as an additional precision condition,",
    ),
    (
        "capability, scorer error, truncation, or multiplicity.",
        "capability, scorer error, truncation, or the number of comparisons tested.",
    ),
    (
        "Crucially, a weakness in any layer can create a treatment effect that is not "
        "caused by the quantizer.",
        "A weakness in any layer can therefore create an apparent effect of quantization "
        "that is not caused by the quantizer itself.",
    ),
    (
        "Twenty model–outcome contrasts create many opportunities",
        "Twenty model–outcome comparisons create many opportunities",
    ),
    (
        "apply a multiplicity correction [10]",
        "apply a correction for multiple comparisons [10]",
    ),
    (
        "which observed changes remain credible after accounting for the number of "
        "comparisons?",
        "which observed changes remain supported after accounting for the number of "
        "comparisons?",
    ),
    (
        "It should state what effect sizes the sample can detect, test whether a scorer is "
        "capable of seeing the relevant construct, and conditions that could obscure an "
        "effect or create a misleading one.",
        "Such a design states which effect sizes the sample can detect, checks that a "
        "scorer can detect the relevant behaviour, and identifies conditions that could "
        "obscure an effect or create a misleading one.",
    ),
    (
        "They do not prove exact preservation, but they make the non-significant result "
        "easier to interpret",
        "They do not show that results are exactly unchanged, but they make a "
        "non-significant result easier to interpret",
    ),
    # -- Chapter 3 ---------------------------------------------------------
    (
        "whereas the treatment member is loaded from the same weights",
        "whereas the quantized member is loaded from the same weights",
    ),
    (
        "The primary treatment is NF4 four-bit quantization",
        "The primary quantization condition is NF4 four-bit quantization",
    ),
    (
        "Model family and scale are observed grouping variables rather than randomised "
        "treatments.",
        "Model family and scale are observed grouping variables rather than randomly "
        "assigned conditions.",
    ),
    (
        "ARC-Challenge provides a larger second anchor with a different task structure",
        "ARC-Challenge provides a larger second capability benchmark with a different task "
        "structure",
    ),
    (
        "can be significant on ARC and unresolved on HarmBench.",
        "can be statistically significant on ARC yet inconclusive on HarmBench.",
    ),
    (
        "This preserves the original analysis while testing whether its conclusion survives "
        "a more semantic scorer.",
        "This preserves the original analysis while testing whether its conclusion still "
        "holds when a more semantic scorer is used.",
    ),
    (
        "contains 20 contrasts: five model pairs multiplied by four outcomes",
        "contains 20 comparisons: five model pairs multiplied by four outcomes",
    ),
    (
        "controls multiplicity across that family [49], and a result is described as "
        "surviving correction only when its adjusted q-value is below 0.05.",
        "corrects for multiple comparisons across that set [49], and a result is described "
        "as statistically significant after correction only when its adjusted q-value is "
        "below 0.05.",
    ),
    (
        "used in the scorer sensitivity analysis [55], and its construct is deliberately "
        "broader than the HarmBench behaviour label.",
        "used in the scorer sensitivity analysis [55], and the behaviour it labels as unsafe "
        "is deliberately broader than the HarmBench behaviour label.",
    ),
    (
        "Exactly three primary contrasts survived Benjamini–Hochberg false-discovery-rate "
        "correction, so scorer validation remains necessary after statistical correction.",
        "Exactly three primary comparisons remained statistically significant after "
        "Benjamini–Hochberg false-discovery-rate correction, so scorer validation remains "
        "necessary even after statistical correction.",
    ),
    (
        "Interpretation labels combine direction and capability context. Alignment "
        "degradation means harmful compliance rises while capability is broadly preserved, "
        "and alignment improvement is the converse. Capability loss with lower harmful "
        "compliance is used when harmful compliance falls alongside capability, whereas "
        "broad degradation describes lower capability without clear evidence of a safety "
        "change. In every case the labels summarise a pattern and do not establish a "
        "mechanism.",
        "Interpretation labels combine the direction of the safety change with the "
        "capability context. Alignment degradation describes harmful compliance rising "
        "while capability shows no substantial decrease, and alignment improvement is the "
        "converse. Capability loss with lower harmful compliance describes harmful "
        "compliance falling alongside capability, so the lower rate should not by itself be "
        "read as a safety gain. Broad degradation describes a clear capability decrease "
        "without clear evidence of a safety change. The labels summarise observed patterns; "
        "they do not establish a mechanism.",
    ),
    (
        "Revised scorers, judge outputs, and diagnostics are instead written",
        "Outputs of revised scorers, judges, and diagnostic procedures are instead written",
    ),
    (
        "but they do not prove that a citation entails a sentence or that a construct is "
        "valid.",
        "but they do not prove that a citation entails a sentence or that a scoring method "
        "measures what it is intended to measure.",
    ),
    (
        "Multiplicity correction is performed from the complete 20-row analysis record",
        "Multiple-comparison correction is performed from the complete 20-row analysis "
        "record",
    ),
    # -- Chapter 4 ---------------------------------------------------------
    (
        "The primary statistical family encompasses 20 NF4 comparisons",
        "The primary analysis covers 20 NF4 comparisons",
    ),
    (
        "Applying multiple-comparison correction clarifies that the primary statistically "
        "defensible effects concern capability degradation. The sole surviving over-refusal "
        "effect is a decrease for Phi-4-mini, which Chapter 5 shows to be most plausibly an "
        "artifact of keyword regex scoring, as it does not survive semantic evaluation.",
        "After multiple-comparison correction, the effects that remain statistically "
        "significant concern capability. The only over-refusal effect that remains "
        "significant is a decrease for Phi-4-mini, which Chapter 5 shows is not reproduced "
        "by the semantic judge and is therefore most plausibly an artifact of keyword regex "
        "scoring.",
    ),
    (
        "Exact p-values and corrected q-values distinguish unadjusted contrasts from "
        "results surviving false-discovery-rate control.",
        "Exact p-values and adjusted q-values distinguish unadjusted results from results "
        "that remain statistically significant after false-discovery-rate control.",
    ),
    (
        "No NF4 harmful-compliance contrast survives multiplicity correction. Figure 4.1 "
        "shows that the five point estimates range from −0.040 to +0.040, with intervals "
        "that cross or touch zero except for the uncorrected Llama decrease. Overall, this "
        "is evidence against a large, consistent increase in harmful compliance under NF4 "
        "in the evaluated setting.",
        "None of the five NF4 harmful-compliance comparisons remains statistically "
        "significant after correction for multiple comparisons. Figure 4.1 shows that the "
        "five estimated changes range from −0.040 to +0.040, with confidence intervals that "
        "cross or touch zero except for the Llama decrease, which is significant before "
        "correction. Overall, these results provide no indication of a large, consistent "
        "increase in harmful compliance under NF4 in the evaluated setting.",
    ),
    (
        "Its individually significant ΔASR is −0.040, a decrease that does not survive the "
        "20-test correction. More importantly, the model also loses ARC accuracy, and a "
        "lower ASR cannot be interpreted as a clean alignment gain when capability also "
        "decreases. Mistral-7B decreases by 0.020 with substantial prompt-level "
        "disagreement and no significant aggregate change.",
        "The estimated change in harmful compliance is −0.040. This difference is "
        "statistically significant before multiple-comparison correction, but it is no "
        "longer statistically significant after the 20 primary comparisons are adjusted "
        "using the Benjamini–Hochberg procedure. The model also loses ARC accuracy, so the "
        "lower harmful-compliance rate should not be interpreted as clear evidence of "
        "improved alignment. Mistral-7B decreases by 0.020 with substantial prompt-level "
        "disagreement and no statistically significant aggregate change.",
    ),
    (
        "Paired deltas remain comparable as treatment effects, but their operational "
        "implications differ.",
        "The paired differences remain useful for comparing the effect of NF4 quantization "
        "within each model, but their practical implications differ.",
    ),
    (
        "Qwen3-4B is the largest point estimate for increased harmful compliance.",
        "Qwen3-4B shows the largest estimated increase in harmful compliance, at 0.040.",
    ),
    (
        "This is therefore a result worth repeating in a larger study, but the present "
        "evidence does not justify a confirmed increase.",
        "This pattern would be worth examining in a larger study, but the present data do "
        "not provide sufficient evidence of an increase.",
    ),
    (
        "that imbalance produces the individually significant decrease. Section 4.4 "
        "explains the caution: ARC accuracy falls by 0.0316 in the same precision contrast.",
        "that imbalance produces the decrease that is statistically significant before "
        "correction. Section 4.4 explains the caution: ARC accuracy falls by 0.0316 in the "
        "same fp16–NF4 comparison.",
    ),
    (
        "No pair shows a statistically significant increase in over-refusal, and four pairs "
        "are non-significant. Phi-4-mini's −0.048 over-refusal decrease under the "
        "pre-specified regex is scorer-dependent, because the independent semantic judge "
        "does not reproduce it. The original scorer reports a decrease rather than an "
        "increase, since the NF4 model refuses fewer benign prompts.",
        "No pair shows a statistically significant increase in over-refusal. Four pairs "
        "show no statistically significant change, and Phi-4-mini shows a statistically "
        "significant decrease of −0.048 under the pre-specified regex. That result depends "
        "on the scoring method, because the independent semantic judge does not reproduce "
        "it. It is also a decrease rather than an increase: the NF4 model refuses fewer "
        "benign prompts.",
    ),
    (
        "The primary family retains the regex result because it was the pre-specified "
        "outcome. By contrast, the validation-informed sensitivity analysis replaces that "
        "score with judge-strict over-refusal. In that validation-informed sensitivity "
        "analysis, exactly two contrasts survived correction, both capability losses. This "
        "distinction avoids rewriting the analysis after seeing the data while still giving "
        "more weight to the scorer that agreed more closely with the human annotations.",
        "The primary analysis retains the regex score because it was pre-specified. The "
        "validation-informed sensitivity analysis instead replaces that score with "
        "judge-strict over-refusal. In that parallel analysis, exactly two comparisons "
        "remained statistically significant after correction, both capability decreases. "
        "Keeping the two analyses separate avoids rewriting the analysis after seeing the "
        "data, while still giving more weight to the scorer that agreed more closely with "
        "the human annotations.",
    ),
    (
        "large enough to survive correction under the original scorer. It is also exactly "
        "the kind of result that requires construct validation, because a small number of "
        "pattern matches can move the rate materially.",
        "large enough to remain statistically significant after correction under the "
        "original scorer. It is also exactly the kind of result that requires validation "
        "that the scoring method is measuring the intended behaviour, because a small "
        "number of pattern matches can move the rate materially.",
    ),
    (
        "representing a substantial loss of 0.090 that survives Benjamini–Hochberg "
        "correction at q = 0.008.",
        "a substantial decrease of 0.090 that remains statistically significant after "
        "Benjamini–Hochberg correction (q = 0.008).",
    ),
    (
        "Llama-3.2-3B has an ARC contrast of −0.032 (unrounded: −0.0316), which survives "
        "BH-FDR correction with q = 0.008. Qwen3-4B ARC decreases by 0.0162 and is "
        "individually significant, but it does not survive the full correction.",
        "Llama-3.2-3B shows an ARC decrease of −0.032 (unrounded: −0.0316), which remains "
        "statistically significant after Benjamini–Hochberg false-discovery-rate correction "
        "with q = 0.008. Qwen3-4B ARC decreases by 0.0162 and is statistically significant "
        "before correction, but it does not remain significant after the full correction.",
    ),
    (
        "The ARC result is individually significant because the benchmark has 1,172 paired "
        "items, but its adjusted q-value is 0.1687. Sample size and multiplicity therefore "
        "change the evidence attached to a small absolute effect.",
        "The ARC result is statistically significant before correction because the "
        "benchmark has 1,172 paired items, but its adjusted q-value is 0.1687. Sample size "
        "and correction for multiple comparisons therefore change the evidence attached to "
        "a small absolute effect.",
    ),
    (
        "estimated more precisely, so ARC survives correction.",
        "estimated more precisely, so the ARC result remains statistically significant "
        "after correction.",
    ),
    (
        "even though only one crosses the inferential threshold.",
        "even though only one reaches the threshold for statistical significance.",
    ),
    (
        "neither of which is significant. These unresolved cases show why a rule that "
        "labels every negative point estimate as capability degradation would overstate the "
        "evidence.",
        "none of which is statistically significant. In these cases the evidence is "
        "inconclusive: a rule that labels every negative estimated change as capability "
        "degradation would overstate the evidence.",
    ),
    (
        "whereas several smaller observed deltas do not. Appendix B therefore reports the "
        "thresholds rather than converting all non-significant effects into claims of "
        "equivalence.",
        "whereas several smaller observed differences do not. Appendix B therefore reports "
        "the thresholds: smaller non-significant differences may simply fall below the "
        "study's reliable detection range, and their non-significant results should be "
        "treated as inconclusive rather than as evidence that fp16 and NF4 are equivalent.",
    ),
    (
        "The MMLU point estimate decreases for every pair, whereas the harmful-compliance "
        "deltas are mixed in sign and none of them survives correction.",
        "The estimated MMLU change is a decrease for every pair, whereas the "
        "harmful-compliance changes differ in sign and none of them remains statistically "
        "significant after correction.",
    ),
    (
        "However, that contrast is much smaller on ARC: the 1.7B decrease is −0.0094 and "
        "the 4B decrease is −0.0162. The evidence therefore supports task-specific "
        "small-model sensitivity, not a universal monotonic scale law.",
        "However, that difference is much smaller on ARC: the 1.7B decrease is −0.0094 and "
        "the 4B decrease is −0.0162. The evidence therefore indicates that the smaller "
        "model's higher sensitivity is specific to MMLU, not a universal monotonic scaling "
        "law.",
    ),
    (
        "This heterogeneity is why results are reported as paired within-model contrasts "
        "rather than as a leaderboard.",
        "This variation across models is why results are reported as paired within-model "
        "comparisons rather than as a ranking.",
    ),
    (
        "The interpretation labels are descriptive summaries of the combined changes. "
        "Qwen3-1.7B is classified as broad degradation because capability falls while ASR "
        "is flat, and Llama is capability loss accompanied by lower harmful compliance "
        "because ASR and capability both fall. The remaining classifications are based only "
        "on the direction of the point estimates: Qwen3-4B and Phi show point estimates in "
        "the direction of higher harmful compliance, while Mistral has a point estimate in "
        "the direction of lower harmful compliance. However, none of those safety "
        "classifications is supported after correction.",
        "The interpretation labels summarise the observed combination of safety and "
        "capability changes; they do not establish a causal mechanism. Qwen3-1.7B shows a "
        "clear capability reduction while harmful compliance is unchanged; the report "
        "labels this pattern broad degradation. For Llama-3.2-3B, both harmful compliance "
        "and capability decrease under NF4; because capability also declines, the lower "
        "harmful-compliance rate should not be read as a safety improvement, and the report "
        "labels this pattern capability loss with lower harmful compliance. The remaining "
        "classifications rest only on the direction of the estimated changes: Qwen3-4B and "
        "Phi show estimated changes towards higher harmful compliance, and Mistral shows an "
        "estimated change towards lower harmful compliance. None of those safety directions "
        "is statistically significant after correction.",
    ),
    (
        "Cross-family response styles also affect measurement. Mistral is the clearest "
        "case: its refusal proxy is far more pessimistic than the semantic classifier. A "
        "family comparison that ignored scorer agreement would mix the effect of "
        "quantization with differences between the benchmarks.",
        "Different model families produce responses in different styles, which can affect "
        "how automatic scorers classify them. Mistral is the clearest case: its refusal "
        "proxy reports much higher attack-success rates than the semantic classifier. A "
        "family comparison that ignored scorer agreement would mix the effect of "
        "quantization with differences in scoring.",
    ),
    (
        "The INT8 comparison asks whether the observed pattern is graded with precision.",
        "The INT8 comparison asks whether the observed pattern changes gradually with "
        "precision.",
    ),
    (
        "harmful compliance does not follow one shared path across fp16, INT8, and NF4:",
        "harmful compliance does not change consistently across fp16, INT8, and NF4:",
    ),
    (
        "every INT8 HarmBench contrast at 512 tokens is non-significant under both the "
        "primary classifier and the API judge.",
        "no INT8 HarmBench comparison at 512 tokens is statistically significant under "
        "either the primary classifier or the API judge.",
    ),
    (
        "practical similarity for the INT8 capability contrasts within the stated ±0.05 "
        "margin",
        "practical similarity for the INT8 capability comparisons within the stated ±0.05 "
        "margin",
    ),
    (
        "For deployment, the evidence favours a staged decision.",
        "For deployment, these results support a staged decision.",
    ),
    (
        "Together they argue against one alignment effect caused by lower precision.",
        "Together they argue against a single alignment effect of lower precision that "
        "applies to all five pairs.",
    ),
    (
        "The point estimates are 0.000, +0.040, −0.040, −0.020, and +0.020 across the five "
        "pairs. Only the Llama decrease is individually significant, and no ASR result "
        "survives BH-FDR. The evidence therefore does not support a corrected increase in "
        "harmful compliance.",
        "The estimated changes are 0.000, +0.040, −0.040, −0.020, and +0.020 across the "
        "five pairs. Only the Llama decrease is statistically significant before "
        "multiple-comparison correction, and no HarmBench result remains statistically "
        "significant after Benjamini–Hochberg correction. The evidence therefore does not "
        "support an increase in harmful compliance once correction is applied.",
    ),
    (
        "The pre-specified regex finds no significant increase, and its only result that "
        "remains significant after correction is the Phi decrease. By contrast, the "
        "semantic judge finds neither a significant increase nor the same Phi decrease.",
        "The pre-specified regex finds no statistically significant increase, and its only "
        "result that remains statistically significant after correction is the Phi decrease. "
        "By contrast, the semantic judge finds neither a statistically significant increase "
        "nor the same Phi decrease.",
    ),
    (
        "Qwen3-1.7B MMLU and Llama ARC survive correction. Qwen3-4B ARC is individually "
        "significant but not corrected, and the remaining estimates are mixed or unresolved. "
        "Capability degradation is therefore real for specific model–task combinations "
        "rather than universal across the matrix.",
        "Qwen3-1.7B MMLU and Llama ARC remain statistically significant after correction. "
        "Qwen3-4B ARC is statistically significant before correction but not after "
        "adjustment, and the remaining estimated differences vary in direction or do not "
        "provide clear statistical evidence of a change. Capability decreases are therefore "
        "established only for specific model–task combinations rather than universally "
        "across the matrix.",
    ),
    (
        "Overall, the evidence supports heterogeneity, not one shared response to "
        "bit-width.",
        "Overall, the effect of lower precision differs across models and benchmarks rather "
        "than following one consistent pattern.",
    ),
    (
        "RQ5 investigates whether the primary conclusions remain robust across alternative "
        "evaluators",
        "RQ5 asks whether the primary conclusions hold across alternative evaluators",
    ),
    (
        "human validation favours the HarmBench classifier over the refusal proxy",
        "human validation supports the HarmBench classifier over the refusal proxy",
    ),
    # -- Chapter 5 ---------------------------------------------------------
    (
        "However, that is not the construct HarmBench defines:",
        "However, that is not the behaviour HarmBench defines:",
    ),
    (
        "so proxy error can itself become treatment-dependent.",
        "so the proxy's error can itself depend on the precision condition.",
    ),
    (
        "broader “unsafe response” construct differs most from",
        "broader “unsafe response” criterion differs most from",
    ),
    (
        "but it is substantially closer to the human construct.",
        "but it agrees substantially more closely with the human annotations.",
    ),
    (
        "The human audit sample purposefully oversamples scorer disagreements to maximise "
        "diagnostic power, meaning raw proportions do not represent baseline population "
        "rates. The key takeaway is the relative fidelity of the evaluators: the official "
        "classifier exhibits substantially higher agreement with the human annotator than "
        "the keyword refusal proxy.",
        "The human-labelled sample deliberately oversamples the items on which the two "
        "scorers disagree, so its raw proportions do not represent baseline population "
        "rates. What the sample shows is the relative accuracy of the two scorers: the "
        "official classifier agrees substantially more closely with the human annotator "
        "than the keyword refusal proxy does.",
    ),
    (
        "Human ground-truth annotation clarifies this discrepancy.",
        "Human annotation clarifies this discrepancy.",
    ),
    (
        "exactly two contrasts survive Benjamini–Hochberg correction across the entire "
        "study: the Qwen MMLU and Llama ARC capability declines. No safety or over-refusal "
        "contrast remains statistically significant.",
        "exactly two comparisons remain statistically significant after Benjamini–Hochberg "
        "correction across the entire study: the Qwen MMLU and Llama ARC capability "
        "decreases. No safety or over-refusal comparison remains statistically significant.",
    ),
    (
        "The earlier signal was therefore an effect of the shorter generation limit rather "
        "than a result that generalised to HarmBench's reference budget.",
        "The earlier apparent increase was therefore an effect of the shorter generation "
        "limit rather than a result that carried over to HarmBench's reference budget.",
    ),
    (
        "selecting either after inspection would produce a misleading narrative.",
        "selecting either after inspection would give a misleading impression.",
    ),
    (
        "greedy point estimate and its five-seed stochastic decoding distribution",
        "greedy estimate and its five-seed stochastic decoding distribution",
    ),
    (
        "demonstrating consistency with expected sampling variance.",
        "which is consistent with expected sampling variance.",
    ),
    (
        "smaller localised effects remain below its resolution threshold.",
        "smaller localised effects fall below the study's reliable detection range.",
    ),
    (
        "The reported null finding is supported along four partly independent dimensions: "
        "no HarmBench contrast survives multiplicity correction, the primary classifier and "
        "two further judges find no significant ASR increases, the 128-token Qwen3-1.7B "
        "artifact does not survive the move to the 512-token reference budget, and no "
        "decoding seed in the multi-seed arm shows a consistent rise in harmful compliance.",
        "Four partly independent checks support the null result: no HarmBench comparison "
        "remains statistically significant after correction for multiple comparisons, the "
        "primary classifier and two further judges find no statistically significant ASR "
        "increases, the 128-token Qwen3-1.7B anomaly does not reappear at the 512-token "
        "reference budget, and no decoding seed in the multi-seed analysis shows a "
        "consistent rise in harmful compliance.",
    ),
    (
        "Capability evaluations possess varying statistical sensitivity due to differing "
        "sample sizes between MMLU and ARC. As detailed in Appendix B, observed capability "
        "changes below the minimum detectable effect threshold are reported as "
        "non-detectable rather than assumed identical, with formal equivalence testing "
        "reserved for the INT8 precision sweep.",
        "Capability evaluations differ in statistical sensitivity because MMLU and ARC have "
        "different sample sizes. As detailed in Appendix B, observed capability changes "
        "below the minimum detectable effect are reported as undetected rather than assumed "
        "to be absent, and formal equivalence testing is reserved for the INT8 precision "
        "comparison.",
    ),
    (
        "The empirical evidence demonstrates the absence of a statistically significant "
        "increase in harmful compliance after correction within the detection limits of the "
        "experimental design, without claiming absolute mathematical invariance across "
        "unmeasured distributions.",
        "Within the detection limits of the experimental design, the study finds no "
        "statistically significant increase in harmful compliance after correction; it does "
        "not claim exact invariance across unmeasured distributions.",
    ),
    (
        "Systematic analysis of scorer divergence provides deeper methodological insights "
        "than uncritical averaging. When evaluators diverge, examining the underlying "
        "criteria identifies measurement biases and justifies reliance on human-validated "
        "instruments.",
        "Analysing why scorers disagree is more informative than averaging their scores. "
        "When evaluators disagree, examining the criteria they apply helps identify "
        "measurement error and shows why human-validated instruments carry more weight.",
    ),
    # -- Chapter 6 ---------------------------------------------------------
    (
        "underscore the necessity of jointly evaluating harmful compliance, capability "
        "preservation, and scorer behaviour.",
        "show why harmful compliance, capability, and scorer behaviour need to be evaluated "
        "together.",
    ),
    (
        "A more accurate characterisation of low-bit quantization is prompt-level decision "
        "boundary instability accompanied by capability degradation. Small weight "
        "perturbations shift borderline outputs in both directions without generating "
        "systematic aggregate alignment failure.",
        "A more accurate description of low-bit quantization in this study is instability "
        "in individual borderline decisions accompanied by capability loss. Small weight "
        "perturbations shift borderline outputs in both directions without producing a "
        "systematic shift in aggregate alignment.",
    ),
    (
        "which reports broad preservation of trustworthiness at four bits and larger damage "
        "under more aggressive compression [31].",
        "which reports that trustworthiness is broadly preserved at four bits and that "
        "larger damage occurs under more aggressive compression [31].",
    ),
    (
        "Simple string rules can therefore fail in both directions depending on the "
        "construct.",
        "Simple string rules can therefore fail in both directions depending on the "
        "behaviour being scored.",
    ),
    (
        "Llama Guard is open-weight but uses a broader policy construct",
        "Llama Guard is open-weight but applies a broader policy definition",
    ),
    (
        "When the estimated treatment effect is small, scorer error can be as large as the "
        "effect.",
        "When the estimated effect of quantization is small, scorer error can be as large "
        "as the effect.",
    ),
    (
        "because its bias depends on how its text patterns relate to the construct.",
        "because its bias depends on how its text patterns relate to the behaviour being "
        "counted.",
    ),
    (
        "Retaining the preregistered analysis preserves methodological transparency, while "
        "reporting the validated semantic judge reflects superior measurement fidelity.",
        "Retaining the preregistered analysis preserves methodological transparency, while "
        "reporting the validated semantic judge reflects its closer agreement with the "
        "human annotations.",
    ),
    (
        "The matched revisions, shared prompts, fixed decoding controls, and paired tests "
        "strengthen causal attribution to the load-time precision change.",
        "The matched revisions, shared prompts, fixed decoding controls, and paired tests "
        "strengthen the attribution of the observed differences to the load-time precision "
        "change.",
    ),
    (
        "Their role is diagnostic, not confirmatory.",
        "Their role is exploratory: they qualify the interpretation but do not confirm it.",
    ),
    (
        "XSTest measures exaggerated safety on a compact diagnostic set.",
        "XSTest measures exaggerated safety on a compact targeted test set.",
    ),
    (
        "HarmBench DirectRequest measures harmful compliance under direct unsafe "
        "instructions.",
        "Construct validity asks whether each benchmark and scorer measures the behaviour "
        "it is intended to measure. HarmBench DirectRequest measures harmful compliance "
        "under direct unsafe instructions.",
    ),
    (
        "which supports construct alignment but creates dependence on one learned "
        "evaluator.",
        "which aligns the scorer with the benchmark's criterion but creates dependence on "
        "one learned evaluator.",
    ),
    (
        "whereas Llama Guard changes both weights and construct.",
        "whereas Llama Guard differs from it in both weights and the behaviour labelled "
        "unsafe.",
    ),
    (
        "pair the statistics, declare multiplicity, and validate the scorer.",
        "pair the statistics, state how many comparisons were corrected, and validate the "
        "scorer.",
    ),
    (
        "which protects the within-pair contrast.",
        "which protects the within-pair comparison.",
    ),
    (
        "retain transparent proxies only as secondary diagnostics.",
        "retain transparent proxies only as secondary checks.",
    ),
    # -- Chapter 7 ---------------------------------------------------------
    (
        "In summary, RQ1 is answered by a robust null: no NF4 change in harmful compliance "
        "remains significant after correcting the 20 primary comparisons",
        "In summary, RQ1 receives a negative answer: no NF4 change in harmful compliance "
        "remains statistically significant after correction of the 20 primary comparisons",
    ),
    (
        "For RQ3, significant capability degradations are observed",
        "For RQ3, statistically significant capability decreases are observed",
    ),
    (
        "though not on every contrast.",
        "though not on every comparison.",
    ),
    # -- Residual sweep (post-first-build review) --------------------------
    (
        "60.3% truncation at 128; signal disappears at 512",
        "60.3% truncation at 128; the effect disappears at 512",
    ),
    (
        "Their human evaluation reveals damage that automatic metrics understate. "
        "Here, human validation reveals scorer error that changes which result "
        "deserves weight.",
        "Their human evaluation shows damage that automatic metrics understate. "
        "Here, human validation identifies scorer error that changes which result "
        "deserves weight.",
    ),
    (
        "RQ4 reveals that capability degradation varies substantially by model "
        "architecture and precision;",
        "RQ4 shows that capability degradation varies substantially by model "
        "architecture and precision;",
    ),
    (
        "The present study uses paired tests and FDR control because its question "
        "is inferential: which observed changes remain supported after accounting "
        "for the number of comparisons?",
        "The present study uses paired tests and FDR control to answer that "
        "question: which observed changes remain supported after accounting for "
        "the number of comparisons?",
    ),
    (
        "The classifier's recall is 0.757 and its precision is 0.609 on the audit "
        "sample.",
        "The classifier's recall is 0.757 and its precision is 0.609 on the "
        "human-labelled sample.",
    ),
    (
        "prompt-level agreement, and a disclosed human audit.",
        "prompt-level agreement, and a disclosed human annotation sample.",
    ),
    (
        "The XSTest human audit deviated from its pre-registered blindness-exclusion "
        "rule.",
        "The XSTest human-labelling exercise deviated from its pre-registered "
        "blindness-exclusion rule.",
    ),
    (
        "so human audits and alternative scorers remain necessary even when the "
        "code is deterministic.",
        "so human annotation and alternative scorers remain necessary even when "
        "the code is deterministic.",
    ),
    (
        "Mistral-7B ARC is the only positive point estimate, at +0.0094, which is "
        "also non-significant.",
        "Mistral-7B ARC is the only positive estimated change, at +0.0094, which "
        "is also non-significant.",
    ),
    (
        "the prompt is fixed and the contrast focuses on the target model.",
        "the prompt is fixed and the comparison focuses on the target model.",
    ),
    (
        "The other capability contrasts are non-significant.",
        "The other capability comparisons are non-significant.",
    ),
    (
        "Its capability contrasts are close to fp16 within the stated equivalence "
        "margin",
        "Its capability comparisons are close to fp16 within the stated equivalence "
        "margin",
    ),
    # -- Tables (prose cells only; no value changes) -----------------------
    (
        "ASR −0.040 to +0.040 (−4.0 to +4.0 points), none surviving correction; MMLU to "
        "−0.090 (−9.0 points)",
        "ASR −0.040 to +0.040 (−4.0 to +4.0 points), none statistically significant after "
        "correction; MMLU to −0.090 (−9.0 points)",
    ),
    # -- Appendices --------------------------------------------------------
    (
        "it estimates comparative scorer fitness rather than population ASR",
        "it estimates how well each scorer matches the annotator rather than population "
        "ASR",
    ),
    (
        "prevents the audit from being described as a fully preregistered confirmation.",
        "prevents the validation from being described as a fully preregistered "
        "confirmation.",
    ),
]

# Shorthand terms that must not appear anywhere in the built v5 body. Each entry
# is reviewed: words with legitimate technical uses ("contrast" as a discourse
# connective, "construct validity" as a named measurement concept, "sensitivity
# analysis", "audit trail") are deliberately absent from this list.
FORBIDDEN_SHORTHAND = (
    "survive",
    "survived",
    "surviving",
    "survivor",
    "corrected family",
    "statistical family",
    "inferential threshold",
    "precision contrast",
    "one shared path",
    "treatment effect",
    "treatment-dependent",
    "scorer-dependent",
    "heterogeneity",
    "unresolved",
)

# Front-matter page-number overrides, filled after rendering v5 and checking the
# cached contents/lists against a real LibreOffice render (same mechanism as
# V4_PAGE_NUMBERS in the v4 builder).
V5_PAGE_NUMBERS: dict[str, int] = {
    "2.4 Refusal, Harmfulness, and Threat Models": 9,
    "Chapter 5: Scorer Validation and Robustness": 30,
    "5.1 HarmBench Classifier versus Refusal Proxy": 30,
    "5.2 Independent HarmBench Judges": 31,
    "5.3 Human Validation of Harmful-Compliance Scorers": 32,
    "5.4 XSTest Scorer Sensitivity and Human Validation": 32,
    "5.5 Generation-Length Robustness": 33,
    "5.6 Multi-Seed Decoding Sensitivity": 34,
    "5.7 Power and the Meaning of the Null": 35,
    "5.8 Robustness Synthesis": 36,
    "Chapter 6: Discussion": 37,
    "6.1 Answer to the Central Question": 37,
    "6.2 Relationship to Prior Work": 37,
    "6.3 Capability Effects and Deployment": 40,
    "6.4 Validity of the Scoring Methods": 41,
    "6.5 Threats to Validity": 41,
    "6.6 Practical Implications": 43,
    "Chapter 7: Conclusion and Future Work": 44,
    "7.1 Conclusions": 44,
    "7.2 Prioritised Future Work": 44,
    "References": 46,
    "Appendix A: Reproducibility Summary": 49,
    "Appendix B: Minimum Detectable Capability Effects": 50,
    "Appendix C: Human-Validation Boundaries": 51,
    "Figure 5.1. HarmBench classifier ASR versus the refusal-regex proxy": 31,
    "Figure 5.2. Qwen3-1.7B greedy result and five-seed HarmBench sensitivity": 35,
    "Table 5.1. Per-pair HarmBench deltas under three judges": 32,
    "Table 5.2. HarmBench classifier deltas at 128 and 512 tokens": 34,
    "Table 5.3. Five-seed HarmBench sensitivity": 34,
    "Table 6.1. Possible explanations for the HarmBench result": 37,
    "Table 6.2. Comparison with published quantization results": 38,
    "Table A.1. Primary run controls": 49,
    "Table B.1. Capability effects and minimum detectable effects": 50,
}


def _load_base_builder():
    spec = importlib.util.spec_from_file_location("build_rewritten_fyp_report_v4_for_v5", BASE_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _all_text_paragraphs(doc: Document):
    """Body paragraphs plus every paragraph inside a table cell."""
    yield from doc.paragraphs
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                yield from cell.paragraphs


def _replace_paragraph_text(paragraph, new_text: str) -> None:
    if not paragraph.runs:
        paragraph.add_run(new_text)
        return
    paragraph.runs[0].text = new_text
    for run in paragraph.runs[1:]:
        run.text = ""


def _expand_ranges(text: str) -> str:
    def expand(match):
        low, high = int(match.group(1)), int(match.group(2))
        if high < low:
            return match.group(0)
        return "".join(f"[{n}]" for n in range(low, high + 1))

    return CITATION_RANGE.sub(expand, text)


def _citation_sequence(doc: Document) -> list[int]:
    sequence = []
    for paragraph in _all_text_paragraphs(doc):
        for match in CITATION.finditer(_expand_ranges(paragraph.text)):
            sequence.append(int(match.group(1)))
    return sequence


def _numeric_inventory(doc: Document) -> list[str]:
    """Numeric tokens outside citation brackets (same shape as the v3 gate).

    Lines containing a tab are dropped first: they are the cached Table of
    Contents / List of Figures / List of Tables page numbers, which are layout
    metadata verified by the pagination checker, not result values.
    """
    text = "\n".join(
        line for line in (p.text for p in _all_text_paragraphs(doc)) if "\t" not in line
    )
    text_without_citations = re.sub(r"\[(?:\d+)(?:\]-\[\d+)?\]", "", text)
    numbers = re.findall(
        r"(?<![A-Za-z])[-+−]?(?:\d{2,}(?:\.\d+)?|\d+\.\d+)%?",
        text_without_citations,
    )
    return [token.replace("-", "−") for token in numbers]


def _apply_language_rewrites(doc: Document) -> int:
    paragraphs = list(_all_text_paragraphs(doc))
    for old, new in LANGUAGE_REWRITES:
        hits = [paragraph for paragraph in paragraphs if old in paragraph.text]
        if len(hits) != 1:
            raise AssertionError(
                f"language rewrite matched {len(hits)} paragraphs, expected exactly 1: {old[:90]!r}"
            )
        _replace_paragraph_text(hits[0], hits[0].text.replace(old, new))
    return len(LANGUAGE_REWRITES)


def _verify_against_source(source: Document, doc: Document) -> None:
    source_numbers = _numeric_inventory(source)
    new_numbers = _numeric_inventory(doc)

    source_counts = {token: source_numbers.count(token) for token in set(source_numbers)}
    new_counts = {token: new_numbers.count(token) for token in set(new_numbers)}
    missing = {
        token: (count, new_counts.get(token, 0))
        for token, count in source_counts.items()
        if new_counts.get(token, 0) < count
    }
    invented = sorted(set(new_numbers) - set(source_counts))
    if missing or invented:
        raise AssertionError(
            f"Numerical inventory changed during language pass: missing={missing!r}, invented={invented!r}"
        )

    if _citation_sequence(source) != _citation_sequence(doc):
        raise AssertionError("Citation marker sequence changed during language pass")

    if len(list(_all_text_paragraphs(source))) != len(list(_all_text_paragraphs(doc))):
        raise AssertionError("Paragraph count changed during language pass")

    source_dashes = sum(p.text.count("\u2014") for p in _all_text_paragraphs(source))
    new_dashes = sum(p.text.count("\u2014") for p in _all_text_paragraphs(doc))
    if new_dashes != source_dashes:
        raise AssertionError(f"Em-dash count changed: {source_dashes} -> {new_dashes}")


def _assert_no_forbidden_shorthand(doc: Document) -> None:
    body = "\n".join(p.text.lower() for p in _all_text_paragraphs(doc))
    remaining = [term for term in FORBIDDEN_SHORTHAND if term in body]
    if remaining:
        raise AssertionError(f"Banned shorthand remains in the v5 body: {remaining}")


def _assert_bibliography_inherited(doc: Document) -> None:
    """Every printed reference must come from the v1 REFERENCES inheritance.

    The v5 line prints the bibliography it inherits from
    scripts/build_rewritten_fyp_report_v1.py. The v4 layer may legitimately
    drop entries that its Chapter 1 rewrite left uncited and may correct one
    entry's venue string, so this asserts membership by normalised prefix
    rather than byte equality, and that a full bibliography is printed.
    """
    spec = importlib.util.spec_from_file_location("build_rewritten_fyp_report_v1_refs_for_v5", V1_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {V1_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    references = getattr(module, "REFERENCES", None)
    if not isinstance(references, list) or not references:
        raise AssertionError("v1 builder exposes no REFERENCES list to inherit")

    def norm(text: str) -> str:
        return re.sub(r"[^0-9a-z]", "", text.lower())

    inherited_prefixes = {norm(str(reference))[:48] for reference in references}
    printed: list[str] = []
    in_references = False
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text == "References":
            in_references = True
            continue
        if in_references:
            if text.startswith("Appendix"):
                break
            if text:
                printed.append(text)

    if len(printed) < 50:
        raise AssertionError(f"Only {len(printed)} reference entries printed; inheritance looks broken")
    absent = [
        text[:60]
        for text in printed
        if norm(re.sub(r"^\[\d+\]\s*", "", text))[:48] not in inherited_prefixes
    ]
    if absent:
        raise AssertionError(f"Printed references that are not part of the inherited bibliography: {absent}")


def _apply_cached_page_numbers(doc: Document) -> int:
    applied = 0
    for paragraph in doc.paragraphs:
        if "\t" not in paragraph.text:
            continue
        title, page = paragraph.text.rsplit("\t", 1)
        if title in V5_PAGE_NUMBERS and page.isdigit() and int(page) != V5_PAGE_NUMBERS[title]:
            _replace_paragraph_text(paragraph, f"{title}\t{V5_PAGE_NUMBERS[title]}")
            applied += 1
    return applied


def build_report() -> Path:
    base = _load_base_builder()
    with tempfile.TemporaryDirectory(prefix="fyp_report_v5_") as tmp:
        tmp_docx = Path(tmp) / "v4_source.docx"
        original_out = base.OUT
        try:
            base.OUT = tmp_docx
            base.build_report()
        finally:
            base.OUT = original_out

        source = Document(tmp_docx)
        doc = Document(tmp_docx)

        applied = _apply_language_rewrites(doc)
        _verify_against_source(source, doc)
        _assert_no_forbidden_shorthand(doc)
        _assert_bibliography_inherited(doc)
        page_fixes = _apply_cached_page_numbers(doc)

        print(f"    language rewrites applied: {applied}")
        print(f"    cached page numbers overridden: {page_fixes}")

        doc.core_properties.subject = "Plain-academic-English FYP report rewrite v5 (language cleanup)"
        doc.core_properties.comments = (
            "Language-cleanup rewrite derived from the verified v4 line. Compressed statistical "
            "shorthand replaced with conventional academic English; numbers, citations, tables, "
            "figures, structure, and conclusions preserved and build-verified."
        )
        OUT.parent.mkdir(parents=True, exist_ok=True)
        doc.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build_report())
