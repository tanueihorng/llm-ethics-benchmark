#!/usr/bin/env python3
"""Pass 9: the 22 live findings from the exhaustive claim-enumeration sweep.

The sweep produced 181 raw candidates; adversarial refutation kept 27; five of
those were already fixed earlier the same session. These are the 22 that were
still live in the builder. Fix wording is mine, not the sweep's, and each was
re-checked against the primary artifact before being applied.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import compress_ch6 as C

EDITS = [
    # 1 + its Chapter 1 twin: a single-regex, single-benchmark result generalised
    ("S1-abstract", 1,
     "Refusal-counting therefore overstates quantization harm.",
     "Refusal-counting therefore overstated the apparent harm from quantization in this study's HarmBench contrasts."),
    ("S1b-ch1", 1,
     "Refusal-counting therefore overstates quantization harm, and that is a cautionary lesson for safety evaluation generally.",
     "Refusal-counting therefore overstated the apparent harm from quantization in this study's HarmBench contrasts, which is a caution worth carrying into other safety evaluations."),

    # 2: [1] is the Qwen3 technical report and does not cover Qwen2.5
    ("S2-qwen25", 1,
     "Models such as Qwen 2.5 and 3.x [1], Llama 3.2 [2],",
     "Models such as Qwen3 [1], Llama 3 [2],"),

    # 3: first clause asserts, second clause hedges — match them
    ("S3-calibration", 1,
     "Quantization degrades the confidence calibration of model predictions even where accuracy is preserved [11],",
     "Quantization can degrade the confidence calibration of model predictions even where accuracy is preserved [11],"),

    # 4, 5, 6, 7: adoption/prevalence framing
    ("S4-dominant-families", 1,
     "The other dominant post-training families, GPTQ [6] and AWQ [5], are excluded",
     "The other post-training families considered, GPTQ [6] and AWQ [5], are excluded"),
    ("S5-mmlu", 1,
     "MMLU [27], the Massive Multitask Language Understanding benchmark, has become a standard general-capability anchor in safety studies.",
     "MMLU [27], the Massive Multitask Language Understanding benchmark, measures multiple-choice accuracy across academic and professional subjects, and is used here as the general-capability anchor."),
    ("S6-practical-standards", 1,
     "Two benchmarks have emerged as the practical standards for measuring opposite sides of this trade-off.",
     "Two benchmarks measure opposite sides of this trade-off, and are the pair used here."),
    ("S7-most-studies", 1,
     "Most quantization studies focus on the seven-billion-to-thirteen-billion parameter range and above.",
     "The quantization studies surveyed in Table 2.1 concentrate on models of seven billion parameters and above."),

    # 8: a universal about the literature, stated from a table that lists 10 studies
    ("S8-tend-not-to", 1,
     "The studies that measure safety carefully tend not to hold a capability anchor on the same models, and the studies that measure capability carefully tend not to measure safety at all.",
     "Across the rows of Table 2.1, the studies that measure safety carefully mostly do not report a capability anchor on the same models, and those that measure capability carefully mostly do not report safety."),

    # 9: an effect on discourse that nothing measures
    ("S9-shifted-conversation", 1,
     "and they have meaningfully shifted the conversation around what constitutes an acceptable safety profile in publicly released models.",
     "and each states explicit criteria for the safety profile it expects of a publicly released model."),

    # 10: an unmeasured claim about where alignment training is strongest
    ("S10-alignment-strongest", 1,
     "Second, they focus on mid-to-large models in the seven-billion-to-seventy-billion parameter range, where alignment training tends to be strongest.",
     "Second, they focus on mid-to-large models in the seven-billion-to-seventy-billion parameter range, leaving the compact regime this study targets under-covered."),

    # 11, 12: XSTest. The protocol sentence is also factually wrong — XSTest's
    # PRIMARY evaluation was manual annotation by three of its authors; the
    # GPT-4 three-way classifier was one of two cheaper alternatives it reported.
    ("S11-xstest-canonical", 1,
     "XSTest is the canonical over-refusal benchmark.",
     "XSTest is the over-refusal test suite used in this study."),
    ("S12-xstest-protocol", 1,
     "XSTest's own protocol relies on LLM-based three-way labelling (full compliance / partial refusal / full refusal) rather than a released fine-tuned classifier.",
     "XSTest's own primary evaluation used manual annotation against a three-way scheme (full compliance / partial refusal / full refusal), with automated string-matching and a GPT-4 classifier reported as cheaper alternatives; no fine-tuned scorer was released with the suite."),

    # 13: causal attribution to a factor the design never varied
    ("S13-calibration-cause", 1,
     "First, baseline safety profiles differ substantially between families even at matched factual capability, so calibration is a property of the release rather than of model size.",
     "First, baseline safety profiles differ substantially between families even at matched factual capability. This study does not vary training or alignment recipe, so it cannot identify which release-level factor produces that difference."),

    # 14, 15: vague appeal to "the broader literature"; name the source
    ("S14-broader-literature", 1,
     "This points the same direction as the broader literature on quantization sensitivity declining with model size.",
     "This points the same direction as Badshah and Sajjad's preprint report [29] that larger models are more resilient to four-bit precision reduction."),
    ("S15-scaling-evidence", 1,
     "The present result extends that picture downward into the compact-deployment regime, where the scaling evidence had suggested alignment would be most fragile.",
     "The present result extends that picture downward into the compact-deployment regime, below the seven-billion-and-above range Hong et al. evaluate."),

    # 16: "field-standard" is an adoption claim; "specified by the paper" is the fact
    # Two instances: Chapter 6 and the Appendix I.5 pointer that relocation created.
    ("S16-field-standard", 2,
     "It is adopted because it is the field-standard reference classifier specified by the HarmBench paper [24],",
     "It is adopted because it is the reference classifier specified by the HarmBench paper [24],"),

    # 17: mechanism asserted as established
    ("S17-mistral-preamble", 1,
     "The reason is that Mistral-7B frequently opens with a refusal-shaped preamble and then complies, so its refuse-versus-comply decision is resolved later than the first token.",
     "A plausible explanation, not tested here, is that Mistral-7B often opens with a refusal-shaped preamble and then complies, which would move its refuse-versus-comply decision past the first token."),

    # 18: attribute the thinking-mode claim to the source that makes it
    ("S18-thinking-mode", 1,
     "and it is through that mode of generation that Qwen3 delivers much of its capability and some of its safety behaviour.",
     "and the Qwen3 technical report [1] attributes a substantial part of the models' reasoning performance to that mode of generation."),

    # 19, 20: "which is where prior work reports effects" universalises; scope it
    ("S19-prior-work", 1,
     "It does not establish that the null would survive optimised adversarial attack, which is where prior quantization-safety work reports effects.",
     "It does not establish that the null would survive optimised adversarial attack, a setting in which some prior quantization-safety work reports effects [35]."),
    ("S20-attack-attribution", 1,
     "and it would directly engage the attack-dependent effects reported by HarmLevelBench [35] and Egashira et al. [12].",
     "and it would directly engage the attack-dependent effects reported by HarmLevelBench [35]. Egashira et al. [12] describe a different threat, in which the quantization step itself is exploited."),

    # 21, 22: "capability essentially preserved" / "preserves capability" for a pair
    # whose ARC contrast IS significant uncorrected (−0.016, p = 0.042)
    ("S21-4b-preserved", 1,
     "The Qwen3-4B pair is directional alignment_degradation (ΔASR +0.040, not significant) with capability essentially preserved.",
     "The Qwen3-4B pair is directional alignment_degradation (ΔASR +0.040, not significant); MMLU is flat, while ARC falls significantly before correction (−0.016, p = 0.042) and does not survive it."),
    ("S22-4b-preserves", 1,
     "By contrast, the 4B pair preserves capability and shows only a directional, non-significant safety nudge (RQ4).",
     "By contrast, no capability contrast for the 4B pair survives correction, and its safety movement is directional and non-significant (RQ4)."),
]

ALLOWED_REMOVALS = {}

if __name__ == "__main__":
    C.run(EDITS, ALLOWED_REMOVALS, label="pass 9 (sweep: 22 live findings)")
