#!/usr/bin/env python3
"""Pass 11: the 5 confirmed Appendix I findings, plus the critic's Table 2.1 point.

The two sampling-parameter findings asked for citations [1] and [2]. Both were
DECLINED as written: the values are model-card material and the reference list
carries the Qwen3 technical report and the Llama 3 herd paper, neither of which
specifies default sampling parameters. Citing them would have replaced a vague
attribution with a false one. Naming the actual source type is the honest fix.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import compress_ch6 as C

EDITS = [
    # i=7 — a property of the method class asserted from five pairs
    ("G7-conceptual", 1,
     "The conceptual conclusion is that quantization's effect on safety is neither bit-width-graded nor budget-robust.",
     "The conceptual conclusion, across the five pairs tested here, is that quantization's observed effect on safety is neither bit-width-graded nor budget-robust."),

    # i=14 — "correctly attributes" upgrades a labelling rule firing as designed
    # into a verified causal attribution. No mediation or ablation test exists,
    # and the appendix's own mechanism section declines a causal reading.
    ("G14-correctly", 1,
     "and the one significant ASR delta, on Llama-3.2-3B, is a decrease that the label scheme correctly attributes to capability loss.",
     "and the one significant ASR delta, on Llama-3.2-3B, is a decrease that the label scheme diagnoses as capability-driven rather than as a genuine alignment gain."),

    # i=19 — the thin-margin analysis was computed on fp16-vs-NF4 flips at a fixed
    # budget, not on the 128-vs-512 INT8 mismatches this sentence explains
    ("G19-attributable", 1,
     "the instability is attributable to cross-run greedy divergence on thin-margin prompts rather than to truncated responses.",
     "the instability is more plausibly attributable to cross-run greedy divergence than to truncated responses, a reading consistent with the thin-margin boundary mechanism of Section 6.10, though that mechanism was not measured for this arm."),

    # i=27, i=28 — name the actual source rather than attaching a reference that
    # does not carry the value
    ("G27-qwen-sampling", 1,
     "That setting is temperature 0.7 with top-p 0.8, Qwen3's published non-thinking recommendation.",
     "That setting is temperature 0.7 with top-p 0.8, the non-thinking sampling values published on the Qwen3 model card."),
    ("G28-llama-sampling", 1,
     "although Meta's Llama guidance is T = 0.6 / top-p 0.9.",
     "although the Llama 3.2 model card's default generation configuration specifies T = 0.6 / top-p 0.9."),

    # Completeness critic: the novelty claim is carried by the table's visual
    # contrast (ten rows of "not reported" against one all-yes row), not by any
    # sentence — so no sentence-level audit can see it. State the bound.
    ("G-critic-table21", 1,
     "\\\"Not reported\\\" means the property is not stated in the source consulted, and should not be read as evidence that the authors did not consider it.",
     "\\\"Not reported\\\" means the property is not stated in the source consulted, and should not be read as evidence that the authors did not consider it. The final row records this study's own design on the same dimensions for comparison; the table is a positioning aid on these specific axes and is not offered as a novelty or priority claim."),
]

ALLOWED_REMOVALS = {}

if __name__ == "__main__":
    C.run(EDITS, ALLOWED_REMOVALS, label="pass 11 (gap-close + critic)")
