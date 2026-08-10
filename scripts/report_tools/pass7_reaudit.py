#!/usr/bin/env python3
"""Pass 7: the independent re-audit's findings (2026-07-26, second Codex pass).

Every one reproduces under the WIDENED epistemic_hedge patterns. Six of the nine
are the same defect class as an already-fixed finding, surviving in a location
neither audit had examined — the D42 sweep failure again: fix the instance, miss
the surface.
"""
import compress_ch6 as C

EDITS = [
    # --- prevalence claims (C11/C12 class) --------------------------------
    ("RA1-ch1-prevalence", 1,
     "However, in practice such models are rarely deployed in full precision. Memory budgets on consumer GPUs, mobile devices, and laptop NPUs make sixteen-bit or higher precision impractical for routine use.",
     "However, full-precision inference is not always practical at this scale. Memory budgets on consumer GPUs, mobile devices, and laptop NPUs make sixteen-bit or higher precision costly to sustain, which is what makes quantization useful for resource-constrained deployment rather than merely convenient."),

    ("RA2-ch2-prevalence", 1,
     'PJ("The deployment reality of these models is that they are almost never used in full precision. Memory and latency constraints on consumer hardware drive routine use of four-bit quantization, often via on-the-fly BitsAndBytes loading or pre-quantized GGUF checkpoints. Therefore, safety claims attached to the unquantized release model do not, in general, transfer to the model that end users actually encounter."),',
     'PJ("Quantization matters for these models because it is what brings them within reach of constrained hardware, whether by on-the-fly BitsAndBytes loading or by pre-quantized GGUF checkpoints. How widely each path is used in practice is not measured here, and no prevalence figure is claimed for it. What matters for the research question is narrower: a safety claim established on the unquantized release does not automatically transfer to the quantized artefact derived from it, and that transfer is what this study tests."),'),

    # --- unearned superlatives (C23 class) --------------------------------
    ("RA3-superlative-isolation", 1,
     "This is the strongest practical isolation available under the recorded matched-pair configuration.",
     "The matched-pair design therefore controls checkpoint provenance within the recorded environment."),

    ("RA4-superlative-harmbench", 1,
     "because it is the most widely adopted harmful-behaviour collection in contemporary safety research and provides a stable definition of attack success",
     "because it is a standardized harmful-behaviour benchmark with a stable, published definition of attack success"),

    # --- INT8: equivalence, not "free" (C14 class) ------------------------
    ("RA5-int8-free", 1,
     "Eight-bit is therefore effectively a free precision point on capability at this resolution, although non-significance bounds a zero effect rather than proving one.",
     "All ten INT8 capability contrasts meet the post-hoc TOST equivalence criterion within ±0.05, so eight-bit capability is equivalent to full precision within that margin rather than merely not significantly different from it."),

    ("RA6-int8-no-delta", 1,
     "while no capability delta appears at the INT8 precision point (RQ3)",
     "while at the INT8 precision point all ten capability contrasts are statistically equivalent to full precision within ±0.05 (RQ3)"),

    # --- HarmBench protocol (C20 class), Chapter 8 instance ---------------
    ("RA7-harmbench-protocol", 1,
     "HarmBench was evaluated in its standard configuration, which presents the curated harmful behaviours directly, with no adversarial attack or jailbreak augmentation applied",
     "HarmBench was evaluated in a direct-request configuration with no adversarial transformation, presenting the curated harmful behaviours as written, with no attack or jailbreak augmentation applied",),

    # --- Appendix I: stale interpretation ---------------------------------
    ("RA8-equivalent-capability", 1,
     "a 15.5 pp difference in favour of Llama-3.2-3B on the safety axis at equivalent capability",
     "a 15.5 pp difference in favour of Llama-3.2-3B on the safety axis at similar observed MMLU accuracy"),

    ("RA9-causal-rlhf", 1,
     "This indicates that two compact models at the same factual capability have different safety calibrations, attributable to differences in instruction-tuning methodology, RLHF recipe, and safety-alignment approach. For these models, family and recipe are a comparable-to-larger safety determinant than quantization.",
     "Two compact models at similar factual capability therefore have different safety calibrations, which may reflect unmeasured differences in instruction-tuning methodology, RLHF recipe, and safety-alignment approach; none of those factors is measured here. For these models the between-family gap is larger than any quantization delta measured in the study."),

    ("RA10-three-to-five", 1,
     "exactly one of the three ΔASR values is statistically significant, and it runs in the safety-improving direction",
     "exactly one of the five ΔASR values is statistically significant, and it runs in the safety-improving direction"),

    # --- typo -------------------------------------------------------------
    ("RA11-typo", 1,
     "The committed aggregate carries carries counts and rates only",
     "The committed aggregate carries counts and rates only"),
]

ALLOWED_REMOVALS = {}

if __name__ == "__main__":
    C.run(EDITS, ALLOWED_REMOVALS, label="pass 7 (re-audit findings)")
