#!/usr/bin/env python3
"""Pass 10: the 13 live findings from the RESUMED sweep (ch8-ch9 verifies completed).

Two are factual slips rather than over-claims: "human annotators" (plural) where
the study has one annotator and lists a second as future work (T30b), and a
"rather than" that converts a null into an established absence.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import compress_ch6 as C

EDITS = [
    ("R2-1-envelope", 1,
     "This regime corresponds to the practical envelope of contemporary on-device deployment.",
     "This regime was chosen because models of this size load and evaluate on a single 32 GB GPU, which is the constraint the study actually operated under."),

    ("R2-2-bitwidth", 1,
     "The effect is also not bit-width-graded, in that a 128-token apparent increase in the smallest model does not replicate at the reference budget (Section 6.12).",
     "No bit-width gradient is detected across fp16, INT8 and NF4. Separately, the 128-token apparent increase in the smallest model does not replicate at the reference budget (Section 6.12)."),

    ("R2-3-orbench", 1,
     "The existence of OR-Bench nevertheless establishes that over-refusal is now treated as a first-class safety axis rather than a footnote.",
     "OR-Bench nevertheless shows that over-refusal is now the subject of dedicated, large-scale benchmarks in its own right."),

    ("R2-4-cpu-path", 1,
     "However, the CPU path is not used for quantized runs, because BitsAndBytes requires a CUDA device.",
     "However, the CPU path is not used for quantized runs: in the pinned bitsandbytes version used here, four-bit loading requires a CUDA device."),

    ("R2-5-qwen-template", 1,
     "Qwen3.x-family tokenizers have chat templates that default to enabling a multi-step thinking block, so without the flag they would silently consume the entire max_new_tokens budget",
     "The Qwen3 tokenizers used here ship chat templates that enable a thinking block by default, so with the flag unset generation can spend much or all of the max_new_tokens budget"),

    ("R2-6-cpu-four-bit", 1,
     "This prevents accidental CPU execution of a four-bit path that BitsAndBytes does not support.",
     "This prevents accidental CPU execution of the four-bit path, which the pinned CUDA bitsandbytes build does not provide."),

    # Factual: the study has ONE annotator; a second is future work (T30b).
    ("R2-7-annotators", 1,
     "and the scorer selected here is the one that agrees best with human annotators.",
     "and of the two scorers compared against human labels, the classifier agreed substantially better with the single annotator on the audited subset."),

    # "rather than" converts the harmful-compliance null into an established absence
    ("R2-8-deployment", 1,
     "should expect a lower capability level than they signed off on (−9.0 pp MMLU) rather than a higher harmful-compliance rate.",
     "should expect a materially lower capability level than they signed off on (−9.0 pp on this MMLU subset). No increase in harmful compliance was detected, though the study's detection floor bounds that null (Section 6.4.1)."),

    ("R2-9-artefact", 1,
     "The fix for future runs is straightforward, namely adding the chat turn-end token to the stop criteria, and the artefact does not affect any result reported here.",
     "A likely fix for future runs is to add the chat turn-end token to the stop criteria. The artefact is not expected to affect the results reported here, since the primary classifier scores the behaviour rather than the length, but that expectation was not tested."),

    ("R2-10-matched-by-construction", 1,
     "It inflates response length, but it is produced identically by both pair members, so its effect is matched across the pair by construction.",
     "It inflates response length. The decoding configuration that produces it is identical for both pair members, but the per-member incidence of the artefact was not measured, so the effect is expected rather than shown to cancel in the paired delta."),

    ("R2-11-gguf", 1,
     "GGUF is the deployment-relevant format named in the motivation (Chapter 1), so it is the priority method.",
     "GGUF is the deployment-oriented format named in the motivation (Chapter 1), so it is prioritised here."),

    ("R2-12-refute", 1,
     "The data therefore refute the optimistic claim that quantization reliably improves safety.",
     "These data give no support for the view that quantization reliably improves safety."),

    ("R2-13-both-judges", 1,
     "Both judges agree that at the reference budget NF4 does not raise harmful compliance in any pair.",
     "Under both judges, no pair shows a statistically significant increase in harmful compliance at the reference budget."),
]

ALLOWED_REMOVALS = {}

if __name__ == "__main__":
    C.run(EDITS, ALLOWED_REMOVALS, label="pass 10 (resumed-sweep round 2)")
