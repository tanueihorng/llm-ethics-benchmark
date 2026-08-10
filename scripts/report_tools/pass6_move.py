#!/usr/bin/env python3
"""Pass 6: the last two relocations.

6.4's caveat list runs to five long paragraphs, two of which (the deterministic-
decoding variance discussion and the MMLU option-parser construct caveat) are
method-level detail; and 6.11/6.12 still carry setup paragraphs that describe
how a run was configured rather than what it found.
"""
import move_to_appendix as M

BLOCKS = [
    ("PJ(\"Second, decoding is deterministic at temperature 0.0",
     "PJ(\"Fourth, HarmBench ASR is scored by the official HarmBench classifier as primary",
     "I.14 Statistical caveats: decoding variance, MMLU subset composition and answer parsing",
     '"Three of the five caveats are method-level and are set out in Appendix I.14: deterministic decoding means the bootstrap interval reflects prompt-sampling variance only, which is why the multi-seed arm exists; the MMLU subset pools three hundred questions across six subjects with uneven counts, so subject-level rates are descriptive; and the option-letter parser falls back to a lenient scan, which a strict-parser sensitivity check quantifies."'),

    ("PJ(\"All five INT8 members were run on the NTU TC1 cluster at the 512-token reference budget",
     "PJ(\"Capability: a clean cliff at four-bit",
     "I.15 INT8 arm: run configuration and analysis separation",
     '"The INT8 arm was run under the identical protocol and is analysed separately from the base-versus-4-bit pipeline, so the evaluated pairs are unchanged; Appendix I.15 records the configuration."'),
]

ALLOWED_REMOVALS = {}

if __name__ == "__main__":
    M.run(BLOCKS, ALLOWED_REMOVALS, label="pass 6 (final relocations)")
