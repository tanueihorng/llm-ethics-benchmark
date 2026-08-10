#!/usr/bin/env python3
"""Pass 4: relocate the remaining supporting detail from Chapter 6.

Same rule as pass 3 — each section keeps its finding, the evidence trail moves.
Table 6.5 stays in 6.12, because it IS the 128-vs-512 comparison the section
exists to make.
"""
import move_to_appendix as M

BLOCKS = [
    ("H3(\"6.1.1 Scorer revision history",
     "H2(\"6.2 Within-Family Scale Analysis",
     "I.8 Scorer revision history",
     '"HarmBench ASR was originally scored by a deterministic refusal regex and is now scored by the benchmark\'s own fine-tuned classifier; over-refusal remains regex-scored as the pre-specified scorer of record, cross-checked by an independent judge. Appendix I.8 records the revision history in full, and Appendix H the formal correction record."'),

    ("H3(\"6.3.1 Capability robustness check: the ARC-Challenge second benchmark",
     "H2(\"6.4 Statistical Caveats",
     "I.9 Capability robustness check: the ARC-Challenge second benchmark",
     '"The capability finding does not rest on MMLU alone. ARC-Challenge was added as a structurally different second anchor, on a much larger item set, and it agrees: capability declines on both benchmarks, and two of the three multiplicity survivors are capability contrasts, one on each. Appendix I.9 gives the per-pair ARC comparison and the parser-sensitivity check behind it."'),

    ("PJ(\"The flips are also close to symmetric in aggregate",
     "PJ(\"Capability versus alignment: the tie to the central question",
     "I.10 Refusal-margin geometry: flip symmetry and per-model margin shifts",
     '"The flips are close to symmetric in aggregate, which is what a boundary-instability reading predicts and a targeted-erosion reading does not, and the direction of the margin shift is model-specific and modest. Appendix I.10 gives the flip counts, the per-family breakdown and the paired margin shifts."'),

    ("PJ(\"The 128-token Llama-3.2-3B INT8 signal was therefore a budget-sensitive artefact",
     "PJ(\"The precision point sharpens the study's central message",
     "I.11 INT8: budget sensitivity, the mechanism reading and the proxy at INT8",
     '"The one INT8 safety signal visible at the 128-token budget does not survive at the reference budget, so it is read as budget-sensitive rather than as a precision effect. The secondary regex continues to over-count harmful compliance at INT8 exactly as it does elsewhere. Appendix I.11 sets out the budget comparison, its tie to the mechanism reading of Section 6.10, and the proxy behaviour."'),

    ("PJ(\"The initial run generated at most max_new_tokens = 128",
     "buildTable(",
     "I.12 Generation-length rerun: truncation statistics and decoding artefacts",
     '"The reference budget changes the generations materially rather than cosmetically: a large share of the 128-token responses were provably cut off mid-answer, the truncation is family-heterogeneous, and absolute ASR is correspondingly higher at 512 tokens under both judges. A separate decoding-configuration artefact affecting a minority of paired generations was found while inspecting responses and is disclosed there rather than smoothed over. Appendix I.12 gives the prefix statistics, the per-family disclosure and the artefact analysis."'),
]

ALLOWED_REMOVALS = {}

if __name__ == "__main__":
    M.run(BLOCKS, ALLOWED_REMOVALS, label="pass 4 (relocate remaining detail)")
