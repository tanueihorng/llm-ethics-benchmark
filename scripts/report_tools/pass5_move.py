#!/usr/bin/env python3
"""Pass 5: consolidate 6.8.1-6.8.5.

Five subsections walk the same three pairs a second time, one dimension at a
time, over numbers Tables 6.1 and 6.2 already carry and that 6.5-6.7 have
already discussed per pair. The findings are kept in one consolidated passage;
the dimension-by-dimension walkthrough moves.
"""
import move_to_appendix as M

BLOCKS = [
    ("H3(\"6.8.1 Baseline safety profiles differ substantially across families",
     "H3(\"6.8.6 Cross-Family Extension: Mistral-7B and Phi-4-mini",
     "I.13 Cross-family comparison, dimension by dimension",
     '"Four findings hold across the original three pairs. First, baseline safety profiles differ substantially between families even at matched factual capability, so calibration is a property of the release rather than of model size. Second, on harmful compliance exactly one ΔASR is individually significant and it is a decrease, and no ΔASR contrast survives the Benjamini-Hochberg correction over the twenty-test NF4 family; the Qwen3-1.7B increase that was the study\'s original headline at the 128-token budget does not reproduce at the reference budget. Third, over-refusal shows no significant increase in any family. Fourth, every one of the three pairs carries at least one significant capability loss, which is what drives each interpretation label. Appendix I.13 works through the four dimensions pair by pair."'),
]

ALLOWED_REMOVALS = {
    "8.1": "section number 6.8.1 — the subsection is renumbered I.13 in the appendix; "
           "no inbound cross-reference to 6.8.1-6.8.5 exists anywhere in the document",
}

if __name__ == "__main__":
    M.run(BLOCKS, ALLOWED_REMOVALS, label="pass 5 (consolidate 6.8.1-6.8.5)")
