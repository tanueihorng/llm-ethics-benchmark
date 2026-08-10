#!/usr/bin/env python3
"""Pass 8: third-round re-audit findings — prevalence/adoption paraphrases.

All six confirmed against the file. Five are the SAME defect class as fixes
already applied in Chapter 1, surviving as paraphrases elsewhere ("commonly
deployed", "routinely integrated", "dominate current practice", "particularly
prevalent"). The Abstract instance directly contradicted the disclaimer this
session added to Chapter 1.
"""
import compress_ch6 as C

EDITS = [
    # Abstract — contradicted Ch1's own "no prevalence figure is claimed" disclaimer
    ("R3-abstract", 1,
     "Compact instruction-tuned language models are commonly deployed quantized, yet their safety is typically validated on the full-precision release.",
     "Quantization is what brings compact instruction-tuned language models within reach of constrained hardware, but a safety evaluation carried out on the full-precision checkpoint does not automatically transfer to the quantized artefact derived from it."),

    # §2.3 — model-release papers establish capability and availability, not integration
    ("R3-integration", 1,
     "These models are routinely integrated into lightweight agent frameworks and consumer applications, where their compact size enables genuinely local execution.",
     "Their compact size is what makes genuinely local execution possible on such hardware. How widely they are in fact integrated into deployed applications is not measured here and is not claimed."),

    # §2.4 — adoption claim
    ("R3-dominate", 1,
     "Several PTQ families dominate current practice.",
     "Several PTQ families recur throughout the compression literature."),

    # §2.4 — QLoRA introduces NF4; it does not measure ecosystem adoption.
    # Justify the choice on a DESIGN ground, which the study can actually defend.
    ("R3-nf4-prevalent", 1,
     "Among PTQ approaches, the NF4 quantization scheme introduced as part of QLoRA [32] has become particularly prevalent in the open-source ecosystem.",
     "Among PTQ approaches, the NF4 quantization scheme introduced as part of QLoRA [32] is the one this study evaluates. It is applied as a load-time option rather than as a separate conversion step, which is the property the matched-pair design depends on, since it allows both members of a pair to come from one unmodified checkpoint."),

    # §2.6 Gap 5 — "commonly" and "never" are universals Table 2.1 cannot support:
    # the table has no scorer-validation column at all.
    ("R3-gap5", 1,
     "Gap 5. Unvalidated scoring. Safety results are commonly reported under a single scorer whose agreement with human judgment is never measured, even though the multilingual evidence shows automatic metrics can understate a quantization effect by an order of magnitude. A safety conclusion is only as trustworthy as the instrument that produced it.",
     "Gap 5. Unvalidated scoring. Table 2.1 records no scorer-validation step for any study listed in it, and this review did not establish whether one was performed in each case. What the multilingual evidence does show is that an automatic metric can understate a quantization effect by roughly an order of magnitude in at least one language [13]. A safety conclusion is only as trustworthy as the instrument that produced it, which is why the scorer is validated here rather than assumed."),

    # §2.1 — weakly attributed, and already stated correctly in the next paragraph
    ("R3-predominantly", 1,
     "First, they predominantly evaluate full-precision checkpoints. Quantized variants are either omitted or treated as a separate, secondary evaluation.",
     "First, quantized variants are either omitted or evaluated separately rather than as a controlled contrast against their own full-precision counterparts."),
]

ALLOWED_REMOVALS = {}

if __name__ == "__main__":
    C.run(EDITS, ALLOWED_REMOVALS, label="pass 8 (round-3 prevalence findings)")
