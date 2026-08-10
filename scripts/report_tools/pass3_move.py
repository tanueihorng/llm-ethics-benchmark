#!/usr/bin/env python3
"""Pass 3: relocate supporting detail from Chapter 6 into Appendix I.

Each Result keeps its headline paragraph in 6.9, so every existing
"Section 6.9, Result N" cross-reference still resolves. What moves is the
evidence trail underneath: per-prompt agreement figures, the annotation
caveats, the residual-caveat discussion and the two robustness arms.
"""
import move_to_appendix as M

BLOCKS = [
    ("H3(\"6.5.1 Multi-seed decoding sensitivity",
     "H3(\"6.5.2 Per-category harmful-compliance profile",
     "I.1 Multi-seed decoding sensitivity, all five pairs",
     '"Robustness to stochastic decoding was checked by re-running all five pairs on HarmBench at temperature 0.7 with top-p 0.8 across five independent seeds, both members of a pair sharing each seed. No pair is robustly positive: every per-seed |ΔASR| is at most 0.055 and every seed-mean |ΔASR| at most 0.029, and the greedy point estimate sits inside the seed range for the three original pairs. The stochastic arm therefore corroborates the RQ1 null across the full matrix. Per-seed deltas, the four individually significant seeds, and the two cross-family pairs whose greedy deltas fall just outside their narrow seed ranges are set out in Appendix I.1."'),

    ("H3(\"6.5.2 Per-category harmful-compliance profile",
     "H2(\"6.6 Observations: Qwen3-4B Pair",
     "I.2 Per-category harmful-compliance profile, Qwen3-1.7B",
     '"Because the aggregate Qwen3-1.7B ΔASR is exactly zero at the reference budget, there is no net increase to decompose by harm type. The judge ASR was nonetheless broken down by HarmBench semantic category to confirm the null is not masking a large offsetting swing between categories. The per-category deltas run in both directions and roughly cancel, and no single category drives a systematic increase. Counts per category are small, so those rates are descriptive rather than individually tested. Appendix I.2 gives the breakdown."'),

    ("PJ(\"The two judges agree strongly at the per-prompt level",
     "PJ(\"Result 5: the classifier tracks human judgment",
     "I.3 Second-judge cross-check: per-prompt agreement and per-pair deltas",
     '"The two judges agree strongly at the per-prompt level, and the per-pair ΔASR direction is reproduced at the reference budget, including for the Qwen3-1.7B effect that the 128-token run had reported, which is null under both judges at 512 tokens. Appendix I.3 gives the per-model agreement coefficients and the per-pair gpt-4o deltas."'),

    ("PJ(\"One input-mismatch caveat applies",
     "PJ(\"Threats and scope",
     "I.4 Human grounding of the HarmBench scorers: protocol, caveat and agreement",
     '"The classifier agrees with the blinded human annotator substantially better than the demoted regex does. This is the human-grounded form of the chapter\'s central result: refusal-counting over-states harmful compliance not only relative to another model, but relative to a person. Appendix I.4 records the annotation protocol, an input-mismatch caveat, and both agreement coefficients."'),

    ("PJ(\"Threats and scope",
     "PJ(\"Result 6: the over-refusal axis is scorer-sensitive",
     "I.5 Scorer threats, residual caveats and the open-weight third judge",
     '"Threats and scope. The judge is itself a model and can err. It is adopted because it is the field-standard reference classifier specified by the HarmBench paper [24], run at full precision with deterministic greedy decoding and zero parse failures on this dataset. The single-judge threat is addressed on two fronts: a second judge from a different provider, and an open-weight third judge, Llama Guard [51], whose per-pair ΔASR is non-significant everywhere. Appendix I.5 sets out the classifier-template comparability bound, the residual caveats attaching to each judge, and the construct caveat that Llama Guard answers a looser question than the behaviour-specific classifier."'),

    ("PJ(\"Strict over-refusal counts full refusals only",
     "buildTable(",
     "I.6 Over-refusal scorer sensitivity: strict and broad mappings",
     '"Appendix I.6 sets out the strict and broad refusal mappings, and the per-pair comparison behind Table 6.4."'),

    ("PJ(\"The human grounding for this axis was completed",
     "H2(\"6.10 Mechanism Probe",
     "I.7 Human grounding of the over-refusal scorers, and the pre-registration deviation",
     '"The human grounding for this axis was completed as well, mirroring Result 5, on a pre-registered two-hundred-item gold set. The judge aligns substantially better with the blinded annotator than the regex does, whose recall on the annotator\'s full refusals is very low, so the single FDR-surviving over-refusal contrast is carried as most plausibly a regex measurement artefact. One protocol deviation is disclosed and the confirmatory pre-registration badge is withdrawn accordingly. Appendix I.7 records the outcome matrix, the recall figures, and the deviation in full."'),
]

ALLOWED_REMOVALS = {}

if __name__ == "__main__":
    M.run(BLOCKS, ALLOWED_REMOVALS, label="pass 3 (relocate detail to Appendix I)")
