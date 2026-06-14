# Progress email draft — Dr. Zhang (2026-06-13)

> Follow-up to the 2026-05-23 email (which promised "another update once I have
> completed benchmark results"). This one delivers the results, including the
> completed multi-seed robustness check (T18).
> Numbers verified against `results/analysis/judge_agreement.json` and
> `results/analysis/sensitivity_multiseed.json` on 2026-06-14.
> Plain-text body below the subject line.

**To:** jiehuang.zhang@ntu.edu.sg
**Subject:** CCDS25-1136 progress update — experiments complete, results in hand

---

Dear Dr. Zhang,

Following up on my May update: the experiments are now complete and I have results in hand.

All runs finished on the TC1 GPU cluster. The smoke run, the full 6-model × 3-benchmark matrix, and the pairwise delta analysis all ran cleanly, after resolving the Hugging Face gating and CUDA/PyTorch issues I had flagged. The study compares three matched baseline-vs-4-bit (NF4) model pairs — Qwen3-1.7B, Qwen3-4B, and Llama-3.2-3B — across three axes: harmful compliance (HarmBench), over-refusal on benign prompts (XSTest), and general capability (MMLU). Both members of each pair load from identical weights, so quantization is the only variable.

The headline is a clean three-way story that I believe is the project's strongest contribution — 4-bit quantization did **not** affect the models uniformly:

- **Qwen3-1.7B (smallest)** degraded on *both* axes: harmful compliance rose significantly (+5.5 pp, McNemar p = 0.027) *and* capability fell significantly (−8.7 pp).
- **Llama-3.2-3B** kept its safety calibration but lost significant capability (−4.3 pp).
- **Qwen3-4B** was the only pair to come through essentially intact on both axes (its safety shift was directional but not statistically significant).

So quantization's harm depends on model size and family, and surfaces on *different* axes for different models. The capability anchor (MMLU) in my framework is what lets me separate a genuine safety regression (Qwen 1.7B) from a capability-driven artifact (Llama) — which is the core question the project set out to answer. Across all three pairs, 4-bit quantization never *reduced* harmful compliance.

One methodological point I am pleased with: in my May email I described HarmBench scoring as deterministic regex-based refusal parsing. On review, that scorer over-counted harmful compliance, so I validated it against HarmBench's own 13B classifier model (also run on TC1). The judge corrected the result and relocated the one statistically significant safety regression from the 4B model to the 1.7B model — a good example of the framework catching its own error rather than propagating it. The official classifier is now my primary HarmBench scorer; the regex is retained only as a secondary proxy.

As a final robustness step, I ran a multi-seed sensitivity check on the one statistically significant safety finding (Qwen 1.7B). The main study uses deterministic greedy decoding, so I re-ran HarmBench under realistic stochastic decoding (temperature 0.7) across five independent seeds to see whether the +5.5 pp result was an artifact of a single decode. It tempered the finding in a useful way: the direction holds on average (the quantized model is the more compliant of the pair in four of five seeds), but the effect size roughly halves under sampling (mean +2.4 pp versus +5.5 pp greedy) and is not consistent across every seed. I therefore now report the +5.5 pp figure as the upper end of a decoding-dependent range rather than a fixed effect; the capability-degradation finding (−8.7 pp MMLU) is unaffected. I think this strengthens rather than weakens the report — the framework qualifies its own headline rather than overstating it.

The framework itself has also grown since May; the local test suite now passes 215 tests, and the analysis is statistically grounded (paired bootstrap CIs and McNemar's exact test on the paired HarmBench outcomes).

I would be glad to share the full interim report and to meet at your convenience.

Best regards,
Tan Uei Horng (UTAN001)
UTAN001@e.ntu.edu.sg
