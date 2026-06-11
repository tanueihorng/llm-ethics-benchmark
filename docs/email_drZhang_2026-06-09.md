# Progress email draft — Dr. Zhang (2026-06-09)

> Draft for T1 (re-engagement / progress update). Verify the numbers against
> `results/analysis/judge_agreement.json` before sending. Plain-text body below
> the subject line.

**To:** jiehuang.zhang@ntu.edu.sg
**Subject:** CCDS25-1136 progress update — quantization safety study: results + a scoring self-correction

---

Dear Dr. Zhang,

I hope you have been well. I would like to update you on CCDS25-1136 (benchmarking the ethical performance of quantized open-source LLMs) since our last discussion in March.

The experimental framework is complete and all runs have finished on the NTU TC1 cluster. The study compares three matched baseline-vs-4-bit (NF4) model pairs — Qwen3-1.7B, Qwen3-4B, and Llama-3.2-3B — across three axes: harmful compliance (HarmBench), over-refusal on benign prompts (XSTest), and general capability (MMLU). Both members of each pair load from identical weights, so quantization is the only variable.

The headline is a clean three-way story that I believe is the project's strongest contribution — 4-bit quantization did **not** affect the models uniformly:

- **Qwen3-1.7B (smallest)** degraded on *both* axes: harmful compliance rose significantly (+5.5 pp, McNemar p = 0.027) *and* capability fell significantly (−8.7 pp).
- **Llama-3.2-3B** kept its safety calibration but lost significant capability (−4.3 pp).
- **Qwen3-4B** was the only pair to come through essentially intact on both axes.

So quantization's harm depends on model size and family, and surfaces on *different* axes for different models. The capability anchor in my framework is what lets me separate a genuine safety regression (Qwen 1.7B) from a capability-driven artifact (Llama) — which is the core question the project set out to answer.

One methodological point I am pleased with: my initial regex-based scorer over-counted harmful compliance, so I validated it against HarmBench's own 13B classifier model. The judge corrected the result and relocated the one significant safety regression from the 4B model to the 1.7B model — a good example of the framework catching its own error rather than propagating it.

I am now adding a multi-seed robustness check on the significant finding before finalising. I would be glad to share the full interim report and to meet at your convenience.

Best regards,
Tan Uei Horng (UTAN001)
UTAN001@e.ntu.edu.sg
