# Thesis Paper — Structure & Writing Guide

> A chapter-by-chapter blueprint for the final FYP thesis, applying how empirical
> ML/NLP work is actually written. Conventions drawn from: the classical empirical
> dissertation arc (problem → background → method → implementation → evaluation →
> discussion); standardized-evaluation framework papers (HELM, Liang et al. 2022,
> arXiv:2211.09110; HarmBench, Mazeika et al. 2024, ICML, arXiv:2402.04249); the
> ML Reproducibility Checklist (Pineau et al. 2021, JMLR); and LLM-as-judge
> validity practice (survey arXiv:2411.15594; "No Free Labels", arXiv:2503.05061).
> Your interim report (`docs/FYP_Report_2026-06-14.docx`) already realizes most of
> this — this guide makes the structure and the conventions explicit so the *final*
> thesis is defensible.

## The mental models to apply (what reviewers/examiners reward)

1. **Separate the artefact from the finding.** HELM and HarmBench read as
   *frameworks first*; their empirical numbers are demonstrations. Give your
   framework its own chapter; keep results in a later chapter. (Your report already
   does this — Ch4 system design vs Ch6 results.)
2. **Statement of need before contribution.** Open by naming the *gap* (does
   quantization shift safety, or capability, and is the scoring even valid?), then
   list contributions as crisp bullets.
3. **Calibrated claims beat strong claims.** The quantization-safety literature
   itself reports "no consistent trend" and method-dependence — cite that, and let
   it license your *modest* framing. Report effect size + uncertainty + the
   significance convention every time; never a bare "significant."
4. **Validate your instrument.** LLM-as-judge work expects chance-corrected
   agreement (κ), ideally human grounding, and multiple judges. Your judge-vs-regex
   κ + second judge is a strength; disclose the missing human-label validation.
5. **Threats to validity as a first-class section** (internal / external /
   construct / statistical-conclusion validity), not an afterthought.
6. **Reproducibility statement + artefact** — map to the Pineau checklist; point
   to the public repo and `docs/REPRODUCIBILITY.md`.

## Chapter blueprint (maps to the existing report)

| Ch | Title | Purpose & conventions to apply | Report § |
|---|---|---|---|
| 1 | **Introduction** | Motivation (compact-LLM deployment ⇒ quantization); the gap; **RQ1–RQ5**; contributions as bullets; thesis structure paragraph. *Lead with need.* | Ch1 |
| 2 | **Background & Related Work** | Three threads woven to a gap: (a) quantization × behaviour/safety, (b) safety benchmarks & red-teaming (HarmBench, XSTest), (c) LLM-as-judge evaluation + validity. End each thread with "…but none does X", motivating your design. | Ch2 |
| 3 | **Methodology** | The *scientific* design: matched-pair rationale, quantization (NF4 + INT8/LLM.int8), benchmarks + metrics, decoding controls, the **capability-anchored interpretation framework**, and the **scoring decision** (classifier-primary, regex demoted; judge-validation protocol). State every control. | Ch3 |
| 4 | **System Design & Implementation** | The *reusable artefact*: package architecture, config schema, plugin contract, model loader (incl. the quantization-engaged guard), matrix orchestration, SLURM, immutable-artifact + redaction discipline, the 329-test suite. This is the engineering contribution. | Ch4 |
| 5 | **Experimental Setup** | Models (5 pairs / 4 families), precisions, benchmarks, sample sizes, hardware (V100), seed, the run plan. One setup table. *(Pineau checklist lives here.)* | Ch5 |
| 6 | **Results** | 6.1 scoring validity (judge vs regex, κ, the relocation); 6.2 main study (5-pair deltas + labels); 6.3 capability anchoring (MMLU + ARC); 6.4 mechanism probe (refusal margin); 6.5 **precision point (fp16→INT8→NF4)**; 6.6 statistical caveats incl. multiple comparisons. Every claim = estimate + CI + test. | Ch6 |
| 7 | **Discussion & Threats to Validity** | What it means for deployment; internal/external/construct/statistical-conclusion validity; why "method-specific, not graded" matters. | Ch7 |
| 8 | **Limitations** | n = 200; uncorrected comparisons; single-method (two bitsandbytes paths); single greedy decode; classifier-as-reference without human labels; the small/non-monotonic INT8 effect. | Ch8 |
| 9 | **Future Work** | Human-grounded judge validation; more families/seeds to test the INT8 effect; GPTQ/AWQ/GGUF; the margin probe across precisions; an open-weight guard. | Ch9 |
| 10 | **Conclusion** | Restate the gap, the three contributions, and the calibrated headline. | Ch10 |
| — | **Reproducibility statement** | Map to the Pineau checklist; cite the repo + `docs/REPRODUCIBILITY.md`; AI-usage disclosure. | App. |
| — | **Appendices** | Config YAML, sbatch example, schema, test inventory, glossary. | App. A–F |

## Your contribution paragraph (drop-in, Introduction)

> This thesis makes three contributions. **First, a methodological one:** a
> capability-anchored, judge-validated procedure for distinguishing genuine
> alignment shifts from capability degradation under quantization — and the
> finding that a refusal-counting scorer systematically over-counts attack
> success, relocating the study's one significant effect once the benchmark's own
> classifier is used. **Second, an empirical one:** across five matched pairs, four
> families, and three precisions, capability loss is a clean cliff at four-bit
> while the safety effect is sparse, model- and *method*-specific rather than a
> smooth function of bit-width. **Third, an engineering one:** an open,
> reproducible, extensible framework (matched-pair loading, a benchmark-plugin
> contract, a judge-validation layer, and SLURM orchestration) that others can
> reuse for their own quantization-safety studies.

## Writing conventions — quick checklist

- [ ] Every empirical claim: point estimate + 95% CI + the test + the significance convention.
- [ ] Never "significant" without "(nominal; uncorrected)" where multiple comparisons apply.
- [ ] Distinguish "the framework" (Ch4) from "the findings" (Ch6) in section headers.
- [ ] Each related-work thread ends in the gap it leaves.
- [ ] Limitations are specific and quantified (n, prompts, precision), not generic.
- [ ] A reproducibility statement + artefact link; cite primary sources (HELM, HarmBench, Pineau) not blogs.
- [ ] Tense: present for established facts/your framework, past for what you did/observed.

## Key references to cite (verified primary sources)

- HELM — Liang et al., *Holistic Evaluation of Language Models*, arXiv:2211.09110 (TMLR 2023).
- HarmBench — Mazeika et al., ICML 2024, arXiv:2402.04249.
- Quantization × safety — Kharinaev et al., arXiv:2502.15799; *A Comprehensive Evaluation of Quantization Strategies for LLMs*, arXiv:2402.16775.
- LLM-as-judge — *A Survey on LLM-as-a-Judge*, arXiv:2411.15594; *No Free Labels*, arXiv:2503.05061.
- Reproducibility — Pineau et al. 2021 (JMLR / NeurIPS 2019 program); Sandve et al. 2013 (PLOS CB 1003285); Wilson et al. 2017 (PLOS CB 1005510); JOSS (Smith et al. 2018, arXiv:1707.02264).
- (Already in your report) Proskurina et al. 2024; Egashira et al. 2024; Belkhiter et al.; Q-resafe/Chen et al. 2025.
