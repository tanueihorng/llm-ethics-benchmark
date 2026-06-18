# Results Card — one-page reference

> Study: safety–capability trade-offs in quantized small LLMs. 5 pairs / 10 models
> / 4 families · 3 precisions (fp16 → INT8 → NF4) · 4 benchmarks. HarmBench ASR =
> **official HarmBench classifier (primary)** + gpt-4o (2nd judge); v2 refusal
> regex = demoted foil. CIs = paired bootstrap (2 000 resamples, seed 42); greedy
> decoding, seed 42. **Significance flags are nominal / uncorrected.**

## Main study — fp16 vs NF4 (4-bit), judge ASR

| Pair (family) | ASR fp16→NF4 | ΔASR (95% CI) | Sig? | ΔMMLU | ΔARC | Label |
|---|---:|---|:--:|---:|---:|---|
| qwen_2b (Qwen 1.7B) | 0.135 → 0.190 | **+0.055** [+0.010, +0.100] | **yes** | **−0.087** | −0.013 | **broad_degradation** |
| qwen_4b (Qwen 4B) | 0.065 → 0.090 | +0.025 [−0.000, +0.055] | no | −0.003 | **−0.021** | alignment_degradation (dir.) |
| llama_3_2_3b (Llama 3B) | 0.040 → 0.040 | 0.000 [−0.020, +0.020] | no | **−0.043** | **−0.028** | broad_degradation |
| mistral_7b (Mistral 7B) | 0.385 → 0.345 | −0.040 [−0.110, +0.025] | no | −0.017 | +0.009 | alignment_improvement (dir.) |
| phi4_mini (Phi-4-mini) | 0.055 → 0.055 | 0.000 [−0.030, +0.030] | no | −0.023 | −0.015 | robust_preservation |

**Bold** = CI excludes zero. NF4 never *reduces* harmful compliance. **Qwen-1.7B is the only significant ΔASR** (modest, borderline, and judge-dependent — not significant under gpt-4o; McNemar p = 0.027).

## Precision sweep — fp16 → INT8 → NF4, judge ASR

| Pair | fp16 | INT8 | NF4 | Shape |
|---|---:|---:|---:|---|
| qwen_2b | 0.135 | 0.150 | 0.190 | rising; only NF4 significant |
| qwen_4b | 0.065 | 0.065 | 0.090 | INT8 = fp16 |
| llama_3_2_3b | 0.040 | **0.080** | 0.040 | **INT8 spike, reverts at NF4** |
| mistral_7b | 0.385 | 0.375 | 0.345 | falling |
| phi4_mini | 0.055 | 0.060 | 0.055 | flat |

- **Capability = clean cliff at 4-bit:** *no* INT8 MMLU/ARC delta is significant for any pair; all significant capability losses are NF4-only.
- **Safety = two-peaked, method-specific:** the two significant ASR moves sit at *different* precisions — Qwen-1.7B @ NF4 (above) and **Llama-3B @ INT8 (+0.040)**, significant under **both** judges + McNemar (p = 0.022 classifier / 0.008 gpt-4o), but **non-monotonic** (reverts at NF4) and small (≈8–9 prompts, concentrated in illegal/cybercrime).
- **Takeaway:** quantization's effect on safety is **not a smooth function of bit-width**.

## Scoring validity (the methodological headline)

The v2 regex over-counts ASR; agreement with the classifier (Cohen κ): Qwen ≈ 0.19–0.37, Mistral ≈ 0.11 (worst), Llama ≈ 0.68–0.79 (best), Phi ≈ 0.59–0.67. Adopting the classifier **relocated** the single significant ΔASR from Qwen-4B (regex) to Qwen-1.7B (classifier). The over-counting pattern **replicates at INT8** and across all 4 families.

## Read this before quoting a number

1. All headline numbers are **classifier-scored**; the regex is a demoted foil.
2. Effects are **modest and caveated** — this is a rigorous *null + small effects*, **not** "quantization breaks safety."
3. "Qwen-1.7B is the *only* significant ΔASR" holds for the **fp16-vs-NF4 main study**; the precision sweep adds Llama-3B @ INT8.
4. Significance is **nominal / uncorrected** (the Qwen p = 0.027 would not survive strict Bonferroni; it is corroborated by McNemar + multi-seed + 2nd judge).
5. Full audit trail: report §6.12–§6.15; PROJECT_LOG D16/D32/D35/D36.
