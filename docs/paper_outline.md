# Workshop Paper / Poster Outline

> A submission-ready skeleton for a short paper (4–8 pp) or poster, derived from
> how standardized-evaluation and quantization-safety papers are structured:
> HELM (Liang et al. 2022, arXiv:2211.09110), HarmBench (Mazeika et al. 2024,
> ICML; arXiv:2402.04249), and the quantization-safety literature (Kharinaev et
> al. 2025, arXiv:2502.15799; "A Comprehensive Evaluation of Quantization
> Strategies for LLMs", arXiv:2402.16775). **Core convention applied: separate
> the reusable *framework* from the empirical *findings*, and lead with a clear
> statement of need.**

## Target venues (short paper / workshop)

- An *-ACL / EMNLP / NeurIPS workshop on **evaluation, safety, or efficient/compressed LLMs** (e.g. workshops on Trustworthy/Safe ML, Efficient NLP, or LLM evaluation).
- The **NeurIPS Datasets & Benchmarks** track or a benchmarks workshop — your framework + judge-validation finding is a natural fit there.
- An NTU FYP **poster showcase** (the poster layout at the end reuses this skeleton).

## Working title (pick one)

1. *"Is it Alignment or Capability? A Matched-Pair, Judge-Validated Study of Quantization Effects on Small-LLM Safety."*
2. *"Quantization Safety is Method-Specific, Not Bit-Width-Graded: Evidence from a Five-Pair, Three-Precision Sweep."*
3. *"When the Scorer is the Story: Refusal-Regex vs. Classifier Judges in Quantization-Safety Evaluation."*

(1) foregrounds the **methodological** contribution; (2) the **empirical** one; (3) the judge-validation finding. For a workshop, (1) is safest — it names the framework *and* the finding.

## Abstract (≈150 words, structured)

> **Context** — Deploying compact LLMs increasingly means quantizing them, but it
> is unclear whether the safety changes reported under quantization are genuine
> alignment shifts or artefacts of capability loss and brittle scoring.
> **Method** — We introduce an open, matched-pair framework (identical weights,
> on-the-fly quantization, deterministic decoding) that evaluates 5 model pairs /
> 4 families across 3 precisions (fp16 → INT8 → NF4) and 4 benchmarks, scoring
> harmful compliance with the official HarmBench classifier (primary) and a second
> judge (cross-check) rather than a refusal regex. A capability-anchored
> interpretation layer separates the two failure modes.
> **Findings** — (i) the refusal regex systematically over-counts ASR, relocating
> the one significant effect; (ii) capability loss is a clean cliff at 4-bit while
> the safety effect is two-peaked and *method-specific*, not bit-width-graded.
> **Contribution** — a reusable framework and a cautionary, judge-validated read
> of quantization safety.

## Section skeleton (with what goes where)

**1. Introduction** — the gap (compact-LLM deployment ⇒ quantization; conflicting safety reports; scoring is itself a confound). State the 3 contributions as bullets. *(HELM/HarmBench convention: name the desiderata you will satisfy up front.)*

**2. Related Work** — three threads: (a) quantization × behaviour/safety (cite the dose-response and "no consistent trend" findings — they *support* your method-specificity claim); (b) safety benchmarks & red-teaming (HarmBench, XSTest); (c) LLM-as-judge evaluation and its validity (survey arXiv:2411.15594; "No Free Labels" arXiv:2503.05061).

**3. Framework (the reusable artefact)** — describe it *as a tool a reader could reuse*: matched-pair design, config schema, benchmark-plugin contract, the judge layer, SLURM orchestration, the immutable-artifact + redaction discipline. *(This is the section that earns the "Datasets & Benchmarks"/artifact framing — keep findings out of it.)*

**4. Experimental Setup** — models (5 pairs / 4 families), precisions (fp16/INT8/NF4), benchmarks (HarmBench/XSTest/MMLU/ARC), decoding (greedy, seed 42), scorers (classifier primary + gpt-4o; v2 regex foil), statistics (paired bootstrap CIs + McNemar). One table.

**5. Results** —
- 5.1 *Scoring validity*: regex vs classifier (κ by family; the relocation of the significant ΔASR). **This is your headline methodological result.**
- 5.2 *Main study (fp16 vs NF4, 512-token primary budget)*: the 5-pair ΔASR/ΔMMLU/ΔARC/ΔOR table + interpretation labels; no significant ΔASR increase (the only significant ΔASR is Llama-3B's decrease; the 128-token Qwen-1.7B +0.055 was a truncation artefact, §6.16).
- 5.3 *Precision point (fp16→INT8→NF4)*: capability cliff at 4-bit; the two-peaked, method-specific safety picture (Qwen-1.7B @ NF4; Llama-3B @ INT8, both judges + McNemar, non-monotonic).

**6. Discussion** — what it means for deployment; why "method-specific, not graded" matters; the capability-anchored reading.

**7. Limitations** — n = 200 CIs; *uncorrected multiple comparisons* (the Qwen p = 0.027 is nominal); single greedy decode (multi-seed arm tempers it); classifier-as-reference without human labels; small/non-monotonic INT8 effect (~8–9 prompts). *(Be explicit — reviewers reward calibrated claims over strong ones.)*

**8. Conclusion + Reproducibility statement** — restate contributions; point to the public repo, the artifact, and the reproducibility kit.

## How your three contributions map to the paper

| Contribution | Lives in | One-line claim |
|---|---|---|
| Methodological (judge-vs-regex validation; capability anchor) | §3 + §5.1 | "Refusal-counting over-states ASR; validating against the benchmark's own classifier relocates the significant finding." |
| Empirical (method-specific, not bit-width-graded) | §5.2–5.3 | "Capability loss is a 4-bit cliff; safety effects are sparse, model- and method-specific." |
| Engineering (open, reusable framework) | §3 + reproducibility | "A matched-pair, judge-validated quantization-eval framework others can extend." |

## Poster layout (A0/A1, reuse the above)

- **Header band**: title + the one-sentence finding ("not bit-width-graded").
- **Left column**: the gap → the framework diagram (matched pair → 3 precisions → 4 benchmarks → judge).
- **Centre**: the 5-pair results table + the fp16→INT8→NF4 sweep figure.
- **Right column**: the judge-vs-regex κ figure (the methodological punchline) + limitations box.
- **Footer**: QR to the repo + the reproducibility statement.

## What reviewers will probe (pre-empt in the text)

1. *"n = 200, uncorrected comparisons — is the effect real?"* → §7 + the McNemar/multi-seed/second-judge convergence.
2. *"Your judge is itself an LLM."* → the second-judge κ + the explicit human-grounding caveat.
3. *"Is this just one quantization library?"* → yes (bitsandbytes NF4 + INT8); name GPTQ/AWQ/GGUF as future work.
4. *"Framework or study?"* → it is both, by design — §3 is the artefact, §5 is the finding.
