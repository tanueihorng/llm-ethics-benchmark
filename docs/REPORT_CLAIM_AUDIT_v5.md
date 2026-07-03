# Report Claim Audit — FYP_Report_2026-07-01_v5 (D43)

**Date:** 2026-07-02 (UTC+8) · **Auditor:** Claude (main agent, source-backed verification loop) · **Scope:** every load-bearing claim in `scripts/build_fyp_report_v5.js` → `docs/FYP_Report_2026-07-01_v5.docx`.

This document is the durable record of the claim-by-claim verification loop run
after the 2026-07-02 adversarial validity audit (PROJECT_LOG §3 D43). Method:
**every empirical number** is machine-checked against the committed analysis
artifacts; **every citation** is checked against the actual paper (arXiv export
API metadata, publisher/venue pages, full text where needed); **every external
fact** is checked against its official source (the HarmBench paper and GitHub
repository, dataset files, model releases). Nothing below is verified by
trusting another prose surface that repeats the claim.

---

## 1. The numeric claim lock (machine-checked, permanent)

`scripts/verify_report_claims.py` asserts **53 checks** (43 on the report; 10
added for the 512-mirrored thesis `build_fyp_thesis_v4.js`, which pins its
Table 6.2/6.3 cells, κ table, truncation block, BH survivors, INT8 point,
multiseed values, and threat-model scoping to the same artifacts), each in two
directions: (a) the report text still contains the claim, and (b) the value
recomputed from the committed artifact equals the claim. It runs via
`make verify-claims` and inside pytest (`tests/test_report_claims.py`), so the
suite fails if the report ever drifts from its evidence. Coverage: the BH-FDR
family (3 survivors, deltas, q-values), all five per-pair judge ΔASR values
with CIs / McNemar p / flip counts at 512 and the 128-era headline, both judges'
deltas, all ten judge-vs-proxy κ, all cross-judge κ ranges at both budgets,
the Mistral proxy levels, the full truncation block (60.3 / 30.5 / 9.2,
per-family ranges, medians, ceiling proxy), the multi-seed aggregates, the
INT8 point at both budgets (deltas + both-judge McNemar), every capability /
over-refusal delta and CI quoted in prose, sample sizes, judge volume, and the
XSTest source file composition. Status at commit: **43 pass, 0 fail, 0 skipped**.

## 2. Discrepancies found by this loop (all fixed, then re-locked)

These survived the 5-examiner audit AND seven prior external review rounds —
found only when every value was mechanically compared to its artifact:

| # | Report said | Artifact says | Where fixed |
|---|-------------|---------------|-------------|
| 1 | Mistral v2 proxy "0.835 rising to 0.900" (§6.13 prose) | `judge_agreement.json` v2_asr **0.825 / 0.890** (Table 6.1 had it right — the delta +0.065 is identical, which is why nothing tripped) | §6.13 prose |
| 2 | Phi second-judge "almost perfectly (κ 0.79 / 0.95)" (§6.13) | 512 cross-judge κ **0.678 / 0.826** — phi baseline is the study's κ *floor*; 0.79/0.95 is the retired **128-era** value (0.7896/0.9497) | §6.13 rewritten, 128 value explicitly scoped |
| 3 | Llama judge ΔASR CI "[−0.070, −0.010]" (8 sites) | `headline_512_vs_128.json` **[−0.075, −0.010]** (no artifact carries −0.070; earlier bootstrap emission drifted on re-run) | all 8 sites incl. Table 6.2 |
| 4 | Mistral judge ΔASR CI "[−0.085, +0.040]" (4 sites) | **[−0.080, +0.040]** | all 4 sites |
| 5 | Phi ΔOR CI "[−0.076, −0.012]" | `pairwise_deltas.json` **[−0.072, −0.016]** | §6.13 |
| 6 | Ref [1] "L. Sun et al., TrustLLM" | Published version (PMLR v235, `huang24x`): first author **Yue Huang**; title carries the "Position:" prefix | reference list |

(Also fixed earlier the same day under D43: the stale 128-era §6.13 Mistral κ
values 0.19/0.11 and the 128-era cross-judge 0.60–0.63 → 512 values 0.28/0.25
and 0.83/0.78; the κ-range lines 0.25–0.41 → 0.25–0.59; LOC ≈2,800/1,800 →
≈8,900/4,600; test counts.)

## 2b. Thesis-mirror verification pass (2026-07-02 evening)

The 512-mirrored thesis (`build_fyp_thesis_v4.js` → `FYP_Thesis_2026-07-02_v4.docx`)
was adversarially verified by an independent agent against the artifacts after
authoring. Every table cell and load-bearing number verified; zero stale-128
residue. Prose findings, all fixed and re-locked:

| # | Was | Artifact says | Fixed in |
|---|-----|---------------|----------|
| 1 | "every point lies below the diagonal" (Fig. 1 caption, both documents) | 8 of 10 points below; llama base (0.100 > 0.090) and phi 4-bit (0.090 > 0.075) sit marginally above | both captions |
| 2 | qwen_2b @512 "ΔASR = 0.000 … under both judges" (thesis) | classifier 0.000; **gpt-4o +0.005** (p = 1.000 under both) | thesis abstract + §6.3 |
| 3 | "relocated the study's single significant effect" (thesis §6.1) | 128-era regex had TWO CI-significant increases (qwen_4b +0.065, mistral +0.055) vs the classifier's one | thesis §6.1, with aggregate subset counts 29 vs 325 |
| 4 | report prose/captions "κ ≈ 0.25–0.41 for Qwen/Mistral" (a 3rd occurrence surviving the earlier fix) | 0.25–0.59 | report §6.12 + caption; range added to the claim lock's banned list |
| 5 | `precision_sweep.png` embedded title "safety is method-specific" (128-era framing baked into the figure) | @512 there is no robust safety move at either step | `make_figures.py` title + figure regenerated (both docx re-embed it; the other five figures re-rendered byte-identical) |
| 6 | "no INT8 capability delta is significant" (thesis) | no committed artifact carries INT8 capability significance tests | softened to the artifact-backed point-estimate form (within ~1.3 pp of fp16) |

**Known artifact-level nuance (documented, not an error):**
`judge_agreement.json` per-pair bootstrap CIs come from a different resampling
run than `headline_512_vs_128.json` (llama [−0.07, −0.01] vs [−0.075, −0.010];
endpoints differ by ≤ 0.005). Both are valid 95% paired-bootstrap intervals;
the locked, reported values are the headline artifact's. Consumers of the other
file (e.g. the dashboard) may show ±0.005 endpoint differences.

## 3. The load-bearing external fact: 512 is HarmBench's own budget

Claim (D41's foundation): *"HarmBench's 512-token reference budget."*
**VERIFIED, two independent official sources:**

- **The HarmBench paper (arXiv:2402.04249, §4):** "the number of tokens
  generated during evaluation can have a drastic effect on ASR … can change
  ASR by up to 30% … **we standardize this parameter to N = 512** to allow the
  metric to converge." (Figure 2 is devoted to exactly this effect.)
- **The official repo:** `evaluate_completions.py --num_tokens` **default 512**
  (completions are clipped to 512 tokens before the official classifier).
  Note: the `generate_completions.py` CLI default is 256 and their shell
  pipeline passes the value explicitly — the *standardized evaluation
  parameter* (per the paper) is 512, which is what the report now cites
  verbatim (§6.16).

## 4. Citation ledger (all 24 references)

Verified against the arXiv export API (title, authors, year, venue-bearing
comment fields) plus targeted venue checks. ✔ = every printed detail matches.

| Ref | Verdict | Venue evidence |
|---|---|---|
| [1] TrustLLM (2401.05561) | ✔ after fix | ICML 2024 ✓ — PMLR v235 `huang24x`; first author corrected to Y. Huang |
| [2] DecodingTrust (2306.11698) | ✔ | arXiv comment: "NeurIPS 2023 Outstanding Paper (Datasets and Benchmarks Track)" |
| [3] SafetyBench (2309.07045) | ✔ | comment: "ACL 2024 Main Conference" |
| [4] HarmBench (2402.04249) | ✔ | Semantic Scholar venue: ICML; comment: harmbench.org |
| [5] XSTest (2308.01263) | ✔ | comment: "Accepted at NAACL 2024 (Main Conference)"; 6 authors match |
| [6] MMLU (2009.03300) | ✔ | comment: "ICLR 2021"; 7 authors match |
| [7] ARC (1803.05457) | ✔ | arXiv-only citation, matches |
| [8] QLoRA (2305.14314) | ✔ | S2 venue: NeurIPS (2023); introduces NF4 ✓ |
| [9] LLM.int8 (2208.07339) | ✔ | comment: "Published at NeurIPS 2022" |
| [10] Qwen3 TR (2505.09388) | ✔ | first author An Yang ✓ |
| [11] Llama 3 (2407.21783) | ✔ | first author Grattafiori ✓ |
| [12] Mistral 7B (2310.06825) | ✔ | A. Q. Jiang ✓ |
| [13] Phi-4 TR (2412.08905) | ✔ (note) | M. Abdin ✓; *optional improvement:* the model evaluated is Phi-4-**mini**-instruct, which has its own TR (arXiv:2503.01743) — no in-text mis-attribution exists (checked), but adding the mini TR would be tighter |
| [14] Kharinaev (2502.15799) | ✔ | cited arXiv-only ✓ (published venue: IEEE Access 2025, per Semantic Scholar — could be added); "66 quantized variants … no single method dominating" matches abstract verbatim |
| [15] Q-resafe (2506.20251) | ✔ | comment: "ICML 2025" |
| [16] Egashira (2405.18137) | ✔ | **NeurIPS 2024 confirmed** ([poster page](https://neurips.cc/virtual/2024/poster/95767), [OpenReview](https://openreview.net/forum?id=ISa7mMe7Vg), [ETH SRI](https://www.sri.inf.ethz.ch/publications/egashira2024quantization)); **NF4-specificity confirmed** from the official attack code ([eth-sri/llm-quantization-attack](https://github.com/eth-sri/llm-quantization-attack): `compute_box_4bit(method="nf4")`, `compute_box_int8`) |
| [17] HarmLevelBench (2411.06835) | ✔ | cited arXiv-only ✓ (venue: NeurIPS 2024 SafeGenAI workshop, per arXiv comment — could be added) |
| [18] Proskurina (2405.00632) | ✔ | comment: "Accepted to NAACL 2024 Findings"; S2 venue NAACL-HLT |
| [19] Arditi (2406.11717) | ✔ | **NeurIPS 2024 confirmed** ([proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf)) |
| [20] LLM-judge survey (2411.15594) | ✔ | arXiv-only citation ✓ |
| [21] Krumdick (2503.05061) | ✔ | arXiv-only citation ✓; first author Michael Krumdick ✓ |
| [22] Llama Guard (2312.06674) | ✔ | Hakan Inan (Meta) ✓ |
| [23] McNemar 1947 | ✔ canonical | Psychometrika 12(2), 153–157 — standard bibliographic record |
| [24] Efron & Tibshirani 1993 | ✔ canonical | Chapman & Hall monograph — standard record |

**Characterisation checks (in-text claims about what cited papers did):**
Kharinaev's full text (ar5iv) uses **OpenSafetyMini, XSafety, SafetyBench,
HotPotQA** with AWQ/QUIK/AQLM/QUIP# — it does **not** use HarmBench, XSTest,
or MMLU, so the report's Gap 2 (integrated harmful-compliance + over-refusal +
capability evaluation) stands; an external auditor's claim to the contrary was
refuted on this evidence. Egashira demonstrates an *adversarial* attack on
bitsandbytes quantization (NF4/int8) — the report correctly frames it as the
adversarial worst case vs. this study's ordinary loading. HarmBench's
"red-teaming framework" identity is precisely why the report now scopes its own
usage: **standard config, direct requests, no attack applied** (Chapter 3
threat-model boundary; verified against `ethical_benchmark/benchmarks/harmbench.py`).

## 5. Dataset / setup facts

| Claim | Source | Verdict |
|---|---|---|
| HarmBench standard config, n=200 per model | walledai/HarmBench standard split; every raw tree carries exactly 200 records/alias (15 aliases) | ✔ |
| XSTest v2: 250 benign evaluated | `data/xstest_v2_prompts.csv`: 450 rows = 250 safe + 200 `contrast_*` (matches Röttger et al.'s 250+200 design) | ✔ (machine-locked) |
| MMLU subset n=300, 6 subjects, zero-shot; ARC-Challenge n=1172 | configs + raw record counts (300/1172) | ✔ (machine-locked) |
| Judge = cais/HarmBench-Llama-2-13b-cls, official template | HF model card template matches the pinned prompt (byte-level regression test in suite) | ✔ |
| 512 judge volume: 15 × 200 = 3,000 classifications, 0 parse errors | committed sidecars (line counts) | ✔ (machine-locked) |
| Model IDs / sizes (Qwen3-1.7B/4B, Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.3 7.2B, Phi-4-mini-instruct 3.8B) | configs + HF releases | ✔ |

## 6. Auditor-hallucination refutations (for the record)

Two findings from the 5-examiner audit were **refuted with primary evidence**
during cross-verification — kept here as a caution that expert reviewers (human
or AI) also confabulate, and that every finding must be re-verified from
sources before acting:

1. *"Kharinaev already evaluates HarmBench+XSTest+MMLU jointly (Gap 2 is not novel)"* —
   full text lists OpenSafetyMini / XSafety / SafetyBench / HotPotQA. Refuted.
2. *"Appendix D.1 sums to 304 across 23 files and omits the INT8 test files"* —
   the table contained all files and summed exactly to its total. Refuted.

## 7. Residual items (tracked, not errors)

- Optional: add Phi-4-mini TR (2503.01743) beside [13]; add published venues to
  [14] (IEEE Access) and [17] (NeurIPS 2024 SafeGenAI workshop).
- The claim lock covers the numeric claim surface; qualitative prose is guarded
  by the stale-text patterns (`configs/artifact_policy.yaml`) which now scan the
  canonical builder source itself.
- T30 human gold set remains the outstanding construct-validity upgrade for the
  judge (tooling ready, awaiting annotation).
