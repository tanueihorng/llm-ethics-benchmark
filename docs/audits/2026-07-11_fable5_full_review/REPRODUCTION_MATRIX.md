# Reproduction Matrix — what was independently recomputed, from what, by whom (2026-07-10 audit)

Legend: MAIN = audit main agent, inline; WF = fresh-context workflow verifier agent.
Every row = an independent recomputation from committed primary artifacts (never from prose).

| # | Published claim | Recomputed from | Result | Verifier |
|---|---|---|---|---|
| 1 | qwen_2b judge ΔASR @512 = 0.000, b/c=16/16, McNemar p=1.000 | scores.judge.harmbench_cls.jsonl (base+4bit) | EXACT match | MAIN + WF results:headline + WF results:agreement |
| 2 | llama ΔASR −0.040, CI [−0.075,−0.010]*, p=0.0215, only individually-sig, a decrease | judge sidecars | delta/p EXACT; CI matches headline_512_vs_128.json; judge_agreement.json variant is [−0.07,−0.01] (2k vs 10k resamples; P3 source-attribution) | MAIN + 2 WF |
| 3 | BH-FDR: 3 survivors (qwen_2b MMLU q=.008, llama ARC q=.008, phi OR q=.049), 0 ASR survivors, family=20 | multiple_comparisons.json contrasts, own step-up | EXACT (all 20 q to 4dp) | MAIN + WF results:headline |
| 4 | All 20 NF4 contrasts (n, b, c, delta, p, sig) | raw score fields + sidecars | EXACT, zero diffs @0.001 | WF results:headline |
| 5 | Judge-vs-regex κ @512, 10 aliases (Qwen .36–.59, Mistral .25–.28, Llama .71–.84, Phi .67–.77) | judge × v2 sidecars join | EXACT to machine precision | WF results:agreement |
| 6 | gpt-4o 2nd-judge κ 0.68–0.95 | api_judge sidecars | EXACT (0.678–0.954) | WF results:agreement |
| 7 | Human gold set κ 0.59/0.11, precision 0.61/0.23, cells 37/46(18,9)/132(101,6) | human_validation.json 2×2 reconstruction | EXACT (arithmetically forced) | MAIN + 2 WF |
| 8 | 128-era qwen_2b +0.055 CI [+.010,+.100] p=0.027 (always 128-scoped) | results/ 128 sidecars | EXACT (p=0.0266) | WF results:agreement |
| 9 | llama INT8 +0.040 both-judge @128 (p=.021/.008); vanishes @512 (+.005/+.010 n.s.) | INT8 sidecars both trees | EXACT | WF results:agreement |
| 10 | INT8 @512: no significant delta on any endpoint | INT8 @512 sidecars/summaries | CONFIRMED (20 endpoints) | WF results:headline |
| 11 | Multiseed @512 per-seed (qwen_2b 0/5, qwen_4b 1/5, llama 2/5-sig-decreases) | results_sensitivity_512 sidecars | EXACT | WF results:headline |
| 12 | 60.3% direct-prefix truncation; 9.2% prefix mismatch; per-family rates | genlen_robustness.json + raw prefix sampling | CONFIRMED (62% proxy construction P3: measured on 512 tree, undocumented) | WF results:genlen |
| 13 | "29 judge-only vs 325 regex-only" one-directionality | sidecar joins across 10 aliases | EXACT | WF results:agreement |
| 14 | HarmBench "we standardize this parameter to N=512" quote | arXiv 2402.04249v2 full text | VERBATIM; context = target generation length | MAIN (WebFetch) |
| 15 | Manifest: 300 pinned raw artifacts | sha256 rehash of all 300 | 300/300 clean | MAIN |
| 16 | Sample counts 200/250/300/1172 × 15 aliases | summary.json sweep | EXACT | MAIN |
| 17 | Humanized variants "every number byte-identical" | number-token diff orig vs humanized ×3 | CONFIRMED (only dash→comma adjacency) | MAIN |
| 18 | Labels follow classify_pair_change mechanically | function replay on artifact deltas | 4/5 EXACT; phi flips on float-vs-exact arithmetic (P1) | WF ×2 + MAIN |

## What could NOT be reproduced / verified (and why)
- Generation itself (needs TC1 V100 + gated weights + offline caches): out of audit scope; raw artifacts taken as evidence, hash-pinned.
- gpt-4o judge calls (unpinned API alias; date not recorded in sidecars): re-scoring would not be byte-reproducible (disclosed in report).
- INT8@512 judge-vs-regex κ ordering (§6.15 scorer note): committed artifact judge_agreement_int8.json @512 is ALL-NULL (n_shared=0); MAIN recompute gives mistral .214 < qwen_2b .387 < phi .530 < qwen_4b .557 < llama .847 → report's "highest for Llama-3B and Phi-4-mini" partially false (Phi is 3rd).
- Human labels themselves (single annotator = author; sheet local-only): aggregate arithmetic verified; label quality not independently verifiable. 2000-char annotation-view truncation discovered (P1).
- §6.14 refusal-margin flip numbers @512: committed results_512 artifact is a byte-copy of the 128 artifact; flips are NOT budget-invariant (P1/P2; margins are).
