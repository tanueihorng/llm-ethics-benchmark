# T44 Phase 2A — recompute results

Script: `phase2a_recompute.py` (this dir). **84/85 checks pass, 1 mismatches.**

| check | recomputed | claimed | ok |
|---|---|---|---|
| A0 128/512 refusal_margin artifacts identical | True | True | ✅ |
| A1 pooled flips 92 | 92 | 92 | ✅ |
| A2 pooled harmful-ward 50 | 50 | 50 | ✅ |
| A3 pooled safe-ward 42 | 42 | 42 | ✅ |
| A4 pooled AUC 0.76 | 0.76 | 0.76 | ✅ |
| A5 z-scored AUC 0.64 | 0.64 | 0.64 | ✅ |
| A6 harmful-ward AUC 0.75 | 0.75 | 0.75 | ✅ |
| A7 safe-ward AUC 0.78 | 0.78 | 0.78 | ✅ |
| A8 within-pair AUC qwen_2b 0.61 | 0.61 | 0.61 | ✅ |
| A9 within-pair AUC mistral 0.54 | 0.54 | 0.54 | ✅ |
| A10 baseline median qwen_2b 3.8 | 3.8 | 3.8 | ✅ |
| A11 baseline median qwen_4b 13.0 | 13.0 | 13.0 | ✅ |
| A12 qwen_2b flips 16/5 (n=21) | (16, 5, 21) | (16, 5, 21) | ✅ |
| A13 qwen_4b flips n=9 | 9 | 9 | ✅ |
| A14 llama flips 2/2 | (2, 2) | (2, 2) | ✅ |
| A15 phi flips 5/5 | (5, 5) | (5, 5) | ✅ |
| A16 mistral flips 20/28 | (20, 28) | (20, 28) | ✅ |
| A17 valid-proxy families 30 vs 14 | (30, 14) | (30, 14) | ✅ |
| A18 mean dm qwen_2b +1.2 | 1.2 | 1.2 | ✅ |
| A19 mean dm qwen_4b +4.0 | 4.0 | 4.0 | ✅ |
| A20 mean dm llama -1.1 | -1.1 | -1.1 | ✅ |
| A21 mean dm phi -0.3 | -0.3 | -0.3 | ✅ |
| A22 Wilcoxon p<0.001 every pair | True | True | ✅ |
| A23 dz min ~ -0.74 | -0.74 | -0.74 | ✅ |
| A24 dz max ~ +1.8 | 1.7 | 1.8 | ❌ MISMATCH |
| A25 gate AUC four families in [0.86,0.90] | True | True | ✅ |
| A26 gate AUC mistral 0.69 | 0.69 | 0.69 | ✅ |
| A27 entropy control +0.21 vs harmful +0.02 | (0.21, 0.02) | (0.21, 0.02) | ✅ |
| B qwen_2b/mmlu paired n | 300 | 300 | ✅ |
| B qwen_2b/mmlu delta==precision_sweep | 0.0133 | 0.0133 | ✅ |
| B qwen_2b/mmlu McNemar p (b=8,c=12) | 0.5034 | report: n.s. | ✅ |
| B qwen_2b/arc paired n | 1172 | 1172 | ✅ |
| B qwen_2b/arc delta==precision_sweep | 0.0128 | 0.0128 | ✅ |
| B qwen_2b/arc McNemar p (b=38,c=53) | 0.1418 | report: n.s. | ✅ |
| B qwen_4b/mmlu paired n | 300 | 300 | ✅ |
| B qwen_4b/mmlu delta==precision_sweep | -0.0067 | -0.0067 | ✅ |
| B qwen_4b/mmlu McNemar p (b=9,c=7) | 0.8036 | report: n.s. | ✅ |
| B qwen_4b/arc paired n | 1172 | 1172 | ✅ |
| B qwen_4b/arc delta==precision_sweep | -0.0026 | -0.0026 | ✅ |
| B qwen_4b/arc McNemar p (b=23,c=20) | 0.7608 | report: n.s. | ✅ |
| B llama_3_2_3b/mmlu paired n | 300 | 300 | ✅ |
| B llama_3_2_3b/mmlu delta==precision_sweep | 0.0067 | 0.0067 | ✅ |
| B llama_3_2_3b/mmlu McNemar p (b=7,c=9) | 0.8036 | report: n.s. | ✅ |
| B llama_3_2_3b/arc paired n | 1172 | 1172 | ✅ |
| B llama_3_2_3b/arc delta==precision_sweep | -0.0068 | -0.0068 | ✅ |
| B llama_3_2_3b/arc McNemar p (b=32,c=24) | 0.3497 | report: n.s. | ✅ |
| B mistral_7b/mmlu paired n | 300 | 300 | ✅ |
| B mistral_7b/mmlu delta==precision_sweep | 0.0133 | 0.0133 | ✅ |
| B mistral_7b/mmlu McNemar p (b=6,c=10) | 0.4545 | report: n.s. | ✅ |
| B mistral_7b/arc paired n | 1172 | 1172 | ✅ |
| B mistral_7b/arc delta==precision_sweep | 0.0009 | 0.0009 | ✅ |
| B mistral_7b/arc McNemar p (b=20,c=21) | 1.0 | report: n.s. | ✅ |
| B phi4_mini/mmlu paired n | 300 | 300 | ✅ |
| B phi4_mini/mmlu delta==precision_sweep | -0.0067 | -0.0067 | ✅ |
| B phi4_mini/mmlu McNemar p (b=7,c=5) | 0.7744 | report: n.s. | ✅ |
| B phi4_mini/arc paired n | 1172 | 1172 | ✅ |
| B phi4_mini/arc delta==precision_sweep | 0.0009 | 0.0009 | ✅ |
| B phi4_mini/arc McNemar p (b=18,c=19) | 1.0 | report: n.s. | ✅ |
| B VERDICT: no INT8 MMLU/ARC delta significant (10 contrasts) | 0 | 0 | ✅ |
| B VERDICT: max |INT8 capability delta| <= ~1.3pp | True | True | ✅ |
| C Table6.5 qwen_2b @128 cls | 0.055 | 0.055 | ✅ |
| C Table6.5 qwen_2b @512 cls | 0.0 | 0.0 | ✅ |
| C Table6.5 qwen_2b @512 gpt4o | 0.005 | 0.005 | ✅ |
| C Table 6.5 parsed | True | True | ✅ |
| D1 BH q-values (own implementation, 20 contrasts) | True | True | ✅ |
| D2 BH survivors set | [('llama_3_2_3b', 'arc_accuracy'), ('phi4_mini', 'xstest_over_refusal'), ('qwen_2b', 'mmlu_accuracy')] | [('llama_3_2_3b', 'arc_accuracy'), ('phi4_mini', 'xstest_over_refusal'), ('qwen_2b', 'mmlu_accuracy')] | ✅ |
| D3 n uncorrected significant | 5 | 5 | ✅ |
| D4 exact McNemar reproduces all 20 contrast p-values | True | True | ✅ |
| D5 MDE formula reproduces per-pair MDEs | True | True | ✅ |
| D6 representative MDE 0.0594 | 0.0594 | 0.0594 | ✅ |
| D7 judge-vs-v2 kappa recomputed from sidecars (10 aliases) | True | True | ✅ |
| D7b max kappa abs error | 2.220446049250313e-16 | <1e-6 | ✅ |
| D8 cross-judge kappa (cls vs gpt-4o) recomputed (10 aliases) | True | True | ✅ |
| D9 LlamaGuard kappa recomputed (10 aliases) | True | True | ✅ |
| D10 human-val classifier kappa 0.59 | 0.59 | 0.59 | ✅ |
| D11 human-val regex kappa 0.11 | 0.11 | 0.11 | ✅ |
| D12 gold-set n | 200 | 200 | ✅ |
| D13 strict kappa regex -0.006 | -0.006 | -0.006 | ✅ |
| D14 strict kappa judge 0.485 | 0.485 | 0.485 | ✅ |
| D15 broad kappa regex 0.054 | 0.054 | 0.054 | ✅ |
| D16 broad kappa judge 0.662 | 0.662 | 0.662 | ✅ |
| D17 full refusals 63 | 63 | 63 | ✅ |
| D18 regex caught 2 of 63 | 2 | 2 | ✅ |
| D19 judge caught 61 of 63 | 61 | 61 | ✅ |
| D20 3-class exact agreement 0.695 | 0.695 | 0.695 | ✅ |
