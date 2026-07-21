# T44 Phase 3 — citations & scholarship results (4 axes)

Session S4, 2026-07-21. Two layers:

## Deterministic layer (orchestrator scripts, zero model tokens) — all clean
- **D54 usage→presence gate:** 12/12 pass.
- **Numbering integrity, all 8 surfaces** (6 builders + 2 tex mirrors): every listed ref is
  bracket-cited, no phantom brackets, all 25 tex bibitems `\cite`d.
- Process note (recorded per the sweep's honesty rule): the orchestrator's first two ad-hoc
  integrity scans BOTH false-alarmed ("refs [22]–[25] uncited") — body-truncation and a bad
  regex lookahead. Caught by cross-verification before filing; no false finding escaped.

## Semantic layer — workflow `wf_8cb7af9e-b96` (47 agents, 0 errors, ~3.2M subagent tokens)
35 finder seats: 31 fitness (one per report ref, Opus xhigh + web), 2× 3b unnamed-methods
(xhigh), 3c conventions + 3d quotes/datasets (Opus low). Every finding → 2 adversarial
Opus-xhigh refuters; kill = both refute; orchestrator re-verified every survivor by direct grep.

**30/35 seats fully clean** — including all 5 recently-added stats refs (BH, Cohen 1960, L&K,
Wilcoxon, Cohen 1988): correct sites, correct renumber cascades, correct bibliography. Zero
hallucinated citations for the second consecutive full fitness audit. Spot-verified attributions
held up under web check (e.g. HarmBench's verbatim "N = 512" quote; XSTest's GPT-4 three-class
judge; TrustLLM's six dimensions).

**6 findings raised → 2 killed, 4 survived (all P3, both refuters concurring each time):**

| FS | Axis | One line |
|---|---|---|
| FS-9 | 3a fitness | Both LaTeX mirrors misattribute "no consistent trend across bit-widths" to Jin et al. (ref4) — Jin actually found 4-bit largely retained; the phrasing matches Kharinaev (ref3). JS surfaces all correct; tex-only defect |
| FS-10 | 3a fitness | Proskurina venue: "Findings of NAACL 2024" on the report, bare arXiv on the other six surfaces |
| FS-11 | 3b completeness | MDE/power method uncited: code docstring names "Connor 1987 / Lachin"; report §6.5.1 quotes the MDE with no bracket; thesis has NO power reference at all. Same class as D54's uncited-BH — caught by the absence-direction pass, proving the D54 corollary in action |
| — | 3b (merged into FS-11) | Thesis "power analysis" naming + lexicon blind spot ("Cohen's d" keying) — one remediation with FS-11 |

**Killed (both refuters):** AUC needs a ROC citation (universal term, no cite conventional);
GPT-4o snapshot pin at third-not-first use (pin present, placement is taste).

## Declared blind spots (per plan)
- 3b's unnamed-method arm is judgment-based; the code-side seat examined the four main analysis
  scripts + metrics helpers but not every scorer utility.
- Fitness agents verified against abstracts/available text via web; paywalled full texts (e.g.
  Landis & Koch 1977 body) checked against standard secondary knowledge.
- 3d skipped self-quotes of the project's own artifacts by design.

**Phase 3 verdict: citation surface is in excellent shape** — zero hallucinations, zero wrong
site→ref mappings post-renumber; the three real defects are a tex-only clause, a venue
normalization, and one uncited methods formula, all P3, all batched to Phase 9.
