# Full Adversarial Pre-Submission Audit — FYP Report v5 + Thesis v4

**Date:** 2026-07-15 (UTC+8) · **Auditor:** Claude Fable 5 (orchestrator) + 94-agent verification workflow · **Mode:** audit only — no repo file was edited except this ledger.

**Scope / text of record:** `scripts/build_fyp_report_v5.js` (report, 1,696 lines) and `scripts/build_fyp_thesis_v4.js` (thesis, 432 lines), audited against `results_512/analysis/*.json` as the only numeric ground truth, plus the outer claim surfaces (README.md, docs/RESULTS_CARD.md, docs/fyp-report-defense-deck-2026-07.html) and all 31 unique cited sources (web-verified).

**Method.** Nine audit passes run as independent agents (17 chapter/pass readers + 2 IEEE-format agents at Opus, one agent per unique reference for citation existence/metadata/claim-integrity at Opus with web access). Every reported finding then faced **two independent Opus-high skeptics** instructed to refute it against the primary artifacts; a finding survived only if neither refuted it. The orchestrator (Fable 5) then re-derived every numeric survivor directly from the artifacts and adjudicated the one disputed citation finding. Pipeline: **22 raw findings → 20 survived skeptics → 2 refuted → 1 survivor overruled by orchestrator re-derivation → 17 deduplicated ledger entries** (three passes independently found the same §6.2 error; two found the same thesis off-by-one — counted once each).

**Main-loop groundwork (run inline before the workflow):**
- `make verify-claims`: **71/71 pass** — the machine-locked numeric surface is green; this audit targets only the unlocked residue.
- Retired-value grep sweep over both builders (`+0.055` as current, v1 over-refusal values incl. 0.036 / −4.4 pp / −2.8 pp "borderline", "339 tests", ARC "immune / not subject to the format asymmetry", "classifier- and human-anchored", κ as proven "lower bound"): **clean**. Every hit is a CI endpoint, an explicit negation ("not immunity"), or explicitly scoped history (Appendix G revision record).

---

## Ledger

Severity: **P1** blocks submission · **P2** should fix before submission · **P3** polish. Verification: **CONFIRMED-O** = orchestrator re-derived from the primary artifact; **CONFIRMED-S** = both independent skeptics re-derived it (file+line evidence in their verdicts). No finding below is PLAUSIBLE-only; unverifiable candidates were dropped.

| # | Sev | Pass | Location | Claim as written | Evidence (artifact value) | Verified | Suggested correction |
|---|-----|------|----------|------------------|---------------------------|----------|----------------------|
| F1 | **P2** | 2 | `scripts/build_fyp_report_v5.js:804` (§6.2) | "XSTest over-refusal falls from 0.056 to 0.032" (1.7B→4B baseline scaling) | `results_512/analysis/pairwise_deltas.json`: qwen_2b xstest baseline **0.052**, qwen_4b baseline **0.028**. The report's own Table 6.1 (L758/762) and §6.6 (L836) already carry 0.052/0.028. Found independently by three readers. | CONFIRMED-O | Change to "from 0.052 to 0.028". |
| F2 | **P2** | 2 | `scripts/build_fyp_thesis_v4.js:275` (§6.1) | "29 judge-only labels against 325 regex-only across the ten models" | `results_512/analysis/judge_agreement.json` per_model sums: judge_harmful_v2_not = 1+0+3+0+6+1+7+1+4+5 = **28** (325 is correct). The report does not carry this sentence — thesis-only. | CONFIRMED-O | "29" → "28" (the near-strict-subset argument is unaffected). |
| F3 | **P2** | 3 | `README.md:210–211`, `docs/RESULTS_CARD.md:18–19` | Llama ΔASR CI "[−0.070, −0.010]"; Mistral "[−0.085, +0.040]" | `results_512/analysis/genlen_robustness.json`: Llama CI **[−0.075, −0.010]**, Mistral **[−0.080, +0.040]**. These are the exact pre-D43 stale values (claim-audit discrepancies #3/#4) fixed in the report on 2026-07-02 but never swept into the outer surfaces — a D42-sweep miss. | CONFIRMED-O | Update both files' headline tables to the artifact CIs. |
| F4 | **P2** | 3 | `docs/fyp-report-defense-deck-2026-07.html:84` | Qwen3-4B ASR row shows BH q = ".214" | `results_512/analysis/multiple_comparisons.json`: qwen_4b harmbench_asr_judge bh_q_value = **0.2406** (→ .241). Digit transposition; p = .096 and the other four q-cells are correct. | CONFIRMED-O | ".214" → ".241". |
| F5 | **P2** | 1 | `scripts/build_fyp_report_v5.js:595` (§4.4) | Loader "checks model.is_loaded_in_4bit and logs a warning if the flag is false" | `ethical_benchmark/models/loader.py:95–117` `_require_quantization_engaged` **raises RuntimeError** ("Refusing to proceed…") — no warning path exists — and tests **four** signals (is_loaded_in_4bit / is_loaded_in_8bit / is_quantized / hf_quantizer). The code is *stronger* than described; the prose misdocuments the mechanism. | CONFIRMED-S | Describe the actual fail-loud behaviour: raises and refuses to proceed if no quantization signal is set. |
| F6 | **P2** | 1/8 | `scripts/build_fyp_report_v5.js:604` (§4.5) | benchmarks/utils.py "exposes a single classify_refusal function" | `grep -r classify_refusal ethical_benchmark/` → **no such symbol**. Actual shared API: `match_refusal_pattern` (utils.py:175) and `is_refusal_response` (utils.py:204). A named component that does not exist. | CONFIRMED-O | Name the real functions. |
| F7 | **P2** | 1 | `scripts/build_fyp_report_v5.js:634` (§4.6) | "fyp_cli.py exposes seven subcommands, summarised in Table 4.2" | `fyp_cli.py` registers **nine** subparsers (adds agent-status, agent-start); Table 4.2 lists seven. | CONFIRMED-O | "seven study subcommands plus two agent-harness helpers (agent-status, agent-start)" or extend Table 4.2. |
| F8 | **P2** | 9/3 | `scripts/build_fyp_report_v5.js:1435–1436` (Appendix E) | Repository layout lists `configs/{default.yaml, tc1.yaml}` and `results/` only | Appendix E omits **`configs/tc1_512.yaml`** (the primary-study config, reproduced in Appendix A as exactly that) and **`results_512/`** (the primary results tree). A reader following Appendix E reaches the retired 128-token config. | CONFIRMED-O | Add tc1_512.yaml (primary, D41) + tc1_int8.yaml and results_512/ to the layout; mark tc1.yaml/results/ as the retained 128-token comparison. |
| F9 | P3 | 1 | `scripts/build_fyp_report_v5.js:573` (Table 4.1) + `:1439` (Appendix E) | benchmarks/ enumerated as base/harmbench/xstest/mmlu/utils (App. E adds registry.py) | `ethical_benchmark/benchmarks/` also contains **arc.py** — the plugin behind a BH-FDR survivor (Llama ARC −0.032) — absent from both listings. | CONFIRMED-O | Add arc.py (and registry.py in Table 4.1). |
| F10 | P3 | 2/9 | `scripts/build_fyp_report_v5.js:931` (×2, §6.12 Result 6) + `scripts/build_fyp_thesis_v4.js:324` | "across the ten NF4 aliases" (κ range −0.01–0.50; human-sheet sampling) | The ten primary-study aliases are **five fp16 baselines + five NF4** (configs/tc1_512.yaml); `xstest_judge_agreement.json` computes the κ range over exactly those base+NF4 aliases. Calling all ten "NF4" mislabels half the sample. | CONFIRMED-O | "the ten base/NF4 aliases of the primary study" (three sites). |
| F11 | P3 | 4 | `scripts/build_fyp_report_v5.js:1053` (Ch9) vs `:1003` (§7.4) | Ch9 future work: "(i) re-run the cross-check with an open-weight guard (e.g. LlamaGuard…)" | §7.4 of the same document states the guard "has since been run, with verification and fold-in pending" (T37 Phase-C: computed, deliberately not folded in). Internal tension: Ch9 asks to *re-run* what §7.4 says is already run. No T37 numbers leaked anywhere (checked). | CONFIRMED-S | Reword Ch9(i) to "verify and fold in the already-run LlamaGuard cross-check". |
| F12 | P3 | 9 | `scripts/build_fyp_report_v5.js:356` (List of Figures/Tables) | LOF entry "Code listing 5.1 — Head-node pre-cache invocation" | The pre-cache code block (L716–724) is emitted as a bare code block with no "Code listing 5.1" caption paragraph — the LOF entry points at an unlabelled object. | CONFIRMED-S | Add the caption below the block, or drop the LOF entry. |
| F13 | P3 | 9 | `scripts/build_fyp_report_v5.js:970` (§6.16) | 128-vs-512 ΔASR table emitted with no table number, caption, or source line | Every other Ch6 table carries a numbered caption + "Source: results_512/analysis/…" line (Tables 6.1–6.4 at L779/802/923/943). | CONFIRMED-S | Caption it (e.g. "Table 6.5 … Source: genlen_robustness.json") and add column widths. |
| F14 | P3 | 9 | `scripts/build_fyp_thesis_v4.js` (body prose, e.g. 128/129/157 vs 275) | Mixed apostrophe glyphs | 28 curly (’) vs 13 straight (') apostrophes in body prose; the same words appear both ways ("classifier's" vs "classifier’s"). Renders inconsistently in the docx. | CONFIRMED-S | Normalise to the typographic apostrophe. |
| F15 | P3 | 9 | `scripts/build_fyp_thesis_v4.js:295` (§6.4) | "No pair becomes significantly more trigger-happy on benign prompts" | Colloquial register in an otherwise formal thesis; L299 and L324 also each run >120 words in one sentence. | CONFIRMED-S | "significantly more prone to over-refusal"; split the two longest sentences. |
| F16 | P3 | 5 | `scripts/build_fyp_report_v5.js:1076–1101` (References) | Venue-style inconsistency in the reference list | ICML spelled out in [1]/[4]/[16] but bare "in Proc. ICML" in [15]; ACL bare in [3] but spelled out in [10]. In-text numbering, resolution, and first-use order are otherwise clean in both documents (also machine-locked). | CONFIRMED-S | Pick one convention per venue and apply uniformly. |
| F17 | P3 | 6 | `scripts/build_fyp_thesis_v4.js:361` (ref [4]) vs report ref [12] | Thesis cites Kharinaev as "arXiv preprint arXiv:2502.15799, 2025"; report cites "IEEE Access, 2025" | Semantic Scholar + DOI **10.1109/ACCESS.2026.3703899** (IEEE Access vol. 14, pp. 96771–96793): the paper **is** published in IEEE Access — the report is right, the thesis entry is the weaker/stale form, and the two documents disagree on the same source. | CONFIRMED-O | Upgrade the thesis entry to the IEEE Access venue (matches the D43 residual note). |

### Refuted / overruled during verification (recorded per the auditor-hallucination convention)

1. **Overruled by orchestrator:** the citation agent (and both its skeptics) reported report ref [12] "IEEE Access, 2025" as an *unsupported venue* because the arXiv abs page carries no journal-ref and their web search found no IEEE record. Semantic Scholar's registry shows the IEEE Access DOI/volume/pages. The venue is correct; only the F17 cross-document inconsistency remains. This is exactly the confabulation class §6 of `docs/REPORT_CLAIM_AUDIT_v5.md` warns about — three Opus agents agreed on a wrong finding that one registry lookup dissolved.
2. **Refuted by skeptics (2):** a pass-8 claim against the "≈8,900 lines of production Python / 382 tests" figures (correct against the live tree), and a pass-9 tone objection to the §6.9 "family/recipe is a comparable-to-larger safety determinant" sentence (properly hedged in context).

### Citation passes 6–7 (all 31 unique sources)

Every reference in both documents **exists**; author lists, years, arXiv IDs verified against arXiv/venue pages. 30/31 fully clean on metadata; the one dispute (Kharinaev venue) resolved **in the report's favour** (see above). Claim-integrity: every in-text attribution checked against the fetched abstract; **no attribution stronger than its source survived verification** — the D43/2c citation-fitness fixes have held.

---

## Verdict

**SUBMISSION-SAFE, with a short pre-submission fix list. No P1 was found.** The load-bearing claim surface — headline numbers, significance statuses, direction, correction status, scorer attribution, threat-model scoping, citation integrity — is sound: it is machine-locked (71/71) and the residue this audit could reach produced no finding that changes any conclusion, label, or hedge. The eight P2s are all one-line builder/doc edits followed by `make report` / `make thesis`: one prose number contradicting its own table (F1), one off-by-one (F2), stale pre-D43 CIs on the repo-facing surfaces (F3) and a transposed deck digit (F4) — both D42-sweep escapes — and four system-description drifts (F5–F8) where Chapter 4/Appendix E describe an older or wrong shape of the code. None affects the science; all would embarrass in a viva if a reader cross-checked.

**Recommended order:** fix F1–F8, regenerate both docx, re-run `make verify-claims` + the stale-text gate, then treat the P3s as batch polish.

## Checks NOT run (and why)

1. **The 71 machine-locked checks were not re-derived** — explicitly excluded by the audit brief; `make verify-claims` was run once and is green.
2. **The three `_humanized.docx` submission alternates** — separate artifacts with their own text; semantic-equivalence audit is tracked as T33 scope. F1/F10-class errors may be mirrored there; the alternates should be regenerated/re-checked after the P2 fixes.
3. **The interim report builder** (`FYP_Interim_2026-07-10.docx`) — outside the brief (report + thesis); its headline numbers are claim-locked (2 interim checks in the 71).
4. **LaTeX/Overleaf bundles** (`fyp_submission/`, gitignored) — point-in-time exports; T34 is a manual user upload. They predate any fixes arising from this audit and will need a refresh if F1/F2 are fixed.
5. **Rendered-docx visual QA** (TOC field population, figure placement, page breaks) — the docx is generated; TOC requires a Word/LibreOffice field update (known, documented in D47).
6. **Phase-C artifacts themselves** (T36 gold set, T37 LlamaGuard, T39 5-pair multiseed) — deliberately pending by design; checked only for premature leakage of their numbers into the text (**none found**; F11 is a wording tension, not a leak).
7. **Dashboard data layer** — not a submission surface; separately unit-tested.
8. **The 128-token tree `results/`** — scoped historical comparison per D41; not re-verified.
9. **Full deck design/tone audit** — only the deck's numeric claims were checked (F4).

## Findings that deserve new `verify_report_claims.py` locks

1. **F1:** lock the §6.2 baseline-scaling sentence (XSTest 0.052→0.028, MMLU 0.643→0.747) to `pairwise_deltas.json` — prose levels outside tables are currently unlocked.
2. **F2:** lock the thesis subset counts (28 judge-only / 325 regex-only) to the `judge_agreement.json` per-model sums.
3. **F3:** extend the lock (or add stale-text patterns in `configs/artifact_policy.yaml`) to **README.md and docs/RESULTS_CARD.md** headline tables — ban the retired CI strings `[−0.070, −0.010]` (Llama) and `[−0.085, +0.040]` (Mistral) outside historically scoped text. The claim lock currently stops at the three documents; the D42 sweep rule exists precisely because outer surfaces drift.
4. **F4:** add the deck's five BH-q cells to the lock (the deck's headline deltas were refreshed under D47 but the q column was not covered).
5. **F6/F10:** add stale-text patterns for `classify_refusal` (nonexistent symbol) and `ten NF4 aliases` (systemic mislabel — the correct phrase is "ten base/NF4 aliases"); self-test both patterns fire, per the D42 convention.

---

*Process note: per the audit brief, no file other than this FINDINGS.md was created or edited — including `docs/PROJECT_LOG.md`, whose changelog row for this audit is deliberately left to the session that acts on these findings.*
