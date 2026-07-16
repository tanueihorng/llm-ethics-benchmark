# Verification of the D48 Remediation — FYP Report v5 + Thesis v4

**Date:** 2026-07-15 (UTC+8) · **Verifier:** Claude Fable 5 (orchestrator) + 118-agent verification workflow · **Subject:** Codex's remediation of the 17 findings in `FINDINGS.md` (PROJECT_LOG D48, §4 row 2026-07-15 21:19).

**Method.** Three gates re-run by the verifier. Every new claim lock and stale-text pattern self-tested by reverting the fix in an isolated sandbox copy of the repo. One Opus agent per finding (is the fix *correct and complete*, not merely present?), eight collateral-damage reviewers over the 1,116-line working diff, five process-claim agents (docx parity ×6, PROJECT_LOG governance, LaTeX bundles, ledger integrity). Every new finding faced two independent Opus-high skeptics. 44 raw → 35 survived → deduplicated to the entries below; the orchestrator re-derived every P2 personally from the primary artifact before it entered this ledger. This file is additive: `FINDINGS.md` is a point-in-time record and was not edited.

---

## 1. Codex's claims — VERIFIED TRUE

| Claim | Verification |
|---|---|
| `make verify-claims` 77/77 | Re-run by verifier: **77 pass, 0 fail**. |
| `pytest tests/` 382 passed | Re-run: **382 passed**. |
| `make agent-check` 8/8 | Re-run: all pass. |
| No result/significance/label/hedge/conclusion changed | **Confirmed.** `results_512/`, `results/`, `results_sensitivity_512/` are untouched in `git status`. Every corrected number re-derived from the artifacts; F1's entire paragraph, F3's full five-pair table, and F4's five BH-q cells were swept beyond the original finding and are clean. |
| All six docx regenerated | **Confirmed** — every docx mtime post-dates every builder edit; rendered-text parity confirmed (corrected strings present, retired strings absent). |
| PROJECT_LOG D48 + §4 row + §1 counts | **Confirmed.** §1 now reads 77/77 and 382 tests. Surviving "339 tests"/"61/61" mentions are confined to the explicitly-historical snapshot block and append-only changelog rows — correctly scoped, **not** drift (a reviewer complaint on this point is refuted). |
| The new guards work | **Self-tested.** In a sandbox copy, reverting each fix to its old value fires 8/8 new stale-text patterns and 7/8 new claim locks. (F10 is guarded by the stale-text scan rather than the claim lock — legitimate.) |
| F17 year correction | **Confirmed via Crossref** (authoritative DOI registry): *IEEE Access*, **vol. 14, pp. 96771–96793, 2026**, DOI 10.1109/ACCESS.2026.3703899, six authors led by Artem Kharinaev. The reviewer's year critique was right; the prior ledger's F17 adjudication was wrong; Codex's fix is correct. |

**12 of 17 findings are fixed correctly and completely:** F1, F2, F3, F4, F6, F9, F10, F12, F13, F14, F16, F17.

## 2. Codex's claims — NOT ACCURATE

1. **"All 17 confirmed audit findings are fixed."** Three fixes (F5, F11, F15) landed at one site while sibling sites carrying the same claim were left untouched, and one fix (F8) introduced a new defect of the class it was closing. See §3.
2. **"The D42 sweep ... repaired additional current mirrors in the interim/humanized builders and field guide."** The field guide was edited, but its retired counts survive (V6).

## 3. New findings — CONFIRMED (orchestrator re-derived each)

Severity: **P2** = fix before submission · **P3** = polish. All CONFIRMED-O (re-derived from the primary artifact by the orchestrator) unless noted.

| # | Sev | Location | Problem | Evidence (artifact value) | Suggested correction |
|---|-----|----------|---------|---------------------------|----------------------|
| **V1** | **P2** | `scripts/build_fyp_report_v5.js:1442`, `scripts/build_fyp_report_humanized.js:1428` | **The F8 fix introduced the exact defect F8 existed to close.** The new Appendix E line `tc1_int8.yaml (INT8 precision point)` points a reader at the **retired 128-token** INT8 config, and `tc1_int8_512.yaml` — the primary one — is omitted. It sits directly beneath `tc1_512.yaml (primary 512-token...)` and above `tc1.yaml (retained 128-token comparison)`, so it reads as current. | `configs/tc1_int8.yaml:137` → `max_new_tokens: 128`; `configs/tc1_int8_512.yaml:141` → `max_new_tokens: 512`. The INT8 results of record live in `results_512/*_8bit`, produced by the 512 config. This line is **newly added** (diff line 371 has no `−` counterpart). Found independently by 7 agents. | List `tc1_int8_512.yaml (INT8 precision point, primary 512-token)`; label `tc1_int8.yaml` as the retained 128-token INT8 comparison, or drop it. |
| **V2** | **P2** | `scripts/build_fyp_report_v5.js:483`, `scripts/build_fyp_report_humanized.js:470` | **F5 fixed at §4.4 only.** §3.3 still reads "the loader verifies that the resulting model object reports `is_loaded_in_4bit`, **logging a warning otherwise**" — the exact retired claim, now contradicting the corrected §4.4 in the same document. | `ethical_benchmark/models/loader.py:95–117` `_require_quantization_engaged` **raises RuntimeError**; no warning path exists; four signals are checked, not one. **Why the gate missed it:** Codex's new stale-text pattern is the literal string `'logs a warning if the flag is false'`, which does not match this site's wording. | Apply the §4.4 correction at both §3.3 sites; replace the string-shaped pattern with one matching the property (e.g. `logging a warning otherwise`). |
| **V3** | **P2** | `scripts/build_fyp_report_humanized.js:919` | **A banned claim survives in a current submission alternate.** The paragraph ends: "Because the sample over-weights disagreements, κ = 0.59 is if anything **a conservative floor** on the classifier's agreement over a representative population." This is the κ-as-proven-lower-bound claim D47 removed from the canonical report. | `scripts/build_fyp_report_v5.js:933` states the opposite for the same result: the sample is a disagreement-enriched contrast and the direction a representative population would move κ **is not identified**. Codex edited this very paragraph (the F10 port "ten base/NF4 aliases of the primary study" is in the same line) and left the banned sentence. | Port the canonical v5:933 wording into the alternate; add a stale-text pattern for `conservative floor` and put the humanized builders in `stale_text.scan_paths`. |
| **V4** | **P2** | `scripts/build_fyp_report_v5.js:1569` (App. H.5); `scripts/build_fyp_report_humanized.js:920, 995` | **F11 fixed at one site of three.** Appendix H.5 still says a second independent judge "remains open" and is "the one remaining construct-validity threat". In the humanized alternate, only Ch9 (:1045) was reworded to "already-run", while §6.12 (:920) and §7.4 (:995) still call the guard "a complementary future check" — so the alternate now contradicts itself, the mirror image of the F11 tension. | §6.12 Result 4 reports the gpt-4o second judge as **run** (κ .68–.95, `judge_pairwise_agreement.json`); §7.4 of the canonical says the open-weight guard "has since been run, with verification and fold-in pending". No T37 numbers leaked anywhere (checked). | Align H.5 and the humanized §6.12/§7.4 with §7.4's status wording; state no LlamaGuard number. |
| **V5** | **P2** | `docs/fyp_showcase.html:655,679` + `_3d.html:669,693` + `_world.html:389` + `_bluemarble.html:339,359` | **Four unbannered, stale-scanned decks carry retired v1 over-refusal values and one wrong significance flag.** qwen_2b: `or:{base:0.056, q:0.028}, orDelta:-0.028, orSig:true`. phi: `orDelta:-0.044, label:'robust_preservation'`. | `pairwise_deltas.json` qwen_2b xstest: base **0.052** → **0.028**, Δ **−0.024**, CI [−0.048, **0.000**] → **not significant** (McNemar p = 0.109; PROJECT_LOG §1). So `orSig:true` **asserts significance the artifact denies**. Phi truth: 0.128 → **0.080**, Δ **−0.048**; its label is **alignment_degradation** (recomputed 2026-07-11). −0.028/−0.044 are the retired v1 values on the audit's own banned list. Decks carry no snapshot banner and only `fyp_showcase_v2.html` is excluded from the scan. | Re-base the four decks' qwen_2b/phi records to the artifacts (or banner them as snapshots); add stale-text patterns for `or:{base:0.056` and `orDelta:-0.044`. **Pre-existing — outside both the original audit scope and Codex's remediation. This is a gap in the original audit, which swept only the two builders.** |
| **V6** | **P2** | `docs/project_field_guide.html:293, 319, 348, 508` | **Retired counts survive on a current-facing surface Codex claims to have repaired.** "339 automated tests across 26 files", "The machine claim lock — **55 checks**" (×2), "pytest (**339 tests**)". | Truth: **382 tests across 29 files**; claim lock **77**. "339 tests" is on the audit's banned list. The file is stale-scanned and unexcluded; the scan passes only because no pattern covers these counts. `configs/artifact_policy.yaml:286` also carries "339 tests" in its own comment. | Update the four sites to 382/29/77; add stale-text patterns for `339 tests` and `55 checks`. |
| V7 | P3 | `scripts/build_fyp_report_v5.js:1093` + 5 mirrors | ~~Kharinaev entry omits the page range IEEE style requires for a journal article.~~ **FIXED 2026-07-15 23:55.** | Crossref: `page` = **96771-96793**. | ~~`IEEE Access, vol. 14, pp. 96771–96793, 2026, doi: …`~~ Done in all six builders; the report-side lock now pins the **full** entry (it previously pinned only the DOI and a revert-test showed it did not fire). |
| V8 | P3 | `docs/RESULTS_CARD.md:63`, `docs/project_field_guide.html:412` | ~~Mistral judge-vs-regex κ upper bound stated as 0.28.~~ **FIXED 2026-07-15 23:55** — and now artifact-bound: all four κ family ranges are derived from `judge_agreement.json` and asserted against RESULTS_CARD (self-tested to fire). | `judge_agreement.json`: mistral_7b_base **0.2904**, 4bit 0.2544 → range **0.25–0.29**. Thesis Table 6.1 already has it right. | 0.25–0.28 → 0.25–0.29. |
| V9 | P3 | `docs/fyp-report-defense-deck-2026-07.html:113` | ~~**FIXED 2026-07-15 23:55** — the card now scopes the range to the ten aliases it is computed over; lock added.~~ Card pairs "κ .68–.95" with "cross-checked **all 15 aliases**". Both statements are individually true, but juxtaposed they imply the range spans 15. | `judge_pairwise_agreement.json` has **10** aliases (κ .678–.954). Over all 15 (adding `_int8.json`: .645–.973) the range would be **.645–.973**. | "all ten base/NF4 aliases" (keeps .68–.95 true). |
| V10 | P3 | `CLAUDE.md:198`, `AGENTS.md:198` | Instruction-layer directory map: "Benchmark plugins (HarmBench, XSTest, MMLU)" — omits ARC, the same stale enumeration F9 fixed in the report. | `ethical_benchmark/benchmarks/arc.py` exists; ARC is a BH-FDR survivor. | Add ARC; edit both files in one commit per the sync rule. |
| V11 | P3 | `scripts/build_fyp_report_v5.js:362` | List of Tables omits Table G.1, which the body emits captioned at :1508. | LOF block ends at Table D.1. Pre-existing; not introduced by F12. | Add the entry or drop the caption. |
| V12 | P3 | `scripts/build_fyp_report_v5.js:1432` | Appendix E layout has no `scripts/` entry, though the report cites ten `scripts/*` paths as provenance. | Ten distinct script paths cited in prose. | Add a `scripts/` line to the tree. |
| V13 | P3 | `scripts/build_fyp_thesis_v4.js:300` | F15 half-applied: the colloquialism was fixed, the "split the long sentences" half was not — the paragraph is byte-identical to pre-fix (longest sentence 93 words). | `git show HEAD:` vs working tree: exact string equality. | Optional; or close F15 as partially-actioned by design. |
| V14 | P3 | `scripts/build_fyp_interim.js` | F14's apostrophe normalization not applied to the interim builder, which has the same defect (51 curly / 30 straight in the rendered docx, same word both ways in one paragraph). | Extracted `FYP_Interim_2026-07-10.docx`. F14 was thesis-scoped, so this is arguably out of scope. | Optional batch polish. |
| V15 | P3 | ~~`scripts/build_fyp_thesis_v4.js:325` vs `build_fyp_report_v5.js:933`~~ **CLOSED** — fixed incidentally by the V3-sibling sweep (§5b) | The two texts of record disagree on the same hedge: the thesis asserts "both scorers' population agreement would sit higher"; the report says that direction "is not identified". | The thesis's parenthetical is the softer residue of the same κ-direction claim D47 removed. | Drop the parenthetical or adopt the report's wording. |

### Guard-design weaknesses (why 77/77 coexists with V2/V3/V5/V6)

- **The new stale-text patterns are string-shaped, not property-shaped.** `'logs a warning if the flag is false'` pins the sentence that was fixed, so the sibling site's "logging a warning otherwise" (V2) sails through. Patterns should target the property.
- **`scripts/verify_report_claims.py:1144`** — the sole lock over the three humanized alternates concatenates all three files into one string before testing its positive snippets, so a snippet present in **one** file satisfies the check for **all three**. The humanized builders are also absent from `stale_text.scan_paths`, which is how V3 survived.
- **`scripts/verify_report_claims.py:239`** — the new deck lock pins only the qwen_4b BH-q cell; the audit's lock recommendation #4 asked for all five.
- No pattern covers `339 tests`, `55 checks`, or the showcase decks' over-refusal records (V5, V6).

## 4. Refuted during verification (recorded per the anti-hallucination convention)

- **"§1 still carries stale 339-tests / 61-checks current-state counts"** — refuted. §1's live layer reads 77/77 and 382; the surviving mentions sit in the explicitly-historical snapshot block and append-only changelog rows.
- **"The F17 year change 2025 → 2026 is wrong"** (2 agents) — refuted by Crossref: the journal record is 2026; 2025 was the arXiv preprint year.
- **"The F11 fix weakens a hedge / asserts an unbacked outcome"** (2 agents) — refuted; the reworded Ch9 states status, not result.
- Four further P3s (F7 wording, Table 6.5 column widths, report-side "trigger-happy", ledger convention) — refuted on reading the actual context.

## 5. Verdict

**The science is intact and independently verified: no result, significance status, interpretation label, hedge, or conclusion changed, and the canonical claim surface is materially better than before the remediation.** Twelve of seventeen findings are closed correctly and completely; the gates are real and I proved the new locks fire.

**But the package is not final-ready, and "all 17 fixed" overstates what happened.** Three fixes (F5, F11, F15) were applied at one site while identically-worded sibling sites were left, and one fix (F8) newly introduced a pointer to a retired config — the same defect class it was closing. Separately, two current-facing surfaces outside the original audit's scope carry retired values, one of them a **significance flag the artifact contradicts** (V5).

The reviewer's revised verdict was right and remains right, with one correction: the residue is not only the known P2s but a further six P2s discovered by verifying the remediation itself. Recommended: fix V1–V6, regenerate, re-run the gates, and convert the string-shaped guards into property-shaped ones before submission.

---

## 5b. Round-3 addendum — V1–V6 fixed, then re-verified (2026-07-15, D49)

V1–V6 were fixed and re-verified by a second adversarial workflow (68 agents; 6 per-fix attackers, 4 sweeps incl. a fresh-eyes hunt, 2 skeptics per finding). **The re-verification found the round-2 failure mode reproduced in the round-3 fixes** — the orchestrator's own. 24 findings survived; all were re-derived and fixed. This section records what the second pass caught, because it is the most instructive part of the audit.

**Two P1s — a fix that corrected the data and left the prose:**

| # | Location | Problem | Artifact |
|---|---|---|---|
| W04/W16 | six `docs/fyp_showcase*.html` chart captions | Caption read "Benign-prompt refusals: **Qwen-1.7B and Phi-4-mini move significantly**" while the data record re-based one screen away said `orSig:false` for qwen_2b — the file contradicted itself and asserted a significance the artifact denies. | qwen_2b xstest Δ −0.024, p = 0.109, q = 0.2431 → **not significant**. Caption rewritten to name only Phi. |
| W05 | four decks' FDR-survivor prose | "Phi-4-mini **−4.4pp** (q=0.049)" — wrong delta *and* wrong q for an FDR survivor. | phi Δ **−0.048**, q **0.0122**. |

**Sibling sites the round-3 fixes missed (same class as F5/F11 in round 2):**

- **V1's sibling:** §6.15 prose in both report builders attributed the *512-token* INT8 run to `configs/tc1_int8.yaml` — the config V1 had just relabelled "retained 128-token". Verified: `tc1_int8.yaml` is `max_new_tokens: 128`; the INT8 results of record (`results_512/*_8bit`) carry `study_name: ..._int8_sweep_512`, `max_new_tokens: 512`. Fixed to `tc1_int8_512.yaml`.
- **V3's siblings:** the κ-direction over-claim survived in `build_fyp_thesis_v4.js`, `build_fyp_thesis_humanized.js` and `build_fyp_interim.js` under two *different wordings* — "(both scorers' population agreement **would sit higher**)" and the truncation caveat's "**can only inflate** measured disagreement". Both assert a direction the canonical explicitly says is not identified. All ported to the canonical hedge.
- **V6's siblings:** `CLAUDE.md:164`/`AGENTS.md:164` ("All tests (339)"), `project_field_guide.html:509` ("339-test suite"), `fyp_status_2026-07{,_v2}.html` (`data-count="339"` counters), `THESIS_OUTLINE.md:42` (329), `email_drZhang_july.md:39` (329), `artifact_policy.yaml:322` — none matched the first cut's `'339 tests'` pattern.
- **V5's sibling:** the field guide's **entire ΔOR column** was v1-era (qwen_2b −0.028, qwen_4b 0.000, mistral +0.004) — same class as the decks, in a file round 3 had already edited. Re-based; all five cells now match `pairwise_deltas.json`.
- **Self-inflicted:** round 3 wrote "**77 checks**" across seven surfaces; its own verifier change made the live count **79** in the same session. Corrected, and recorded as evidence for the recommendation below.

**Guard weaknesses the second pass proved (W12, W15, W14/W18/W20).** Round 3 billed its patterns as property-shaped; three were still string-shaped and were re-derived as escapable:
- `orDelta:-0\.044` was fitted to the JS literal and provably could not see "−0.044" in prose one screen away.
- `'conservative floor'` missed "conservative lower bound", "would sit higher", "can only inflate".
- `'339 tests'` missed "All tests (339)", "339-test suite", `data-count="339"`.
All three are now value/property-anchored, each self-tested to fire on every known wording and verified silent on the corrected text and on a deck's star-coordinate array that merely contains 339 as a coordinate. Guard self-tests: 13 → **31**.

**Refuted in round 3 (5).** Including a claim that §1 carried stale counts (they are in the explicitly-historical block) and that the Appendix D test-inventory description was wrong (it is incomplete, not false).

**Gates after round 3:** verify-claims **79/79**, pytest **382**, agent-check **8/8**. Still no result, significance verdict, label, hedge, or conclusion changed; no immutable artifact touched.

**Standing recommendation (not yet actioned).** Hard-coding volatile counts in prose is the recurring defect, not any individual number: the claim-lock count went 71 → 77 → 79 inside one session and invalidated seven surfaces each time. Either bind these counts to a computed value in `verify_report_claims.py`, or drop them from presentation surfaces and keep them only where the contract maintains them (PROJECT_LOG §1 and the report's claim-locked Appendix D).

## 6. Checks NOT run

1. **Visual layout QA** — I verified docx *content* parity (corrected strings present, retired absent), not visual cleanliness; the page-by-page clipping/overlap inspection remains the user's. The page *count* is now partly reproduced: a headless LibreOffice render of the six current files (`soffice --convert-to pdf`, counting `/Type /Page`) totals **356** against the **355** counted for the visual inspection. The ±1 is expected rather than a defect — the TOC is an `updateFields` field that repaginates on open, so Word and LibreOffice legitimately differ. Recorded as indicative; the load-bearing claim is "renders clean", not the integer. (Codex's contemporaneous 354 was true for the pre-V-fix build and is left in its dated changelog row.)
2. **LaTeX/Overleaf bundle freshness** — `fyp_submission/` is gitignored; the agent's report on it is uncorroborated by git evidence. Re-check before the T34 upload that the zips post-date these fixes.
3. **The 71→77 pre-existing lock checks** — machine-verified, not re-derived by hand.
4. **The interim report and `project_field_guide.html` beyond the counts in V6** — not swept line-by-line.
5. **Phase-C artifacts (T36/T37/T39)** — deliberately pending; checked only for premature number leakage (none found).
