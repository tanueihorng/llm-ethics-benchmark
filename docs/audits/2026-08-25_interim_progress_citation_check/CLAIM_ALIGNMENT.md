# Claim-Alignment Audit — report claims vs PAPER_SUMMARIES.md vs sources

Date: 2026-08-25 (goal follow-up to goal-166fd06f). Artifact audited: `docs/rewritten_fyp_interim_progress.docx`, SHA-256 `cfef9af1…` (current file; SHA changed from `96f0e930…` due to the logged F1 §3.1 repair — body citing locations re-censused and UNCHANGED).

## Method
All 57 references' citing sentences were extracted from the submitted docx body (93 claim sentences; bracket citations incl. ranges `[a]–[b]` expanded per-ref). Six independent agents judged every claim-ref pair against the paper summaries in `PAPER_SUMMARIES.md`, dropping to the raw pdftotext extracts (`tmp/extract/ref_<n>.txt`) wherever the summary was silent or a number/causal phrase needed source confirmation. The parent independently pre-verified ~17 refs' claims (incl. all numeric ones) BEFORE reading agent outputs, then cross-checked agent findings against extracts after.

## Verdicts (93 judgments)

| Verdict | Count | Meaning |
|---|---|---|
| ALIGNED | 60 | Summary directly supports the claim |
| ALIGNED-WITH-NUANCE | 22 | Supported; wording-level precision notes below |
| NOT-IN-SUMMARY | 2 | Summary silent, but extract confirms the claim ([39] binary-scale scoping; [24] NF4 calibration-freeness) |
| NO-SUMMARY | 9 | The 7 statistics classics ([47],[48],[49],[50],[51],[54],[56]) have no local copy — canonical attributions, plausibility-only (tracked as T68) |
| MISALIGNED | **0** | Nothing contradicted by its summary or source |

Per-group detail: `tmp/claim_audit/audit_{A,B,C,D,E,F}.md` (A: refs 1–9 · B: 10–17 · C: 18–26 · D: 27–34 · E: 35–43 · F: 44–57).

## Actionable wording suggestions for the REPORT (all optional; none is an error)

1. **[22] "representative samples AND an optimisation pass"** — SmoothQuant is verbatim "**training-free**"; it needs only a calibration step. Suggest: "Some methods require representative samples and an optimisation or calibration pass". (Parent re-verified against ref_22.txt.)
2. **[29] HarmLevelBench "grades the level of harm in a response"** — harm-level grading applies to *queries* (7 topics × 8 severity levels); responses get 3-class graded compliance. Suggest: "grades queries by harm level and scores responses for graded compliance".
3. **[15]/[10]** SafetyBench headlines MC↔open-ended *correlation* (r=0.99/0.91); the sharp knowledge≠behaviour dissociation is Kharinaev's result — cite `[10]` alongside if that point is load-bearing.
4. **[11]** "pragmatic triggers" is your gloss; XSTest says "lexical" (substance unaffected).
5. **[33]** "durable quantization-conditioned failure" — scope as *more durable than the ELQ baseline*; the paper's epoch panel shows decay (ASR 0.92→0.11 over five Alpaca epochs) and says "delays catastrophic amnesia".
6. **[35]/[36]** neither refusal-direction nor shallow-alignment papers mention quantization (grep-zero); the bridge is your own hedged inference ("plausible mechanism") — correctly carried.
7. **[27]** "difficult or low-confidence inputs": only "low-confidence" (pre-quantization confidence bins) is the paper's variable.
8. **[31]** "most similar" holds at design level (4-bit broad preservation), not setting level.
9. **Checkpoint-level reference gaps (claim-side):** [5] Llama 3 herd paper documents 8B/70B/405B only — no Llama-3.2-3B coverage; [7] Mistral paper predates the v0.3 Instruct checkpoint actually run. Family-level attribution fine.
10. **[52]/[51]** Card et al. prescribe simulation-based power analysis; the literal name "paired-proportion power analysis" rests on Connor [51].

## Verification integrity
- Parent cross-checked agent findings against raw extracts before accepting: SmoothQuant training-free ✓, AWQ 0.1%–1% ✓, MT-Bench >80% ✓, OR-Bench all five figures ✓, Do-Not-Answer 6 action categories ✓ (distinct from its five risk areas), SafetyBench r-values ✓.
- Two extractor artifacts were caught and contained (scaffolding text glued before some citation clauses in the auto-generated claims file); no claim content was lost.
- No SUMMARY-DEFECT invalidated any verdict. Minor summary-completeness gaps noted: [39] binary-scale clause, [40] system-prompt standardisation — both extract-backed; candidates for a future summary patch.
