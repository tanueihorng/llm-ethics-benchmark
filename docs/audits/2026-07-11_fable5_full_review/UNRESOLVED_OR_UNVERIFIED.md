# Unresolved or Unverified — honest residue of the 2026-07-10 audit

## Not verifiable from a repo clone (structural)
1. **Generation itself**: raw generations exist only locally (hash-pinned, 300 files verified clean this audit); an independent group cannot re-score or audit the actual text without TC1 access + gated weights. Disclosed in report ("re-run-from-scratch or replay, not recompute-from-raw on a clone").
2. **gpt-4o second-judge calls**: unpinned API alias, no call-date in sidecars — not byte-reproducible even in principle. Disclosed as residual caveat.
3. **Human labels**: single annotator (the author), sheet gitignored. Aggregate arithmetic verified; label quality unverifiable. T30b (second annotator) remains open.
4. **Exact software environment**: recorded nowhere machine-readable (requirements floors only). Report Table 5.2 versions are prose-only and cannot be confirmed against any artifact.
5. **TC1 execution logs**: slurm logs gitignored; job numbers cited in PROJECT_LOG cannot be independently confirmed from the repo.

## Checked but not fully resolved in this audit
6. **Batch-composition counterfactual** for the qwen_2b_base @512 smoke+resume composite: bounding argument only (inside CI; McNemar p=1.000); a clean re-run would be needed to measure the actual effect. Not attempted (TC1 compute).
7. **Harness-vs-model-card classifier template delta**: direction of bias known qualitatively (harness variant more permissive on hedged compliance); magnitude unmeasured. A one-alias re-score with the harness template would bound it. Not attempted (GPU).
8. **Post-stratified human-validation κ**: verifier's reweighting (classifier ≈0.8, regex ≈0.5) is an estimate from the sheet's strata composition; exact population κ depends on labels for un-sampled strata. The order (classifier ≫ regex) is stable under any reweighting.
9. **Historical CI runs**: GitHub Actions history not queried; CI coverage judged from the workflow file only.
10. **resvg/figure regeneration drift**: figures outside the claim lock; a rebuilt PNG could differ silently. Not tested.

## Coverage gaps (explicitly NOT reviewed, with reasons)
11. `docs/archive/**` (18 files) — bannered history, out of current-facing scope.
12. `.claude/skills/html-diagram/references/**` (26 files) — vendored third-party examples.
13. `docs/agent_tasks/**` — historical task packets, policy-excluded from scans.
14. `docs/figures*/**` PNG binary content — provenance reviewed via make_figures.py; pixels not audited (except the one stale precision_sweep.png title found by the claim-surface agent).
15. `results_sensitivity/` (128-era multiseed, 20 raw files) — enumerated, flagged as unpinned/untracked-by-manifest (finding); contents not scanned beyond counts.
16. `node_modules/`, `pdfthumbs/`, `.understand-anything/` intermediates — third-party/tool caches (privacy-scanned only).
17. Rendered-page visual inspection of the docx (fonts, page breaks, TOC pagination) was NOT performed by eye; structure was audited from extracted text + builder source. A human print-preview pass before submission is still advisable.

## Open user actions the audit cannot do
18. ROTATE the OpenAI key pasted in earlier chats (still open; not in repo — verified again this audit).
19. Decide the canonical submission channel (docx vs the school-recommended Overleaf LaTeX) — the P1 stale-LaTeX finding blocks submission until decided/refreshed.
20. NTU GenAI-declaration policy: confirm the school's current requirement; the LaTeX thesis currently has NO AI-use declaration (deleted 2026-06-27 at user request) while all docx variants retain one. Consistency required whichever channel is submitted.
