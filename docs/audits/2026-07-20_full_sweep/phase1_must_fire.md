# T44 Phase 1a — Gate must-fire results

Live evidence P0: agent-check project-log-update fired correctly during Phase 0 (see BASELINE.md Notes).

## Perturbation run — 2026-07-20 22:31
| # | family | gate FIRED? | verdict |
|---|---|---|---|
| P1 kappa-value-drift | verify-claims/computed | YES | ok (must-fire proven) |
| P2 ieee-first-use-break | verify-claims/ieee | YES | ok (must-fire proven) |
| P3 pinned-ref-drift | verify-claims/pins | YES | ok (must-fire proven) |
| P4 registry-drift | claim-registry | YES | ok (must-fire proven) |
| P5 tex-vs-zip-drift | surfaces/zip+freshness | YES | ok (must-fire proven) |
| P6 pdf-copy-drift | surfaces/byte-match | YES | ok (must-fire proven) |
| P7 stale-text-inject | agent-check/stale-text | YES | ok (must-fire proven) |
| P8 doc-desync | agent-check/doc-sync | YES | ok (must-fire proven) |
| P9 immutable-mutation | agent-check/immutable | YES | ok (must-fire proven) |
| P10 redaction-leak | agent-check/redaction | YES | ok (must-fire proven) |
| P11 freshness-break | agent-check/report-freshness | YES | ok (must-fire proven) |

Post-run tree check: CLEAN

## Phase 1a summary

**12 check families proven to fire** (11 perturbations + 1 live P0 event); **0 fail-to-fire findings**;
tree verified byte-clean after every restore. Additionally covered without new perturbation:
- pytest gate: self-evidencing (449 tests executed at baseline)
- harness-eval: its 17 tests ARE must-fire fixtures (superseded-builder guards, synthetic leaked
  sidecar, etc.) — all passed at baseline
- citation-completeness gate: proven against the real pre-fix builder at HEAD during D54
  (fired on all four historical defects) + 4 in-suite must-fire fixtures

**Declared blind spots (enumeration basis = the perturbation list; what it cannot see):**
- `volatile_free` / `registered` snapshot-classification profiles: not perturbed (target is a
  gitignored email/deck; profile logic shares the pattern-scan family with P7 — accepted risk)
- Generated claim-surface BLOCKS in docs (sync_claim_surfaces): only the registry half was
  perturbed (P4); the doc-side block-currency check is untested this session
- Per-check exhaustiveness: 1 representative perturbation per FAMILY, not per check (86+264
  individual checks were not each perturbed) — a check could share a family and still be vacuous
