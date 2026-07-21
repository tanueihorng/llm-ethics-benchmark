# T44 Phase 4 — cross-document consistency results

Session S5, 2026-07-21. Workflow `wf_eb1681e8-cc2`: 12 surface seats diffed against the
canonical story (report v5 + results_512 artifacts) in both directions (contradiction/drift AND
material absence of load-bearing caveats); 2 adversarial Opus-xhigh refuters per finding;
orchestrator re-verified every survivor by direct grep. 34 agents, 0 errors, ~2.5M subagent
tokens. Known findings FS-1..FS-11 excluded from re-filing.

## Seat verdicts

| Surface | Verdict | Notes |
|---|---|---|
| thesis builder | **consistent, 0 findings** | full number/verdict/caveat diff clean — every shared value matches artifacts; all five load-bearing caveats present at canon strength |
| interim builder | minor_drift | science fully consistent + properly 512-re-based; one provenance finding (FS-14) |
| tex thesis mirror | minor_drift | FS-12 (stale κ-table labels) + Egashira venue (FS-10 note) |
| tex interim mirror | **material_drift** | FS-12 (labels + 0.28≠0.29) + FS-13 (front matter lists done work as planned — internally inconsistent with its own results section) |
| humanized report | minor_drift | FS-15 (abstract keeps two pre-hedge overstatements) |
| humanized thesis | consistent | 2 findings killed (legitimate restructuring/typography) |
| humanized interim | consistent, 0 findings | |
| README | consistent | 2 findings killed (legitimate front-door compression; caveats one click away) |
| RESULTS_CARD | consistent, 0 findings | |
| decks | consistent, 0 findings | snapshot notices present where required |
| dashboard data layer | consistent, 0 findings | results_512 preference + judge-primary rebuild confirmed |
| instruction layer (CLAUDE/AGENTS + LOG §1) | consistent, 0 findings | |

## Findings: 11 raised → 4 killed → 7 survived → filed as FS-12..FS-16 (+1 FS-10 note)

- **FS-12 (P2)** — both tex mirrors' κ tables still carry pre-T43 labels ("poor to moderate",
  "worst"); interim tex adds a numeric drift (0.25–0.28 vs 0.29). T43 reached the tex citations
  but not the tex tables.
- **FS-13 (P2)** — tex interim's abstract/methodology/table present BH correction, the GPT-4o
  second judge, ARC anchor etc. as "planned"/"will be reported" while its own results section
  reports them done. The JS interim was re-based; the tex front matter was not.
- **FS-14 (P3)** — interim dated "10 July 2026" (cover, declaration, filename) over 18–19 July
  content, with no revision note (report v5 carries one for the identical situation).
- **FS-15 (P3)** — humanized report abstract keeps "quantization is the only variable" and
  "almost always deployed quantized" where the canon deliberately hedged both.
- **FS-16 (P2 lock gap)** — root cause of the tex cluster: NO gate compares tex mirrors to their
  JS sources (the zip gate byte-matches zips vs the same tex = self-consistency only). Four
  independent tex defects this sweep (FS-9, FS-12, FS-13, FS-10-tex) share this one hole.
- FS-10 status note — Egashira also loses its NeurIPS venue in tex (same normalization pass).

**Killed (both refuters):** README RQ1-null without bounded-null caveat + T36 without deviation
note (legitimate front-door compression, full story one click away); humanized-thesis +0.055
paragraph relocation and title punctuation (meaning-preserving).

## The theme

The JS surfaces are in genuinely excellent sync — the thesis diff came back completely clean,
and README/RESULTS_CARD/decks/dashboard/instruction-layer all consistent. Every surviving
defect but FS-15 lives in the **hand-maintained LaTeX mirrors**, which sit outside all gate
coverage (FS-16). Phase 9: R1 fixes the tex content in one pass; R2 closes the structural hole
so it cannot re-open.
