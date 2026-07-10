# Report format and style audit

## Current documents

| Artifact | Builder | Render status | Visual coverage |
|---|---|---|---|
| `FYP_Report_2026-07-01_v5.docx` | `build_fyp_report_v5.js` | 102 pp rendered | Cover/body/reference targeted; not every page. |
| Report humanized | `build_fyp_report_humanized.js` | 102 pp rendered | Cover only. |
| `FYP_Thesis_2026-07-02_v4.docx` | `build_fyp_thesis_v4.js` | 32 pp rendered | Cover/reference targeted. |
| Thesis humanized | humanized builder | 32 pp rendered | Not visually inspected. |
| `FYP_Interim_2026-07-10.docx` | `build_fyp_interim.js` | 33 pp rendered | Cover/conclusion targeted. |
| Interim humanized | humanized builder | 33 pp rendered | Not visually inspected. |

No clipping/overlap was observed in inspected pages. This is targeted review, not complete visual certification of all 334 pages.

## Findings

### P1 — canonical document identity

**READ:** the report cover says “Interim Report”; its footer says “FYP Interim Report · 24 May 2026”; core title is “FYP Interim Report”; cover date is 2 July. This conflicts with current filename/log status as the canonical full report. The humanized alternate inherits it. Evidence: `build_fyp_report_v5.js:253,1553,1623`.

### P2 — interim declaration date

**READ:** interim cover is 10 July 2026; declaration says 2 July. Use a consistent signing date or label a statement date.

### P3 — bibliography style assertion

**READ:** full report calls its list IEEE numbered style but renders `1.` rather than strict `[1]` labels; metadata is not uniformly complete. Thesis/interim use brackets. Repair or name it adapted numbered style.

The claim lock checks selected numbers and first-use numbering. It does not establish semantic citation support, humanized alternate fidelity, complete visual layout or NTU/CCDS template compliance.
