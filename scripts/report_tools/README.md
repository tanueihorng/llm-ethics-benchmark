# report_tools — gated editors for the FYP report builders

## Version-agnostic audit pass (`audit_report.py`)
`make report-audit AUDIT_TARGET=<alias|registered path>` resolves the target from
`configs/report_versions.yaml` and audits the rendered DOCX: coverage reconciled
against every text node, citations both ways, cross-references, floats, contents,
front matter, RQ tracing, headline numbers, residue, the omission rules in
`configs/report_audit_rules.yaml`, and the target's claim locks. Output stays in
gitignored `tmp/report_audit/<stem>/`. It is Stage 1 of `.agents/skills/fyp-report-audit/SKILL.md`, not a complete
audit; the original skill (`fyp-report-audit-old`) does not use it. Tests: `tests/test_report_audit.py`.

Rescued from a session scratchpad on 2026-07-27 (they would otherwise have been
lost). These are the tools that made the T47-T51 report work safe.

## The gate battery (`apply_audit_fixes.py`)
Content-keyed exact-string replacement. NOTHING is written unless every gate passes:

| Gate | Checks |
|---|---|
| G1 | each `old` string occurs exactly the expected number of times |
| G2 | no model family lost, no model identity invented (allowlist) |
| G3 | no number disappears from the document |
| G4 | no reference orphaned (every `[n]` still cited) |
| G5 | reference numbering still strictly by first use |

Usage: import it, set `EDITS = [(id, count, old, new), ...]`, call `main()`.

## The compression battery (`compress_ch6.py`)
Same gates, but G3 is evaluated on the **RENDERED docx**, not the builder source,
and permits declared removals:

```python
C.run(EDITS, ALLOWED_REMOVALS, label="...")     # ALLOWED_REMOVALS = {"60980": "reason"}
```

Why rendered: registry-driven tables (`CLAIMS.render.*`) are not literals in the
`.js`, so a source-level check reports a number as DELETED at the moment a table
starts carrying it. It flagged Llama's NF4 MMLU `0.573` exactly that way.

## Block relocation (`move_to_appendix.py`)
Moves whole blocks from a chapter into Appendix I verbatim and leaves a pointer.
G5 is deferred during a move (relocation legitimately changes citation first-use
order), then `renumber_refs.py` runs and the order is re-verified.

## `renumber_refs.py`
Resolves `[[key]]` placeholders and reorders `const refs` strictly by first use.
**It rebuilds the refs array including `H1("References")` and the
`...numberedList([...])` wrapper.** An earlier version emitted bare strings;
docx-js silently drops non-paragraph children, and the report shipped with NO
BIBLIOGRAPHY for a whole session while every source-level citation gate stayed
green. Do not "simplify" that wrapper away — `tests/test_claim_fitness.py`
now gates the rendered bibliography.

## Passes
`pass1_perpair.py` .. `pass8_round3.py` are the applied edit sets, kept as a
record of what changed and why. They are not re-runnable against the current
builder (their anchors are consumed); they are documentation.
