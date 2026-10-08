---
name: fyp-report-audit
description: Audit any registered FYP report version (final, canonical, submitted, review or a registered path) for evidence, numbers, citations, omissions, cross-references, front matter, whole-report consistency, examiner readiness and academic prose. Use for "audit the report", "is the final report ready", "check the citations/numbers", or a pre-submission review.
---

# FYP Report Audit

The skill is version-agnostic: it never assumes which report is current. The
target is resolved by purpose from `configs/report_versions.yaml`, and every
check reads the produced document, not a remembered version.

It is also agent-agnostic. The instructions are plain Markdown plus repository
commands (`make`, `python`), with no agent-specific tools. The source is
`.agents/skills/fyp-report-audit/`, which Codex reads. Claude Code reads the
byte-identical mirror in `.claude/skills/`, and any other agent reaches it through
the task route in `AGENTS.md`. Where a step names another skill, open that skill's
`SKILL.md` at the path given. Parallel reviewers are optional; if you use them,
recheck what they report against the primary artifacts. Edit only the source, then
copy it to `.claude/skills/` and to any global copy (for example
`~/.codex/skills/`); `tests/test_agent_harness.py` fails when a copy drifts.

**Thoroughness over speed.** A full audit of a report this size is a long task
with hundreds of items to check, and a run that finishes in minutes has not done
one. Never trade depth for speed without saying so. If you must stop early,
report a partial audit with its coverage counts, never a verdict.

## Every audit is a fresh, independent audit

This rule has no exceptions and applies in every mode.

- Audit the current artifact from its primary sources. Nothing carries over from an earlier audit: not its findings, its coverage, its ledger or its verdict, even when the document hash is unchanged.
- Treat everything an earlier audit produced as an unverified lead. That includes findings summarised in PROJECT_LOG or task packets, verification folders and earlier ledgers. Do not open prior audit records until your own stages are complete.
- Each finding you report must rest on evidence gathered in this audit: the path and field, page, command or calculation you ran. Citing an earlier audit is never evidence.
- Use only the blank ledger this audit's Stage 1 run wrote; never copy statuses from another ledger. If the work spans sessions, continue that same ledger and record its path. It is still one audit.

## Scope

- **Full audit (default).** Any request to audit, review or check the readiness of a report means a full audit, unless it explicitly asks for less. Do every stage and fill the whole ledger.
- **Narrow audit.** Only when asked for numbers, citations or prose alone. Do that stage completely, and just as fresh, then list the stages left unchecked.

Name the scope in the first line of the result.

## Stage 0 — Scope, target and identity

1. Apply `AGENTS.md`; search `docs/PROJECT_LOG.md` for the target and topic (decisions, open tasks, superseding entries). Resolve provenance from live sources and recorded decisions, never from assumed chronology.
2. Resolve the target. `make reports-list` shows the purpose aliases (`final`, `canonical`, `submitted`, `review`); use the alias or registered path the request names. If it is ambiguous which document is meant, ask; never substitute another version. Frozen copies (`frozen_outputs`) are audited read-only.
3. Record the artifact identity (path, SHA-256, modified time, builder, git HEAD) so a later rebuild cannot silently invalidate the audit. `audit_report.py` records this.
4. **Prior audits are leads, not evidence** (see the rule above). PROJECT_LOG is read for context and decisions. Findings it records from earlier audits are unverified claims, to be reconciled only after your own stages: say which of them you confirmed, which you did not reproduce, and which findings are new. An identical hash exempts no stage.
5. Check whether the DOCX was saved in Word after its last build (PROJECT_LOG, a changed hash with no builder change, cover or wording differences). Word edits are not in the builder, so a rebuild silently reverts them. Port them into the builder before any rebuild.

## Stage 1 — Mechanical pass and worklist

```bash
make report-audit AUDIT_TARGET=final
```

```bash
make verify-claims
```

- `scripts/report_tools/audit_report.py` writes `tmp/report_audit/<stem>/AUDIT.md`, `findings.json` and a worklist ledger, which stay local because report text is local-only (D94). It reconciles every text node in the package against its extraction (the coverage denominator is independent of the extractor). It checks citations in both directions, cross-references, floats, the contents list against the headings, front matter, research-question tracing, headline numbers, drafting residue, tracked changes and comments. It also applies the omission rules in `configs/report_audit_rules.yaml` and runs the target's claim locks.
- **Severity:** `fail` means a mechanical defect or an uncovered surface. `review` needs a human verdict; it may be legitimate. `info` is inventory. Adjudicate every `fail` and `review` with evidence. The mechanical pass is the start of the audit, not the audit.
- **The ledger is the denominator.** Each run writes a new blank `ledger-<hash8>-<timestamp>.csv`, with one row per section, numeric unit, citation-bearing unit, float, reference entry and research question, all derived from the document. Stages 2–6 fill it. `status` is one of `verified`, `defect`, `gap` (source unavailable), `n/a` or `sampled` (not checked; say why), and `evidence` gives a path and field, page, command or calculation. Existing ledgers are never overwritten, and never copied into a new audit. Check progress with:

```bash
python scripts/report_tools/audit_report.py --ledger-summary tmp/report_audit/<stem>/ledger-<hash8>-<timestamp>.csv
```

- If `claim-lock` fails as **unregistered**, the document's load-bearing numbers are not machine-checked. Register it as a `docx_primary` surface in `configs/claim_surfaces.yaml` before relying on a green gate; this is a harness change. A green `make verify-claims` proves only the registered properties: not citation entailment, and not unregistered numbers or omissions.
- If the builder is newer than the output, rebuild or audit the existing file and say which.

## Stage 2 — Numbers, statistics and material claims (ledger: `numbers`, `section`)

- Primary-study claims come from `results_512/analysis/`. The prespecified family is in `multiple_comparisons.json`. `multiple_comparisons_judge_strict.json` holds only the explicitly validation-informed sensitivity family. Use `results/analysis/` only for explicitly identified 128-token historical comparisons. Keep budgets, models, precision, scorers, populations and statistical families labelled.
- HarmBench is classifier-primary; judge claims use the 512-token judge agreement and sidecar artifacts. Never treat regex `pair_interpretations` as judge-primary ASR. XSTest keeps its prespecified v2 scorer, with the validation-informed sensitivity analysis labelled separately.
- Trace each material sentence, table cell and caption to the source fields or passage that support it, including non-numeric claims about methods, scope and procedure. Recompute numbers independently where the inputs allow, checking the denominator, exclusions, sign, rounding, intervals, and p against adjusted q. Values generated from a shared source prove consistency, not independent correctness.
- Check interpretation as well as arithmetic. Nonsignificance does not establish equivalence; association does not establish cause; post-hoc evidence keeps its timing and limitations. The same quantity must agree across the abstract, results, tables, discussion and conclusion. Missing required qualifications count as omissions even when every value matches.
- Search for counterevidence and missing qualifications in the source artifacts, not only in the report.

## Stage 3 — Citations and reference list (ledger: `citations`, `references`)

Take the required style from the brief, the template or the document's own consistent convention; do not impose a different one. Keep citation mechanics and source support as separate checks.

- **Mechanics.** The tool covers marker↔entry resolution in both directions, uncited entries, numbering gaps, first-use order, duplicate identifiers or titles, and preprint-like entries. Distinguish citation occurrences, unique cited works and reference-list entries when you report counts. Flag uncited entries for review rather than deleting them. For author–year styles, which the tool does not link mechanically, check author spelling, year, same-year suffixes and consistent abbreviated author lists by hand.
- **Identity.** For each entry, confirm the authors, title, year, venue or version, and the DOI or other identifier against an authoritative source. Detect the same work under different identifiers, distinguish preprints from published versions, and label preprints. Check completeness and consistent formatting without inventing missing bibliographic fields.
- **Support.** For each material citation, read the sentence or clause it is attached to and check its placement. When a sentence carries several claims, or a paragraph ends with a single citation, make sure it is clear which claim the citation supports. Verify support against the relevant original passage: a correct bibliography entry alone does not establish support. Check that the source's qualifications, population, experimental setting and publication status match the wording. Keep direct quotations faithful and check locators where they are required, without reproducing restricted source text in audit output.
- Record each citation as identity-verified, abstract-only support, full-text support, contradicted or unavailable. A related paper or a secondary citation cannot silently replace the cited original. Look for uncited attributable claims, named methods (which need bibliography entries in the same edit, D54/D56/D61), and borrowed figures or tables with their attribution and adaptation labels. Distinguish these from the study's own results and from ordinary background statements.

## Stage 4 — Whole report, read as an examiner (ledger: `rqs`, `floats`, `section`)

- **Research questions:** each RQ has a method, evidence and an answer in the conclusion, and the answer's strength matches the evidence.
- **Consistency:** the abstract and conclusion claim nothing the results do not show, and the contribution claims match what was actually done.
- **Figures and tables:** plotted values, axes, legends and captions match the text and the source artifacts. Captions stand alone, and every float is referenced and discussed.
- **Validity:** threats to validity, limitations and the threat-model scope appear where claims are made, not only in a limitations section.
- **Reproducibility:** the models, revisions, configurations, budgets and implementation steps needed to rerun the study are present or pointed to.
- **Ethics:** harmful content is handled with IDs and aggregates only.
- **Measured scope:** look for claims that go beyond what was measured, such as privacy, deployment or trust statements.
- **Delegated findings:** recheck anything a reviewer or earlier audit reported against the primary artifacts rather than inheriting its verdict.

## Stage 5 — Rendered output and programme requirements

- Microsoft Word is the pagination authority (D62). Render or export the document and check the contents and list page numbers, table breaks, figure legibility and captions on the rendered pages, not only in the XML.
- Front matter, declarations, page or word limits and formatting come from the programme brief or template. Cite the source requirement; never invent one. Author-specific content such as acknowledgements and declarations is the student's to write.
- The tool flags external fields (for example `INCLUDEPICTURE` URLs) and a static contents list; confirm the exported PDF is correct.

## Stage 6 — Prose (ledger: `section`)

Include this stage in a full audit and respect a narrower request. This is a writing-quality review, not AI-authorship detection: wording, punctuation and detector scores cannot establish who wrote a passage. Read connected paragraphs, including captions and discussion; do not search only for a list of words. Aim for clear, conventional academic prose that preserves the author's voice.

Patterns to look for:

- Inflated significance, promotional claims, vague praise, or generic conclusions that exceed the evidence.
- Repeated stock transitions, paragraph openings, section previews, and summaries that add no information.
- Forced contrasts, repeated three-part lists, dramatic fragments, and rhythms that make every point sound like a slogan.
- Vague subjects, stacked abstract nouns, unnecessary jargon, compressed statistical shorthand, and synonym changes that blur a stable technical term.
- Paragraphs that list findings without relating them, repeat table values without interpreting them, or jump from evidence to conclusion.
- Chat-style commentary, instructions to the reader, invented personal voice, and drafting residue.

Judge each passage in context. Formal vocabulary, the passive voice, technical hyphens, an occasional transition or dash, and standard signposting are not defects by themselves. Prefer precise subjects, direct verbs, consistent terminology, and paragraphs whose sentences develop one argument. Do not make the report casual, flatten technical distinctions, or introduce deliberate imperfections.

Report prose findings separately from factual defects. Give the location, a short safe excerpt, the problem and a rewrite that preserves the claim. Group recurring patterns with representative examples, state which sections you read, and never claim a full prose review from a word scan. Leave clear passages alone. A rewrite must not change numbers, citation attachment, methods, scorer or family labels, causal strength, uncertainty or limitations; removing filler must not strengthen a claim. A rewrite that needs new evidence is substantive and goes back through Stage 2.

## Project pitfalls (the machine form is `configs/report_audit_rules.yaml`)

| Pitfall | Requirement | Source |
|---|---|---|
| INT8 equivalence | Disclosed as post hoc in the same paragraph as the claim | T73 |
| HarmBench safety claims | Scoped to direct requests (no jailbreak or attack) | D43 |
| 128-token +0.055 | Always 128-scoped; 512 tokens is primary | D41 |
| Phi-4-mini over-refusal −0.048 | Keyword-rule measurement caveat | T36, D51 |
| Judge-strict family | Labelled validation-informed and reported separately | D51 |
| Human labels | Single-annotator limitation disclosed | T30 |
| Detectable-effect figures | Approximate, post hoc, unadjusted | D95 |
| 128-token truncation count (1,206 / 60.3%) | Given with the 184 unclassified comparisons, or as "at least" | `genlen_robustness.json`, T80 |
| Phi decrease attributed to the keyword rule | Worded as plausible or unresolved; scorer dependence is what is shown | T80 |
| GPT-4o judge | Unrecorded snapshot disclosed | T22 |
| Jev judge | Separate named family, never primary; present only if T75 adds it | T75, D86, D87 |
| Regex/keyword HarmBench rates | Secondary, never judge-primary ASR | D16 |

Put each qualifier in the same paragraph as its claim, or anchor it to a sentence about the same subject. A document-wide search is satisfied by unrelated prose: "post-hoc scoring" once hid an undisclosed INT8 claim, and "mismatched generalization" once hid a truncation omission.

When an audit finds a new recurring failure, add a rule with must-fire and must-pass fixtures in `tests/test_report_audit.py`, using `.agents/skills/fyp-harness-maintenance/SKILL.md`. Never loosen a rule or claim lock to clear a finding: correct the report, or record a justified allow or waiver entry.

## Findings and completion

A full audit is complete only when:

- `--ledger-summary` exits 0: every row carries a status, and every non-`n/a` row carries evidence or a note;
- every `fail` and `review` from Stage 1 has been adjudicated;
- Stages 4–6 each have a written scope.

Report in this order:

1. **Scope, target, hash and ledger path.**
2. **Coverage.** Checked/total per ledger stage, which items were sampled and why, sources that were unavailable, and surfaces the tool does not read (headers and footers, images).
3. **Findings**, ordered by severity, then by document order:
   - **P1:** a wrong or misleading claim, an integrity or redaction breach, or a submission blocker (claim-lock failure, unresolved citation, missing required section).
   - **P2:** an omitted qualification, unsupported citation, broken cross-reference or inconsistency between chapters.
   - **P3:** clarity, style or an optional improvement.

   Each finding gives its ID and severity, location and a short safe excerpt, the claim, the evidence consulted, the consequence and a proposed correction.
4. **Reconciliation with prior audits**, done last: which of their findings this audit confirmed, did not reproduce, or found new.

Separate verified defects, access gaps and optional improvements. Never call an unavailable source false, or treat a green gate as a complete audit. Never expose raw HarmBench prompt, behavior, generation or response text; use IDs, labels and aggregates. Raw artifacts are immutable, and existing scorer sidecars must not be overwritten.

**Pre-submission triage.** Triage orders the fixes, not the audit. Complete the audit first, then fix in this order: P1 findings, green claim locks, citations resolving to the right works, cross-references and front matter, P2 omissions, then prose.

## Authorized fixes

- Edit the registered builder, never the generated DOCX. Port any Word-saved edits into the builder first (Stage 0), then rebuild with `python scripts/reports.py --target <alias> --build` (the canonical alias uses `make report`). Add bibliography entries in the same edit as any named method.
- Verify the rebuilt artifact, not only its source. Rerun `make verify-claims`, re-extract the affected claims, reconcile coverage, and inspect the rendered pages when layout, tables, figures or pagination changed. Before calling the report ready, run a new fresh, full audit of the rebuilt artifact. After citation edits, recheck marker-to-entry links, numbering and order, and source support; a valid bibliography does not prove valid in-text citations.
- When results or the primary configuration change, sweep the whole claim surface for retired claims and their paraphrases, and label historical snapshots (D42).
- Record the change in PROJECT_LOG with the files touched and what was rebuilt. Preserve frozen and historical copies.
- Changes to checker behaviour or policy go through `.agents/skills/fyp-harness-maintenance/SKILL.md`, with meaningful bad-value and omission fixtures and without weakening guards.
