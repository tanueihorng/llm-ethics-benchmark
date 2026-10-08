---
name: fyp-report-audit-old
description: Original (pre-2026-09-27) version of the FYP report audit, kept alongside fyp-report-audit. Audit FYP report evidence, citations, numerical claims and omissions, plus prose clarity and natural academic writing in the selected document. Use only when the old audit skill is requested by name.
---

# FYP Report Audit

## Resolve scope and deliverable

Apply the repository AGENTS.md and locate relevant PROJECT_LOG entries by task or topic. Resolve the requested artifact and builder in `configs/report_versions.yaml`; use a matching task packet only if it fits the request. Read the selected produced document before stating its content. Do not substitute the canonical report for a requested rewrite or submission variant.

Inventory the requested scope from the actual document and task: prose, tables, figures/captions, notes, appendices, and references as applicable. Reconcile the inventory with the audit extractor; derive expected coverage independently. For a full citation audit, count citation occurrences and bibliography entries separately, including table cells and notes. Missing required qualifications count as omissions even if all extracted values match. Disclose samples, inaccessible sources, and uninspected surfaces; continue independent checks.

## Trace claims to evidence

- Current primary-study claims use `results_512/analysis/`. Use `multiple_comparisons.json` for the prespecified family and `multiple_comparisons_judge_strict.json` only for the explicitly validation-informed sensitivity family.
- HarmBench is classifier-primary. Use the 512 judge agreement/sidecar artifacts for judge claims. Never treat regex `pair_interpretations` as judge-primary ASR. XSTest keeps its prespecified v2 scorer and separately labeled validation-informed sensitivity analysis.
- Use `results/analysis/` only for explicitly identified 128-token historical comparisons. Keep budgets, models, precision, scorers, populations, and statistical families labeled.
- Trace each material sentence, cell, or caption to the source fields or passage that support it. Independently recompute numerical claims where inputs allow; check denominator, exclusions, sign, rounding, intervals, and p versus adjusted q. Shared generated values prove consistency, not independent correctness.
- Check interpretation as well as arithmetic: nonsignificance does not establish equivalence; association does not establish cause; post-hoc evidence must retain its timing and limitations. Resolve provenance from live sources and recorded decisions rather than assumed chronology.
- Resolve citation markers to bibliography entries, confirm identifier/title/authors and publication status, and inspect the original claim-relevant passage. Label preprints. Separate identity verification, abstract-only support, full-text support, contradiction, and unavailable evidence. A related paper or secondary citation cannot silently replace the cited original.
- Search for counterevidence and missing qualifications. Recheck prior or delegated findings against primary artifacts; do not inherit verdicts. `make verify-claims` checks its registered properties, not citation entailment or unregistered omissions.

Keep evidence reproducible with document location, source path/field or page, and commands or calculations used. Record artifact identity (hash when useful) so later rebuilds cannot silently invalidate the audit. Never expose raw HarmBench prompt, behavior, generation, or response text in evidence logs or chat; use IDs, labels, and aggregates. Raw artifacts remain immutable and existing scorer sidecars must not be overwritten.

## Cross-check in-text citations and the reference list

For a general report audit or a requested citation audit, inspect the selected document's actual citation markers and bibliography. Establish its required citation style from the brief, template, or consistent existing convention; do not impose a different style. Keep citation mechanics and source support as separate checks.

- Inventory citation occurrences in prose, table cells and notes, figure captions, footnotes/endnotes, and appendices. Expand grouped markers and numeric ranges into individual references. Reconcile extraction with document inspection so unsupported marker formats or omitted surfaces cannot disappear from the coverage count.
- Check both directions: every in-text marker resolves to the intended bibliography entry; every bibliography entry has an in-text use or an explicit permitted role such as further reading. Flag uncited entries for review rather than automatically deleting them. Distinguish citation occurrences, unique cited works, and reference-list entries when reporting counts.
- For numeric styles, check missing/duplicate identifiers, broken ranges, and citation order where the required style demands first-appearance numbering. For author–year styles, check author spelling, year, same-year suffixes, and consistent use of abbreviated author lists. Do not infer that an unusual ordering is wrong without checking the style.
- Confirm each work's authors, title, year, publication venue/version, and DOI or other identifier against an authoritative source. Detect duplicate works under different identifiers and distinguish preprints from published versions. Check completeness and consistent formatting without inventing missing bibliographic fields.
- Read the sentence or clause attached to each material citation. Check that placement makes the supported claim clear, especially when a sentence contains several claims or a paragraph ends with one citation. Verify support from the relevant original passage; bibliographic accuracy alone does not establish support. Keep direct quotations faithful and check locators where required, without reproducing restricted source text in audit output.
- Look for uncited attributable claims, named methods, borrowed figures/tables, and literature assertions; distinguish these from the study's own results and ordinary background statements. Verify figure/table attribution and adaptation labels when relevant. Check that a source's qualifications, population, experimental setting, and publication status match the wording.

For each issue, report the document location and marker, resolved reference (or failure to resolve), the mismatch or unsupported clause, the evidence consulted, and the proposed correction. Record unavailable full text separately from a confirmed mismatch. After authorized edits, recheck marker-to-entry links, affected numbering/order, source support, and coverage in the rebuilt deliverable; a valid bibliography does not prove valid in-text citations.

## Review prose for natural academic writing

Include an editorial pass in a general FYP report audit; respect a narrower request limited to evidence, numbers, or citations. Review connected paragraphs in the selected document, including captions and discussion, rather than searching only for a list of words. Aim for clear, conventional academic prose that preserves the author's voice. This is a writing-quality review, not AI-authorship detection: wording, punctuation, or detector scores cannot establish who wrote a passage.

Look for patterns that obscure the argument or make it formulaic:

- Inflated significance, promotional claims, vague praise, or generic conclusions that exceed the evidence.
- Repeated stock transitions, paragraph openings, section previews, and summaries that add no information.
- Forced contrasts, repeated three-part lists, dramatic fragments, and sentence rhythms that make every point sound like a slogan.
- Vague subjects, stacked abstract nouns, unnecessary jargon, awkward compressed statistical shorthand, and synonym changes that blur a stable technical term.
- Paragraphs that list findings without explaining their relationship, repeat table values without interpretation, or make a logical jump between evidence and conclusion.
- Chat-style commentary, instructions to the reader, invented personal voice, and drafting residue that do not belong in the report.

Judge each passage in context. Formal vocabulary, passive voice, technical hyphens, an occasional transition or dash, and standard chapter signposting are not defects by themselves. Prefer precise subjects, direct verbs, consistent terminology, and paragraphs whose sentences develop one argument. Do not make the report casual, flatten technical distinctions, or introduce deliberate imperfections to sound human.

Report material prose findings separately from factual defects. Give the document location, a short safe excerpt, the concrete readability problem, and a proposed rewrite that preserves the claim. Group recurring patterns with representative examples and state the sections checked; do not claim a full prose review from a word scan. Leave clear passages alone.

For authorized rewrites, compare original and revised text for changes to numbers, citation attachment, methods, scorer/family labels, causal strength, uncertainty, and limitations. Preserve necessary qualifications such as post-hoc status and the distinction between nonsignificance and equivalence; removing filler must not strengthen the scientific claim. A wording fix that needs new evidence is a substantive change and must undergo the evidence checks above. Apply edits through the registered builder and inspect the regenerated document as described below.

## Findings and authorized fixes

For read-only audits, return findings ordered by impact with exact claim location, supporting evidence, consequence, and proposed correction. Separate verified defects from access gaps and optional improvements. State the inspected scope and omissions; do not call an unavailable source false or treat a green gate as a complete audit.

For authorized edits, change the selected registered builder and regenerate that deliverable. Add bibliography entries in the same edit as named methods. Run `make report` when canonical report content changes. Sweep the whole current-facing claim surface for retired claims and their paraphrases when results or primary configuration change; label historical snapshots. Update PROJECT_LOG with actual files and whether `make report` ran.

Verify the final rebuilt artifact, not only its source: re-extract affected claims, reconcile coverage, and inspect rendered pages when layout, tables, figures, or pagination are involved. Re-run affected checks; report pre-existing failures separately. For changes to checker behavior or policy, use the harness-maintenance skill and meaningful bad-value/omission fixtures without weakening guards.
