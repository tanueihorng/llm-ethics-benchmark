"""Must-fire and must-pass fixtures for the version-agnostic report audit.

Every check is exercised on a synthetic DOCX package so the suite runs on a
clean clone; the real-deliverable smoke test skips when the local-only report
is absent (D94).
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from report_tools.audit_report import main, resolve_target, run_audit  # noqa: E402

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
STYLES = {"Heading1": "heading 1", "Heading2": "heading 2", "Caption": "Caption", "Reference": "Reference"}


def _p(text: str, style: str = "", raw: str = "") -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{ppr}<w:r><w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r>{raw}</w:p>"


def _table(rows: list[list[str]], header: bool = True) -> str:
    out = "<w:tbl>"
    for index, row in enumerate(rows):
        trpr = "<w:trPr><w:tblHeader/></w:trPr>" if header and index == 0 else ""
        out += "<w:tr>" + trpr + "".join(f"<w:tc>{_p(cell)}</w:tc>" for cell in row) + "</w:tr>"
    return out + "</w:tbl>"


def _docx(path: Path, body: str, extra: dict[str, str] | None = None) -> Path:
    styles = "".join(f'<w:style w:styleId="{sid}"><w:name w:val="{name}"/></w:style>' for sid, name in STYLES.items())
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml",
                         f'<?xml version="1.0"?><w:document xmlns:w="{W_NS}"><w:body>{body}</w:body></w:document>')
        archive.writestr("word/styles.xml", f'<?xml version="1.0"?><w:styles xmlns:w="{W_NS}">{styles}</w:styles>')
        for name, xml in (extra or {}).items():
            archive.writestr(name, xml)
    return path


def _report(*chapters: str, references: int = 2, contents: list[str] | None = None) -> str:
    front = _p("Abstract", "Heading1") + _p("This study reports 0.25 and finds no increase in the direct-request setting.")
    if contents is not None:
        front += _p("Table of Contents", "Heading1") + "".join(_p(f"{entry}\t{i + 1}") for i, entry in enumerate(contents))
    refs = _p("References", "Heading1") + "".join(
        _p(f"[{n}] A. Author, “Title number {n} of a cited work,” in Proc. Venue, 2024.", "Reference")
        for n in range(1, references + 1))
    return front + "".join(chapters) + refs


BASE = (_p("Chapter 1: Introduction", "Heading1")
        + _p("Prior work [1] motivates RQ1. HarmBench is used in the direct-request setting [2]. Table 1.1 lists RQ1.")
        + _p("Table 1.1. Research questions", "Caption") + _table([["RQ", "Question"], ["RQ1", "Does it change?"]])
        + _p("Chapter 2: Results", "Heading1") + _p("For RQ1 the rate is 0.25 (Section 1.1 gives context).")
        + _p("1.1 Context", "Heading2")
        + _p("Chapter 3: Conclusion", "Heading1") + _p("RQ1 is answered: the rate stays at 0.25."))


def _checks(tmp_path: Path, body: str, extra=None) -> list[tuple[str, str, str]]:
    path = _docx(tmp_path / "report.docx", body, extra)
    audit = run_audit(path, {"status": "fixture"})
    return [(f.check, f.severity, f.message) for f in audit.findings if f.check != "claim-lock"]


def _fails(findings) -> list[tuple[str, str]]:
    return [(check, message) for check, severity, message in findings if severity == "fail"]


def test_clean_fixture_has_no_fail(tmp_path: Path) -> None:
    findings = _checks(tmp_path, _report(BASE))
    assert _fails(findings) == []


def test_citations_both_directions(tmp_path: Path) -> None:
    body = _report(BASE + _p("Further work [2]–[4] and [7] exists."), references=5)
    findings = _checks(tmp_path, body)
    assert ("cite-unresolved", "[7] has no bibliography entry") in _fails(findings)
    assert ("bib-uncited", "review", "entry [5] is never cited in the text (confirm a permitted role)") in findings
    assert not any(check == "bib-uncited" and "[3]" in message for check, _, message in findings)  # range expanded


def test_first_use_order_and_numbering(tmp_path: Path) -> None:
    body = _report(BASE.replace("Prior work [1]", "Prior work [2] and [1]"))
    findings = _checks(tmp_path, body)
    assert any(check == "cite-order" for check, _, _ in findings)
    gap = _report(BASE).replace("[2] A. Author", "[3] A. Author")
    assert any(check == "bib-numbering" for check, _ in _fails(_checks(tmp_path, gap)))


def test_duplicate_work_is_flagged(tmp_path: Path) -> None:
    body = _report(BASE).replace("Title number 2", "Title number 1")
    assert any(check == "bib-duplicate" for check, _, _ in _checks(tmp_path, body))


def test_dangling_and_unreferenced_floats(tmp_path: Path) -> None:
    body = _report(BASE + _p("See Table 2.9 and Section 9.9.") + _p("Figure 2.1. Unused plot", "Caption"))
    fails = _fails(_checks(tmp_path, body))
    assert ("xref-dangling", "'Table 2.9' has no matching caption") in fails
    assert ("xref-dangling", "'Section 9.9' has no matching heading") in fails
    findings = _checks(tmp_path, body)
    assert ("float-unreferenced", "review", "figure 2.1 is never referred to in the text") in findings


def test_cited_works_float_is_review_not_fail(tmp_path: Path) -> None:
    body = _report(BASE + _p("The original paper reports this in its Table 14 [1]."))
    findings = _checks(tmp_path, body)
    assert not any(check == "xref-dangling" for check, _ in _fails(findings))
    assert any(check == "xref-dangling" and severity == "review" for check, severity, _ in findings)


def test_prose_sentence_is_not_a_caption(tmp_path: Path) -> None:
    body = _report(BASE + _p("Figure 1.1 gives an overview of the results."))
    assert ("xref-dangling", "'Figure 1.1' has no matching caption") in _fails(_checks(tmp_path, body))


def test_duplicate_caption_fails(tmp_path: Path) -> None:
    body = _report(BASE + _p("Table 1.1. Second table with the same label", "Caption"))
    assert any(check == "float-duplicate" for check, _ in _fails(_checks(tmp_path, body)))


def test_contents_must_match_headings(tmp_path: Path) -> None:
    good = ["Chapter 1: Introduction", "Chapter 2: Results", "1.1 Context", "Chapter 3: Conclusion"]
    assert not any(c.startswith("contents") for c, _ in _fails(_checks(tmp_path, _report(BASE, contents=good))))
    stale = good + ["Chapter 4: Removed Chapter"]
    assert ("contents-orphan", "contents entry has no matching heading") in _fails(
        _checks(tmp_path, _report(BASE, contents=stale)))


def test_front_matter_lists_are_not_contents_entries(tmp_path: Path) -> None:
    good = ["Chapter 1: Introduction", "Chapter 2: Results", "1.1 Context", "Chapter 3: Conclusion", "References"]
    lists = _p("List of Figures", "Heading1") + _p("Figure 2.1. Unused plot\t3") + _p("List of Tables", "Heading1") + _p("Table 1.1. Research questions\t2")
    body = _report(BASE, contents=good).replace(_p("Chapter 1: Introduction", "Heading1"), lists + _p("Chapter 1: Introduction", "Heading1"), 1)
    findings = _checks(tmp_path, body)
    # Every real heading is listed, so any contents-missing finding would be a list heading.
    assert not any(check == "contents-missing" for check, _, _ in findings)


def test_word_heading_numbering_is_rendered(tmp_path: Path) -> None:
    """Headings numbered by a Word multilevel list carry no number in w:t; the audit must
    read them as Word draws them, or every section cross-reference looks dangling."""
    numbering = (f'<w:numbering xmlns:w="{W_NS}"><w:abstractNum w:abstractNumId="0">'
                 '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:suff w:val="space"/><w:lvlText w:val="Chapter %1:"/></w:lvl>'
                 '<w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:suff w:val="space"/><w:lvlText w:val="%1.%2"/></w:lvl>'
                 '</w:abstractNum><w:num w:numId="7"><w:abstractNumId w:val="0"/></w:num></w:numbering>')
    def style_xml(sid: str, name: str) -> str:
        num = ""
        if sid in ("Heading1", "Heading2"):
            lvl = '<w:ilvl w:val="1"/>' if sid == "Heading2" else ""
            num = f'<w:pPr><w:numPr>{lvl}<w:numId w:val="7"/></w:numPr></w:pPr>'
        return f'<w:style w:styleId="{sid}"><w:name w:val="{name}"/>{num}</w:style>'
    styles = "".join(style_xml(sid, name) for sid, name in STYLES.items())
    body = (_p("Introduction", "Heading1") + _p("Context", "Heading2") + _p("Scope", "Heading2")
            + _p("Results", "Heading1") + _p("Details", "Heading2") + _p("See Section 2.1 and Chapter 1.")
            + '<w:p><w:pPr><w:pStyle w:val="Heading1"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr></w:pPr><w:r><w:t>References</w:t></w:r></w:p>')
    path = tmp_path / "numbered.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", f'<?xml version="1.0"?><w:document xmlns:w="{W_NS}"><w:body>{body}</w:body></w:document>')
        archive.writestr("word/styles.xml", f'<?xml version="1.0"?><w:styles xmlns:w="{W_NS}">{styles}</w:styles>')
        archive.writestr("word/numbering.xml", numbering)
    from report_tools.audit_report import extract
    headings = [u.text for u in extract(path)[0] if u.heading_level]
    assert headings == ["Chapter 1: Introduction", "1.1 Context", "1.2 Scope", "Chapter 2: Results", "2.1 Details", "References"]
    audit = run_audit(path, None)
    assert not [f for f in audit.findings if f.check == "xref-dangling"]


def test_style_separator_is_not_hidden_text(tmp_path: Path) -> None:
    separator = ('<w:p><w:pPr><w:pStyle w:val="Caption"/><w:rPr><w:vanish/><w:specVanish/></w:rPr></w:pPr>'
                 '<w:r><w:t>Figure 2.1. Short title.</w:t></w:r></w:p>'
                 '<w:p><w:pPr><w:pStyle w:val="Caption"/></w:pPr><w:r><w:t xml:space="preserve"> Longer description.</w:t></w:r></w:p>')
    hidden = '<w:p><w:r><w:rPr><w:vanish/></w:rPr><w:t>secret</w:t></w:r></w:p>'
    audit = run_audit(_docx(tmp_path / "a.docx", _report(BASE + separator)), None)
    assert audit.inventory["hidden_runs"] == 0 and audit.inventory["style_separators"] == 1
    audit = run_audit(_docx(tmp_path / "b.docx", _report(BASE + hidden)), None)
    assert audit.inventory["hidden_runs"] == 1


def test_residue_comments_and_tracked_changes(tmp_path: Path) -> None:
    body = _report(BASE + _p("TODO: add the final figure."))
    assert any(check == "residue" for check, _ in _fails(_checks(tmp_path, body)))
    # Ordinary lower-case words are not residue.
    assert not _fails(_checks(tmp_path, _report(BASE + _p("The report does not insert new values."))))
    tracked = _report(BASE) + '<w:p><w:ins w:id="1" w:author="x"><w:r><w:t>new text</w:t></w:r></w:ins></w:p>'
    assert any("tracked changes" in message for _, message in _fails(_checks(tmp_path, tracked)))
    comments = {"word/comments.xml": f'<w:comments xmlns:w="{W_NS}"><w:comment w:id="0"><w:p/></w:comment></w:comments>'}
    assert any("comments" in message for _, message in _fails(_checks(tmp_path, _report(BASE), comments)))


def test_coverage_counts_text_outside_extracted_units(tmp_path: Path) -> None:
    hidden = '<w:unknownWrapper><w:r><w:t>orphan text node</w:t></w:r></w:unknownWrapper>'
    fails = _fails(_checks(tmp_path, _report(BASE) + hidden))
    assert any(check == "coverage" for check, _ in fails)
    footnotes = {"word/footnotes.xml": f'<w:footnotes xmlns:w="{W_NS}"><w:footnote w:id="1">'
                 f'<w:p><w:r><w:t>See [9].</w:t></w:r></w:p></w:footnote></w:footnotes>'}
    # Footnote citations are part of the citation scan, not silently skipped.
    assert ("cite-unresolved", "[9] has no bibliography entry") in _fails(_checks(tmp_path, _report(BASE), footnotes))


def test_research_question_trace(tmp_path: Path) -> None:
    body = _report(BASE.replace("RQ1 is answered: the rate stays at 0.25.", "The rate stays at 0.25."))
    assert ("rq-trace", "review", "RQ1 is never answered by label in a conclusion chapter") in _checks(tmp_path, body)


def test_headline_number_absent_from_body(tmp_path: Path) -> None:
    body = _report(BASE.replace("the rate stays at 0.25.", "the rate stays at 0.31."))
    assert any(check == "headline-number" and "0.31" in message for check, _, message in _checks(tmp_path, body))


@pytest.mark.parametrize("sentence, fires", [
    ("An equivalence test placed all ten INT8 comparisons within 0.05 of FP16.", True),
    ("An equivalence test placed all ten INT8 comparisons within 0.05 of FP16; it is post hoc.", False),
    ("An equivalence test placed all ten INT8 comparisons within 0.05; it was added after the INT8 results were known.", False),
])
def test_int8_equivalence_needs_same_paragraph_disclosure(tmp_path: Path, sentence: str, fires: bool) -> None:
    # Unrelated "post-hoc" prose elsewhere must not satisfy the rule (thesis v4 shape).
    body = _report(BASE + _p("All post-hoc scoring writes derived sidecars.") + _p(sentence))
    fired = any(check == "rule:int8-equivalence-post-hoc" for check, _ in _fails(_checks(tmp_path, body)))
    assert fired is fires


def test_direct_request_scope_rule(tmp_path: Path) -> None:
    body = _report(BASE).replace("the direct-request setting", "the standard setting").replace(
        "in the direct-request setting", "")
    assert any(check == "rule:harmbench-direct-request-scope" for check, _ in _fails(_checks(tmp_path, body)))


def test_retired_value_rule_uses_header_scope_and_exempts_intervals(tmp_path: Path) -> None:
    unscoped = _report(BASE + _table([["Pair", "Delta"], ["Qwen3-1.7B", "+0.055"]]))
    assert any(check == "rule:retired-128-token-headline" for check, _ in _fails(_checks(tmp_path, unscoped)))
    scoped = _report(BASE + _table([["Pair", "128 tokens"], ["Qwen3-1.7B", "+0.055"]]))
    assert not any(check == "rule:retired-128-token-headline" for check, _ in _fails(_checks(tmp_path, scoped)))
    interval = _report(BASE + _p("The change is 0.000 [−0.055, +0.055]."))
    assert not any(check == "rule:retired-128-token-headline" for check, _ in _fails(_checks(tmp_path, interval)))


@pytest.mark.parametrize("sentence, fires", [
    ("At 128 tokens, 1,206 of the 2,000 responses were cut off.", True),
    ("At 128 tokens, 1,206 of the 2,000 responses were confirmed truncated, 610 stopped and 184 were unclassified.", False),
    ("At least 60.3% of 128-token responses were truncated by the prefix check.", False),
])
def test_truncation_count_needs_its_unclassified_group(tmp_path: Path, sentence: str, fires: bool) -> None:
    unrelated = _p("Wei et al. attribute such failures to mismatched generalization [1].")
    fired = any(check == "rule:truncation-three-way-classification"
                for check, _ in _fails(_checks(tmp_path, _report(BASE + unrelated + _p(sentence)))))
    assert fired is fires


def test_document_scope_qualifiers_must_be_about_their_subject(tmp_path: Path) -> None:
    # Generic words elsewhere ("an artefact of scoring", "approximately") must not
    # satisfy a disclosure that belongs to a specific result.
    elsewhere = _p("A change may be an artefact of scoring; models have approximately 2 billion parameters.")
    phi = _p("Phi-4-mini over-refusal fell by 0.048 under the keyword rule.")
    mde = _p("The minimum detectable effect is 0.06.")
    fails = {check for check, _ in _fails(_checks(tmp_path, _report(BASE + elsewhere + phi + mde)))}
    assert {"rule:phi-over-refusal-scorer-artefact", "rule:detectable-effect-approximate"} <= fails
    fixed = _report(BASE + _p("Phi-4-mini over-refusal fell by 0.048; the semantic judge does not reproduce it.")
                    + _p("The minimum detectable effect is an approximate guide of 0.06."))
    fails = {check for check, _ in _fails(_checks(tmp_path, fixed))}
    assert not fails & {"rule:phi-over-refusal-scorer-artefact", "rule:detectable-effect-approximate"}


def test_absent_rule_is_reported_as_info(tmp_path: Path) -> None:
    findings = _checks(tmp_path, _report(BASE))
    assert any(check == "rule:jev-judge-integration" and severity == "info" for check, severity, _ in findings)


def test_unregistered_document_is_not_claim_locked(tmp_path: Path) -> None:
    path = _docx(tmp_path / "draft.docx", _report(BASE))
    audit = run_audit(path, None)
    assert any(f.check == "claim-lock" and f.severity == "fail" for f in audit.findings)
    assert any(f.check == "registry" for f in audit.findings)


def test_targets_resolve_by_purpose_not_version() -> None:
    path, entry = resolve_target("final")
    assert path == ROOT / "submission/final_report.docx"
    assert entry["builder"] == "submission/build/build_final_report.py"
    with pytest.raises(ValueError, match="neither a purpose alias"):
        resolve_target("docs/not_a_report.docx")


def test_cli_writes_outputs_and_exit_status(tmp_path: Path) -> None:
    path = _docx(tmp_path / "draft.docx", _report(BASE + _p("TODO")))
    out = tmp_path / "out"
    assert main(["--unregistered", str(path), "--out", str(out), "--quiet"]) == 1
    assert (out / "AUDIT.md").read_text().startswith("# Report audit")
    assert (out / "findings.json").exists()


@pytest.mark.report
def test_final_report_audit_is_fully_covered_and_claim_locked(tmp_path: Path) -> None:
    try:
        path, entry = resolve_target("final")
    except ValueError:
        pytest.skip("no final alias")
    if not path.exists():
        pytest.skip("final report is local-only (D94)")
    audit = run_audit(path, entry)
    assert audit.inventory["text_nodes_uncovered"] == 0
    assert audit.inventory["claim_lock"]["failures"] == 0
    assert not [f for f in audit.findings if f.check.startswith(("cite-", "bib-numbering"))]


# --- Worklist ledger ---------------------------------------------------------


def test_ledger_rows_are_derived_from_the_document(tmp_path: Path) -> None:
    from report_tools.audit_report import build_ledger, extract

    path = _docx(tmp_path / "report.docx", _report(BASE + _p("A result of 0.31 without a citation.")))
    rows = build_ledger(extract(path)[0])
    by_stage = {stage: [r for r in rows if r["stage"] == stage] for stage in {r["stage"] for r in rows}}
    assert len(by_stage["references"]) == 2
    assert len(by_stage["rqs"]) == 1
    assert len(by_stage["floats"]) == 1
    assert [r["markers"] for r in by_stage["citations"]] == ["[1] [2]"]
    # Cross-references and citation markers alone do not make a unit numeric.
    assert not any("Table 1.1 lists" in r["excerpt"] for r in by_stage["numbers"])
    assert any("0.31" in r["excerpt"] for r in by_stage["numbers"])
    assert {"Chapter 1: Introduction", "1.1 Context", "Abstract"} <= {r["excerpt"] for r in by_stage["section"]}


def test_each_audit_gets_a_fresh_ledger_and_summary_demands_evidence(tmp_path: Path) -> None:
    import csv

    path = _docx(tmp_path / "draft.docx", _report(BASE))
    out = tmp_path / "out"
    main(["--unregistered", str(path), "--out", str(out), "--quiet"])
    (ledger,) = out.glob("ledger-*.csv")
    rows = list(csv.DictReader(ledger.open()))
    assert main(["--ledger-summary", str(ledger)]) == 1  # everything unchecked

    for row in rows:
        row["status"], row["evidence"] = "verified", "fixture"
    rows[0]["evidence"] = ""  # verified with no evidence is not accepted
    with ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert main(["--ledger-summary", str(ledger)]) == 1
    rows[0]["evidence"] = "fixture"
    with ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert main(["--ledger-summary", str(ledger)]) == 0

    # A new audit gets its own blank ledger and never touches an earlier one.
    main(["--unregistered", str(path), "--out", str(out), "--quiet"])
    ledgers = sorted(out.glob("ledger-*.csv"))
    assert len(ledgers) == 2
    assert all(r["status"] == "verified" for r in csv.DictReader(ledger.open()))
    (fresh,) = [p for p in ledgers if p != ledger]
    assert all(r["status"] == "" for r in csv.DictReader(fresh.open()))


# --- Skill requirement locks --------------------------------------------------
# 2026-09-27: a condensing rewrite of this skill (D96) first dropped much of what
# made the original precise, and a Codex run then "audited" the final report in
# about two minutes by reusing an earlier audit. The student kept both versions
# (D99): `fyp-report-audit` (staged, fresh-audit rule, worklist ledger) and
# `fyp-report-audit-old` (the original text). Removing or condensing any locked
# requirement in either skill fails.

SKILL_REQUIREMENTS = {
    "fyp-report-audit": (
        'every audit is a fresh, independent audit',
        'thoroughness over speed',
        'prior audits are leads, not evidence',
        'never copy statuses',
        'gathered in this audit',
        'exempts no stage',
        'ledger-summary',
        'exits 0',
        'resolve provenance from live sources',
        'each material sentence',
        'shared source prove consistency',
        'missing required qualifications count as omissions',
        'nonsignificance does not establish equivalence',
        'association does not establish cause',
        'search for counterevidence',
        'citation occurrences, unique cited works',
        'same work under different identifiers',
        'without inventing missing bibliographic fields',
        'several claims',
        'population, experimental setting',
        'direct quotations faithful',
        'locators',
        'related paper or a secondary citation',
        'uncited attributable claims',
        'attribution and adaptation labels',
        'not ai-authorship detection',
        'inflated significance',
        'stock transitions',
        'forced contrasts',
        'synonym changes',
        'chat-style commentary',
        'deliberate imperfections',
        'never claim a full prose review from a word scan',
        'removing filler must not strengthen',
        'verify the rebuilt artifact',
        'a valid bibliography does not prove valid in-text citations',
        'retired claims and their paraphrases',
        'never expose raw harmbench',
        'scorer sidecars must not be overwritten',
    ),
    "fyp-report-audit-old": (
        'read the selected produced document before stating its content',
        'reconcile the inventory with the audit extractor; derive expected coverage independently',
        'count citation occurrences and bibliography entries separately',
        'missing required qualifications count as omissions',
        'disclose samples, inaccessible sources, and uninspected surfaces',
        'never treat regex `pair_interpretations` as judge-primary asr',
        'trace each material sentence, cell, or caption',
        'shared generated values prove consistency, not independent correctness',
        'nonsignificance does not establish equivalence',
        'association does not establish cause',
        'resolve provenance from live sources and recorded decisions',
        'a related paper or secondary citation cannot silently replace the cited original',
        'search for counterevidence and missing qualifications',
        'do not inherit verdicts',
        'record artifact identity',
        'expand grouped markers and numeric ranges',
        'every bibliography entry has an in-text use',
        'same-year suffixes',
        'duplicate works under different identifiers',
        'without inventing missing bibliographic fields',
        'especially when a sentence contains several claims',
        'bibliographic accuracy alone does not establish support',
        'keep direct quotations faithful and check locators',
        'uncited attributable claims',
        'population, experimental setting, and publication status',
        'a valid bibliography does not prove valid in-text citations',
        'not ai-authorship detection',
        'inflated significance',
        'repeated stock transitions',
        'forced contrasts',
        'synonym changes that blur a stable technical term',
        'chat-style commentary',
        'introduce deliberate imperfections',
        'do not claim a full prose review from a word scan',
        'leave clear passages alone',
        'removing filler must not strengthen the scientific claim',
        'a wording fix that needs new evidence is a substantive change',
        'do not call an unavailable source false or treat a green gate as a complete audit',
        'verify the final rebuilt artifact, not only its source',
        'sweep the whole current-facing claim surface for retired claims and their paraphrases',
        'without weakening guards',
    ),
}


@pytest.mark.parametrize("skill", sorted(SKILL_REQUIREMENTS))
def test_skill_keeps_its_audit_requirements(skill: str) -> None:
    text = (ROOT / ".agents/skills" / skill / "SKILL.md").read_text(encoding="utf-8").lower()
    text = " ".join(text.split())
    missing = [phrase for phrase in SKILL_REQUIREMENTS[skill] if phrase not in text]
    assert not missing, f"{skill} lost required instructions: {missing}"
