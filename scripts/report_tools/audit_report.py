"""Mechanical first pass of a report audit, for any registered DOCX version.

The target is resolved from ``configs/report_versions.yaml`` by purpose alias
(``final``, ``canonical``, ``submitted``, ``review``) or by registered path, so
nothing here names a report version. The pass reads the rendered DOCX package,
not the builder, and reports what a script can establish:

* coverage: every text node in ``word/document.xml`` is reconciled against the
  units the walker extracted (an independent denominator), and other text
  parts (footnotes, headers, comments, text boxes) are inventoried;
* numeric citations in both directions, first-use order, bibliography
  numbering, duplicate works and preprint entries;
* internal cross-references against captions and headings, float numbering
  and unreferenced floats, and the contents list against the headings;
* front-matter presence, research-question traceability, headline numbers in
  the abstract/conclusion that never appear in the body;
* drafting residue, comments, tracked changes, hidden text and external fields;
* the project omission rules in ``configs/report_audit_rules.yaml``;
* whether the document is claim-locked in ``configs/claim_surfaces.yaml``, and
  the result of those locks when it is.

It also writes a fresh worklist ledger for every run
(``ledger-<hash8>-<timestamp>.csv``): one row per section, numeric unit,
citation-bearing unit, float, reference entry and research question, all blank.
The ledger is the denominator of one audit. Every audit starts from its own
blank ledger and never inherits another audit's statuses; existing ledgers are
never overwritten. ``--ledger-summary`` reports what remains unchecked.

It does not judge source entailment, prose quality, figure content, page
layout or programme formatting rules; ``.agents/skills/fyp-report-audit/SKILL.md``
covers those manual stages for any agent, and ``AUDIT.md`` lists them.

Usage::

    python scripts/report_tools/audit_report.py --target final
    python scripts/report_tools/audit_report.py --target docs/rewritten_fyp_report_v5.docx
    python scripts/report_tools/audit_report.py --unregistered path/to/draft.docx

Outputs go to ``tmp/report_audit/<stem>/`` (gitignored; report text stays
local under D94). Exit status is 1 when any ``fail`` finding exists.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from report_tools.workflows import load_workflows  # noqa: E402
from verify_claim_surfaces import _is_header_row  # noqa: E402

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{W_NS}}}"
NS = {"w": W_NS}
RULES_PATH = ROOT / "configs/report_audit_rules.yaml"
SURFACES_PATH = ROOT / "configs/claim_surfaces.yaml"
EXCERPT = 100


# --------------------------------------------------------------------------
# Model


@dataclass
class Unit:
    index: int
    kind: str          # paragraph | table-row | footnote | endnote
    text: str
    style: str = ""
    zone: str = "body"  # front | abstract | contents | body | references | appendix
    chapter: str = ""
    heading_level: int = 0
    table: int | None = None


@dataclass
class Finding:
    check: str
    severity: str      # fail | review | info
    message: str
    location: str = ""
    excerpt: str = ""


@dataclass
class Audit:
    target: str
    identity: dict
    findings: list[Finding] = field(default_factory=list)
    inventory: dict = field(default_factory=dict)
    units: list[Unit] = field(default_factory=list)

    def add(self, check: str, severity: str, message: str, unit: Unit | None = None, excerpt: str = "") -> None:
        location = ""
        if unit is not None:
            location = f"{unit.kind} {unit.index}" + (f" · {unit.chapter}" if unit.chapter else "")
            excerpt = excerpt or unit.text
        self.findings.append(Finding(check, severity, message, location, _short(excerpt)))


def _short(text: str, limit: int = EXCERPT) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------
# Target resolution


def resolve_target(target: str, root: Path = ROOT) -> tuple[Path, dict]:
    """Purpose alias or registered path -> (path, registry entry)."""
    registry = load_workflows(root)
    path = registry.get("entrypoints", {}).get(target, target)
    entry = registry["owners"].get(path)
    if entry is None:
        standalone = {doc["path"]: doc for doc in registry.get("standalone_documents", [])}
        entry = standalone.get(path)
    if entry is None:
        aliases = ", ".join(registry.get("entrypoints", {}))
        raise ValueError(f"{target!r} is neither a purpose alias ({aliases}) nor a registered document; "
                         "use --unregistered for an unregistered draft")
    return root / path, entry


# --------------------------------------------------------------------------
# Extraction


def _text_of(element: ElementTree.Element, consumed: set[int]) -> str:
    parts: list[str] = []
    for node in element.iter():
        if node.tag == f"{W}t":
            consumed.add(id(node))
            parts.append(node.text or "")
        elif node.tag == f"{W}tab":
            parts.append("\t")
        elif node.tag in (f"{W}br", f"{W}cr"):
            parts.append(" ")
    return "".join(parts)


def _style_names(archive: zipfile.ZipFile) -> dict[str, str]:
    if "word/styles.xml" not in archive.namelist():
        return {}
    root = ElementTree.fromstring(archive.read("word/styles.xml"))
    names = {}
    for style in root.findall("w:style", NS):
        name = style.find("w:name", NS)
        names[style.get(f"{W}styleId", "")] = name.get(f"{W}val", "") if name is not None else ""
    return names


def _numbering(archive: zipfile.ZipFile) -> tuple[dict, dict]:
    """numId -> abstract levels {ilvl: (start, numFmt, lvlText, suff)}, and styleId -> (numId, ilvl)."""
    levels: dict[str, dict[int, tuple[int, str, str, str]]] = {}
    if "word/numbering.xml" in archive.namelist():
        root = ElementTree.fromstring(archive.read("word/numbering.xml"))
        abstract = {}
        for an in root.findall("w:abstractNum", NS):
            lv = {}
            for lvl in an.findall("w:lvl", NS):
                val = lambda name, default="": (lvl.find(f"w:{name}", NS).get(f"{W}val")  # noqa: E731
                                                if lvl.find(f"w:{name}", NS) is not None else default)
                lv[int(lvl.get(f"{W}ilvl"))] = (int(val("start", "1")), val("numFmt", "decimal"),
                                                 val("lvlText"), val("suff", "tab"))
            abstract[an.get(f"{W}abstractNumId")] = lv
        for num in root.findall("w:num", NS):
            ref = num.find("w:abstractNumId", NS)
            if ref is not None:
                levels[num.get(f"{W}numId")] = abstract.get(ref.get(f"{W}val"), {})
    styled: dict[str, tuple[str, int]] = {}
    if "word/styles.xml" in archive.namelist():
        for style in ElementTree.fromstring(archive.read("word/styles.xml")).findall("w:style", NS):
            num = style.find("w:pPr/w:numPr", NS)
            if num is not None and num.find("w:numId", NS) is not None:
                ilvl = num.find("w:ilvl", NS)
                styled[style.get(f"{W}styleId", "")] = (num.find("w:numId", NS).get(f"{W}val"),
                                                         int(ilvl.get(f"{W}val")) if ilvl is not None else 0)
    return levels, styled


def _format_number(value: int, fmt: str) -> str:
    if fmt in {"lowerLetter", "upperLetter"}:
        text = chr(ord("a") + (value - 1) % 26)
        return text.upper() if fmt == "upperLetter" else text
    if fmt in {"lowerRoman", "upperRoman"}:
        pairs = ((1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"), (50, "l"),
                 (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"))
        out = ""
        for size, glyph in pairs:
            while value >= size:
                out, value = out + glyph, value - size
        return out.upper() if fmt == "upperRoman" else out
    return str(value)


def _heading_level(style: str) -> int:
    lowered = style.lower()
    match = re.match(r"heading\s*(\d)", lowered)
    if match:
        return int(match.group(1))
    if lowered in {"title", "toc heading"} or "heading" in lowered:
        return 1
    return 0


_ZONE_HEADINGS = (
    (re.compile(r"^abstract$", re.I), "abstract"),
    (re.compile(r"^(table of )?contents$", re.I), "contents"),
    (re.compile(r"^(references|bibliography|works cited)$", re.I), "references"),
    (re.compile(r"revision history", re.I), "history"),
    (re.compile(r"^appendi(x|ces)\b", re.I), "appendix"),
    (re.compile(r"^list of (figures|tables)\b", re.I), "lists"),
    (re.compile(r"^(acknowledg(e)?ments?|declaration|statement of originality|list of (abbreviations|acronyms))\b", re.I), "front"),
)


def extract(path: Path) -> tuple[list[Unit], dict]:
    with zipfile.ZipFile(path) as archive:
        styles = _style_names(archive)
        num_levels, styled_numbering = _numbering(archive)
        document = ElementTree.fromstring(archive.read("word/document.xml"))
        parts = {name: archive.read(name) for name in archive.namelist()
                 if re.match(r"word/(footnotes|endnotes|comments|header\d*|footer\d*|settings)\.xml$", name)}
        media = [name for name in archive.namelist() if name.startswith("word/media/")]

    consumed: set[int] = set()
    units: list[Unit] = []
    counters: dict[str, dict[int, int]] = {}

    def list_label(paragraph: ElementTree.Element, style_id: str) -> str:
        """The number Word draws before a numbered heading (automatic list numbering is not
        in w:t, so a heading such as "Chapter 4: Results" would otherwise read "Results")."""
        num = paragraph.find("w:pPr/w:numPr", NS)
        num_id, ilvl = styled_numbering.get(style_id, (None, 0))
        if num is not None and num.find("w:numId", NS) is not None:
            num_id = num.find("w:numId", NS).get(f"{W}val")
            ilvl = int(num.find("w:ilvl", NS).get(f"{W}val")) if num.find("w:ilvl", NS) is not None else ilvl
        lv = num_levels.get(num_id or "", {})
        if not num_id or num_id == "0" or ilvl not in lv or lv[ilvl][1] in {"bullet", "none"}:
            return ""
        count = counters.setdefault(num_id, {})
        count[ilvl] = count.get(ilvl, lv[ilvl][0] - 1) + 1
        for deeper in [k for k in count if k > ilvl]:
            del count[deeper]
        label = lv[ilvl][2]
        for k in range(ilvl + 1):
            label = label.replace(f"%{k + 1}", _format_number(count.get(k, lv.get(k, (1,))[0]), lv.get(k, (1, "decimal"))[1]))
        return label + ("" if lv[ilvl][3] == "nothing" else " ")
    body = document.find("w:body", NS)
    table_index = 0

    def walk(container: ElementTree.Element) -> None:
        nonlocal table_index
        for child in container:
            tag = child.tag.removeprefix(W)
            if tag == "p":
                style_el = child.find("w:pPr/w:pStyle", NS)
                style_id = style_el.get(f"{W}val", "") if style_el is not None else ""
                style = styles.get(style_id, style_id)
                text = _text_of(child, consumed).strip()
                level = _heading_level(style)
                if level:
                    text = (list_label(child, style_id) + text).strip()
                if text:
                    units.append(Unit(len(units), "paragraph", text, style, heading_level=level))
            elif tag == "tbl":
                # Same column-label binding as the claim-surface gate: a value in a
                # data row carries its own header-row label ("128 tokens: +0.055").
                table_index += 1
                header: list[str] | None = None
                for row in child.findall(".//w:tr", NS):
                    cells = [_text_of(cell, consumed).strip() for cell in row.findall("w:tc", NS)]
                    if _is_header_row(row, NS):
                        header = header or cells
                    elif header and len(header) == len(cells):
                        cells = [f"{label}: {value}" if label and value and label != value else value
                                 for label, value in zip(header, cells)]
                    if any(cells):
                        units.append(Unit(len(units), "table-row", " | ".join(cells), table=table_index))
            elif tag in {"sdt", "sdtContent", "customXml", "smartTag"}:
                walk(child)

    walk(body)
    all_text_nodes = [node for node in document.iter(f"{W}t")]
    uncovered = [node.text or "" for node in all_text_nodes if id(node) not in consumed and (node.text or "").strip()]

    for kind in ("footnotes", "endnotes"):
        raw = parts.get(f"word/{kind}.xml")
        if raw is None:
            continue
        for note in ElementTree.fromstring(raw):
            if note.get(f"{W}type") in {"separator", "continuationSeparator", "continuationNotice"}:
                continue
            text = _text_of(note, set()).strip()
            if text:
                units.append(Unit(len(units), kind[:-1], text))

    _assign_zones(units)

    xml = ElementTree.tostring(document, encoding="unicode")
    comments = 0
    if "word/comments.xml" in parts:
        comments = len(ElementTree.fromstring(parts["word/comments.xml"]).findall("w:comment", NS))
    settings = parts.get("word/settings.xml", b"").decode("utf-8", "replace")
    instructions = [text.strip() for text in re.findall(r"<(?:\w+:)?instrText[^>]*>([^<]*)<", xml)]
    instructions += re.findall(r'instr="([^"]*)"', xml)
    header_text = sum(len(_text_of(ElementTree.fromstring(raw), set()))
                      for name, raw in parts.items() if re.search(r"(header|footer)\d*\.xml$", name))
    inventory = {
        "text_nodes": len(all_text_nodes),
        "text_nodes_uncovered": len(uncovered),
        "uncovered_sample": [_short(text, 60) for text in uncovered[:5]],
        "paragraph_units": sum(u.kind == "paragraph" for u in units),
        "table_rows": sum(u.kind == "table-row" for u in units),
        "tables": table_index,
        "footnotes": sum(u.kind == "footnote" for u in units),
        "endnotes": sum(u.kind == "endnote" for u in units),
        "text_boxes": xml.count("txbxContent>") // 2,
        "images": len(media),
        "header_footer_chars": header_text,
        "comments": comments,
        "tracked_insertions": len(document.findall(".//w:ins", NS)),
        "tracked_deletions": len(document.findall(".//w:del", NS)),
        # Hidden text is a run-level w:vanish. A vanished paragraph mark with w:specVanish is a
        # Word style separator (Ctrl+Alt+Enter): it hides no content, so it is inventoried apart.
        "hidden_runs": len(document.findall(".//w:r/w:rPr/w:vanish", NS)),
        "style_separators": len(document.findall(".//w:pPr/w:rPr/w:specVanish", NS)),
        "field_instructions": sorted({re.split(r"\s+", text)[0].upper() for text in instructions if text}),
        "external_fields": [_short(text, 80) for text in instructions if re.search(r"https?://", text)],
        "update_fields_on_open": "updateFields" in settings,
    }
    return units, inventory


def _assign_zones(units: list[Unit]) -> None:
    zone, chapter = "front", ""
    seen_chapter = False
    for unit in units:
        if unit.kind != "paragraph":
            unit.zone, unit.chapter = (zone if unit.kind == "table-row" else "notes"), chapter
            continue
        if unit.heading_level:
            label = re.sub(r"\s+", " ", unit.text).strip()
            for pattern, named in _ZONE_HEADINGS:
                if pattern.search(label):
                    zone = named
                    if named in {"references", "appendix", "history"}:
                        chapter = label
                    break
            else:
                if unit.heading_level == 1:
                    zone, chapter, seen_chapter = "body", label, True
        elif zone in {"contents", "lists"} and not re.search(r"\t\s*[\divxlc]+\s*$", unit.text, re.I) and seen_chapter:
            zone = "body"
        unit.zone, unit.chapter = zone, chapter


# --------------------------------------------------------------------------
# Checks


CITE = re.compile(
    r"\[(\d+(?:\s*[-–—]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–—]\s*\d+)?)*)\]"
    r"(?:\s*[-–—]\s*\[(\d+)\])?"
)
AUTHOR_YEAR = re.compile(r"\([A-Z][A-Za-z'’-]+(?: et al\.)?(?:,| and [A-Z][A-Za-z'’-]+,)? (?:19|20)\d{2}[a-z]?\)")


def _expand(group: str, tail: str | None) -> list[int]:
    numbers: list[int] = []
    for part in group.split(","):
        bounds = [int(x) for x in re.split(r"\s*[-–—]\s*", part.strip()) if x]
        numbers.extend(range(bounds[0], bounds[-1] + 1) if len(bounds) == 2 else bounds)
    if tail:
        numbers.extend(range(numbers[-1] + 1, int(tail) + 1))
    return numbers


def check_citations(audit: Audit, units: list[Unit]) -> None:
    entries: dict[int, Unit] = {}
    counter = 0
    for unit in units:
        if unit.zone != "references" or unit.heading_level:
            continue
        match = re.match(r"^\[(\d+)\]", unit.text)
        counter = int(match.group(1)) if match else counter + 1
        if counter in entries:
            audit.add("bib-numbering", "fail", f"bibliography number [{counter}] is used twice", unit)
        entries[counter] = unit

    occurrences: list[tuple[int, Unit]] = []
    for unit in units:
        if unit.zone in {"references", "contents", "lists", "history"} or unit.heading_level:
            continue
        for match in CITE.finditer(unit.text):
            occurrences.extend((number, unit) for number in _expand(match.group(1), match.group(2)))

    surfaces: dict[str, int] = {}
    for _, unit in occurrences:
        key = "caption" if "caption" in unit.style.lower() else unit.kind
        surfaces[key] = surfaces.get(key, 0) + 1
    audit.inventory["citations"] = {
        "occurrences": len(occurrences),
        "unique_cited": len({n for n, _ in occurrences}),
        "bibliography_entries": len(entries),
        "by_surface": surfaces,
    }

    if not occurrences:
        author_year = sum(len(AUTHOR_YEAR.findall(u.text)) for u in units if u.zone != "references")
        if author_year:
            audit.add("citation-style", "info", f"{author_year} author–year markers and no numeric markers: "
                      "marker-to-entry linkage is not checked mechanically for this style")
        return
    if not entries:
        audit.add("bib-missing", "fail", "numeric citations are used but no reference list was found")
        return

    for number in sorted({n for n, _ in occurrences} - set(entries)):
        unit = next(u for n, u in occurrences if n == number)
        audit.add("cite-unresolved", "fail", f"[{number}] has no bibliography entry", unit)
    cited = {n for n, _ in occurrences}
    for number in sorted(set(entries) - cited):
        audit.add("bib-uncited", "review", f"entry [{number}] is never cited in the text (confirm a permitted role)",
                  entries[number])
    missing = sorted(set(range(1, max(entries) + 1)) - set(entries))
    if missing:
        audit.add("bib-numbering", "fail", f"bibliography numbering has gaps: {missing[:10]}")

    first_use: list[int] = []
    for number, _ in occurrences:
        if number not in first_use:
            first_use.append(number)
    out_of_order = [(a, b) for a, b in zip(first_use, first_use[1:]) if b < a]
    if out_of_order:
        a, b = out_of_order[0]
        audit.add("cite-order", "review", f"{len(out_of_order)} first-use inversions (first: [{b}] first cited after [{a}]); "
                  "an issue only if the style numbers by first appearance")

    seen: dict[str, int] = {}
    for number, unit in sorted(entries.items()):
        keys = re.findall(r"arXiv[:\s]*(\d{4}\.\d{4,5})|doi[:\s./]*(10\.\S+)", unit.text, re.I)
        keys = [a or b.rstrip(".") for a, b in keys]
        title = re.search(r"[“\"]([^”\"]{12,})[”\"]", unit.text)
        if title:
            keys.append(re.sub(r"\W+", " ", title.group(1)).strip().lower())
        for key in keys:
            if key in seen and seen[key] != number:
                audit.add("bib-duplicate", "review", f"entries [{seen[key]}] and [{number}] share an identifier or title",
                          unit)
            seen.setdefault(key, number)
    preprints = [n for n, u in entries.items() if re.search(r"arXiv|preprint|bioRxiv|SSRN|OpenReview", u.text, re.I)]
    audit.inventory["citations"]["preprint_like_entries"] = preprints


_LABEL = r"(?:[A-Z](?:\.\d+)+|[A-Z]?\d+(?:\.\d+)*[a-z]?)"
_CAPTION = re.compile(rf"^(Table|Figure|Fig\.)\s+({_LABEL})[.:]?\s", re.I)
_REF_KIND = {"table": "table", "tables": "table", "figure": "figure", "figures": "figure", "fig.": "figure",
             "figs.": "figure", "section": "section", "sections": "section", "§": "section",
             "chapter": "chapter", "chapters": "chapter", "appendix": "appendix", "appendices": "appendix"}
_ID = rf"(?:{_LABEL}|[A-Z](?![a-z.]))"
_XREF = re.compile(
    rf"(?<![\w-])(Tables?|Figures?|Figs?\.|Sections?|§|Chapters?|Appendi(?:x|ces))\s*({_ID}"
    rf"(?:\s*(?:,|and|or|to|–|-|&)\s*{_ID})*)"
)


def _is_caption(unit: Unit, match: re.Match) -> bool:
    """Caption style, or a label followed by '.', ':', a double space or a
    capitalised word; "Figure 1.1 gives an overview" is prose."""
    if "caption" in unit.style.lower():
        return True
    rest = unit.text[match.end(2):]
    return bool(re.match(r"[.:]|\s{2,}|\s+[A-Z]", rest))


def _targets(units: list[Unit]) -> dict[str, dict[str, Unit]]:
    targets: dict[str, dict[str, Unit]] = {"table": {}, "figure": {}, "section": {}, "chapter": {}, "appendix": {}}
    for unit in units:
        if unit.kind != "paragraph" or unit.zone in {"contents", "lists", "front"}:
            continue
        caption = _CAPTION.match(unit.text)
        if caption and not _is_caption(unit, caption):
            caption = None
        if caption and ("caption" in unit.style.lower() or not unit.heading_level):
            kind = "table" if caption.group(1).lower() == "table" else "figure"
            if kind in targets and caption.group(2) in targets[kind]:
                unit.style = unit.style or "caption"
            targets[kind].setdefault(caption.group(2), unit)
            targets[kind].setdefault(f"#dup:{caption.group(2)}:{unit.index}", unit)
            continue
        if unit.heading_level:
            chapter = re.match(r"^Chapter\s+(\d+)\b", unit.text, re.I)
            section = re.match(r"^((?:[A-Z]|\d+)(?:\.\d+)+)\s", unit.text)
            appendix = re.match(r"^Appendix\s+([A-Z0-9]+)\b", unit.text, re.I)
            bare = re.match(r"^(\d+)\s+\S", unit.text)
            if chapter:
                targets["chapter"][chapter.group(1)] = unit
            elif section:
                targets["section"][section.group(1)] = unit
            elif appendix:
                targets["appendix"][appendix.group(1)] = unit
            elif bare and unit.heading_level == 1:
                targets["chapter"][bare.group(1)] = unit
    return targets


def check_cross_references(audit: Audit, units: list[Unit]) -> None:
    targets = _targets(units)
    captions = {kind: {k: u for k, u in found.items() if not k.startswith("#dup:")}
                for kind, found in targets.items()}
    for kind in ("table", "figure"):
        labels = [k.split(":")[1] for k in targets[kind] if k.startswith("#dup:")]
        for label in sorted({x for x in labels if labels.count(x) > 1}):
            audit.add("float-duplicate", "fail", f"{kind} {label} is captioned more than once", captions[kind][label])
        by_chapter: dict[str, list[int]] = {}
        for label in dict.fromkeys(labels):
            if re.fullmatch(r"\d+\.\d+", label):
                chapter, number = label.split(".")
                by_chapter.setdefault(chapter, []).append(int(number))
        for chapter, numbers in by_chapter.items():
            expected = list(range(1, max(numbers) + 1))
            if sorted(numbers) != expected:
                audit.add("float-sequence", "review", f"{kind} numbering in chapter {chapter} is {sorted(numbers)}")
        for label, unit in captions[kind].items():
            chapter_number = re.match(r"^(?:Chapter\s+)?(\d+)\b", unit.chapter or "")
            if chapter_number and re.fullmatch(r"\d+\.\d+", label) and label.split(".")[0] != chapter_number.group(1):
                audit.add("float-chapter", "review", f"{kind} {label} sits in '{unit.chapter}'", unit)

    referenced: dict[str, set[str]] = {kind: set() for kind in targets}
    sections = set(captions["section"]) | set(captions["chapter"])
    for unit in units:
        if unit.zone in {"contents", "lists", "front", "references", "history"} or unit.heading_level:
            continue
        if unit.kind == "paragraph" and "caption" in unit.style.lower():
            continue
        for match in _XREF.finditer(unit.text):
            kind = _REF_KIND[match.group(1).lower()]
            ids = re.findall(_ID, match.group(2))
            dotted = "." in ids[0]
            for ident in ids:
                if dotted != ("." in ident) or (kind != "appendix" and re.fullmatch(r"[A-Z]", ident)):
                    continue
                referenced[kind].add(ident)
                known = {"section": sections, "chapter": set(captions["chapter"]),
                         "appendix": set(captions["appendix"]) | set(captions["section"])}.get(kind, set(captions[kind]))
                if ident in known:
                    continue
                window = unit.text[max(0, match.start() - 40): match.end() + 40]
                external = re.search(r"\[\d+\]|\bet al\b|\bof the (?:paper|original)", window)
                if kind in {"table", "figure"} and captions[kind] and not any(
                        ("." in ident) == ("." in label) for label in captions[kind]):
                    external = True  # numbering scheme this document does not use: likely a cited work's float
                audit.add("xref-dangling", "review" if external else "fail",
                          f"'{match.group(1)} {ident}' has no matching "
                          f"{'heading' if kind in {'section', 'chapter', 'appendix'} else 'caption'}"
                          + (" (near a citation: may refer to a cited work)" if external else ""), unit)
    for kind in ("table", "figure"):
        for label, unit in captions[kind].items():
            if label not in referenced[kind]:
                audit.add("float-unreferenced", "review", f"{kind} {label} is never referred to in the text", unit)
    audit.inventory["floats"] = {kind: len(captions[kind]) for kind in ("table", "figure")}


def check_contents(audit: Audit, units: list[Unit]) -> None:
    entries = [u for u in units if u.zone == "contents" and not u.heading_level]
    # Front-matter lists (List of Figures/Tables) are never contents entries.
    headings = [u for u in units if u.heading_level and u.zone not in {"contents", "front", "abstract", "lists"}]
    if not entries:
        audit.add("contents", "review", "no contents entries were extracted (field without cached text, or no contents)")
        return
    normalise = lambda text: re.sub(r"\s+", " ", re.sub(r"\t.*$", "", text)).strip().lower()  # noqa: E731
    heading_text = {normalise(u.text) for u in headings}
    listed = {normalise(u.text) for u in entries}
    for unit in entries:
        if normalise(unit.text) not in heading_text:
            audit.add("contents-orphan", "fail", "contents entry has no matching heading", unit)
    deepest = max((u.heading_level for u in headings if normalise(u.text) in listed), default=0)
    for unit in headings:
        if unit.heading_level <= deepest and normalise(unit.text) not in listed:
            audit.add("contents-missing", "review", "heading is not listed in the contents", unit)
    captions = {f"{m.group(1).lower()} {m.group(2)}" for u in units if u.zone not in {"contents", "lists", "front"}
                and u.kind == "paragraph" and (m := _CAPTION.match(u.text)) and _is_caption(u, m)}
    listed_floats = set()
    for unit in (u for u in units if u.zone == "lists" and not u.heading_level):
        match = _CAPTION.match(unit.text)
        if not match:
            continue
        label = f"{match.group(1).lower()} {match.group(2)}"
        listed_floats.add(label)
        if label not in captions:
            audit.add("lists-orphan", "fail", "list-of-figures/tables entry has no matching caption", unit)
    if listed_floats:
        for label in sorted(captions - listed_floats):
            kind = label.split()[0]
            if any(entry.startswith(kind) for entry in listed_floats):
                audit.add("lists-missing", "review", f"{label} is captioned but not in its list")
    static = "TOC" not in audit.inventory.get("field_instructions", [])
    audit.add("contents-pages", "info", ("static contents text: page numbers must be re-checked against a Word render"
                                         if static else "contents is a TOC field: update fields in Word before export")
              + " (Word is the pagination authority, D62)")


_FRONT = {
    "abstract": r"^abstract$",
    "contents": r"^(table of )?contents$",
    "acknowledgements": r"^acknowledg(e)?ments?$",
    "declaration / originality": r"declaration|originality|academic integrity",
    "list of figures": r"^list of figures$",
    "list of tables": r"^list of tables$",
    "references": r"^(references|bibliography)$",
}


def check_front_matter(audit: Audit, units: list[Unit]) -> None:
    headings = [u.text.strip() for u in units if u.heading_level or (u.zone == "front" and len(u.text) < 60)]
    present = {name: any(re.search(p, h, re.I) for h in headings) for name, p in _FRONT.items()}
    audit.inventory["front_matter"] = present
    for name, found in present.items():
        if not found:
            audit.add("front-matter", "review", f"no '{name}' section found; check the programme's report template")
    abstract = [u for u in units if u.zone == "abstract" and not u.heading_level]
    audit.inventory["abstract_words"] = sum(len(u.text.split()) for u in abstract)


def check_research_questions(audit: Audit, units: list[Unit]) -> None:
    mentions: dict[str, list[Unit]] = {}
    for unit in units:
        if unit.zone == "contents":
            continue
        for rq in set(re.findall(r"\bRQ\s?(\d+)\b", unit.text)):
            mentions.setdefault(rq, []).append(unit)
    if not mentions:
        audit.add("rq-trace", "info", "no RQ labels found; trace the research questions to answers manually")
        return
    audit.inventory["research_questions"] = {
        f"RQ{rq}": sorted({u.chapter for u in found if u.chapter}) for rq, found in sorted(mentions.items())}
    for rq, found in sorted(mentions.items()):
        chapters = " ".join(u.chapter for u in found).lower()
        if "conclusion" not in chapters:
            audit.add("rq-trace", "review", f"RQ{rq} is never answered by label in a conclusion chapter")
        if "result" not in chapters:
            audit.add("rq-trace", "review", f"RQ{rq} is never named in a results chapter")


_NUMBER = re.compile(r"(?<![\w.])[+\-−]?(\d+\.\d+|\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s?%)")


def check_headline_numbers(audit: Audit, units: list[Unit]) -> None:
    headline = [u for u in units if not u.heading_level and
                (u.zone == "abstract" or (u.zone == "body" and "conclusion" in u.chapter.lower()))]
    pool = " ".join(u.text for u in units if u.zone in {"body", "appendix"} and "conclusion" not in u.chapter.lower())
    pool_numbers = {m.group(1).replace(" ", "") for m in _NUMBER.finditer(pool)}
    for unit in headline:
        for match in _NUMBER.finditer(unit.text):
            token = match.group(1).replace(" ", "")
            if token not in pool_numbers:
                audit.add("headline-number", "review", f"{token} in the {unit.zone if unit.zone == 'abstract' else 'conclusion'} "
                          "does not appear in the body or tables", unit, excerpt=unit.text[max(0, match.start() - 50): match.end() + 30])


# Capitalised markers are matched case-sensitively so ordinary words ("does
# not insert") never fire; phrases that only occur as residue ignore case.
_RESIDUE = re.compile(
    r"\bTODO\b|\bTBD\b|\bFIXME\b|\bXXX\b|\bINSERT\b|\bPLACEHOLDER\b|\?\?|\[\[|\]\]"
    r"|(?i:lorem ipsum|\[citation needed\]|\[[^\]]{0,40}placeholder[^\]]{0,40}\]|Error! (?:Reference source|Bookmark)"
    r"|\bas an AI\b|\bI hope this helps\b)"
)


def check_residue(audit: Audit, units: list[Unit]) -> None:
    for unit in units:
        if unit.zone == "history":
            continue
        match = _RESIDUE.search(unit.text)
        if match:
            audit.add("residue", "fail", f"drafting residue '{match.group(0)}'", unit,
                      excerpt=unit.text[max(0, match.start() - 50): match.end() + 30])
    inv = audit.inventory
    if inv["comments"]:
        audit.add("residue", "fail", f"{inv['comments']} Word comments remain in the package")
    if inv["tracked_insertions"] or inv["tracked_deletions"]:
        audit.add("residue", "fail", f"tracked changes remain ({inv['tracked_insertions']} insertions, "
                  f"{inv['tracked_deletions']} deletions)")
    if inv["hidden_runs"]:
        audit.add("residue", "review", f"{inv['hidden_runs']} hidden-text runs; confirm none carries content")
    for field_text in inv["external_fields"]:
        audit.add("external-field", "review", "field fetches an external URL when fields update", excerpt=field_text)
    if inv["text_nodes_uncovered"]:
        audit.add("coverage", "fail", f"{inv['text_nodes_uncovered']} text nodes are outside the extracted units; "
                  "the audit does not cover them", excerpt="; ".join(inv["uncovered_sample"]))
    if inv["text_boxes"]:
        audit.add("coverage", "info", f"{inv['text_boxes']} text boxes: their text is read inline with the anchoring paragraph")
    if inv["header_footer_chars"]:
        audit.add("coverage", "info", "headers/footers are inventoried, not audited for claims or citations")


def check_rules(audit: Audit, units: list[Unit], rules_path: Path = RULES_PATH) -> None:
    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))["rules"]
    body = [u for u in units if u.zone in {"abstract", "body", "appendix", "notes"} and not u.heading_level]
    document = " ".join(u.text for u in body)
    for rule in rules:
        flags = re.IGNORECASE
        trigger = re.compile(rule["trigger"], flags)
        also = re.compile(rule["also"], flags) if rule.get("also") else None
        strip = re.compile(rule["strip"]) if rule.get("strip") else None
        probe = (lambda text: strip.sub(" ", text)) if strip else (lambda text: text)
        hits = [u for u in body if trigger.search(probe(u.text)) and (also is None or also.search(u.text))]
        if not hits:
            if rule.get("absent"):
                audit.add(f"rule:{rule['id']}", "info", rule["absent"])
            continue
        require = re.compile(rule["require"], flags) if rule.get("require") else None
        if require is None:
            for unit in hits[:15]:
                text = probe(unit.text)
                match = trigger.search(text)
                audit.add(f"rule:{rule['id']}", rule["severity"], f"{rule['message']} [{rule['source']}]", unit,
                          excerpt=text[max(0, match.start() - 40): match.end() + 40])
            if len(hits) > 15:
                audit.add(f"rule:{rule['id']}", "info", f"{len(hits) - 15} further matches not listed")
            continue
        if rule.get("scope") == "document":
            if not require.search(document):
                audit.add(f"rule:{rule['id']}", rule["severity"], f"{rule['message']} [{rule['source']}]", hits[0])
        else:
            for unit in hits:
                if not require.search(unit.text):
                    audit.add(f"rule:{rule['id']}", rule["severity"], f"{rule['message']} [{rule['source']}]", unit)


def check_claim_locks(audit: Audit, rel: str, root: Path = ROOT) -> None:
    manifest = yaml.safe_load((root / "configs/claim_surfaces.yaml").read_text(encoding="utf-8"))
    surface = next((s for s in manifest["surfaces"] if s["path"] == rel), None)
    if surface is None or "docx_primary" not in surface.get("profiles", []):
        audit.add("claim-lock", "fail", "the document is not claim-locked (no docx_primary surface in "
                  "configs/claim_surfaces.yaml): its load-bearing numbers are not machine-checked")
        audit.inventory["claim_lock"] = "unregistered"
        return
    from claim_registry import build_registry
    from verify_claim_surfaces import validate_surface

    results = validate_surface(surface, build_registry(root), root)
    failures = [(name, detail) for name, ok, detail in results if not ok]
    audit.inventory["claim_lock"] = {"surface": surface["id"], "checks": len(results), "failures": len(failures),
                                     "waived": sorted(surface.get("waived_claims", {}))}
    for name, detail in failures:
        audit.add("claim-lock", "fail", f"{name}: {detail}")
    audit.add("claim-lock", "info", f"{len(results) - len(failures)}/{len(results)} registered claim-surface checks pass; "
              "they cover registered properties only, not every number or citation")


# --------------------------------------------------------------------------
# Driver


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_audit(path: Path, entry: dict | None, root: Path = ROOT, rules_path: Path = RULES_PATH) -> Audit:
    rel = path.resolve().relative_to(root.resolve()).as_posix() if path.resolve().is_relative_to(root.resolve()) else str(path)
    identity = {
        "path": rel,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "modified": dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "registry": {k: entry[k] for k in ("status", "builder", "decisions", "label") if entry and k in entry},
        "audited": dt.datetime.now().isoformat(timespec="seconds"),
    }
    builder = (entry or {}).get("builder")
    if builder and (root / builder).is_file():
        identity["builder_sha256"] = _sha256(root / builder)
        identity["builder_newer_than_output"] = (root / builder).stat().st_mtime > path.stat().st_mtime
    try:
        identity["git_head"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        identity["git_head"] = "unavailable"

    units, inventory = extract(path)
    audit = Audit(rel, identity, inventory=inventory, units=units)
    if entry is None:
        audit.add("registry", "review", "document is not registered in configs/report_versions.yaml")
    if identity.get("builder_newer_than_output"):
        audit.add("freshness", "fail", "the registered builder is newer than this output: rebuild before auditing")
    check_residue(audit, units)
    check_front_matter(audit, units)
    check_contents(audit, units)
    check_citations(audit, units)
    check_cross_references(audit, units)
    check_research_questions(audit, units)
    check_headline_numbers(audit, units)
    check_rules(audit, units, rules_path)
    check_claim_locks(audit, rel, root)
    return audit


_ORDER = {"fail": 0, "review": 1, "info": 2}

# --------------------------------------------------------------------------
# Worklist ledger: the denominator of a full audit. One row per item a human
# (or agent) must check, derived from the document, never from a prior audit.

LEDGER_FIELDS = ["id", "stage", "location", "excerpt", "markers", "status", "evidence", "note"]
LEDGER_STATUSES = {"verified", "defect", "gap", "n/a", "sampled"}
LEDGER_STAGES = ("section", "numbers", "citations", "floats", "references", "rqs")
_READ_ZONES = {"abstract", "body", "appendix", "notes"}


def build_ledger(units: list[Unit]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(stage: str, unit: Unit | None, excerpt: str, markers: str = "") -> None:
        location = "" if unit is None else f"{unit.kind} {unit.index}" + (f" · {unit.chapter}" if unit.chapter else "")
        rows.append({"id": f"{stage[:3].upper()}-{sum(r['stage'] == stage for r in rows) + 1:03d}", "stage": stage,
                     "location": location, "excerpt": _short(excerpt, 160), "markers": markers,
                     "status": "", "evidence": "", "note": ""})

    for unit in units:
        if unit.heading_level and unit.zone in _READ_ZONES | {"abstract"}:
            add("section", unit, unit.text)
    for unit in units:
        if unit.zone not in _READ_ZONES or unit.heading_level:
            continue
        probe = re.sub(r"\bRQ\s?\d+\b", " ", _XREF.sub(" ", CITE.sub(" ", unit.text)))
        if unit.kind == "paragraph" and (match := _CAPTION.match(unit.text)) and _is_caption(unit, match):
            probe = probe[match.end():]
        if re.search(r"\d", probe):
            add("numbers", unit, unit.text)
    for unit in units:
        if unit.zone in {"references", "contents", "lists", "history"} or unit.heading_level:
            continue
        markers = [n for m in CITE.finditer(unit.text) for n in _expand(m.group(1), m.group(2))]
        if markers:
            add("citations", unit, unit.text, " ".join(f"[{n}]" for n in markers))
    for unit in units:
        if (unit.kind == "paragraph" and unit.zone not in {"contents", "lists", "front"}
                and (match := _CAPTION.match(unit.text)) and _is_caption(unit, match)):
            add("floats", unit, unit.text)
    for unit in units:
        if unit.zone == "references" and not unit.heading_level:
            add("references", unit, unit.text)
    for rq in sorted({rq for u in units if u.zone != "contents" for rq in re.findall(r"\bRQ\s?(\d+)\b", u.text)}, key=int):
        add("rqs", None, f"RQ{rq}: method -> evidence -> answer in the conclusion")
    return rows


def write_ledger(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarise_ledger(path: Path) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Per-stage status counts, plus problems: blank or unknown statuses, and
    non-verified rows without evidence or a note."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary: dict[str, dict[str, int]] = {}
    problems: list[str] = []
    for row in rows:
        status = row.get("status", "").strip().lower()
        stage = summary.setdefault(row["stage"], {"total": 0})
        stage["total"] += 1
        key = status or "unchecked"
        stage[key] = stage.get(key, 0) + 1
        if status and status not in LEDGER_STATUSES:
            problems.append(f"{row['id']}: unknown status '{status}'")
        elif status and status != "n/a" and not (row.get("evidence", "").strip() or row.get("note", "").strip()):
            problems.append(f"{row['id']}: '{status}' without evidence or note")
    unchecked = sum(stage.get("unchecked", 0) for stage in summary.values())
    if unchecked:
        problems.insert(0, f"{unchecked} rows unchecked")
    return summary, problems


def render_markdown(audit: Audit) -> str:
    counts = {level: sum(f.severity == level for f in audit.findings) for level in _ORDER}
    lines = [
        f"# Report audit — {audit.target}",
        "",
        "Mechanical first pass by `scripts/report_tools/audit_report.py`. A clean run is not a complete audit: "
        "work through the manual stages below (`.agents/skills/fyp-report-audit/SKILL.md`) and record their scope.",
        "",
        "## Artifact identity",
        "",
        *[f"- {key}: `{value}`" for key, value in audit.identity.items()],
        "",
        "## Coverage inventory",
        "",
        *[f"- {key}: {json.dumps(value, ensure_ascii=False)}" for key, value in audit.inventory.items()],
        "",
        f"## Mechanical findings — {counts['fail']} fail, {counts['review']} review, {counts['info']} info",
        "",
        "| Severity | Check | Location | Finding | Excerpt |",
        "|---|---|---|---|---|",
    ]
    for f in sorted(audit.findings, key=lambda f: (_ORDER[f.severity], f.check)):
        cells = [f.severity, f.check, f.location, f.message, f.excerpt]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    lines += [
        "",
        "## Worklist ledger (the denominator of a full audit)",
        "",
        "Fill `status` (verified, defect, gap, n/a, sampled) and `evidence` for every row of the blank ledger "
        "this run wrote; never copy statuses from another ledger or audit. Summarise with `--ledger-summary`. "
        "A full audit reports checked/total per stage from it.",
        "",
        *[f"- {stage}: {sum(r['stage'] == stage for r in build_ledger(audit.units))} rows" for stage in LEDGER_STAGES],
        "",
        "## Manual stages (record scope, evidence and verdict for each)",
        "",
        "- [ ] Numbers: every table cell and numeric sentence traced to `results_512/analysis/` fields and recomputed where inputs allow",
        "- [ ] Citations: each material citation's claim checked against the original passage; identity, version and preprint status",
        "- [ ] Whole-report consistency: abstract ↔ results ↔ discussion ↔ conclusion; RQ → evidence → answer",
        "- [ ] Figures: plotted values, axes, legends and captions match the text and source artifacts",
        "- [ ] Rendered output: Word render, contents/list page numbers, table breaks, figure legibility",
        "- [ ] Programme requirements: template, front matter, declarations, length limits (from the brief, not assumed)",
        "- [ ] Examiner pass: contribution, limitations, threats to validity, reproducibility, ethics of harmful-content handling",
        "- [ ] Prose: connected-paragraph review with safe excerpts and claim-preserving rewrites",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", help="Purpose alias (final, canonical, ...) or registered document path")
    group.add_argument("--unregistered", help="Audit a DOCX that is not in the report registry")
    group.add_argument("--ledger-summary", help="Summarise a filled worklist ledger CSV; exit 1 while rows are unchecked")
    parser.add_argument("--out", help="Output directory (default tmp/report_audit/<stem>/)")
    parser.add_argument("--quiet", action="store_true", help="Print only the summary line")
    args = parser.parse_args(argv)
    if args.ledger_summary:
        summary, problems = summarise_ledger(Path(args.ledger_summary))
        for stage, counts in summary.items():
            detail = ", ".join(f"{k} {v}" for k, v in counts.items() if k != "total")
            print(f"{stage:11} {counts['total']:4} rows: {detail}")
        for problem in problems[:30]:
            print(f"  ! {problem}")
        return 1 if problems else 0
    try:
        if args.target:
            path, entry = resolve_target(args.target)
        else:
            path, entry = Path(args.unregistered).resolve(), None
        if not path.is_file():
            raise ValueError(f"{path} is not present locally (report deliverables are local-only, D94)")
        audit = run_audit(path, entry)
    except (ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"report audit error: {exc}\n")

    out = Path(args.out) if args.out else ROOT / "tmp/report_audit" / path.stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "findings.json").write_text(json.dumps({
        "target": audit.target, "identity": audit.identity, "inventory": audit.inventory,
        "findings": [asdict(f) for f in audit.findings]}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "AUDIT.md").write_text(render_markdown(audit), encoding="utf-8")
    # Every audit is fresh and independent: each run writes a new blank ledger and
    # never touches an existing one, so no audit can inherit another's statuses.
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    ledger = out / f"ledger-{audit.identity['sha256'][:8]}-{stamp}.csv"
    suffix = 1
    while ledger.exists():
        suffix += 1
        ledger = out / f"ledger-{audit.identity['sha256'][:8]}-{stamp}-{suffix}.csv"
    write_ledger(ledger, build_ledger(audit.units))
    print(f"worklist for this audit: {ledger.relative_to(ROOT) if ledger.is_relative_to(ROOT) else ledger}")

    counts = {level: sum(f.severity == level for f in audit.findings) for level in _ORDER}
    if not args.quiet:
        for f in sorted(audit.findings, key=lambda f: (_ORDER[f.severity], f.check)):
            if f.severity != "info":
                print(f"{f.severity.upper():6} {f.check:34} {f.location:28} {f.message}")
    print(f"{audit.target}: {counts['fail']} fail, {counts['review']} review, {counts['info']} info -> "
          f"{(out / 'AUDIT.md').relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
