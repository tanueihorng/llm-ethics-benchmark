"""Report version inventory + structural diff tooling.

Answers, per FYP document version pair: which chapters/sections/paragraphs
were ADDED, REMOVED, or EDITED — at chapter/paragraph granularity, with
removed material called out explicitly.

Registry : configs/report_versions.yaml   (families, chains, statuses)
Outputs  : docs/report_versions/inventory.json        (committed, metadata only)
           docs/report_versions/changes/<edge>.summary.md  (committed, quotes <=90 chars)
           docs/report_versions/changes/<edge>.full.md     (gitignored, full prose)
           docs/report_versions/changes/<edge>.full.json   (gitignored, machine-complete)
           docs/REPORT_VERSIONS.md                          (committed, generated index)

The .full.* files quote report prose wholesale; they are local-only so the
public repository does not republish submission text (D60 rationale).
Usage:
    python scripts/report_tools/report_version_diff.py all       # regenerate everything
    python scripts/report_tools/report_version_diff.py check     # registry<->disk consistency only
    python scripts/report_tools/report_version_diff.py extract <docx> [--json]
    python scripts/report_tools/report_version_diff.py diff <from.docx> <to.docx>
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "configs/report_versions.yaml"
OUTPUT_DIRNAME = Path("docs/report_versions")
INDEX_PATH = ROOT / "docs/REPORT_VERSIONS.md"

QUOTE_CAP = 90          # max chars of prose quoted in COMMITTED summaries
PAIR_RATIO = 0.55       # min similarity to call a replace-pair an edit (vs remove+add)
RENUMBER_RATIO = 0.999  # citation-stripped equality => renumber-only edit
MIN_BLOCKS = 5          # extraction sanity floor
TRANSIENT_RE = re.compile(r"^(~\$|\.~lock\.|#.*#$|.*~$)")


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

@dataclass
class Block:
    kind: str            # 'heading' | 'paragraph' | 'table'
    level: int           # 1..3 for headings, else 0
    style: str
    text: str            # whitespace-normalised (tables: cells joined ' | ')
    region: str          # 'front' | 'main' | 'back'
    chapter: str         # current Heading-1 text ('' in front matter)
    section: str         # current Heading-2 text ('' if none yet)
    index: int = 0       # ordinal among non-empty blocks

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def sha8(self) -> str:
        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()[:8]

    @property
    def loc(self) -> str:
        base = self.chapter or "(front matter)"
        return f"{base}" + (f" § {self.section}" if self.section else "")

    def citeless(self) -> str:
        return _strip_cites(self.text)


_HEADING_RE = re.compile(r"^heading\s*(\d)$", re.I)
_CHAPTER_START_RE = re.compile(r"^chapter\b", re.I)
_BACK_RE = re.compile(r"^(references|appendix)\b", re.I)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_cites(text: str) -> str:
    return re.sub(r"\[\d+(?:\s*[–,-]\s*\d+)*\]", "", text)


def _style_of(paragraph: Paragraph) -> str:
    try:
        style = paragraph.style
    except Exception:  # malformed styles.xml defensively
        return "<none>"
    return style.name if style is not None and style.name else "<none>"


def _table_text(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells = [_norm(c.text) for c in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    return " || ".join(rows)


def iter_body_blocks(doc):
    """Yield Paragraph/Table wrappers in true body order."""
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def extract_blocks(docx_path: Path) -> list[Block]:
    """Ordered content blocks with chapter/section attribution."""
    doc = Document(str(docx_path))
    blocks: list[Block] = []
    region = "front"
    chapter = ""
    section = ""
    idx = 0
    for item in iter_body_blocks(doc):
        if isinstance(item, Paragraph):
            text = _norm(item.text)
            if not text:
                continue
            style = _style_of(item)
            m = _HEADING_RE.match(style or "")
            kind = "heading" if m else "paragraph"
            level = int(m.group(1)) if m else 0
            if kind == "heading":
                if level == 1:
                    if region == "front" and _CHAPTER_START_RE.match(text):
                        region = "main"
                    elif region == "main" and _BACK_RE.match(text):
                        region = "back"
                    chapter = text if region != "front" else chapter
                    section = ""
                elif level == 2:
                    section = text
            elif region == "front" and chapter == "" and _CHAPTER_START_RE.match(text):
                # body paragraph before any heading (defensive)
                region = "main"
            idx += 1
            blocks.append(Block(kind, level, style, text, region,
                                chapter if region != "front" else "", section,
                                idx))
        else:  # Table
            text = _table_text(item)
            if not text:
                continue
            idx += 1
            blocks.append(Block("table", 0, "TABLE", text, region,
                                chapter if region != "front" else "", section, idx))
    return blocks


# --------------------------------------------------------------------------- #
# Diff
# --------------------------------------------------------------------------- #

@dataclass
class Op:
    op: str              # added | removed | edited | renumber_only | renamed_heading
                       # | chapter_added | chapter_removed | section_added | section_removed
    kind: str            # paragraph | table | heading
    location_new: str = ""
    location_old: str = ""
    old_text: str = ""
    new_text: str = ""
    old_sha8: str = ""
    new_sha8: str = ""
    old_words: int = 0
    new_words: int = 0


def _pair_replace(old_span: list[Block], new_span: list[Block]) -> tuple[list[Op], list[Block], list[Block]]:
    """Greedily pair similar blocks across a replace span."""
    scores = []
    for i, o in enumerate(old_span):
        for j, n in enumerate(new_span):
            r = difflib.SequenceMatcher(None, o.text, n.text).ratio()
            if r >= PAIR_RATIO:
                scores.append((r, i, j))
    scores.sort(reverse=True)
    used_o: set[int] = set()
    used_n: set[int] = set()
    pairs: list[tuple[Block, Block, float]] = []
    for r, i, j in scores:
        if i in used_o or j in used_n:
            continue
        used_o.add(i)
        used_n.add(j)
        pairs.append((old_span[i], new_span[j], r))
    pairs.sort(key=lambda t: t[0].index)
    leftovers_o = [o for i, o in enumerate(old_span) if i not in used_o]
    leftovers_n = [n for j, n in enumerate(new_span) if j not in used_n]
    return pairs, leftovers_o, leftovers_n


def diff_blocks(old: list[Block], new: list[Block]) -> list[Op]:
    sm = difflib.SequenceMatcher(
        a=[b.sha8 for b in old], b=[b.sha8 for b in new], autojunk=False
    )
    ops: list[Op] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        ospan, nspan = old[i1:i2], new[j1:j2]
        if tag == "equal":
            continue
        if tag == "delete":
            ops.extend(_op_removed(b) for b in ospan)
        elif tag == "insert":
            ops.extend(_op_added(b) for b in nspan)
        else:  # replace
            pairs, lo, ln = _pair_replace(ospan, nspan)
            for o, n, _r in pairs:
                ops.append(_op_edited(o, n))
            for b in lo:
                ops.append(_op_removed(b))
            for b in ln:
                ops.append(_op_added(b))
    return ops


def _op_added(b: Block) -> Op:
    op = "chapter_added" if b.kind == "heading" and b.level == 1 else (
         "section_added" if b.kind == "heading" else "added")
    return Op(op, b.kind, location_new=b.loc, new_text=b.text,
              new_sha8=b.sha8, new_words=b.words)


def _op_removed(b: Block) -> Op:
    op = "chapter_removed" if b.kind == "heading" and b.level == 1 else (
         "section_removed" if b.kind == "heading" else "removed")
    return Op(op, b.kind, location_old=b.loc, old_text=b.text,
              old_sha8=b.sha8, old_words=b.words)


def _op_edited(o: Block, n: Block) -> Op:
    if o.kind == "heading":
        return Op("renamed_heading", o.kind, n.loc, o.loc, o.text, n.text,
                  o.sha8, n.sha8, o.words, n.words)
    cls = "renumber_only" if o.citeless() == n.citeless() else "edited"
    return Op(cls, o.kind, n.loc, o.loc, o.text, n.text, o.sha8, n.sha8,
              o.words, n.words)


_TOC_ENTRY_RE = re.compile(r"^(chapter\b|appendix\b|references\b|list of\b|\d+(\.\d+)*\s).*\s\d{1,3}$", re.I)


def _is_toc_line(text: str) -> bool:
    """A static-TOC / list-of-figures entry: title text ending in a page number."""
    return bool(_TOC_ENTRY_RE.match(text.strip()))


def _edit_snippet(old: str, new: str, before: int = 55) -> str:
    """Quote the changed window of an edited pair, not its identical prefix."""
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    ops_ = [o for o in sm.get_opcodes() if o[0] != "equal"]
    if not ops_:
        return _cap(old)
    start = min(o[1] for o in ops_)
    end = max(o[2] for o in ops_)
    lo = max(0, start - before)
    hi = min(len(old), end + 25)
    head = ("…" if lo > 0 else "") + old[lo:start]
    tail = old[end:hi] + ("…" if hi < len(old) else "")
    mid = ""
    for tag, i1, i2, j1, j2 in ops_:
        if tag in ("delete", "replace"):
            mid += f"[−{_esc(old[i1:i2])}]"
        if tag in ("insert", "replace"):
            mid += f"[+{_esc(new[j1:j2])}]"
    # Hard cap: many small hunks can balloon the window; committed summaries
    # never carry more than QUOTE_CAP chars of prose (privacy gate, D60/D74).
    return _cap(head + mid + tail)


def summarize_ops(ops: list[Op], old_front: int, new_front: int) -> dict:
    """Aggregate view used by both renderers."""
    out: dict = {
        "totals": {},
        "chapters": {},
        "front": {"added": 0, "removed": 0, "edited": 0, "renumber_only": 0},
        "front_notable_removed": [],
        "front_notable_added": [],
        "chapters_removed": [],
        "chapters_added": [],
        "sections_removed": [],
        "sections_added": [],
        "headings_renamed": [],
        "tables": {"added": 0, "removed": 0, "edited": 0},
        "front_matter_changed": old_front != new_front,
        "front_matter": {"old_blocks": old_front, "new_blocks": new_front},
    }
    for o in ops:
        out["totals"][o.op] = out["totals"].get(o.op, 0) + 1
        is_front = o.location_new.startswith("(front matter)") or \
            o.location_old.startswith("(front matter)")
        if o.op.startswith("chapter_"):
            (out["chapters_added"] if o.op == "chapter_added" else out["chapters_removed"]) \
                .append(o.new_text or o.old_text)
        elif o.op == "section_added":
            out["sections_added"].append(f"{o.location_new} — {o.new_text}")
        elif o.op == "section_removed":
            out["sections_removed"].append(f"{o.location_old} — {o.old_text}")
        elif o.op == "renamed_heading":
            out["headings_renamed"].append({"old": o.old_text, "new": o.new_text})
        if o.kind == "table":
            key = {"added": "added", "removed": "removed"}.get(o.op)
            if key:
                out["tables"][key] += 1
            elif o.op in ("edited", "renumber_only"):
                out["tables"]["edited"] += 1
        if o.op in ("added", "removed", "edited", "renumber_only"):
            if is_front:
                out["front"][o.op] += 1
                text = o.new_text or o.old_text
                if not _is_toc_line(text):
                    target = "front_notable_added" if o.op == "added" else "front_notable_removed"
                    if len(out[target]) < 24:
                        out[target].append(text)
            else:
                chap = (o.location_new or o.location_old).split(" § ")[0] or "(unattributed)"
                c = out["chapters"].setdefault(chap, {"added": 0, "removed": 0,
                                                      "edited": 0, "renumber_only": 0})
                c[o.op] += 1
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _cap(text: str, limit: int = QUOTE_CAP) -> str:
    text = text.replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


_esc = _md_escape


def edge_slug(family_id: str, from_label: str, to_label: str) -> str:
    def s(t: str) -> str:
        t = re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_").lower()
        return t[:40]
    return f"{family_id}__{s(from_label)}_to_{s(to_label)}"


def render_summary_md(edge: dict, stats: dict, ops: list[Op]) -> str:
    L = []
    L.append(f"## {edge['family_label']}: {edge['from_label']} → {edge['to_label']}")
    L.append("")
    L.append(f"*From:* `{edge['from_path']}`  ")
    L.append(f"*To:* `{edge['to_path']}`  ")
    if edge.get("decisions"):
        L.append(f"*Decisions:* {', '.join(edge['decisions'])}  ")
    fm = stats["front_matter"]
    L.append("")
    tot = stats["totals"]
    L.append(
        f"**Paragraph/body blocks:** +{tot.get('added', 0)} added · "
        f"−{tot.get('removed', 0)} removed · ~{tot.get('edited', 0)} edited · "
        f"⟳{tot.get('renumber_only', 0)} citation-renumber-only · "
        f"tables: +{stats['tables']['added']} −{stats['tables']['removed']} "
        f"~{stats['tables']['edited']} · front matter blocks {fm['old_blocks']}→{fm['new_blocks']}"
    )
    fr = stats["front"]
    if fr["added"] or fr["removed"] or fr["edited"]:
        L.append("")
        L.append(f"*Front matter (cover/abstract/TOC/lists):* +{fr['added']} −{fr['removed']} "
                 f"~{fr['edited']} — mostly static-TOC rebuilds; non-TOC items below.")
        for t in stats["front_notable_removed"][:12]:
            L.append(f"  - ❌ {_cap(t, 100)}")
        for t in stats["front_notable_added"][:12]:
            L.append(f"  - ➕ {_cap(t, 100)}")
    L.append("")
    if stats["chapters_removed"]:
        L.append("### Chapters / parts REMOVED")
        for t in stats["chapters_removed"]:
            L.append(f"- ❌ {_cap(t, 120)}")
        L.append("")
    if stats["chapters_added"]:
        L.append("### Chapters / parts ADDED")
        for t in stats["chapters_added"]:
            L.append(f"- ➕ {_cap(t, 120)}")
        L.append("")
    if stats["sections_removed"]:
        L.append("### Sections removed")
        for t in stats["sections_removed"][:40]:
            L.append(f"- ❌ {_cap(t, 120)}")
        if len(stats["sections_removed"]) > 40:
            L.append(f"- …and {len(stats['sections_removed']) - 40} more (see .full.json)")
        L.append("")
    if stats["sections_added"]:
        L.append("### Sections added")
        for t in stats["sections_added"][:40]:
            L.append(f"- ➕ {_cap(t, 120)}")
        if len(stats["sections_added"]) > 40:
            L.append(f"- …and {len(stats['sections_added']) - 40} more (see .full.json)")
        L.append("")
    if stats["chapters"]:
        L.append("### Per-chapter delta (body)")
        L.append("")
        L.append("| Chapter | added | removed | edited | renumber-only |")
        L.append("|---|---|---|---|---|")
        body_chapters = {k: v for k, v in stats["chapters"].items()
                         if k != "(front matter)"}
        for chap in sorted(body_chapters):
            c = body_chapters[chap]
            L.append(f"| {_cap(chap, 60)} | {c['added']} | {c['removed']} | "
                     f"{c['edited']} | {c['renumber_only']} |")
        L.append("")
    removed = [o for o in ops if o.op == "removed"
               and not o.location_old.startswith("(front matter)")]
    if removed:
        L.append(f"### Removed paragraphs ({len(removed)} body; "
                 f"{stats['front'].get('removed', 0)} front matter)")
        L.append("")
        L.append("<details><summary>show</summary>")
        L.append("")
        for o in removed:
            L.append(f"- `{o.old_sha8}` ({o.old_words}w) {o.location_old} — “{_cap(o.old_text)}”")
        L.append("")
        L.append("</details>")
        L.append("")
    added = [o for o in ops if o.op == "added"
             and not o.location_new.startswith("(front matter)")]
    if added:
        L.append(f"### Added paragraphs ({len(added)})")
        L.append("")
        L.append("<details><summary>show</summary>")
        L.append("")
        for o in added:
            L.append(f"- `{o.new_sha8}` ({o.new_words}w) {o.location_new} — “{_cap(o.new_text)}”")
        L.append("")
        L.append("</details>")
        L.append("")
    edited = [o for o in ops if o.op == "edited"]
    if edited:
        L.append(f"### Edited paragraphs ({len(edited)}; change window shown)")
        L.append("")
        L.append("<details><summary>show</summary>")
        L.append("")
        for o in edited[:120]:
            L.append(f"- {o.location_new}: “{_edit_snippet(o.old_text, o.new_text)}”")
        if len(edited) > 120:
            L.append(f"- …and {len(edited) - 120} more (see .full.json)")
        L.append("")
        L.append("</details>")
        L.append("")
    renum = [o for o in ops if o.op == "renumber_only"]
    if renum:
        L.append(f"### Citation-renumber-only paragraphs: {len(renum)} "
                 "(text identical once [n] markers stripped; details in .full.json)")
        L.append("")
    ren_h = stats["headings_renamed"]
    if ren_h:
        L.append(f"### Renamed headings ({len(ren_h)})")
        for h in ren_h[:30]:
            L.append(f"- “{_cap(h['old'], 70)}” → “{_cap(h['new'], 70)}”")
        L.append("")
    L.append("*Full prose detail is in the gitignored `.full.md` / `.full.json` siblings "
             "of this file (regenerate via `make report-versions`).*")
    return "\n".join(L) + "\n"


def render_full_md(edge: dict, stats: dict, ops: list[Op]) -> str:
    L = [f"# FULL DIFF — {edge['family_label']}", "",
         f"FROM: {edge['from_path']}", f"TO:   {edge['to_path']}", ""]
    for title, group in (("REMOVED", [o for o in ops if o.op in ("removed", "section_removed", "chapter_removed")]),
                         ("ADDED", [o for o in ops if o.op in ("added", "section_added", "chapter_added")]),
                         ("EDITED", [o for o in ops if o.op == "edited"]),
                         ("CITATION-RENUMBER-ONLY", [o for o in ops if o.op == "renumber_only"]),
                         ("RENAMED HEADINGS", [o for o in ops if o.op == "renamed_heading"])):
        L.append(f"\n## {title} ({len(group)})\n")
        for o in group:
            L.append(f"### [{o.op}] {o.location_old or o.location_new}")
            if o.old_text:
                L.append(f"OLD: {o.old_text}")
            if o.new_text:
                L.append(f"NEW: {o.new_text}")
            L.append("")
    return "\n".join(L) + "\n"


def ops_to_json(ops: list[Op]) -> list[dict]:
    return [vars(o) for o in ops]


# --------------------------------------------------------------------------- #
# Registry / inventory / checks
# --------------------------------------------------------------------------- #

def load_registry() -> dict:
    with open(REGISTRY_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def rel(p: str) -> Path:
    return ROOT / p


def sha256_12(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def edges_of(registry: dict) -> list[dict]:
    edges = []
    for fam in registry["families"]:
        versions = fam.get("versions", [])
        if fam.get("diff") != "consecutive":
            continue
        for a, b in zip(versions, versions[1:]):
            edges.append({
                "family_id": fam["id"],
                "family_label": fam.get("label", fam["id"]),
                "from": a, "to": b,
                "from_label": a.get("label", a["path"]),
                "to_label": b.get("label", b["path"]),
                "from_path": a["path"], "to_path": b["path"],
                "decisions": sorted(set((a.get("decisions") or []) + (b.get("decisions") or []))),
            })
    return edges


def build_inventory(registry: dict, problems: list[str]) -> dict:
    inv: dict = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generator": "scripts/report_tools/report_version_diff.py",
        "registry": "configs/report_versions.yaml",
        "note": "Rendered docx are gitignored local artifacts (D60); builders are tracked.",
        "documents": [],
        "edges": [],
        "chain_notes": [],
    }
    seen: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for fam in registry["families"]:
        for v in fam.get("versions", []):
            entry = _document_entry(v, fam["id"], seen, problems, aliases=aliases)
            inv["documents"].append(entry)
        for note in fam.get("chain_notes", []):
            inv["chain_notes"].append({"family": fam["id"], "note": " ".join(str(note).split())})
    for v in registry.get("standalone_documents", []):
        entry = _document_entry(v, "(standalone)", seen, problems, light=True, aliases=aliases)
        inv["documents"].append(entry)
    # disk coverage: every docx under docs/ must be registered; exactly once as an
    # owner, plus optionally any number of role-tagged (`alias`) reuses such as a
    # canonical base doubling as the `base` member of a humanized pair.
    disk = [p for p in (ROOT / "docs").rglob("*.docx") if not TRANSIENT_RE.match(p.name)]
    for p in sorted(disk):
        rp = str(p.relative_to(ROOT))
        owner = seen.get(rp) or aliases.get(rp)
        if owner is None:
            problems.append(f"UNREGISTERED docx on disk: {rp}")
        else:
            seen.pop(rp, None)
    for leftover in sorted(seen):
        problems.append(f"REGISTERED but not found on disk: {leftover}")
    return inv


def _structure_summary(path: Path) -> dict:
    blocks = extract_blocks(path)
    h1 = [b.text for b in blocks if b.kind == "heading" and b.level == 1]
    words = sum(b.words for b in blocks)
    return {"blocks": len(blocks), "headings_h1": len(h1),
            "first_chapter": next((t for t in h1 if _CHAPTER_START_RE.match(t)), ""),
            "words": words}


def _document_entry(v: dict, family: str, seen: dict, problems: list[str],
                    light: bool = False, aliases: dict[str, str] | None = None) -> dict:
    path = rel(v["path"])
    entry = {
        "path": v["path"], "family": family,
        "label": v.get("label", ""), "status": v.get("status", ""),
        "builder": v.get("builder", ""), "date": str(v.get("date", "") or ""),
        "decisions": [str(d) for d in v.get("decisions", [])], "alias": v.get("alias", ""),
        "note": " ".join(v.get("note", "").split()),
    }
    if v.get("builder_provenance"):
        entry["builder_provenance"] = v["builder_provenance"]
    if v["path"] in seen and not v.get("alias"):
        problems.append(f"DUPLICATE registration: {v['path']} ({seen[v['path']]} and {family})")
    elif not v.get("alias"):
        seen[v["path"]] = family
    elif aliases is not None:
        aliases.setdefault(v["path"], family)
    if path.exists():
        entry["exists"] = True
        entry["size_bytes"] = path.stat().st_size
        entry["mtime"] = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        entry["sha256_12"] = sha256_12(path)
        if not light:
            s = _structure_summary(path)
            entry["structure"] = s
            if s["headings_h1"] < 1 or s["blocks"] < MIN_BLOCKS:
                problems.append(f"SUSPICIOUS extraction (H1={s['headings_h1']}, "
                                f"blocks={s['blocks']}): {v['path']}")
    else:
        entry["exists"] = False
        problems.append(f"MISSING artifact: {v['path']}")
    return entry


# --------------------------------------------------------------------------- #
# Index (docs/REPORT_VERSIONS.md)
# --------------------------------------------------------------------------- #

STATUS_ICON = {
    "canonical": "⭐ canonical",
    "selected": "✅ selected",
    "current-reference": "📘 current reference",
    "comparison-draft": "🔍 comparison draft",
    "fallback": "🧯 fallback",
    "reference": "📎 reference",
    "archived": "🗄 archived",
}


def render_index(registry: dict, inv: dict) -> str:
    L = []
    L.append("# Report Versions — Complete Map")
    L.append("")
    L.append("> **Generated file** (`make report-versions`); edit "
             "`configs/report_versions.yaml`, not this file.")
    L.append("> Provenance: statuses and decision IDs were transcribed from "
             "`docs/PROJECT_LOG.md` §1/§3. Rendered docx are **gitignored local "
             "artifacts** (D60) — the tracked source of every document is its builder script.")
    L.append("")
    L.append("Every FYP report/thesis/interim version ever built is listed below, "
             "grouped into lineages. Each consecutive pair in a lineage has a change "
             "record under [`docs/report_versions/changes/`](report_versions/changes/) "
             "stating what was added, **removed**, or edited down to individual "
             "chapters, sections and paragraphs.")
    L.append("")
    for fam in registry["families"]:
        L.append(f"## {fam['label']}  (`{fam['id']}`)")
        L.append("")
        L.append(" ".join(str(fam.get("description", "")).split()))
        L.append("")
        L.append("| Version | Status | Date | Builder | Decisions | Artifact |")
        L.append("|---|---|---|---|---|---|")
        for v in fam.get("versions", []):
            doc = next((d for d in inv["documents"] if d["path"] == v["path"]), {})
            icon = STATUS_ICON.get(v.get("status", ""), v.get("status", ""))
            builder = v.get("builder", "—")
            prov = " *(name-inferred)*" if v.get("builder_provenance") == "name-inferred" else ""
            art = f"✅ {doc.get('size_bytes', '?'):,} B" if doc.get("exists") else "❌ missing"
            dec = ", ".join(v.get("decisions", [])) or "—"
            note = v.get("note", "")
            label = v.get("label", "")
            if note:
                label += f" — {' '.join(note.split())}"
            L.append(f"| **{label}** | {icon} | {v.get('date', '—')} "
                     f"| `{builder}`{prov} | {dec} | {art} |")
        L.append("")
        for note in fam.get("chain_notes", []):
            L.append(f"> ⚠️ Chain note: {' '.join(str(note).split())}")
            L.append("")
        changes = [e for e in inv.get("edges", []) if e["family_id"] == fam["id"]]
        if changes:
            L.append("**Change records:**")
            L.append("")
            for e in changes:
                slug = e["slug"]
                L.append(f"- [`{e['from_label']}` → `{e['to_label']}`]"
                         f"(report_versions/changes/{slug}.summary.md) — "
                         f"+{e['counts'].get('added', 0)} / −{e['counts'].get('removed', 0)} / "
                         f"~{e['counts'].get('edited', 0)}¶")
            L.append("")
    if inv.get("chain_notes"):
        pass
    L.append("## Standalone documents (inventoried, no version chain)")
    L.append("")
    L.append("| Document | Note |")
    L.append("|---|---|")
    for d in inv["documents"]:
        if d["family"] == "(standalone)":
            L.append(f"| `{d['path']}` | {d.get('note','')} |")
    L.append("")
    L.append("## How to read / regenerate")
    L.append("")
    L.append("```bash")
    L.append("make report-versions                    # regenerate this map, the inventory and all change records")
    L.append("python scripts/report_tools/report_version_diff.py extract docs/FYP_Report_2026-07-01_v5.docx")
    L.append("python scripts/report_tools/report_version_diff.py diff <old.docx> <new.docx>")
    L.append("```")
    L.append("")
    L.append("- `changes/<edge>.summary.md` — committed overview; removed/added/edited "
             f"paragraph lists with ≤{QUOTE_CAP}-char quotes and content hashes.")
    L.append("- `changes/<edge>.full.md` / `.full.json` — **gitignored local files** with "
             "complete prose (kept out of git so the public repo never republishes "
             "submission text, same rationale as D60).")
    L.append("- Paragraph identity: `sha8` is the first 8 hex of SHA-1 over the "
             "whitespace-normalised paragraph text; `NNw` is word count.")
    L.append("- Consistency between this map, the YAML registry and the artifacts on "
             "disk is enforced by `tests/test_report_versions.py`.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_all(verbose: bool = True) -> tuple[dict, list[str]]:
    registry = load_registry()
    problems: list[str] = []
    out_dir = ROOT / OUTPUT_DIRNAME
    chg_dir = out_dir / "changes"
    chg_dir.mkdir(parents=True, exist_ok=True)

    inv = build_inventory(registry, problems)

    # stale-output hygiene: remove outputs whose edge no longer exists
    live_slugs = set()
    for edge in edges_of(registry):
        a, b = edge["from"], edge["to"]
        fa, fb = rel(a["path"]), rel(b["path"])
        if not (fa.exists() and fb.exists()):
            problems.append(f"EDGE SKIPPED (missing artifact): {edge['from_path']} → {edge['to_path']}")
            continue
        slug = edge_slug(edge["family_id"], edge["from_label"], edge["to_label"])
        live_slugs.add(slug)
        old_blocks = extract_blocks(fa)
        new_blocks = extract_blocks(fb)
        ops = diff_blocks(old_blocks, new_blocks)
        old_front = sum(1 for x in old_blocks if x.region == "front")
        new_front = sum(1 for x in new_blocks if x.region == "front")
        stats = summarize_ops(ops, old_front, new_front)
        counts = {k: stats["totals"].get(k, 0)
                  for k in ("added", "removed", "edited", "renumber_only")}
        edge["slug"] = slug
        edge["counts"] = counts
        inv["edges"].append({k: edge[k] for k in
                             ("family_id", "family_label", "from_label", "to_label",
                              "from_path", "to_path", "decisions", "slug", "counts")})
        (chg_dir / f"{slug}.summary.md").write_text(
            render_summary_md(edge, stats, ops), encoding="utf-8")
        (chg_dir / f"{slug}.full.md").write_text(
            render_full_md(edge, stats, ops), encoding="utf-8")
        (chg_dir / f"{slug}.full.json").write_text(
            json.dumps({"edge": {k: edge[k] for k in ("family_id", "from_path", "to_path",
                                                      "from_label", "to_label")},
                        "stats": stats, "ops": ops_to_json(ops)},
                       indent=1, ensure_ascii=False), encoding="utf-8")
        if verbose:
            print(f"  edge {slug}: +{counts['added']} -{counts['removed']} "
                  f"~{counts['edited']} renum {counts['renumber_only']}")

    for stale in sorted(chg_dir.glob("*")):
        base = re.sub(r"\.(summary\.md|full\.md|full\.json)$", "", stale.name)
        if base not in live_slugs:
            problems.append(f"STALE output for unknown edge: {stale.name}")

    (out_dir / "inventory.json").write_text(
        json.dumps(inv, indent=1, ensure_ascii=False), encoding="utf-8")
    INDEX_PATH.write_text(render_index(registry, inv), encoding="utf-8")
    return inv, problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("all", help="regenerate inventory, change records and index")
    sub.add_parser("check", help="registry↔disk consistency only (no writes)")
    p_ext = sub.add_parser("extract", help="dump structural blocks of one docx")
    p_ext.add_argument("docx")
    p_ext.add_argument("--json", action="store_true")
    p_dif = sub.add_parser("diff", help="ad-hoc diff of two docx (prints summary)")
    p_dif.add_argument("frm")
    p_dif.add_argument("to")
    args = ap.parse_args(argv)

    if args.cmd in ("all", "check", None):
        if args.cmd in ("check", None):
            registry = load_registry()
            problems: list[str] = []
            build_inventory(registry, problems)
            for p in problems:
                print("PROBLEM:", p)
            print(f"check: {len(problems)} problem(s)")
            return 1 if problems else 0
        inv, problems = run_all()
        for p in problems:
            print("PROBLEM:", p)
        print(f"all: {len(inv['documents'])} documents, {len(inv['edges'])} change records, "
              f"{len(problems)} problem(s)")
        return 1 if problems else 0

    if args.cmd == "extract":
        blocks = extract_blocks(Path(args.docx))
        if args.json:
            print(json.dumps([{"kind": b.kind, "level": b.level, "style": b.style,
                               "region": b.region, "chapter": b.chapter,
                               "section": b.section, "text": b.text}
                              for b in blocks], indent=1, ensure_ascii=False))
        else:
            for b in blocks:
                marker = "#" * b.level if b.kind == "heading" else (
                    "T" if b.kind == "table" else "·")
                print(f"{b.index:>4} {marker:<4} [{b.region[:1]}] {b.loc[:60]} :: {b.text[:100]}")
        return 0

    if args.cmd == "diff":
        old = extract_blocks(Path(args.frm))
        new = extract_blocks(Path(args.to))
        ops = diff_blocks(old, new)
        stats = summarize_ops(ops,
                              sum(1 for x in old if x.region == "front"),
                              sum(1 for x in new if x.region == "front"))
        edge = {"family_label": f"{args.frm} → {args.to}", "from_label": args.frm,
                "to_label": args.to, "from_path": args.frm, "to_path": args.to,
                "decisions": []}
        print(render_summary_md(edge, stats, ops))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
