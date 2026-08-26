"""Check the hand-maintained front-matter page numbers against a real render.

The rewrite builder types its Table of Contents, List of Figures and List of
Tables page numbers by hand. Word repaginates the TOC field on open, but the two
lists are plain paragraphs: nothing corrects them, and nothing has ever checked
them. Any edit that changes body length silently drifts all three.

Microsoft Word is the submission renderer and therefore the pagination
authority. This renders the docx with LibreOffice for the repeatable automated
check, then applies a short, explicit map for the five entries whose Word pages
were verified to differ from LibreOffice on 2026-08-19. It reads the claimed
numbers out of the builder source with ``ast`` so the builder stays the single
source.

Usage::

    python scripts/report_tools/check_pagination.py            # check
    python scripts/report_tools/check_pagination.py --json     # machine-readable

Exit status is 1 when any entry is off, so it can be wired into a gate.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER = REPO_ROOT / "scripts" / "build_rewritten_fyp_report_v1.py"
REPORT = REPO_ROOT / "docs" / "rewritten_fyp_report_v1.docx"

SOFFICE_CANDIDATES = (
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    "soffice",
    "libreoffice",
)

# Verified in Microsoft Word for Mac on 2026-08-19 after updating the TOC field.
# Word keeps 58 physical pages but places these entries one printed page earlier
# than LibreOffice. Lists of figures/tables are plain paragraphs, so Figure 4.2
# needs the same explicit authority as the four cached TOC entries.
WORD_AUTHORITATIVE_PAGES = {
    ("contents", "4.7 Research-Question Synthesis"): 26,
    ("contents", "6.2 Relationship to Prior Work"): 35,
    ("contents", "6.4 Scorer Validity as a Research Result"): 38,
    ("contents", "6.5 Threats to Validity"): 39,
    ("figures", "Figure 4.2. Precision sweep across fp16, INT8, and NF4"): 26,
}



class PaginationError(RuntimeError):
    """Raised when the render or the source parse cannot be completed."""


# --------------------------------------------------------------------------
# claimed numbers, read out of the builder source
# --------------------------------------------------------------------------


def _string_pairs(node: ast.AST) -> list[tuple[str, str]]:
    """Pull (title, page) out of a list of 2- or 3-element string tuples."""
    pairs: list[tuple[str, str]] = []
    if not isinstance(node, (ast.List, ast.Tuple)):
        return pairs
    for element in node.elts:
        if not isinstance(element, ast.Tuple) or len(element.elts) < 2:
            continue
        title, page = element.elts[0], element.elts[1]
        if isinstance(title, ast.Constant) and isinstance(page, ast.Constant):
            if isinstance(title.value, str) and isinstance(page.value, str):
                pairs.append((title.value, page.value))
    return pairs


def claimed_entries(builder: Path = BUILDER) -> dict[str, list[tuple[str, str]]]:
    """Return the three hand-maintained lists as they appear in the builder."""
    tree = ast.parse(builder.read_text(encoding="utf-8"))
    found: dict[str, list[tuple[str, str]]] = {}

    for node in ast.walk(tree):
        # toc_entries = [...]
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "toc_entries" in targets:
                found["contents"] = _string_pairs(node.value)
        # for title, page in [...]:  -- the figure and table lists
        if isinstance(node, ast.For) and isinstance(node.iter, ast.List):
            pairs = _string_pairs(node.iter)
            if not pairs:
                continue
            first = pairs[0][0]
            if first.startswith("Figure"):
                found["figures"] = pairs
            elif first.startswith("Table"):
                found["tables"] = pairs

    missing = {"contents", "figures", "tables"} - set(found)
    if missing:
        raise PaginationError(
            f"could not locate {sorted(missing)} in {builder.name}; "
            "the builder's front-matter lists moved or changed shape"
        )
    return found


# --------------------------------------------------------------------------
# actual pages, read out of a real render
# --------------------------------------------------------------------------


def find_soffice() -> str:
    for candidate in SOFFICE_CANDIDATES:
        resolved = shutil.which(candidate) or (
            candidate if Path(candidate).exists() else None
        )
        if resolved:
            return resolved
    raise PaginationError(
        "LibreOffice not found; install it or pass --pdf with a pre-made render"
    )


def render_pdf(docx: Path, outdir: Path) -> Path:
    soffice = find_soffice()

    def _convert(extra: tuple[str, ...] = ()) -> None:
        subprocess.run(
            [soffice, "--headless", *extra, "--convert-to", "pdf", "--outdir", str(outdir), str(docx)],
            check=True,
            capture_output=True,
            timeout=300,
        )

    try:
        _convert()
    except subprocess.CalledProcessError:
        # A corrupted or locked default LibreOffice profile aborts every
        # conversion with com::sun::star::deployment::DeploymentException
        # (observed 2026-08-25). Retry once with an isolated throwaway
        # profile: same renderer, no shared user state.
        with tempfile.TemporaryDirectory(prefix="lo_profile_") as profile:
            _convert(("-env:UserInstallation=" + Path(profile).as_uri(),))
    pdf = outdir / f"{docx.stem}.pdf"
    if not pdf.exists():
        raise PaginationError(f"LibreOffice produced no PDF for {docx.name}")
    return pdf


def page_texts(pdf: Path) -> list[str]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PaginationError("PyMuPDF (fitz) is required to read the render") from exc
    with fitz.open(pdf) as doc:
        return [page.get_text("text") for page in doc]


def printed_labels(texts: list[str]) -> list[int | None]:
    """The page number each page prints on itself, or None for front matter.

    Body pages carry a running header followed by the page number on its own
    line. Reading that label beats deriving an offset: it stays correct if the
    front matter ever changes length.
    """
    labels: list[int | None] = []
    for text in texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        label = None
        for line in lines[:3]:
            if line.isdigit():
                label = int(line)
                break
        labels.append(label)

    if not any(label is not None for label in labels):
        raise PaginationError(
            "no page carried a printed number; the running header changed shape"
        )
    return labels


LABEL_RE = re.compile(r"((?:Figure|Table)\s+[A-Z0-9]+\.\d+\.)")

# Dropped before measuring caption overlap: they carry no identifying signal and
# would let a stray sentence score highly on "the/of/and" alone.
_STOPWORDS = frozenset(
    "a an and are as at be by for from in into is it its of on or the their this to "
    "with within under over across using used versus vs per".split()
)


def _content_words(text: str) -> list[str]:
    return [
        w for w in re.findall(r"[0-9A-Za-zͰ-Ͽ∆-]+", text.lower())
        if len(w) > 1 and w not in _STOPWORDS
    ]


def entry_probe(kind: str, title: str) -> str:
    """The shortest string that uniquely identifies this entry in the body.

    Figure and table list titles are abbreviated relative to the real captions,
    so those match on the label only. Contents entries reproduce the heading
    text exactly. The label alone is NOT sufficient on its own -- see
    ``caption_words`` and ``locate``.
    """
    if kind in {"figures", "tables"}:
        match = LABEL_RE.match(title)
        if match:
            return match.group(1)
    return title


def caption_words(kind: str, title: str) -> list[str] | None:
    """Content words following the label, used to confirm a line is the caption.

    ``None`` for contents entries, whose probe is already the full heading.
    """
    if kind not in {"figures", "tables"}:
        return None
    match = LABEL_RE.match(title)
    if not match:
        return None
    return _content_words(title[match.end():])


def locate(
    texts: list[str],
    labels: list[int | None],
    probe: str,
    caption_words: list[str] | None = None,
    min_overlap: float = 0.6,
) -> int | None:
    """Printed page of the first numbered page whose line *is* this entry.

    Line-start matching separates a caption from most in-text cross-references:
    "Table A.1. Primary run controls" opens a line, "summarised in Table A.1"
    usually does not. **Usually is not always**, and that was T57: the Chapter
    3-4 flow pass reflowed a sentence ending "in Table A.1." onto a line start,
    and the checker reported Table A.1 on page 18 instead of 46 -- a false
    mismatch with no pagination error behind it. Line-start position is a
    property of the current line breaks, so it cannot carry the whole check.

    The fix strengthens the probe rather than relaxing it (D55/D56): when
    ``caption_words`` is supplied, the line must also repeat enough of the
    caption's own wording. Overlap rather than an exact prefix, because the
    front-matter list is an abbreviation of the caption -- "Table 2.1.
    Positioning of representative prior work" in the list against "Table 2.1.
    Scoped positioning of representative prior work..." on the page -- so an
    exact match would fail on real, correct captions.
    """
    required = 0
    if caption_words:
        required = max(1, math.ceil(min_overlap * len(caption_words)))

    for text, label in zip(texts, labels):
        if label is None:
            continue  # front matter: the contents lists live here themselves
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith(probe):
                continue
            if required:
                rest = set(_content_words(stripped[len(probe):]))
                if sum(1 for w in caption_words if w in rest) < required:
                    continue  # a cross-reference that merely opens a line
            return label
    return None


# --------------------------------------------------------------------------


def claimed_from_document(docx: Path) -> dict[str, list[tuple[str, str]]]:
    """Read the three front-matter lists out of the RENDERED document.

    ``claimed_entries`` parses the builder source, which only works for v1: the
    v3 and v4 builders derive their front matter from v1 and patch individual
    page numbers, so they declare no lists of their own and the ``ast`` walk
    raises. That left the whole v3/v4 front-matter class ungated, and the
    2026-08-23 audit found four wrong entries in v4 as a result -- including a
    List of Tables line, which is a plain paragraph that Word never corrects.

    Reading the document instead of the builder covers every line that actually
    ships, whichever builder produced it.
    """
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PaginationError("python-docx is required to read a document's front matter") from exc

    sections = {
        "Table of Contents": "contents",
        "List of Figures": "figures",
        "List of Tables": "tables",
    }
    found: dict[str, list[tuple[str, str]]] = {"contents": [], "figures": [], "tables": []}
    current: str | None = None
    entry = re.compile(r"^(?P<title>.*?)\t(?P<page>\d+)$")

    for paragraph in Document(docx).paragraphs:
        text = paragraph.text.strip("\n")
        stripped = text.strip()
        if stripped in sections:
            current = sections[stripped]
            continue
        # Stop at the real Chapter 1 heading. It is distinguished from the
        # Table of Contents' own first line -- "Chapter 1: Introduction\t1" --
        # by carrying no tabbed page number.
        if stripped.startswith("Chapter 1:") and "\t" not in text:
            break
        if current is None:
            continue
        match = entry.match(text)
        if match:
            found[current].append((match.group("title").strip(), match.group("page")))

    empty = [kind for kind, entries in found.items() if not entries]
    if empty:
        raise PaginationError(
            f"no front-matter entries found for {empty} in {docx.name}; "
            "the lists moved or changed shape"
        )
    return found


def check(docx: Path = REPORT, pdf: Path | None = None, builder: Path | None = None) -> dict:
    docx = Path(docx).resolve()
    is_v1 = docx == REPORT.resolve()

    # Read the claimed pages from the builder when the builder is the authority
    # for THIS document -- that is v1, whose builder declares the three lists,
    # or an explicitly supplied one. For every other document read the rendered
    # file: v1's lists describe v1's body, and asserting them about a longer
    # document compares two different books.
    if builder is not None:
        claimed = claimed_entries(builder)
    elif is_v1:
        claimed = claimed_entries(BUILDER)
    else:
        claimed = claimed_from_document(docx)

    with tempfile.TemporaryDirectory() as tmp:
        render = pdf or render_pdf(docx, Path(tmp))
        texts = page_texts(render)

    labels = printed_labels(texts)
    numbered = [label for label in labels if label is not None]
    results = []
    for kind, entries in claimed.items():
        for title, page in entries:
            probe = entry_probe(kind, title)
            rendered_actual = locate(texts, labels, probe, caption_words(kind, title))
            # The Word overrides were measured against v1's layout on
            # 2026-08-19. Applying them to a longer document (v3/v4) would
            # assert v1's page numbers about a different body and invent
            # mismatches, so they are scoped to the document they came from.
            override = WORD_AUTHORITATIVE_PAGES.get((kind, title)) if is_v1 else None
            actual = override if override is not None else rendered_actual
            results.append(
                {
                    "list": kind,
                    "title": title,
                    "claimed": int(page),
                    "actual": actual,
                    "rendered_actual": rendered_actual,
                    "authority": "microsoft_word" if override is not None else "libreoffice",
                    "status": (
                        "not_found"
                        if actual is None
                        else "ok"
                        if actual == int(page)
                        else "mismatch"
                    ),
                }
            )

    return {
        "physical_pages": len(texts),
        "front_matter_pages": len(texts) - len(numbered),
        "printed_pages": max(numbered) if numbered else 0,
        "entries": results,
        "mismatches": [r for r in results if r["status"] != "ok"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docx", type=Path, default=REPORT)
    parser.add_argument("--pdf", type=Path, default=None, help="use an existing render")
    parser.add_argument(
        "--builder",
        type=Path,
        default=None,
        help="builder to read claimed pages from; omit to fall back to the document itself",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = check(args.docx, args.pdf, args.builder)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"{report['physical_pages']} physical pages "
            f"({report['front_matter_pages']} front matter + {report['printed_pages']} numbered)"
        )
        for row in report["entries"]:
            if row["status"] == "ok":
                continue
            actual = row["actual"] if row["actual"] is not None else "NOT FOUND"
            print(f"  {row['status']:>9}  [{row['list']}] {row['title']}: says {row['claimed']}, is {actual}")
        total = len(report["entries"])
        bad = len(report["mismatches"])
        print(f"{total} front-matter entries: {total - bad} correct, {bad} wrong")

    return 1 if report["mismatches"] else 0


if __name__ == "__main__":
    sys.exit(main())
