"""Check the hand-maintained front-matter page numbers against a real render.

The rewrite builder types its Table of Contents, List of Figures and List of
Tables page numbers by hand. Word repaginates the TOC field on open, but the two
lists are plain paragraphs: nothing corrects them, and nothing has ever checked
them. Any edit that changes body length silently drifts all three.

This renders the docx with LibreOffice, finds where each entry actually lands,
and reports the mismatches. It reads the claimed numbers out of the builder
source with ``ast`` so the builder stays the single source.

Usage::

    python scripts/report_tools/check_pagination.py            # check
    python scripts/report_tools/check_pagination.py --json     # machine-readable

Exit status is 1 when any entry is off, so it can be wired into a gate.
"""

from __future__ import annotations

import argparse
import ast
import json
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
    subprocess.run(
        [find_soffice(), "--headless", "--convert-to", "pdf", "--outdir", str(outdir), str(docx)],
        check=True,
        capture_output=True,
        timeout=300,
    )
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


def entry_probe(kind: str, title: str) -> str:
    """The shortest string that uniquely identifies this entry in the body.

    Figure and table list titles are abbreviated relative to the real captions,
    so those match on the label only. Contents entries reproduce the heading
    text exactly.
    """
    if kind in {"figures", "tables"}:
        match = re.match(r"((?:Figure|Table)\s+[A-Z0-9]+\.\d+\.)", title)
        if match:
            return match.group(1)
    return title


def locate(texts: list[str], labels: list[int | None], probe: str) -> int | None:
    """Printed page of the first numbered page with a line *starting* with ``probe``.

    Line-start matching is what separates a caption or heading from an in-text
    cross-reference: "Table A.1. Primary run controls" opens a line, while
    "summarised in Table A.1" does not.
    """
    for text, label in zip(texts, labels):
        if label is None:
            continue  # front matter: the contents lists live here themselves
        for line in text.splitlines():
            if line.strip().startswith(probe):
                return label
    return None


# --------------------------------------------------------------------------


def check(docx: Path = REPORT, pdf: Path | None = None) -> dict:
    claimed = claimed_entries()

    with tempfile.TemporaryDirectory() as tmp:
        render = pdf or render_pdf(docx, Path(tmp))
        texts = page_texts(render)

    labels = printed_labels(texts)
    numbered = [label for label in labels if label is not None]
    results = []
    for kind, entries in claimed.items():
        for title, page in entries:
            probe = entry_probe(kind, title)
            actual = locate(texts, labels, probe)
            results.append(
                {
                    "list": kind,
                    "title": title,
                    "claimed": int(page),
                    "actual": actual,
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
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = check(args.docx, args.pdf)

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
