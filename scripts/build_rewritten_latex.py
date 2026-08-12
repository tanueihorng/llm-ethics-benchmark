"""Overleaf/LaTeX mirror of the rewrite line, generated from its Python builder.

The two hand-maintained mirrors under ``fyp_submission/report_latex/`` mirror the
*older* document line: ``final_thesis/`` tracks ``build_fyp_thesis_v4.js`` and
``interim_report/`` tracks ``build_fyp_interim.js``. Neither tracks the rewrite,
which is the submission candidate (T52/D59), so the LaTeX surface has been
mirroring documents that are no longer the ones being submitted.

This builder closes that gap **without adding a fifth hand-maintained copy of the
prose**. Hand-maintenance is the documented failure mode here: T44 FS-16 found four
independent tex defects living in exactly the hole where a mirror can drift from
its source with no gate firing. So the prose is not copied — it is *captured*.
``build_rewritten_fyp_report_v1.py`` is imported unchanged, its content helpers are
swapped for recorders with the same signatures, and the chapter functions are run
against a stub document. Whatever the docx says, the tex says, by construction.

That is the construction ``build_rewritten_fyp_interim_v1.py`` uses for the interim
docx and ``build_viva_deck_html.py`` uses for the web deck, and for the same reason
(D56): two copies drift, and a correction applied to one is silently absent from
the other.

Two trees are written, differing only in how the document names itself:

  fyp_submission/report_latex/rewritten_final/     <- 19 Oct submission (T15)
  fyp_submission/report_latex/rewritten_interim/   <- 24 Aug submission (T53)

plus an Overleaf zip beside each. The template assets (``Thesis.cls``, ``Styles/``)
are copied from the existing ``interim_report/`` mirror so both lines compile
against the same NTU class file.

    python scripts/build_rewritten_latex.py
    python scripts/build_rewritten_latex.py --compile   # verify with tectonic
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_BUILDER = ROOT / "scripts" / "build_rewritten_fyp_report_v1.py"
LATEX_ROOT = ROOT / "fyp_submission" / "report_latex"
# Styles/ come from the interim mirror; the CLASS comes from the thesis mirror,
# because the interim one hard-codes "INTERIM REPORT" into its title page.
TEMPLATE_SOURCE = LATEX_ROOT / "interim_report"
CLASS_SOURCE = LATEX_ROOT / "final_thesis" / "Thesis.cls"
FINAL_DIR = LATEX_ROOT / "rewritten_final"
INTERIM_DIR = LATEX_ROOT / "rewritten_interim"

# How the document names itself. In LaTeX this lives on the title page, which the
# CLASS file draws -- not the .tex. A guard that reads only mythesis.tex would have
# passed while the final tree rendered a title page reading "INTERIM REPORT"; that
# is exactly what happened on the first build here.
FINAL_COVER = "FINAL YEAR PROJECT REPORT"
INTERIM_COVER = "FINAL YEAR PROJECT --- INTERIM REPORT"
FINAL_SUBMISSION_LINE = (
    "\\large A final year project report submitted to the Nanyang Technological "
    "University \\\\ in partial fulfilment of the requirements for the degree of\\\\ "
    "\\degreename\\\\[1cm]"
)
INTERIM_SUBMISSION_LINE = (
    "\\large An interim report submitted to the Nanyang Technological "
    "University \\\\ in partial fulfilment of a Final Year Project for the degree of\\\\ "
    "\\degreename\\\\[1cm]"
)

# Anchors in Thesis.cls's \maketitle. Each must match exactly once in the head
# half of the file, or the build fails rather than shipping an unpatched cover.
ANCHOR_AUTHOR = "  \\textbf{\\Large \\authornames}\\\\%[1.2cm]"
ANCHOR_SUBMISSION = (
    "  \\large A thesis submitted to the Nanyang Technological University "
    "\\\\ in partial fulfillment of the requirements for the degree of\\\\ "
    "\\degreename\\\\[1cm] % University requirement text"
)
SPLIT_MARKER = "\\newcommand\\maketitleforreview"

TITLE = "Benchmarking Ethical Performance of Open-Source LLMs"
SUBTITLE = (
    "A Matched-Pair Study of Safety and Capability in Quantized Compact Language Models"
)

# ---------------------------------------------------------------------------
# LaTeX escaping
# ---------------------------------------------------------------------------
# The non-ASCII set is closed and was measured from the rendered docx rather than
# guessed: 13 distinct codepoints. `unknown_non_ascii` below fails the build if a
# future edit introduces a 14th, so this map cannot silently fall behind the prose.
UNICODE_MAP = {
    "\u2212": "$-$",          # MINUS SIGN (91x) - numeric minus, not a hyphen
    "\u201c": "``",           # LEFT DOUBLE QUOTATION MARK
    "\u201d": "''",           # RIGHT DOUBLE QUOTATION MARK
    "\u2013": "--",           # EN DASH
    "\u2192": "$\\rightarrow$",
    "\u0394": "$\\Delta$",
    "\u00b1": "$\\pm$",
    "\u00dc": '\\"{U}',
    "\u00fc": '\\"{u}',
    "\u00f6": '\\"{o}',
    "\u00ed": "\\'{i}",
    "\u03ba": "$\\kappa$",
    "\u03c4": "$\\tau$",
    # Present in builder source though not in the current render; mapped so an
    # edit that reintroduces them does not trip the unknown-character guard.
    "\u2014": "---",
    "\u00b7": "$\\cdot$",
    "\u2248": "$\\approx$",
    "\u2264": "$\\leq$",
    "\u2265": "$\\geq$",
    "\u00d7": "$\\times$",
    "\u2018": "`",
    "\u2019": "'",
    "\u03b1": "$\\alpha$",
    "\u03c3": "$\\sigma$",
    "\u2026": "\\ldots",
    "\u00a0": "~",
}

# LaTeX specials, longest-first so the backslash rule cannot eat its own output.
SPECIALS = [
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
]

# Placeholder so unicode replacements (which legitimately contain $ and \) are not
# re-escaped by the specials pass.
_SENTINEL = "\x00{}\x00"


def tex_escape(text: str) -> str:
    """Escape body text for LaTeX, unicode-aware and idempotent within one call."""
    substitutions: list[str] = []

    def stash(replacement: str) -> str:
        substitutions.append(replacement)
        return _SENTINEL.format(len(substitutions) - 1)

    out = "".join(stash(UNICODE_MAP[ch]) if ch in UNICODE_MAP else ch for ch in text)
    for raw, escaped in SPECIALS:
        out = out.replace(raw, escaped)
    for index, replacement in enumerate(substitutions):
        out = out.replace(_SENTINEL.format(index), replacement)
    return out


def unknown_non_ascii(text: str) -> set[str]:
    """Non-ASCII characters with no mapping. Non-empty means the map is stale."""
    return {ch for ch in text if ord(ch) > 126 and ch not in UNICODE_MAP}


# ---------------------------------------------------------------------------
# Recording stubs
# ---------------------------------------------------------------------------
class _Format:
    """Absorbs paragraph_format mutations the builder performs for docx layout."""

    def __init__(self) -> None:
        self.tab_stops = _TabStops()

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)


class _TabStops:
    def add_tab_stop(self, *args, **kwargs):
        return None


class _Run:
    def __init__(self, event: dict, text: str) -> None:
        self._event = event
        self.font = _Font()
        self.bold = False
        self.italic = False
        event["text"] += text

    def add_picture(self, *args, **kwargs):
        return None


class _Font:
    def __init__(self) -> None:
        self.name = None
        self.size = None


class _Paragraph:
    """Registers itself in the event stream immediately, so order is preserved."""

    def __init__(self, events: list[dict], text: str, style) -> None:
        self._events = events
        self._event = {"kind": "para", "style": _style_name(style), "text": text}
        events.append(self._event)
        self.paragraph_format = _Format()
        self.alignment = None
        self.style = style

    def add_run(self, text: str = "") -> _Run:
        return _Run(self._event, text)


def _style_name(style) -> str:
    if style is None:
        return ""
    return style if isinstance(style, str) else getattr(style, "name", "")


class _Section:
    def __init__(self) -> None:
        self.footer = _Container()
        self.header = _Container()


class _Container:
    def __init__(self) -> None:
        self.paragraphs = [_Paragraph([], "", None)]
        self.is_linked_to_previous = False

    def add_paragraph(self, text: str = "", style=None):
        return _Paragraph([], text, style)


class RecorderDoc:
    """Stands in for python-docx's Document during a capture run."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.sections = [_Section()]

    # -- calls the builder makes directly on the document --------------------
    def add_paragraph(self, text: str = "", style=None) -> _Paragraph:
        return _Paragraph(self.events, text, style)

    def add_heading(self, text: str = "", level: int = 1) -> _Paragraph:
        self.events.append({"kind": "raw_heading", "text": text, "level": level})
        return _Paragraph([], "", None)

    def add_page_break(self) -> None:
        return None

    def add_section(self, *args, **kwargs) -> _Section:
        return _Section()

    def add_table(self, rows: int = 1, cols: int = 1):  # pragma: no cover - patched away
        raise AssertionError("add_table should be intercepted at the helper level")


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
def load_report_builder():
    spec = importlib.util.spec_from_file_location(
        "build_rewritten_fyp_report_v1", REPORT_BUILDER
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def capture(rpt) -> list[dict]:
    """Run the report builder's content functions against recording helpers.

    Only the *emitting* helpers are swapped. Every function that computes a number
    from the claim registry runs untouched, so the tex carries the same values the
    docx does -- they are produced by the same code path, not transcribed.
    """
    doc = RecorderDoc()
    events = doc.events
    originals = {}

    def patch(name, fn):
        originals[name] = getattr(rpt, name)
        setattr(rpt, name, fn)

    patch("add_body", lambda _doc, text, **kw: events.append({"kind": "body", "text": text}))
    patch(
        "add_bullets",
        lambda _doc, items: events.append({"kind": "bullets", "items": list(items)}),
    )
    patch(
        "add_chapter",
        lambda _doc, number, title: events.append(
            {"kind": "chapter", "number": number, "title": title}
        ),
    )
    patch(
        "add_heading",
        lambda _doc, number, title, level=2: events.append(
            {"kind": "heading", "number": number, "title": title, "level": level}
        ),
    )
    patch(
        "add_table",
        lambda _doc, number, title, headers, rows, widths_cm, **kw: events.append(
            {
                "kind": "table",
                "number": number,
                "title": title,
                "headers": list(headers),
                "rows": [list(r) for r in rows],
                "caption": kw.get("show_caption", True),
            }
        ),
    )
    patch(
        "add_figure",
        lambda _doc, number, title, filename, width_cm=14.4: events.append(
            {
                "kind": "figure",
                "number": number,
                "title": title,
                "filename": filename,
                "width_cm": width_cm,
            }
        ),
    )
    # The typed contents/figure/table lists are a docx workaround; LaTeX builds
    # them from the document itself, so they are dropped rather than transcribed.
    patch("add_contents_entry", lambda *a, **kw: None)
    patch("add_table_caption", lambda *a, **kw: None)
    patch("add_figure_caption", lambda *a, **kw: None)
    # Word field plumbing (the TOC field, page-number fields). These manipulate
    # raw docx XML and have no LaTeX counterpart -- \tableofcontents replaces them.
    patch("add_field", lambda *a, **kw: None)
    patch("begin_multiline_field", lambda *a, **kw: None)
    patch("end_multiline_field", lambda *a, **kw: None)

    try:
        # add_cover is deliberately skipped: \maketitle owns the title page.
        rpt.add_front_matter(doc)
        rpt.add_chapter_one(doc)
        rpt.add_chapter_two(doc)
        rpt.add_chapter_three(doc)
        rpt.add_chapter_four(doc)
        rpt.add_chapter_five(doc)
        rpt.add_chapter_six(doc)
        rpt.add_chapter_seven(doc)
        rpt.add_references(doc)
        rpt.add_appendices(doc)
    finally:
        for name, fn in originals.items():
            setattr(rpt, name, fn)

    return [e for e in events if e.get("kind") != "para" or e["text"].strip()]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
PREAMBLE = r"""%% !TEX program = pdflatex
%% ============================================================================
%% {cover_upper}  --  CCDS25-1136  --  Tan Uei Horng (UTAN001)
%% NTU thesis template (wang-chen/thesis_template_ntu).
%%
%% GENERATED FILE -- do not edit by hand.
%%   source : scripts/build_rewritten_fyp_report_v1.py (prose, tables, numbers)
%%   builder: scripts/build_rewritten_latex.py
%% The prose is captured from the docx builder, not copied, so this mirror cannot
%% drift from the submitted document. Regenerate with:
%%   python scripts/build_rewritten_latex.py
%%
%% Compile: tectonic mythesis.tex   (or latexmk -pdf on Overleaf)
%% ============================================================================
%% Two deliberate departures from the template defaults, both driven by the school
%% page band (40--65 pages) rather than taste:
%%   oneside        - `book` defaults to twoside, which inserted 16 blank versos.
%%   setstretch 1.08 - at the template's 1.5 this body renders 97 pages and at
%%                     1.15 it renders 68, both outside the band; the docx of the
%%                     same prose is 57. The bibliography drops to 1.0 below.
\documentclass[12pt,a4paper,oneside]{{Thesis}}

\graphicspath{{{{./Pictures/}}{{./Figures/}}}}

\DeclareMathOperator{{\Tr}}{{Tr}}
\let\savedegree\degree
\let\degree\relax
\let\rem\relax

\usepackage{{adjustbox}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{multirow}}
\usepackage{{enumitem}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{float}}
\usepackage{{longtable}}
\usepackage[square, numbers, comma, sort&compress]{{natbib}}
\usepackage{{Styles/mydefs}}

%% Float placement. The defaults sent three tables to near-empty pages of their
%% own (6, 22 and 2 words), which is how a 57-page docx became a 67-page PDF.
\renewcommand{{\topfraction}}{{0.9}}
\renewcommand{{\bottomfraction}}{{0.8}}
\renewcommand{{\textfraction}}{{0.07}}
\renewcommand{{\floatpagefraction}}{{0.8}}
\setlength{{\intextsep}}{{10pt plus 2pt minus 2pt}}

%% ---- particulars ----------------------------------------------------------
\def\ttitle{{{title}}}
\def\authornames{{Tan Uei Horng (UTAN001)}}
\def\deptname{{College of Computing and Data Science}}
\def\degreename{{Bachelor of Engineering (Computer Science)}}
\def\supname{{Dr. Zhang Jiehuang}}
\def\univname{{Nanyang Technological University}}

\title{{\ttitle}}

\begin{{document}}

\frontmatter
\setstretch{{1.08}}

\pagestyle{{fancy}}
\fancyhead{{}}
\fancyhead[LO]{{\sl{{\leftmark}}}}
\fancyhead[RE]{{\sl{{\rightmark}}}}
\fancyhead[LE,RO]{{\thepage}}

%%----------------------------------------------------------------------------
%% The report type, subtitle and project code are drawn by Thesis.cls, which this
%% builder patches per tree -- see render_class(). Putting them here instead cost a
%% whole extra page and produced a title page and a cover line that disagreed.
\maketitle
"""

TAIL = r"""
\end{document}
"""


def render_body(events: list[dict], figures: dict[str, str]) -> str:
    out: list[str] = []
    in_front_matter = True
    in_bibliography = False
    in_appendix = False

    def flush_bibliography() -> None:
        nonlocal in_bibliography
        if in_bibliography:
            out.append("\\end{thebibliography}\n")
            in_bibliography = False

    for event in events:
        kind = event["kind"]

        if kind == "para" and event["style"] == "Front Matter Heading":
            title = event["text"].strip()
            # TOC / figure / table lists are generated by LaTeX, not transcribed.
            if title in {"Table of Contents", "List of Figures", "List of Tables"}:
                continue
            out.append(f"\\addtotoc{{{tex_escape(title)}}}")
            out.append(f"\\chapter*{{{tex_escape(title)}}}\n")
            continue

        if kind == "para" and event["style"] == "Reference":
            if not in_bibliography:
                # Reference lists are set solid in the existing mirrors too.
                out.append("\\setstretch{1}")
                out.append("\\begin{thebibliography}{99}")
                in_bibliography = True
            # Strip the builder's literal "[n] " marker; \bibitem numbers itself.
            text = re.sub(r"^\[\d+\]\s*", "", event["text"].strip())
            index = len(
                [line for line in out if line.startswith("\\bibitem")]
            ) + 1
            out.append(f"\\bibitem{{ref{index}}} {tex_escape(text)}")
            continue

        if kind == "chapter":
            if in_front_matter:
                out.append(
                    "\n\\tableofcontents\n\\listoffigures\n\\listoftables\n\n\\mainmatter\n"
                )
                in_front_matter = False
            flush_bibliography()
            out.append(f"\n\\chapter{{{tex_escape(event['title'])}}}\n")
            continue

        if kind == "raw_heading":
            text = event["text"].strip()
            if text == "References":
                flush_bibliography()
                out.append("\n\\backmatter\n\\addtotoc{References}")
                out.append("\\chapter*{References}\n")
                continue
            if text.startswith("Appendix"):
                flush_bibliography()
                if not in_appendix:
                    out.append("\n\\appendix\n\\mainmatter\n")
                    in_appendix = True
                # "Appendix A: Reproducibility Summary" -> the title alone;
                # \appendix supplies the letter.
                title = text.split(":", 1)[1].strip() if ":" in text else text
                out.append(f"\n\\chapter{{{tex_escape(title)}}}\n")
                continue
            out.append(f"\n\\section{{{tex_escape(text)}}}\n")
            continue

        if kind == "heading":
            flush_bibliography()
            command = "subsection" if event["level"] >= 3 else "section"
            out.append(f"\n\\{command}{{{tex_escape(event['title'])}}}\n")
            continue

        if kind in {"body", "para"}:
            flush_bibliography()
            out.append(tex_escape(event["text"].strip()) + "\n")
            continue

        if kind == "bullets":
            flush_bibliography()
            out.append("\\begin{itemize}[leftmargin=*,itemsep=2pt,topsep=4pt]")
            for item in event["items"]:
                out.append(f"  \\item {tex_escape(item)}")
            out.append("\\end{itemize}\n")
            continue

        if kind == "table":
            flush_bibliography()
            out.append(render_table(event))
            continue

        if kind == "figure":
            flush_bibliography()
            stem = figures[event["filename"]]
            width = min(event["width_cm"] / 16.0, 0.95)
            out.append(
                "\\begin{figure}[htbp]\n  \\centering\n"
                f"  \\includegraphics[width={width:.2f}\\textwidth]{{{stem}}}\n"
                f"  \\caption{{{tex_escape(event['title'])}}}\n"
                "\\end{figure}\n"
            )
            continue

    flush_bibliography()
    return "\n".join(out)


def render_table(event: dict) -> str:
    headers = event["headers"]
    rows = event["rows"]
    # Right-align columns whose body cells are consistently numeric.
    numeric = re.compile(r"^[\s$+\u2212<>=~-]*[\d.,%\[\]/ ]+$")
    spec = "".join(
        "r" if all(numeric.match(str(r[i])) for r in rows if i < len(r)) and rows else "l"
        for i in range(len(headers))
    )
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\footnotesize",
    ]
    if event.get("caption", True):
        lines.append(f"\\caption{{{tex_escape(event['title'])}}}")
    lines += [
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{tex_escape(h)}}}" for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(str(v)) for v in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def render(events: list[dict], figures: dict[str, str], *, interim: bool) -> str:
    cover = INTERIM_COVER if interim else FINAL_COVER
    head = PREAMBLE.format(title=TITLE, cover_upper=cover)
    return head + "\n" + render_body(events, figures) + TAIL


def render_class(*, interim: bool) -> str:
    """Patch Thesis.cls's title page to name this document correctly.

    The class is split at \\maketitleforreview so only the \\maketitle copy is
    touched -- the submission-line string appears in both, and patching blind
    would rewrite a command this document never calls.
    """
    source = CLASS_SOURCE.read_text(encoding="utf-8")
    if SPLIT_MARKER not in source:
        raise SystemExit(f"Thesis.cls: split marker {SPLIT_MARKER!r} not found")
    head, tail = source.split(SPLIT_MARKER, 1)

    for anchor in (ANCHOR_AUTHOR, ANCHOR_SUBMISSION):
        if head.count(anchor) != 1:
            raise SystemExit(
                f"Thesis.cls: anchor matched {head.count(anchor)} times, expected 1:\n  {anchor}"
            )

    cover = INTERIM_COVER if interim else FINAL_COVER
    submission = INTERIM_SUBMISSION_LINE if interim else FINAL_SUBMISSION_LINE

    head = head.replace(
        ANCHOR_AUTHOR,
        "  \\textbf{\\large " + cover + "}\\\\[6pt]\n"
        "  {\\itshape " + SUBTITLE + "}\\\\[6pt]\n"
        "  {\\normalsize Project Code: CCDS25-1136}\\\\\n"
        "  \\vfill\n" + ANCHOR_AUTHOR,
        1,
    )
    head = head.replace(ANCHOR_SUBMISSION, submission + " % report type", 1)
    return head + SPLIT_MARKER + tail


# ---------------------------------------------------------------------------
# Tree assembly
# ---------------------------------------------------------------------------
def collect_figures(rpt, events: list[dict], dest: Path) -> dict[str, str]:
    """Copy every referenced figure into the tree; return filename -> tex stem."""
    figures_dir = dest / "Figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for event in events:
        if event["kind"] != "figure":
            continue
        filename = event["filename"]
        if filename in mapping:
            continue
        source = rpt.resolve_figure(filename)
        shutil.copy2(source, figures_dir / source.name)
        mapping[filename] = Path(source.name).stem
    return mapping


def write_tree(dest: Path, tex: str, cls: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "Thesis.cls").write_text(cls, encoding="utf-8")
    styles = dest / "Styles"
    if styles.exists():
        shutil.rmtree(styles)
    shutil.copytree(TEMPLATE_SOURCE / "Styles", styles)
    (dest / "mythesis.tex").write_text(tex, encoding="utf-8")


def write_zip(dest: Path, zip_path: Path) -> None:
    """Bundle the tree for Overleaf.

    Members are prefixed with the tree name, matching the existing bundles so the
    ``zip_matches_tex`` surface profile addresses them the same way. Build
    intermediates are excluded; the compiled PDF is not part of the source bundle.
    """
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(dest.rglob("*")):
            if path.is_dir() or path.suffix in {".pdf", ".aux", ".log", ".out", ".toc"}:
                continue
            archive.write(path, (Path(dest.name) / path.relative_to(dest)).as_posix())


def compile_tex(dest: Path) -> bool:
    tectonic = shutil.which("tectonic")
    if not tectonic:
        print("  tectonic not found - skipping compile", file=sys.stderr)
        return False
    result = subprocess.run(
        [tectonic, "--keep-logs", "--print", "mythesis.tex"],
        cwd=dest,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr[-4000:])
        return False
    return True


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", help="compile with tectonic")
    args = parser.parse_args()

    rpt = load_report_builder()
    events = capture(rpt)

    # A stale unicode map would silently emit mojibake, so fail loudly instead.
    joined = "".join(
        str(v)
        for e in events
        for v in ([e.get("text", "")] + list(e.get("items", [])) + [e.get("title", "")])
    )
    for event in events:
        for row in event.get("rows", []):
            joined += "".join(str(v) for v in row)
        joined += "".join(str(h) for h in event.get("headers", []))
    unknown = unknown_non_ascii(joined)
    if unknown:
        raise SystemExit(
            "Unmapped non-ASCII characters: "
            + ", ".join(f"U+{ord(c):04X} {c!r}" for c in sorted(unknown))
            + "\nAdd them to UNICODE_MAP in scripts/build_rewritten_latex.py."
        )

    chapters = [e for e in events if e["kind"] == "chapter"]
    if len(chapters) != 7:
        raise SystemExit(f"expected 7 chapters, captured {len(chapters)}")

    written = []
    for dest, interim in ((FINAL_DIR, False), (INTERIM_DIR, True)):
        figures = collect_figures(rpt, events, dest)
        tex = render(events, figures, interim=interim)

        cls = render_class(interim=interim)

        # Fail-loud guard, applied to the surface that actually renders the cover.
        # The first build of this script checked only mythesis.tex and shipped a
        # "final" tree whose title page read INTERIM REPORT; the class is the
        # surface, so the class is what gets checked.
        cover = INTERIM_COVER if interim else FINAL_COVER
        other = FINAL_COVER if interim else INTERIM_COVER
        if cls.count(cover) != 1:
            raise SystemExit(f"{dest.name}: expected 1 '{cover}' in Thesis.cls, got {cls.count(cover)}")
        if other in cls:
            raise SystemExit(f"{dest.name}: Thesis.cls also names itself '{other}'")
        if "interim" in cls.lower().split("maketitleforreview")[0] and not interim:
            raise SystemExit(f"{dest.name}: 'interim' survives in the final title page")

        write_tree(dest, tex, cls)
        zip_path = dest.parent / f"{dest.name}_overleaf.zip"
        write_zip(dest, zip_path)
        written.append((dest, zip_path))
        print(f"{dest / 'mythesis.tex'}  ({len(tex.splitlines())} lines)")
        print(f"{zip_path}")

    if args.compile:
        for dest, _ in written:
            ok = compile_tex(dest)
            print(f"  compile {dest.name}: {'OK' if ok else 'FAILED'}")
            if not ok:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
