"""Generate presentation-grade architecture diagrams for the FYP repo."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import shutil
import subprocess
import textwrap
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "architecture"
SVG_PATH = OUT_DIR / "fyp_quant_agentic_architecture.svg"
DRAWIO_PATH = OUT_DIR / "fyp_quant_agentic_architecture.drawio"
PNG_PATH = OUT_DIR / "fyp_quant_agentic_architecture.png"

WIDTH = 1800
HEIGHT = 1240


@dataclass(frozen=True)
class Card:
    key: str
    title: str
    body: tuple[str, ...]
    x: int
    y: int
    w: int
    h: int
    fill: str
    stroke: str


@dataclass(frozen=True)
class Section:
    title: str
    subtitle: str
    x: int
    y: int
    w: int
    h: int
    fill: str
    stroke: str


SECTIONS = [
    Section("Human + Agent Entry", "How work enters the repo", 60, 160, 380, 830, "#EEF5FF", "#7AA7D9"),
    Section("Repo Memory + Control Plane", "Small context, strong rules", 480, 160, 420, 830, "#EFFAF6", "#69B99A"),
    Section("Research Execution Pipeline", "What the benchmark runs", 940, 160, 420, 830, "#FFF6E8", "#D7A75D"),
    Section("Evidence + Reporting", "What becomes defensible output", 1400, 160, 340, 830, "#F7F0FF", "#A88AD8"),
]

CARDS = [
    Card(
        "human",
        "Human researcher",
        ("Defines goal", "Handles VPN/MFA/auth", "Approves risky choices"),
        90,
        235,
        320,
        118,
        "#FFFFFF",
        "#3178C6",
    ),
    Card(
        "agents",
        "Agent surfaces",
        ("Codex / Claude / Cursor", "AGENTS.md + PROJECT_LOG", "Ask repo for live state"),
        90,
        382,
        320,
        128,
        "#FFFFFF",
        "#3178C6",
    ),
    Card(
        "cli",
        "CLI + Makefile",
        ("fyp_cli.py", "make agent-check", "report / analyze / cluster"),
        90,
        540,
        320,
        128,
        "#FFFFFF",
        "#3178C6",
    ),
    Card(
        "startup",
        "Task-specific startup",
        ("agent-status", "agent-start --task --agent", "No giant prompt"),
        90,
        700,
        320,
        128,
        "#FFFFFF",
        "#3178C6",
    ),
    Card(
        "truth",
        "Single source of truth",
        ("docs/PROJECT_LOG.md", "Decisions + open actions", "Changelog for each change"),
        510,
        235,
        360,
        118,
        "#FFFFFF",
        "#28966F",
    ),
    Card(
        "context",
        "On-demand context",
        ("docs/agent_tasks/*", ".agents/skills/*", "Load only relevant packet"),
        510,
        382,
        360,
        118,
        "#FFFFFF",
        "#28966F",
    ),
    Card(
        "subagents",
        "Context isolation",
        (".codex/agents/*", "Report / artifact / TC1 / judge", "Findings to main agent"),
        510,
        528,
        360,
        130,
        "#FFFFFF",
        "#28966F",
    ),
    Card(
        "hooks",
        "Hooks + CI",
        (".codex/hooks.json", ".github/workflows/agent-harness.yml", "Warnings + lightweight checks"),
        510,
        688,
        360,
        130,
        "#FFFFFF",
        "#28966F",
    ),
    Card(
        "gate",
        "Finish gate",
        ("make agent-check", "186 tests", "Redaction + artifact checks"),
        510,
        848,
        360,
        118,
        "#E6FFF5",
        "#1F7A5A",
    ),
    Card(
        "configs",
        "Experiment config",
        ("configs/default.yaml", "configs/tc1.yaml", "Matched baseline vs NF4"),
        970,
        235,
        360,
        118,
        "#FFFFFF",
        "#C2842E",
    ),
    Card(
        "loader",
        "Model load + generation",
        ("HFModelLoader", "BitsAndBytes NF4", "Greedy decoding"),
        970,
        382,
        360,
        118,
        "#FFFFFF",
        "#C2842E",
    ),
    Card(
        "benchmarks",
        "Benchmark plugins",
        ("HarmBench", "XSTest", "MMLU subset"),
        970,
        528,
        360,
        118,
        "#FFFFFF",
        "#C2842E",
    ),
    Card(
        "pipeline",
        "Pipeline orchestration",
        ("run_quant_benchmark.py", "run_quant_matrix.py", "Model reuse"),
        970,
        674,
        360,
        118,
        "#FFFFFF",
        "#C2842E",
    ),
    Card(
        "tc1",
        "TC1 SLURM execution",
        ("Prefetch on head node", "Compute via sbatch", "Offline HF env vars"),
        970,
        840,
        360,
        126,
        "#FFF9EA",
        "#C2842E",
    ),
    Card(
        "raw",
        "Immutable raw evidence",
        ("results/*/*/raw.jsonl", "results/*/*/summary.json", "Hash manifest"),
        1430,
        235,
        280,
        130,
        "#FFFFFF",
        "#8D68C7",
    ),
    Card(
        "sidecars",
        "Derived scoring sidecars",
        ("scores.v2.jsonl", "scores.judge.*", "Redacted IDs + booleans"),
        1430,
        398,
        280,
        130,
        "#FFFFFF",
        "#8D68C7",
    ),
    Card(
        "analysis",
        "Analysis outputs",
        ("pairwise deltas", "judge agreement", "interpretation labels"),
        1430,
        560,
        280,
        122,
        "#FFFFFF",
        "#8D68C7",
    ),
    Card(
        "report",
        "Report builder",
        ("build_fyp_report.js", "make report", "FYP_Report.docx"),
        1430,
        710,
        280,
        130,
        "#FFFFFF",
        "#8D68C7",
    ),
    Card(
        "present",
        "Presentation artifacts",
        ("Meetup brief", "Architecture SVG", "Dashboard / handoff"),
        1430,
        852,
        280,
        114,
        "#F3EAFE",
        "#8D68C7",
    ),
]

EDGES = [
    ("human", "agents", "#3178C6", "intent"),
    ("agents", "cli", "#3178C6", "commands"),
    ("cli", "startup", "#3178C6", "small packet"),
    ("startup", "truth", "#3178C6", "grounding"),
    ("truth", "context", "#28966F", "current state"),
    ("context", "subagents", "#28966F", "delegate narrowly"),
    ("subagents", "hooks", "#28966F", "findings"),
    ("hooks", "gate", "#28966F", "verify"),
    ("configs", "loader", "#C2842E", "model spec"),
    ("loader", "benchmarks", "#C2842E", "responses"),
    ("benchmarks", "pipeline", "#C2842E", "scores"),
    ("pipeline", "tc1", "#C2842E", "batch jobs"),
    ("tc1", "raw", "#C2842E", "TC1 outputs"),
    ("raw", "sidecars", "#8D68C7", "derived only"),
    ("sidecars", "analysis", "#8D68C7", "aggregates"),
    ("analysis", "report", "#8D68C7", "tables + claims"),
    ("report", "present", "#8D68C7", "deliverables"),
    ("gate", "truth", "#1F7A5A", "log update"),
    ("gate", "report", "#1F7A5A", "if report-worthy"),
]


def _svg_text(
    lines: list[str],
    *,
    x: int,
    y: int,
    size: int,
    fill: str,
    weight: int = 400,
    line_height: int | None = None,
) -> str:
    line_height = line_height or int(size * 1.35)
    chunks = []
    for idx, line in enumerate(lines):
        chunks.append(
            f'<text x="{x}" y="{y + idx * line_height}" '
            f'font-family="Inter, Arial, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}">{escape(line)}</text>'
        )
    return "\n".join(chunks)


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False) or [text]


def _card_svg(card: Card) -> str:
    shadow = f'<rect x="{card.x + 5}" y="{card.y + 7}" width="{card.w}" height="{card.h}" rx="18" fill="#0F172A" opacity="0.08"/>'
    rect = (
        f'<rect x="{card.x}" y="{card.y}" width="{card.w}" height="{card.h}" rx="18" '
        f'fill="{card.fill}" stroke="{card.stroke}" stroke-width="2"/>'
    )
    title = _svg_text([card.title], x=card.x + 22, y=card.y + 35, size=20, fill="#102033", weight=700)
    body_lines: list[str] = []
    for item in card.body:
        body_lines.extend(_wrap(f"- {item}", 31 if card.w < 300 else 40))
    body = _svg_text(body_lines, x=card.x + 22, y=card.y + 68, size=15, fill="#334155", line_height=21)
    return "\n".join([shadow, rect, title, body])


def _section_svg(section: Section) -> str:
    rect = (
        f'<rect x="{section.x}" y="{section.y}" width="{section.w}" height="{section.h}" rx="26" '
        f'fill="{section.fill}" stroke="{section.stroke}" stroke-width="2"/>'
    )
    title = _svg_text([section.title], x=section.x + 24, y=section.y + 38, size=24, fill="#0F172A", weight=800)
    subtitle = _svg_text([section.subtitle], x=section.x + 24, y=section.y + 65, size=15, fill="#475569")
    return "\n".join([rect, title, subtitle])


def _center(card: Card) -> tuple[int, int]:
    return card.x + card.w // 2, card.y + card.h // 2


def _edge_svg(source: Card, target: Card, color: str, label: str) -> str:
    sx, sy = _center(source)
    tx, ty = _center(target)
    if source.x < target.x:
        sx = source.x + source.w
        tx = target.x
    elif source.x > target.x:
        sx = source.x
        tx = target.x + target.w
    elif source.y < target.y:
        sy = source.y + source.h
        ty = target.y
    else:
        sy = source.y
        ty = target.y + target.h

    mx = (sx + tx) // 2
    path = f"M {sx} {sy} C {mx} {sy}, {mx} {ty}, {tx} {ty}"
    return "\n".join(
        [
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3" marker-end="url(#{_marker_id(color)})" opacity="0.9"/>',
        ]
    )


def _marker_id(color: str) -> str:
    return "arrow_" + color.replace("#", "")


def render_svg() -> str:
    card_by_key = {card.key: card for card in CARDS}
    markers = []
    for color in sorted({edge[2] for edge in EDGES}):
        markers.append(
            f'<marker id="{_marker_id(color)}" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">'
            f'<path d="M2,2 L10,6 L2,10 Z" fill="{color}"/></marker>'
        )
    header = "\n".join(
        [
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#F8FAFC"/>',
        f'<rect x="0" y="0" width="{WIDTH}" height="126" fill="url(#headerGrad)"/>',
            _svg_text(["fyp_quant: Agentic Research Harness Architecture"], x=62, y=58, size=38, fill="#FFFFFF", weight=800),
            _svg_text(
                ["How the repo turns Codex/Claude from chat assistants into a verified research operating system"],
                x=64,
                y=91,
                size=18,
                fill="#DDEBFF",
                weight=500,
            ),
            '<text x="1510" y="52" font-family="Inter, Arial, sans-serif" font-size="16" font-weight="700" fill="#DDEBFF">Presentation view</text>',
            '<text x="1510" y="80" font-family="Inter, Arial, sans-serif" font-size="14" fill="#BFD7FF">Editable source: .drawio</text>',
        ]
    )
    bottom = "\n".join(
        [
            '<rect x="60" y="1028" width="1680" height="118" rx="24" fill="#FFF1F2" stroke="#E06B78" stroke-width="2"/>',
            _svg_text(["Guardrail Contract"], x=92, y=1070, size=23, fill="#991B1B", weight=800),
            _svg_text(
                [
                    "Raw TC1 artifacts are immutable. Revised scoring writes derived sidecars only.",
                    "Handoffs and dashboards are redacted. PROJECT_LOG records decisions and every repo change.",
                    "make agent-check is the final gate: docs sync, project-log discipline, artifact hashes, stale text, redaction, diff check, pytest.",
                ],
                x=320,
                y=1058,
                size=16,
                fill="#7F1D1D",
                line_height=27,
            ),
        ]
    )
    legend = "\n".join(
        [
            '<rect x="60" y="1170" width="1680" height="1" fill="#CBD5E1"/>',
            '<circle cx="72" cy="1192" r="6" fill="#3178C6"/><text x="88" y="1197" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">Agent control loop</text>',
            '<circle cx="238" cy="1192" r="6" fill="#C2842E"/><text x="254" y="1197" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">Benchmark execution path</text>',
            '<circle cx="448" cy="1192" r="6" fill="#8D68C7"/><text x="464" y="1197" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">Evidence/reporting path</text>',
            '<circle cx="640" cy="1192" r="6" fill="#1F7A5A"/><text x="656" y="1197" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">Verification feedback</text>',
        ]
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '<linearGradient id="headerGrad" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#0F3B66"/><stop offset="55%" stop-color="#0B5E6F"/><stop offset="100%" stop-color="#513C80"/></linearGradient>',
        *markers,
        "</defs>",
        header,
        *[_section_svg(section) for section in SECTIONS],
        *[_edge_svg(card_by_key[src], card_by_key[dst], color, label) for src, dst, color, label in EDGES],
        *[_card_svg(card) for card in CARDS],
        bottom,
        legend,
        "</svg>",
    ]
    return "\n".join(parts)


def _mx_cell(container: ET.Element, cell_id: str, **attrs: str) -> ET.Element:
    attrs.setdefault("id", cell_id)
    return ET.SubElement(container, "mxCell", attrs)


def _drawio_label(card: Card | Section) -> str:
    if isinstance(card, Section):
        return f"<b>{escape(card.title)}</b><br><font color='#475569'>{escape(card.subtitle)}</font>"
    return "<br>".join([f"<b>{escape(card.title)}</b>", *[escape(item) for item in card.body]])


def _style(fill: str, stroke: str, rounded: bool = True) -> str:
    rounded_part = "rounded=1;arcSize=12;" if rounded else "rounded=0;"
    return (
        f"{rounded_part}whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
        "fontFamily=Inter;fontSize=14;fontColor=#102033;align=left;verticalAlign=top;"
        "spacingLeft=16;spacingTop=12;shadow=1;"
    )


def render_drawio() -> str:
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "modified": "2026-06-08T18:00:00+08:00",
            "agent": "Codex",
            "version": "24.7.17",
            "type": "device",
        },
    )
    diagram = ET.SubElement(mxfile, "diagram", {"id": "fyp-agentic-architecture", "name": "Architecture Overview"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "1800",
            "dy": "1120",
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": str(WIDTH),
            "pageHeight": str(HEIGHT),
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    _mx_cell(root, "0")
    _mx_cell(root, "1", parent="0")

    _mx_cell(
        root,
        "title",
        value="fyp_quant: Agentic Research Harness Architecture<br><font style='font-size: 16px'>Repo-native memory, isolated subagents, verified evidence, and report generation</font>",
        style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#0F3B66;strokeColor=#0F3B66;fontColor=#FFFFFF;fontFamily=Inter;fontSize=28;fontStyle=1;spacingLeft=24;spacingTop=20;",
        vertex="1",
        parent="1",
    ).append(ET.Element("mxGeometry", {"x": "40", "y": "30", "width": "1720", "height": "88", "as": "geometry"}))

    for idx, section in enumerate(SECTIONS, start=1):
        cell = _mx_cell(
            root,
            f"section-{idx}",
            value=_drawio_label(section),
            style=_style(section.fill, section.stroke),
            vertex="1",
            parent="1",
        )
        cell.append(
            ET.Element(
                "mxGeometry",
                {"x": str(section.x), "y": str(section.y), "width": str(section.w), "height": str(section.h), "as": "geometry"},
            )
        )

    for card in CARDS:
        cell = _mx_cell(
            root,
            card.key,
            value=_drawio_label(card),
            style=_style(card.fill, card.stroke),
            vertex="1",
            parent="1",
        )
        cell.append(
            ET.Element(
                "mxGeometry",
                {"x": str(card.x), "y": str(card.y), "width": str(card.w), "height": str(card.h), "as": "geometry"},
            )
        )

    for idx, (src, dst, color, label) in enumerate(EDGES, start=1):
        cell = _mx_cell(
            root,
            f"edge-{idx}",
            value="",
            style=f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor={color};strokeWidth=3;fontColor={color};fontStyle=1;endArrow=block;endFill=1;",
            edge="1",
            parent="1",
            source=src,
            target=dst,
        )
        cell.append(ET.Element("mxGeometry", {"relative": "1", "as": "geometry"}))

    guard = _mx_cell(
        root,
        "guardrails",
        value=(
            "<b>Guardrail Contract</b><br>"
            "Raw TC1 artifacts are immutable. Revised scoring writes derived sidecars only.<br>"
            "Handoffs are redacted. PROJECT_LOG records decisions and every repo change.<br>"
            "make agent-check is the final gate: docs sync, artifact hashes, stale text, redaction, diff check, pytest."
        ),
        style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFF1F2;strokeColor=#E06B78;fontColor=#7F1D1D;fontFamily=Inter;fontSize=15;fontStyle=0;spacingLeft=22;spacingTop=15;shadow=1;",
        vertex="1",
        parent="1",
    )
    guard.append(ET.Element("mxGeometry", {"x": "60", "y": "1028", "width": "1680", "height": "118", "as": "geometry"}))

    ET.indent(mxfile, space="  ")
    return ET.tostring(mxfile, encoding="unicode", xml_declaration=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(render_svg(), encoding="utf-8")
    DRAWIO_PATH.write_text(render_drawio(), encoding="utf-8")
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {DRAWIO_PATH}")
    if shutil.which("sips"):
        subprocess.run(
            ["sips", "-s", "format", "png", str(SVG_PATH), "--out", str(PNG_PATH)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Wrote {PNG_PATH}")
    else:
        print("Skipped PNG export because sips is not available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
