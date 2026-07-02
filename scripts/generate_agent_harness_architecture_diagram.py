"""Generate a presentation diagram for the fyp_quant agent harness."""

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
SVG_PATH = OUT_DIR / "fyp_quant_agent_harness_architecture.svg"
DRAWIO_PATH = OUT_DIR / "fyp_quant_agent_harness_architecture.drawio"
PNG_PATH = OUT_DIR / "fyp_quant_agent_harness_architecture.png"

WIDTH = 1920
HEIGHT = 1320


@dataclass(frozen=True)
class Card:
    key: str
    title: str
    subtitle: str
    items: tuple[str, ...]
    x: int
    y: int
    w: int
    h: int
    fill: str
    stroke: str
    badge: str


CONTROL_CARDS = [
    Card(
        "orientation",
        "Orientation Kernel",
        "Always-loaded law and live state",
        ("AGENTS.md / CLAUDE.md", "docs/PROJECT_LOG.md", "agent-status snapshot"),
        130,
        380,
        520,
        185,
        "#EEF6FF",
        "#2F80C5",
        "1",
    ),
    Card(
        "routing",
        "Task Router",
        "Bounded startup for one job",
        ("fyp_cli.py agent-start", "docs/agent_tasks/", "recommended subagent"),
        700,
        380,
        520,
        185,
        "#F2FAF5",
        "#2C9A67",
        "2",
    ),
    Card(
        "isolation",
        "Context Isolation",
        "Small context on purpose",
        (".agents/skills/", ".codex/agents/", "only files named in packet"),
        1270,
        380,
        520,
        185,
        "#F7F0FF",
        "#8A63C8",
        "3",
    ),
    Card(
        "guardrails",
        "Guardrail Contract",
        "Rules become machine checks",
        ("configs/artifact_policy.yaml", "Codex hooks + GitHub Action", "raw artifact manifest"),
        130,
        635,
        520,
        185,
        "#FFF5E8",
        "#C6812D",
        "4",
    ),
    Card(
        "verification",
        "Verification Gate",
        "Finish with evidence",
        ("make agent-check", "harness_eval + pytest", "stale text / redaction scans"),
        700,
        635,
        520,
        185,
        "#EFFAFB",
        "#2494A3",
        "5",
    ),
    Card(
        "recovery",
        "Recovery Layer",
        "Next session resumes cleanly",
        ("docs/HANDOFF.md", "docs/AGENT_DASHBOARD.md", "docs/TC1_AGENT_CHECKLIST.md"),
        1270,
        635,
        520,
        185,
        "#FFF0F3",
        "#C85C74",
        "6",
    ),
]

BOTTOM_CARDS = [
    Card(
        "evidence",
        "Evidence Protected",
        "raw artifacts stay fixed",
        ("raw.jsonl", "summary.json", "hash manifest"),
        120,
        990,
        405,
        165,
        "#FFFFFF",
        "#2F80C5",
        "",
    ),
    Card(
        "sidecars",
        "Derived Sidecars",
        "post-hoc scoring is auditable",
        ("scores.v2.*", "scores.judge.*", "redacted outputs"),
        555,
        990,
        405,
        165,
        "#FFFFFF",
        "#8A63C8",
        "",
    ),
    Card(
        "deliverables",
        "Report + Analysis",
        "deliverables remain reproducible",
        ("build_fyp_report_v5.js", "results_512/analysis/", "architecture visuals"),
        990,
        990,
        405,
        165,
        "#FFFFFF",
        "#2C9A67",
        "",
    ),
    Card(
        "usp",
        "Meetup USP",
        "repo memory beats chat memory",
        ("repo remembers", "agents verify", "human keeps authority"),
        1425,
        990,
        405,
        165,
        "#FFFFFF",
        "#C6812D",
        "",
    ),
]

ALL_CARDS = CONTROL_CARDS + BOTTOM_CARDS

EDGES = [
    ("orientation", "routing", "#64748B"),
    ("routing", "isolation", "#64748B"),
    ("orientation", "guardrails", "#64748B"),
    ("routing", "verification", "#64748B"),
    ("isolation", "recovery", "#64748B"),
    ("guardrails", "verification", "#64748B"),
    ("verification", "recovery", "#64748B"),
]


def _marker_id(color: str) -> str:
    return "arrow_" + color.replace("#", "")


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False) or [text]


def _text(
    lines: list[str],
    x: int,
    y: int,
    size: int,
    fill: str,
    weight: int = 400,
    line_height: int | None = None,
    anchor: str = "start",
) -> str:
    line_height = line_height or int(size * 1.35)
    return "\n".join(
        f'<text x="{x}" y="{y + idx * line_height}" text-anchor="{anchor}" '
        f'font-family="Inter, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{escape(line)}</text>'
        for idx, line in enumerate(lines)
    )


def _pill(x: int, y: int, w: int, h: int, fill: str, stroke: str, title: str, subtitle: str) -> str:
    return "\n".join(
        [
            f'<rect x="{x + 6}" y="{y + 8}" width="{w}" height="{h}" rx="{h // 2}" fill="#0F172A" opacity="0.10"/>',
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h // 2}" fill="{fill}" stroke="{stroke}" stroke-width="2.2"/>',
            _text([title], x + w // 2, y + 34, 22, "#0F172A", 800, anchor="middle"),
            _text([subtitle], x + w // 2, y + 60, 15, "#475569", 500, anchor="middle"),
        ]
    )


def _card_svg(card: Card) -> str:
    shadow = f'<rect x="{card.x + 6}" y="{card.y + 8}" width="{card.w}" height="{card.h}" rx="20" fill="#0F172A" opacity="0.09"/>'
    rect = (
        f'<rect x="{card.x}" y="{card.y}" width="{card.w}" height="{card.h}" rx="20" '
        f'fill="{card.fill}" stroke="{card.stroke}" stroke-width="2.2"/>'
    )
    accent = f'<rect x="{card.x}" y="{card.y}" width="8" height="{card.h}" rx="4" fill="{card.stroke}"/>'
    badge = ""
    title_x = card.x + 34
    if card.badge:
        badge = "\n".join(
            [
                f'<circle cx="{card.x + 42}" cy="{card.y + 42}" r="20" fill="{card.stroke}"/>',
                _text([card.badge], card.x + 42, card.y + 50, 19, "#FFFFFF", 800, anchor="middle"),
            ]
        )
        title_x = card.x + 78
    title = _text([card.title], title_x, card.y + 39, 23, "#0F172A", 800)
    subtitle = _text([card.subtitle], title_x, card.y + 68, 14, "#475569", 600)
    body_lines: list[str] = []
    wrap_width = 46 if card.w >= 500 else 34
    for item in card.items:
        body_lines.extend(_wrap(f"- {item}", wrap_width))
    body = _text(body_lines, card.x + 34, card.y + 105, 15, "#334155", 500, line_height=22)
    return "\n".join([shadow, rect, accent, badge, title, subtitle, body])


def _center(card: Card) -> tuple[int, int]:
    return card.x + card.w // 2, card.y + card.h // 2


def _right_mid(card: Card) -> tuple[int, int]:
    return card.x + card.w, card.y + card.h // 2


def _left_mid(card: Card) -> tuple[int, int]:
    return card.x, card.y + card.h // 2


def _bottom_mid(card: Card) -> tuple[int, int]:
    return card.x + card.w // 2, card.y + card.h


def _top_mid(card: Card) -> tuple[int, int]:
    return card.x + card.w // 2, card.y


def _edge_svg(source: Card, target: Card, color: str) -> str:
    if source.y == target.y and source.x < target.x:
        sx, sy = _right_mid(source)
        tx, ty = _left_mid(target)
        path = f"M {sx} {sy} C {sx + 55} {sy}, {tx - 55} {ty}, {tx} {ty}"
    elif source.x == target.x and source.y < target.y:
        sx, sy = _bottom_mid(source)
        tx, ty = _top_mid(target)
        path = f"M {sx} {sy} C {sx} {sy + 45}, {tx} {ty - 45}, {tx} {ty}"
    else:
        sx, sy = _bottom_mid(source)
        tx, ty = _top_mid(target)
        mid_y = (sy + ty) // 2
        path = f"M {sx} {sy} C {sx} {mid_y}, {tx} {mid_y}, {tx} {ty}"
    return f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3" marker-end="url(#{_marker_id(color)})" opacity="0.72"/>'


def _down_arrow(x1: int, y1: int, x2: int, y2: int, color: str) -> str:
    mid_y = (y1 + y2) // 2
    path = f"M {x1} {y1} C {x1} {mid_y}, {x2} {mid_y}, {x2} {y2}"
    return f'<path d="{path}" fill="none" stroke="{color}" stroke-width="4" marker-end="url(#{_marker_id(color)})" opacity="0.82"/>'


def render_svg() -> str:
    card_by_key = {card.key: card for card in ALL_CARDS}
    markers = [
        f'<marker id="{_marker_id(color)}" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">'
        f'<path d="M2,2 L10,6 L2,10 Z" fill="{color}"/></marker>'
        for color in sorted({"#334155", "#475569", "#64748B"})
    ]
    header = "\n".join(
        [
            f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#F8FAFC"/>',
            f'<rect x="0" y="0" width="{WIDTH}" height="132" fill="url(#headerGrad)"/>',
            _text(["fyp_quant Agent Harness Architecture"], 64, 58, 40, "#FFFFFF", 800),
            _text(
                ["How a fresh coding agent is oriented, scoped, isolated, checked, and handed off"],
                66,
                93,
                18,
                "#DDEBFF",
                500,
            ),
            '<text x="1550" y="54" font-family="Inter, Arial, sans-serif" font-size="16" font-weight="700" fill="#DDEBFF">Control-plane view</text>',
            '<text x="1550" y="82" font-family="Inter, Arial, sans-serif" font-size="14" fill="#BFD7FF">Editable source: .drawio</text>',
        ]
    )
    entry = _pill(
        520,
        165,
        880,
        82,
        "#FFFFFF",
        "#CBD5E1",
        "Human goal enters Codex / Claude / Cursor",
        "The repo supplies durable memory, task scope, and finish gates",
    )
    control_panel = "\n".join(
        [
            '<rect x="80" y="295" width="1760" height="570" rx="30" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="2.2"/>',
            '<rect x="104" y="319" width="1712" height="64" rx="20" fill="#F1F5F9"/>',
            _text(["Repo-Native Agent Harness Control Plane"], 128, 357, 26, "#0F172A", 850),
            _text(
                ["Thin startup context; task-specific packets and skills are pulled only when needed"],
                740,
                356,
                17,
                "#475569",
                600,
            ),
            '<rect x="1510" y="336" width="260" height="30" rx="15" fill="#E0F2FE" stroke="#7DD3FC"/>',
            _text(["context pressure controlled"], 1640, 357, 13, "#075985", 800, anchor="middle"),
        ]
    )
    bottom = "\n".join(
        [
            '<rect x="80" y="920" width="1760" height="250" rx="30" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="2.2"/>',
            _text(["What the Harness Protects"], 120, 963, 26, "#0F172A", 850),
            _text(
                ["Research integrity stays in the repo: evidence, scoring changes, deliverables, and next-session state are all inspectable."],
                520,
                962,
                17,
                "#475569",
                600,
            ),
        ]
    )
    guide = "\n".join(
        [
            '<rect x="80" y="1185" width="1760" height="86" rx="24" fill="#0F172A" opacity="0.95"/>',
            _text(["Talk track"], 120, 1222, 22, "#FFFFFF", 850),
            _text(
                [
                    "1. The agent starts from durable repo state, not chat memory.  2. agent-start loads a small task packet.  3. subagents isolate audits.  4. make agent-check turns the harness into evidence before handoff.",
                ],
                300,
                1221,
                15,
                "#DDEBFF",
                600,
            ),
        ]
    )
    entry_to_panel = _down_arrow(960, 247, 960, 295, "#334155")
    panel_to_bottom = _down_arrow(960, 865, 960, 920, "#334155")
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
            "<defs>",
            '<linearGradient id="headerGrad" x1="0%" y1="0%" x2="100%"><stop offset="0%" stop-color="#12355B"/><stop offset="48%" stop-color="#0F766E"/><stop offset="100%" stop-color="#6D4C8D"/></linearGradient>',
            *markers,
            "</defs>",
            header,
            entry,
            entry_to_panel,
            control_panel,
            *[_edge_svg(card_by_key[src], card_by_key[dst], color) for src, dst, color in EDGES],
            *[_card_svg(card) for card in CONTROL_CARDS],
            panel_to_bottom,
            bottom,
            *[_card_svg(card) for card in BOTTOM_CARDS],
            guide,
            "</svg>",
        ]
    )


def _mx_cell(container: ET.Element, cell_id: str, **attrs: str) -> ET.Element:
    attrs.setdefault("id", cell_id)
    return ET.SubElement(container, "mxCell", attrs)


def _node_label(card: Card) -> str:
    title = f"<b>{escape(card.badge + '. ' if card.badge else '')}{escape(card.title)}</b>"
    return "<br>".join([title, escape(card.subtitle), *[escape(item) for item in card.items]])


def _style(card: Card) -> str:
    return (
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
        f"fillColor={card.fill};strokeColor={card.stroke};fontColor=#102033;"
        "fontFamily=Inter;fontSize=14;align=left;verticalAlign=top;spacingLeft=18;spacingTop=14;shadow=1;"
    )


def render_drawio() -> str:
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "modified": "2026-06-08T23:45:00+08:00",
            "agent": "Codex",
            "version": "24.7.17",
            "type": "device",
        },
    )
    diagram = ET.SubElement(mxfile, "diagram", {"id": "fyp-agent-harness", "name": "Agent Harness Architecture"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": str(WIDTH),
            "dy": str(HEIGHT),
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

    title = _mx_cell(
        root,
        "title",
        value=(
            "fyp_quant Agent Harness Architecture<br>"
            "<font style='font-size: 16px'>How a fresh coding agent is oriented, scoped, isolated, checked, and handed off</font>"
        ),
        style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#12355B;strokeColor=#12355B;fontColor=#FFFFFF;fontFamily=Inter;fontSize=28;fontStyle=1;spacingLeft=24;spacingTop=20;",
        vertex="1",
        parent="1",
    )
    title.append(ET.Element("mxGeometry", {"x": "40", "y": "30", "width": "1840", "height": "92", "as": "geometry"}))

    entry = _mx_cell(
        root,
        "entry",
        value="<b>Human goal enters Codex / Claude / Cursor</b><br>The repo supplies durable memory, task scope, and finish gates",
        style="rounded=1;arcSize=50;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;fontColor=#0F172A;fontFamily=Inter;fontSize=16;align=center;verticalAlign=middle;shadow=1;",
        vertex="1",
        parent="1",
    )
    entry.append(ET.Element("mxGeometry", {"x": "520", "y": "165", "width": "880", "height": "82", "as": "geometry"}))

    panel = _mx_cell(
        root,
        "panel",
        value="<b>Repo-Native Agent Harness Control Plane</b><br>Thin startup context; task-specific packets and skills are pulled only when needed",
        style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;fontColor=#0F172A;fontFamily=Inter;fontSize=18;align=left;verticalAlign=top;spacingLeft=26;spacingTop=18;",
        vertex="1",
        parent="1",
    )
    panel.append(ET.Element("mxGeometry", {"x": "80", "y": "295", "width": "1760", "height": "570", "as": "geometry"}))

    bottom = _mx_cell(
        root,
        "protected",
        value="<b>What the Harness Protects</b><br>Evidence, scoring changes, deliverables, and next-session state are inspectable in the repo.",
        style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#CBD5E1;fontColor=#0F172A;fontFamily=Inter;fontSize=18;align=left;verticalAlign=top;spacingLeft=26;spacingTop=18;",
        vertex="1",
        parent="1",
    )
    bottom.append(ET.Element("mxGeometry", {"x": "80", "y": "920", "width": "1760", "height": "250", "as": "geometry"}))

    for card in ALL_CARDS:
        cell = _mx_cell(root, card.key, value=_node_label(card), style=_style(card), vertex="1", parent="1")
        cell.append(
            ET.Element(
                "mxGeometry",
                {"x": str(card.x), "y": str(card.y), "width": str(card.w), "height": str(card.h), "as": "geometry"},
            )
        )

    for idx, (src, dst, color) in enumerate(EDGES, start=1):
        cell = _mx_cell(
            root,
            f"edge-{idx}",
            value="",
            style=f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor={color};strokeWidth=3;endArrow=block;endFill=1;",
            edge="1",
            parent="1",
            source=src,
            target=dst,
        )
        cell.append(ET.Element("mxGeometry", {"relative": "1", "as": "geometry"}))

    for idx, (src, dst) in enumerate((("entry", "panel"), ("panel", "protected")), start=1):
        cell = _mx_cell(
            root,
            f"flow-{idx}",
            value="",
            style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#334155;strokeWidth=4;endArrow=block;endFill=1;",
            edge="1",
            parent="1",
            source=src,
            target=dst,
        )
        cell.append(ET.Element("mxGeometry", {"relative": "1", "as": "geometry"}))

    guide = _mx_cell(
        root,
        "talk-track",
        value=(
            "<b>Talk track</b><br>"
            "The agent starts from durable repo state, agent-start loads a small task packet, "
            "subagents isolate audits, and make agent-check creates evidence before handoff."
        ),
        style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#0F172A;strokeColor=#0F172A;fontColor=#DDEBFF;fontFamily=Inter;fontSize=15;spacingLeft=24;spacingTop=18;",
        vertex="1",
        parent="1",
    )
    guide.append(ET.Element("mxGeometry", {"x": "80", "y": "1185", "width": "1760", "height": "86", "as": "geometry"}))

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
