"""Generate a top-down repository hierarchy diagram for fyp_quant."""

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
SVG_PATH = OUT_DIR / "fyp_quant_repo_hierarchy.svg"
DRAWIO_PATH = OUT_DIR / "fyp_quant_repo_hierarchy.drawio"
PNG_PATH = OUT_DIR / "fyp_quant_repo_hierarchy.png"

WIDTH = 1800
HEIGHT = 1320


@dataclass(frozen=True)
class Node:
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


NODES = [
    Node(
        "root",
        "fyp_quant/",
        "Repo-native FYP research operating system",
        ("Four major areas below",),
        600,
        150,
        600,
        140,
        "#0F3B66",
        "#0F3B66",
    ),
    Node(
        "agent_os",
        "Agent Operating System",
        "How Codex/Claude know what to do",
        ("AGENTS.md + CLAUDE.md", "docs/PROJECT_LOG.md", "docs/HANDOFF.md + dashboard", ".agents/skills/ + .codex/agents/ + hooks", "make agent-check + CI"),
        70,
        370,
        390,
        250,
        "#EEF5FF",
        "#3178C6",
    ),
    Node(
        "research_core",
        "Research Core",
        "The Python framework that runs the study",
        ("ethical_benchmark/benchmarks", "ethical_benchmark/models", "ethical_benchmark/pipeline", "ethical_benchmark/analysis", "ethical_benchmark/judges", "ethical_benchmark/cluster"),
        505,
        370,
        390,
        250,
        "#EFFAF6",
        "#28966F",
    ),
    Node(
        "inputs_execution",
        "Inputs + Execution",
        "Config, data, SLURM, and commands",
        ("configs/default.yaml + tc1.yaml", "data/xstest_v2_prompts.csv", "slurm/jobs_tc1* + judge sbatch", "scripts/prefetch_tc1.py", "fyp_cli.py + Makefile"),
        940,
        370,
        390,
        250,
        "#FFF6E8",
        "#C2842E",
    ),
    Node(
        "evidence",
        "Evidence + Deliverables",
        "What is produced and defended",
        ("results/*/*/raw.jsonl", "results/*/*/summary.json", "scores.v2 + scores.judge sidecars", "results/analysis/*", "docs/FYP_Report_*.docx", "docs/architecture/*"),
        1375,
        370,
        355,
        250,
        "#F7F0FF",
        "#8D68C7",
    ),
    Node(
        "agent_files",
        "Agent Files",
        "Small context, reusable work packets",
        ("docs/agent_tasks/", ".agents/skills/", ".codex/agents/", ".codex/hooks/", ".github/workflows/"),
        70,
        710,
        390,
        250,
        "#FFFFFF",
        "#3178C6",
    ),
    Node(
        "core_modules",
        "Core Modules",
        "Separation of concerns inside ethical_benchmark/",
        ("benchmarks: HarmBench / XSTest / MMLU", "models: HF loading + generation", "pipeline: run + matrix orchestration", "analysis: deltas + labels", "judges: HarmBench classifier validation", "cluster: SLURM helpers"),
        505,
        710,
        390,
        250,
        "#FFFFFF",
        "#28966F",
    ),
    Node(
        "execution_files",
        "Execution Files",
        "Where runs are configured and launched",
        ("configs/: model pairs + TC1 policy", "slurm/: tracked sbatch scripts", "scripts/: prefetch, judge, report, diagrams", "Makefile: short commands", "fyp_cli.py: unified CLI"),
        940,
        710,
        390,
        250,
        "#FFFFFF",
        "#C2842E",
    ),
    Node(
        "artifact_contract",
        "Artifact Contract",
        "The rule that protects scientific integrity",
        ("Raw TC1 outputs are immutable", "Revisions use derived sidecars", "Judge sidecars are redacted", "PROJECT_LOG records decisions", "Report is generated from source"),
        1375,
        710,
        355,
        250,
        "#FFFFFF",
        "#8D68C7",
    ),
]

EDGES = [
    ("root", "agent_os", "#3178C6"),
    ("root", "research_core", "#28966F"),
    ("root", "inputs_execution", "#C2842E"),
    ("root", "evidence", "#8D68C7"),
    ("agent_os", "agent_files", "#3178C6"),
    ("research_core", "core_modules", "#28966F"),
    ("inputs_execution", "execution_files", "#C2842E"),
    ("evidence", "artifact_contract", "#8D68C7"),
    ("core_modules", "execution_files", "#1F7A5A"),
    ("execution_files", "artifact_contract", "#1F7A5A"),
]


def _marker_id(color: str) -> str:
    return "arrow_" + color.replace("#", "")


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False) or [text]


def _text(lines: list[str], x: int, y: int, size: int, fill: str, weight: int = 400, line_height: int | None = None) -> str:
    line_height = line_height or int(size * 1.35)
    return "\n".join(
        f'<text x="{x}" y="{y + idx * line_height}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(line)}</text>'
        for idx, line in enumerate(lines)
    )


def _node_svg(node: Node) -> str:
    is_root = node.key == "root"
    title_fill = "#FFFFFF" if is_root else "#0F172A"
    subtitle_fill = "#DDEBFF" if is_root else "#475569"
    body_fill = "#EAF5FF" if is_root else "#334155"
    shadow = f'<rect x="{node.x + 6}" y="{node.y + 8}" width="{node.w}" height="{node.h}" rx="24" fill="#0F172A" opacity="0.09"/>'
    rect = (
        f'<rect x="{node.x}" y="{node.y}" width="{node.w}" height="{node.h}" rx="24" '
        f'fill="{node.fill}" stroke="{node.stroke}" stroke-width="2.2"/>'
    )
    folder_tab = ""
    if not is_root:
        folder_tab = (
            f'<path d="M {node.x + 24} {node.y - 10} H {node.x + 128} '
            f'L {node.x + 150} {node.y + 12} H {node.x + 24} Z" '
            f'fill="{node.stroke}" opacity="0.18" stroke="{node.stroke}" stroke-width="1.5"/>'
        )
    title = _text([node.title], node.x + 28, node.y + 42, 24 if is_root else 22, title_fill, 800)
    subtitle = _text([node.subtitle], node.x + 28, node.y + 72, 15, subtitle_fill, 500)
    body_lines: list[str] = []
    wrap_width = 42 if node.w >= 390 else 35
    for item in node.items:
        body_lines.extend(_wrap(f"- {item}", wrap_width))
    body = _text(body_lines, node.x + 28, node.y + 107, 14 if node.h < 245 else 15, body_fill, 500, line_height=22)
    return "\n".join([shadow, folder_tab, rect, title, subtitle, body])


def _center_bottom(node: Node) -> tuple[int, int]:
    return node.x + node.w // 2, node.y + node.h


def _center_top(node: Node) -> tuple[int, int]:
    return node.x + node.w // 2, node.y


def _edge_svg(source: Node, target: Node, color: str) -> str:
    sx, sy = _center_bottom(source)
    tx, ty = _center_top(target)
    mid_y = (sy + ty) // 2
    path = f"M {sx} {sy} C {sx} {mid_y}, {tx} {mid_y}, {tx} {ty}"
    return f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3" marker-end="url(#{_marker_id(color)})" opacity="0.85"/>'


def render_svg() -> str:
    node_by_key = {node.key: node for node in NODES}
    markers = [
        f'<marker id="{_marker_id(color)}" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">'
        f'<path d="M2,2 L10,6 L2,10 Z" fill="{color}"/></marker>'
        for color in sorted({color for _, _, color in EDGES})
    ]
    header = "\n".join(
        [
            f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#F8FAFC"/>',
            f'<rect x="0" y="0" width="{WIDTH}" height="126" fill="url(#headerGrad)"/>',
            _text(["fyp_quant Repository Hierarchy"], 62, 58, 40, "#FFFFFF", 800),
            _text(["Top-down map: what each folder/file group does and how the repo hangs together"], 64, 92, 18, "#DDEBFF", 500),
            '<text x="1510" y="52" font-family="Inter, Arial, sans-serif" font-size="16" font-weight="700" fill="#DDEBFF">Folder-first view</text>',
            '<text x="1510" y="80" font-family="Inter, Arial, sans-serif" font-size="14" fill="#BFD7FF">Editable source: .drawio</text>',
        ]
    )
    read_order = "\n".join(
        [
            '<rect x="70" y="1025" width="1660" height="150" rx="24" fill="#F1F5F9" stroke="#CBD5E1" stroke-width="2"/>',
            _text(["How to read this"], 102, 1064, 24, "#0F172A", 800),
            _text(
                [
                    "1. Start at fyp_quant/: the repo is both a research benchmark and an agent operating system.",
                    "2. The middle row shows the four big jobs: agent control, research code, execution setup, and evidence/reporting.",
                    "3. The lower row shows the concrete folders/files a new agent or human should inspect.",
                    "4. The green links mean verification/control: checks connect the agent harness back to the evidence contract.",
                ],
                355,
                1052,
                16,
                "#334155",
                500,
                line_height=28,
            ),
        ]
    )
    legend = "\n".join(
        [
            '<rect x="70" y="1205" width="1660" height="1" fill="#CBD5E1"/>',
            '<circle cx="82" cy="1227" r="6" fill="#3178C6"/><text x="98" y="1232" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">Agent harness</text>',
            '<circle cx="230" cy="1227" r="6" fill="#28966F"/><text x="246" y="1232" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">Research framework</text>',
            '<circle cx="420" cy="1227" r="6" fill="#C2842E"/><text x="436" y="1232" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">Execution setup</text>',
            '<circle cx="585" cy="1227" r="6" fill="#8D68C7"/><text x="601" y="1232" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">Evidence and deliverables</text>',
            '<circle cx="805" cy="1227" r="6" fill="#1F7A5A"/><text x="821" y="1232" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">Verification/control link</text>',
        ]
    )
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
            "<defs>",
            '<linearGradient id="headerGrad" x1="0%" y1="0%" x2="100%"><stop offset="0%" stop-color="#0F3B66"/><stop offset="55%" stop-color="#0B5E6F"/><stop offset="100%" stop-color="#513C80"/></linearGradient>',
            *markers,
            "</defs>",
            header,
            *[_edge_svg(node_by_key[src], node_by_key[dst], color) for src, dst, color in EDGES],
            *[_node_svg(node) for node in NODES],
            read_order,
            legend,
            "</svg>",
        ]
    )


def _mx_cell(container: ET.Element, cell_id: str, **attrs: str) -> ET.Element:
    attrs.setdefault("id", cell_id)
    return ET.SubElement(container, "mxCell", attrs)


def _node_label(node: Node) -> str:
    return "<br>".join([f"<b>{escape(node.title)}</b>", escape(node.subtitle), *[escape(item) for item in node.items]])


def _style(node: Node) -> str:
    font_color = "#FFFFFF" if node.key == "root" else "#102033"
    return (
        "rounded=1;arcSize=12;whiteSpace=wrap;html=1;"
        f"fillColor={node.fill};strokeColor={node.stroke};fontColor={font_color};"
        "fontFamily=Inter;fontSize=14;align=left;verticalAlign=top;spacingLeft=18;spacingTop=14;shadow=1;"
    )


def render_drawio() -> str:
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "modified": "2026-06-08T18:20:00+08:00",
            "agent": "Codex",
            "version": "24.7.17",
            "type": "device",
        },
    )
    diagram = ET.SubElement(mxfile, "diagram", {"id": "fyp-repo-hierarchy", "name": "Repo Hierarchy"})
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
        value="fyp_quant Repository Hierarchy<br><font style='font-size: 16px'>Top-down map of folders, files, and responsibilities</font>",
        style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#0F3B66;strokeColor=#0F3B66;fontColor=#FFFFFF;fontFamily=Inter;fontSize=28;fontStyle=1;spacingLeft=24;spacingTop=20;",
        vertex="1",
        parent="1",
    )
    title.append(ET.Element("mxGeometry", {"x": "40", "y": "30", "width": "1720", "height": "88", "as": "geometry"}))

    for node in NODES:
        cell = _mx_cell(root, node.key, value=_node_label(node), style=_style(node), vertex="1", parent="1")
        cell.append(ET.Element("mxGeometry", {"x": str(node.x), "y": str(node.y), "width": str(node.w), "height": str(node.h), "as": "geometry"}))

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

    guide = _mx_cell(
        root,
        "read-order",
        value=(
            "<b>How to read this</b><br>"
            "Start at fyp_quant/. The middle row shows the four big responsibilities. "
            "The lower row shows concrete folders/files. Green links are verification/control."
        ),
        style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F1F5F9;strokeColor=#CBD5E1;fontColor=#334155;fontFamily=Inter;fontSize=15;spacingLeft=22;spacingTop=18;shadow=1;",
        vertex="1",
        parent="1",
    )
    guide.append(ET.Element("mxGeometry", {"x": "70", "y": "1025", "width": "1660", "height": "150", "as": "geometry"}))
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
