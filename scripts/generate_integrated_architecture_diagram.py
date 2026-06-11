"""Generate one detailed architecture poster for the fyp_quant agentic stack."""

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
SVG_PATH = OUT_DIR / "fyp_quant_integrated_agentic_stack.svg"
DRAWIO_PATH = OUT_DIR / "fyp_quant_integrated_agentic_stack.drawio"
PNG_PATH = OUT_DIR / "fyp_quant_integrated_agentic_stack.png"

WIDTH = 2560
HEIGHT = 1820


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
    accent: str


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
    eyebrow: str


SECTIONS = [
    Section(
        "Repository Folder Hierarchy",
        "What lives where in fyp_quant/",
        60,
        220,
        650,
        1180,
        "#FFFFFF",
        "#CBD5E1",
        "folder map",
    ),
    Section(
        "Agent Harness Control Plane",
        "How every fresh coding session is scoped, constrained, checked, and handed off",
        750,
        220,
        1040,
        760,
        "#FFFFFF",
        "#CBD5E1",
        "repo-native memory",
    ),
    Section(
        "Context-Isolated Multi-Agent Team",
        "Project subagents and skills keep review context small",
        1830,
        220,
        670,
        760,
        "#FFFFFF",
        "#CBD5E1",
        "subagents",
    ),
    Section(
        "Research Execution + Evidence Contract",
        "The benchmark pipeline runs below the harness; raw evidence stays protected",
        750,
        1165,
        1750,
        350,
        "#FFFFFF",
        "#CBD5E1",
        "scientific integrity",
    ),
]


FOLDER_GROUPS = [
    (
        "Agent OS",
        "#2F80C5",
        (
            "AGENTS.md / CLAUDE.md",
            "docs/PROJECT_LOG.md",
            "docs/HANDOFF.md",
            "docs/AGENT_DASHBOARD.md",
            "docs/agent_tasks/",
            ".agents/skills/",
            ".codex/agents/",
            ".codex/hooks/",
            ".github/workflows/",
        ),
    ),
    (
        "Research Core",
        "#2C9A67",
        (
            "ethical_benchmark/benchmarks/",
            "ethical_benchmark/models/",
            "ethical_benchmark/pipeline/",
            "ethical_benchmark/analysis/",
            "ethical_benchmark/judges/",
            "ethical_benchmark/cluster/",
            "ethical_benchmark/quant/",
            "ethical_benchmark/metrics/",
        ),
    ),
    (
        "Inputs + Execution",
        "#C6812D",
        (
            "configs/default.yaml",
            "configs/tc1.yaml",
            "data/xstest_v2_prompts.csv",
            "slurm/jobs_tc1*/",
            "scripts/",
            "fyp_cli.py",
            "Makefile",
        ),
    ),
    (
        "Evidence + Deliverables",
        "#8A63C8",
        (
            "results/*/*/raw.jsonl",
            "results/*/*/summary.json",
            "scores.v2.* sidecars",
            "scores.judge.* sidecars",
            "results/analysis/",
            "docs/FYP_Report_*.docx",
            "docs/architecture/",
        ),
    ),
]

HARNESS_CARDS = [
    Card(
        "entry",
        "1. Session Entry",
        "Human asks Codex / Claude / Cursor",
        ("Start from repo, not chat memory", "Use task intent + current git state"),
        800,
        335,
        295,
        145,
        "#EEF6FF",
        "#2F80C5",
        "#2F80C5",
    ),
    Card(
        "orientation",
        "2. Orientation Kernel",
        "Durable law + live state",
        ("AGENTS.md / CLAUDE.md", "docs/PROJECT_LOG.md", "agent-status"),
        1130,
        335,
        295,
        145,
        "#F8FAFC",
        "#64748B",
        "#2F80C5",
    ),
    Card(
        "router",
        "3. Task Router",
        "Small startup packet",
        ("agent-start --task <T>", "docs/agent_tasks/", "recommended subagent"),
        1460,
        335,
        280,
        145,
        "#F2FAF5",
        "#2C9A67",
        "#2C9A67",
    ),
    Card(
        "context",
        "4. Context Isolation",
        "Pull only relevant context",
        (".agents/skills/", ".codex/agents/", "task-named files only"),
        800,
        545,
        295,
        145,
        "#F7F0FF",
        "#8A63C8",
        "#8A63C8",
    ),
    Card(
        "guardrails",
        "5. Guardrails",
        "Rules become checks",
        ("artifact_policy.yaml", "Codex hooks", "GitHub Actions"),
        1130,
        545,
        295,
        145,
        "#FFF5E8",
        "#C6812D",
        "#C6812D",
    ),
    Card(
        "verify",
        "6. Verification Gate",
        "Finish with evidence",
        ("make agent-check", "harness_eval", "pytest: 186"),
        1460,
        545,
        280,
        145,
        "#EFFAFB",
        "#2494A3",
        "#2494A3",
    ),
    Card(
        "handoff",
        "7. Recovery Layer",
        "Next agent resumes safely",
        ("docs/HANDOFF.md", "AGENT_DASHBOARD.md", "TC1 checklist"),
        985,
        755,
        370,
        145,
        "#FFF0F3",
        "#C85C74",
        "#C85C74",
    ),
]

AGENT_CARDS = [
    Card(
        "main-agent",
        "Main Codex Session",
        "Plans, edits, verifies, logs",
        ("owns final changes", "updates PROJECT_LOG", "runs finish gate"),
        2030,
        350,
        280,
        145,
        "#EEF6FF",
        "#2F80C5",
        "#2F80C5",
    ),
    Card(
        "report-auditor",
        "fyp-report-auditor",
        "Report/log consistency",
        ("judge-primary claims", "stale text", "report freshness"),
        1870,
        545,
        260,
        135,
        "#F8FAFC",
        "#64748B",
        "#64748B",
    ),
    Card(
        "artifact-guardian",
        "fyp-artifact-guardian",
        "Evidence-chain audit",
        ("raw immutability", "redaction leaks", "artifact policy"),
        2170,
        545,
        260,
        135,
        "#FFF5E8",
        "#C6812D",
        "#C6812D",
    ),
    Card(
        "tc1-ops",
        "fyp-tc1-ops",
        "Cluster workflow safety",
        ("sbatch-only", "offline env vars", "SLURM scripts"),
        1870,
        720,
        260,
        135,
        "#EFFAFB",
        "#2494A3",
        "#2494A3",
    ),
    Card(
        "judge-reviewer",
        "fyp-judge-reviewer",
        "Scoring validation",
        ("judge sidecars", "agreement analysis", "second judge"),
        2170,
        720,
        260,
        135,
        "#F7F0FF",
        "#8A63C8",
        "#8A63C8",
    ),
    Card(
        "meetup-story",
        "fyp-meetup-story",
        "Human explanation",
        ("USP narrative", "demo + memory"),
        2030,
        875,
        280,
        95,
        "#F2FAF5",
        "#2C9A67",
        "#2C9A67",
    ),
]

PIPELINE_CARDS = [
    Card(
        "configs",
        "Config + Data",
        "Study inputs",
        ("configs/*.yaml", "XSTest CSV", "HF cached datasets"),
        800,
        1295,
        230,
        135,
        "#FFF5E8",
        "#C6812D",
        "#C6812D",
    ),
    Card(
        "loader",
        "Model Loading",
        "Matched baseline / NF4",
        ("HFModelLoader", "BitsAndBytesConfig", "dtype/device"),
        1060,
        1295,
        230,
        135,
        "#EEF6FF",
        "#2F80C5",
        "#2F80C5",
    ),
    Card(
        "benchmarks",
        "Benchmarks",
        "Plugins score outputs",
        ("HarmBench", "XSTest", "MMLU subset"),
        1320,
        1295,
        230,
        135,
        "#F2FAF5",
        "#2C9A67",
        "#2C9A67",
    ),
    Card(
        "artifacts",
        "Raw Artifacts",
        "TC1-original evidence",
        ("raw.jsonl", "summary.json", "immutable manifest"),
        1580,
        1295,
        230,
        135,
        "#FFF0F3",
        "#C85C74",
        "#C85C74",
    ),
    Card(
        "sidecars",
        "Derived Scoring",
        "No overwrites",
        ("scores.v2.*", "scores.judge.*", "redacted only"),
        1840,
        1295,
        230,
        135,
        "#F7F0FF",
        "#8A63C8",
        "#8A63C8",
    ),
    Card(
        "analysis",
        "Analysis + Report",
        "Defensible output",
        ("pairwise deltas", "judge agreement", "FYP report"),
        2100,
        1295,
        230,
        135,
        "#EFFAFB",
        "#2494A3",
        "#2494A3",
    ),
]

CHECK_ITEMS = (
    ("agent-doc-sync", "AGENTS/CLAUDE synced"),
    ("project-log-update", "PROJECT_LOG changed"),
    ("immutable-artifacts", "36 raw files hashed"),
    ("redaction", "no raw HarmBench text"),
    ("stale-text", "no stale current claims"),
    ("report-freshness", "docx required when report-worthy"),
    ("git-diff-check", "clean whitespace"),
    ("pytest", "186 tests"),
)


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


def _section_svg(section: Section) -> str:
    return "\n".join(
        [
            f'<rect x="{section.x + 8}" y="{section.y + 10}" width="{section.w}" height="{section.h}" rx="30" fill="#0F172A" opacity="0.08"/>',
            f'<rect x="{section.x}" y="{section.y}" width="{section.w}" height="{section.h}" rx="30" fill="{section.fill}" stroke="{section.stroke}" stroke-width="2.2"/>',
            f'<rect x="{section.x + 24}" y="{section.y + 24}" width="{section.w - 48}" height="64" rx="20" fill="#F1F5F9"/>',
            _text([section.title], section.x + 46, section.y + 62, 25, "#0F172A", 850),
            _text([section.subtitle], section.x + 46, section.y + 88, 14, "#475569", 600),
            f'<rect x="{section.x + section.w - 184}" y="{section.y + 38}" width="140" height="28" rx="14" fill="#E0F2FE" stroke="#7DD3FC"/>',
            _text([section.eyebrow], section.x + section.w - 114, section.y + 58, 12, "#075985", 800, anchor="middle"),
        ]
    )


def _card_svg(card: Card, title_size: int = 19, body_size: int = 13, wrap_width: int = 30) -> str:
    shadow = f'<rect x="{card.x + 5}" y="{card.y + 7}" width="{card.w}" height="{card.h}" rx="18" fill="#0F172A" opacity="0.08"/>'
    rect = (
        f'<rect x="{card.x}" y="{card.y}" width="{card.w}" height="{card.h}" rx="18" '
        f'fill="{card.fill}" stroke="{card.stroke}" stroke-width="2"/>'
    )
    accent = f'<rect x="{card.x}" y="{card.y}" width="7" height="{card.h}" rx="4" fill="{card.accent}"/>'
    title = _text([card.title], card.x + 24, card.y + 32, title_size, "#0F172A", 820)
    subtitle = _text([card.subtitle], card.x + 24, card.y + 56, 12, "#475569", 650)
    body_lines: list[str] = []
    for item in card.items:
        body_lines.extend(_wrap(f"- {item}", wrap_width))
    body = _text(body_lines, card.x + 24, card.y + 86, body_size, "#334155", 520, line_height=19)
    return "\n".join([shadow, rect, accent, title, subtitle, body])


def _mini_card(x: int, y: int, w: int, title: str, subtitle: str, color: str) -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{w}" height="54" rx="14" fill="#FFFFFF" stroke="{color}" stroke-width="1.8"/>',
            f'<circle cx="{x + 24}" cy="{y + 27}" r="8" fill="{color}"/>',
            _text([title], x + 44, y + 24, 13, "#0F172A", 760),
            _text([subtitle], x + 44, y + 42, 11, "#475569", 550),
        ]
    )


def _center(card: Card) -> tuple[int, int]:
    return card.x + card.w // 2, card.y + card.h // 2


def _right(card: Card) -> tuple[int, int]:
    return card.x + card.w, card.y + card.h // 2


def _left(card: Card) -> tuple[int, int]:
    return card.x, card.y + card.h // 2


def _top(card: Card) -> tuple[int, int]:
    return card.x + card.w // 2, card.y


def _bottom(card: Card) -> tuple[int, int]:
    return card.x + card.w // 2, card.y + card.h


def _arrow(x1: int, y1: int, x2: int, y2: int, color: str = "#64748B", width: int = 3, dashed: bool = False) -> str:
    if abs(y1 - y2) < 8:
        path = f"M {x1} {y1} C {x1 + 60} {y1}, {x2 - 60} {y2}, {x2} {y2}"
    elif abs(x1 - x2) < 8:
        path = f"M {x1} {y1} C {x1} {y1 + 55}, {x2} {y2 - 55}, {x2} {y2}"
    else:
        mid_y = (y1 + y2) // 2
        path = f"M {x1} {y1} C {x1} {mid_y}, {x2} {mid_y}, {x2} {y2}"
    dash_attr = ' stroke-dasharray="9 7"' if dashed else ""
    return f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}"{dash_attr} marker-end="url(#{_marker_id(color)})" opacity="0.82"/>'


def _render_folder_tree() -> str:
    parts = [
        '<rect x="92" y="328" width="586" height="58" rx="18" fill="#0F3B66"/>',
        _text(["fyp_quant/"], 124, 365, 25, "#FFFFFF", 850),
        _text(["research repo + agent operating system"], 300, 365, 14, "#DDEBFF", 600),
    ]
    group_y = 420
    for title, color, items in FOLDER_GROUPS:
        parts.append(f'<rect x="92" y="{group_y}" width="586" height="206" rx="18" fill="#F8FAFC" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<rect x="92" y="{group_y}" width="7" height="206" rx="4" fill="{color}"/>')
        parts.append(_text([title], 120, group_y + 34, 19, "#0F172A", 820))
        item_y = group_y + 65
        left_items = items[:5]
        right_items = items[5:]
        for idx, item in enumerate(left_items):
            parts.append(_text([f"- {item}"], 120, item_y + idx * 25, 13, "#334155", 540))
        for idx, item in enumerate(right_items):
            parts.append(_text([f"- {item}"], 365, item_y + idx * 25, 13, "#334155", 540))
        parts.append(_arrow(385, 386 if group_y == 420 else group_y - 8, 385, group_y, color, 2))
        group_y += 235
    return "\n".join(parts)


def _render_checks() -> str:
    x0, y0 = 800, 910
    parts = [
        '<rect x="800" y="910" width="940" height="44" rx="16" fill="#0F172A" opacity="0.95"/>',
        _text(["make agent-check = finish gate"], 830, 939, 18, "#FFFFFF", 850),
        _text(["docs sync, log discipline, artifact immutability, privacy, stale text, report freshness, diff, tests"], 1120, 938, 13, "#DDEBFF", 600),
    ]
    for idx, (title, subtitle) in enumerate(CHECK_ITEMS):
        row = idx // 4
        col = idx % 4
        parts.append(_mini_card(x0 + col * 238, y0 + 68 + row * 66, 218, title, subtitle, "#2494A3"))
    return "\n".join(parts)


def _render_skills() -> str:
    return "\n".join(
        [
            '<rect x="1870" y="998" width="590" height="150" rx="22" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="2"/>',
            _text(["Repo-Scoped Skills"], 1902, 1034, 20, "#0F172A", 850),
            _mini_card(1902, 1056, 260, "fyp-report-audit", "load only report workflow", "#64748B"),
            _mini_card(2178, 1056, 248, "fyp-second-judge", "judge extension workflow", "#8A63C8"),
            _mini_card(1902, 1120, 260, "fyp-meetup-brief", "presentation story", "#2C9A67"),
            _mini_card(2178, 1120, 248, "fyp-harness-maintenance", "harness changes", "#C6812D"),
        ]
    )


def _render_contract_band() -> str:
    return "\n".join(
        [
            '<rect x="60" y="1595" width="2440" height="175" rx="28" fill="#0F172A" opacity="0.96"/>',
            _text(["Core USP"], 96, 1643, 28, "#FFFFFF", 850),
            _text(["Memory is not magic; make the repo remember."], 96, 1680, 18, "#DDEBFF", 700),
            _mini_card(545, 1632, 360, "Durable state", "PROJECT_LOG is source of truth", "#2F80C5"),
            _mini_card(930, 1632, 360, "Context economy", "task packets + skills on demand", "#8A63C8"),
            _mini_card(1315, 1632, 360, "Safety rails", "immutable raw + redacted sidecars", "#C6812D"),
            _mini_card(1700, 1632, 360, "Verification", "make agent-check before finish", "#2494A3"),
            _mini_card(2085, 1632, 360, "Recoverability", "handoff/dashboard for next agent", "#C85C74"),
            _text(
                [
                    "Presentation line: I do not just prompt Codex; I built a repo-native operating system that scopes agents, protects evidence, and makes each session recoverable.",
                ],
                545,
                1728,
                16,
                "#E2E8F0",
                650,
            ),
        ]
    )


def render_svg() -> str:
    cards = {card.key: card for card in HARNESS_CARDS + AGENT_CARDS + PIPELINE_CARDS}
    marker_colors = {"#64748B", "#334155", "#2F80C5", "#2C9A67", "#8A63C8", "#C6812D", "#2494A3", "#C85C74"}
    markers = [
        f'<marker id="{_marker_id(color)}" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">'
        f'<path d="M2,2 L10,6 L2,10 Z" fill="{color}"/></marker>'
        for color in sorted(marker_colors)
    ]
    header = "\n".join(
        [
            f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#F8FAFC"/>',
            f'<rect x="0" y="0" width="{WIDTH}" height="168" fill="url(#headerGrad)"/>',
            _text(["fyp_quant Integrated Agentic Research Stack"], 74, 72, 44, "#FFFFFF", 850),
            _text(["One poster: folder hierarchy, harness control plane, multi-agent roles, benchmark pipeline, and evidence guardrails"], 76, 112, 19, "#DDEBFF", 560),
            '<rect x="2050" y="54" width="350" height="48" rx="24" fill="#FFFFFF" opacity="0.14" stroke="#DDEBFF"/>',
            _text(["software stack diagram"], 2225, 85, 16, "#EAF5FF", 850, anchor="middle"),
            _text(["ImageGen-inspired styling; exact labels rendered in SVG"], 2055, 124, 14, "#BFD7FF", 600),
        ]
    )
    harness_arrows = [
        _arrow(*_right(cards["entry"]), *_left(cards["orientation"]), "#64748B"),
        _arrow(*_right(cards["orientation"]), *_left(cards["router"]), "#64748B"),
        _arrow(*_bottom(cards["entry"]), *_top(cards["context"]), "#64748B"),
        _arrow(*_bottom(cards["orientation"]), *_top(cards["guardrails"]), "#64748B"),
        _arrow(*_bottom(cards["router"]), *_top(cards["verify"]), "#64748B"),
        _arrow(*_right(cards["context"]), *_left(cards["guardrails"]), "#64748B"),
        _arrow(*_right(cards["guardrails"]), *_left(cards["verify"]), "#64748B"),
        _arrow(*_bottom(cards["guardrails"]), *_top(cards["handoff"]), "#64748B"),
        _arrow(*_bottom(cards["verify"]), *_top(cards["handoff"]), "#64748B"),
        _arrow(710, 822, 800, 822, "#2F80C5", 3, dashed=True),
        _arrow(1740, 822, 1830, 822, "#8A63C8", 3, dashed=True),
        _arrow(1260, 900, 1260, 1165, "#2494A3", 3),
    ]
    agent_arrows = [
        _arrow(*_bottom(cards["main-agent"]), *_top(cards["report-auditor"]), "#64748B", 2, dashed=True),
        _arrow(*_bottom(cards["main-agent"]), *_top(cards["artifact-guardian"]), "#64748B", 2, dashed=True),
        _arrow(*_bottom(cards["report-auditor"]), *_top(cards["tc1-ops"]), "#64748B", 2, dashed=True),
        _arrow(*_bottom(cards["artifact-guardian"]), *_top(cards["judge-reviewer"]), "#64748B", 2, dashed=True),
        _arrow(*_bottom(cards["tc1-ops"]), *_top(cards["meetup-story"]), "#64748B", 2, dashed=True),
        _arrow(*_bottom(cards["judge-reviewer"]), *_top(cards["meetup-story"]), "#64748B", 2, dashed=True),
    ]
    pipeline_arrows = []
    for left, right in zip(PIPELINE_CARDS, PIPELINE_CARDS[1:]):
        pipeline_arrows.append(_arrow(*_right(left), *_left(right), "#64748B", 3))
    side_notes = "\n".join(
        [
            '<rect x="800" y="1448" width="730" height="58" rx="18" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="2"/>',
            _text(["Runtime path"], 830, 1474, 14, "#0F172A", 800),
            _text(["make smoke/run/matrix -> fyp_cli.py -> pipeline -> results"], 955, 1474, 13, "#334155", 600),
            '<rect x="1570" y="1448" width="745" height="58" rx="18" fill="#FFF5E8" stroke="#C6812D" stroke-width="2"/>',
            _text(["Artifact rule"], 1600, 1474, 14, "#0F172A", 800),
            _text(["raw/summary immutable; rescoring writes sidecars; docs log every change"], 1730, 1474, 13, "#334155", 600),
        ]
    )
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
            "<defs>",
            '<linearGradient id="headerGrad" x1="0%" y1="0%" x2="100%"><stop offset="0%" stop-color="#12355B"/><stop offset="38%" stop-color="#0F766E"/><stop offset="72%" stop-color="#4C5D95"/><stop offset="100%" stop-color="#6D4C8D"/></linearGradient>',
            *markers,
            "</defs>",
            header,
            *[_section_svg(section) for section in SECTIONS],
            _render_folder_tree(),
            *harness_arrows,
            *[_card_svg(card, title_size=17, body_size=12, wrap_width=28) for card in HARNESS_CARDS],
            _render_checks(),
            *agent_arrows,
            *[_card_svg(card, title_size=16, body_size=12, wrap_width=27) for card in AGENT_CARDS],
            _render_skills(),
            *pipeline_arrows,
            *[_card_svg(card, title_size=16, body_size=12, wrap_width=24) for card in PIPELINE_CARDS],
            side_notes,
            _render_contract_band(),
            "</svg>",
        ]
    )


def _mx_cell(container: ET.Element, cell_id: str, **attrs: str) -> ET.Element:
    attrs.setdefault("id", cell_id)
    return ET.SubElement(container, "mxCell", attrs)


def _drawio_label(card: Card) -> str:
    return "<br>".join([f"<b>{escape(card.title)}</b>", escape(card.subtitle), *[escape(item) for item in card.items]])


def _drawio_style(card: Card) -> str:
    return (
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
        f"fillColor={card.fill};strokeColor={card.stroke};fontColor=#102033;"
        "fontFamily=Inter;fontSize=13;align=left;verticalAlign=top;spacingLeft=16;spacingTop=12;shadow=1;"
    )


def render_drawio() -> str:
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "modified": "2026-06-09T00:00:00+08:00",
            "agent": "Codex",
            "version": "24.7.17",
            "type": "device",
        },
    )
    diagram = ET.SubElement(mxfile, "diagram", {"id": "fyp-integrated-agentic-stack", "name": "Integrated Agentic Stack"})
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
            "fyp_quant Integrated Agentic Research Stack<br>"
            "<font style='font-size: 16px'>Folder hierarchy + agent harness + multi-agent roles + benchmark/evidence pipeline</font>"
        ),
        style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#12355B;strokeColor=#12355B;fontColor=#FFFFFF;fontFamily=Inter;fontSize=28;fontStyle=1;spacingLeft=24;spacingTop=20;",
        vertex="1",
        parent="1",
    )
    title.append(ET.Element("mxGeometry", {"x": "40", "y": "30", "width": "2480", "height": "112", "as": "geometry"}))

    for idx, section in enumerate(SECTIONS, start=1):
        cell = _mx_cell(
            root,
            f"section-{idx}",
            value=f"<b>{escape(section.title)}</b><br>{escape(section.subtitle)}",
            style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;fontColor=#0F172A;fontFamily=Inter;fontSize=18;align=left;verticalAlign=top;spacingLeft=24;spacingTop=20;",
            vertex="1",
            parent="1",
        )
        cell.append(
            ET.Element(
                "mxGeometry",
                {"x": str(section.x), "y": str(section.y), "width": str(section.w), "height": str(section.h), "as": "geometry"},
            )
        )

    all_cards = HARNESS_CARDS + AGENT_CARDS + PIPELINE_CARDS
    for card in all_cards:
        cell = _mx_cell(root, card.key, value=_drawio_label(card), style=_drawio_style(card), vertex="1", parent="1")
        cell.append(
            ET.Element(
                "mxGeometry",
                {"x": str(card.x), "y": str(card.y), "width": str(card.w), "height": str(card.h), "as": "geometry"},
            )
        )

    folder = _mx_cell(
        root,
        "folder-tree",
        value="<b>fyp_quant/</b><br>" + "<br>".join(
            [
                "<b>Agent OS</b>: AGENTS, PROJECT_LOG, handoff, tasks, skills, subagents, hooks, CI",
                "<b>Research Core</b>: benchmarks, models, pipeline, analysis, judges, cluster, quant, metrics",
                "<b>Inputs + Execution</b>: configs, data, slurm, scripts, fyp_cli.py, Makefile",
                "<b>Evidence + Deliverables</b>: raw artifacts, sidecars, analysis, report, visuals",
            ]
        ),
        style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#0F3B66;fontColor=#0F172A;fontFamily=Inter;fontSize=16;align=left;verticalAlign=top;spacingLeft=18;spacingTop=18;shadow=1;",
        vertex="1",
        parent="1",
    )
    folder.append(ET.Element("mxGeometry", {"x": "92", "y": "328", "width": "586", "height": "990", "as": "geometry"}))

    check = _mx_cell(
        root,
        "agent-check",
        value="<b>make agent-check</b><br>agent-doc-sync, project-log-update, immutable-artifacts, redaction, stale-text, report-freshness, git-diff-check, pytest",
        style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#0F172A;strokeColor=#0F172A;fontColor=#DDEBFF;fontFamily=Inter;fontSize=15;spacingLeft=20;spacingTop=15;",
        vertex="1",
        parent="1",
    )
    check.append(ET.Element("mxGeometry", {"x": "800", "y": "910", "width": "940", "height": "160", "as": "geometry"}))

    skills = _mx_cell(
        root,
        "skills",
        value="<b>Repo-Scoped Skills</b><br>fyp-report-audit, fyp-second-judge, fyp-meetup-brief, fyp-harness-maintenance",
        style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#CBD5E1;fontColor=#0F172A;fontFamily=Inter;fontSize=15;spacingLeft=20;spacingTop=15;",
        vertex="1",
        parent="1",
    )
    skills.append(ET.Element("mxGeometry", {"x": "1870", "y": "998", "width": "590", "height": "150", "as": "geometry"}))

    usp = _mx_cell(
        root,
        "usp-band",
        value="<b>Core USP</b><br>Memory is not magic; make the repo remember. The repo scopes agents, protects evidence, verifies claims, and makes each session recoverable.",
        style="rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#0F172A;strokeColor=#0F172A;fontColor=#DDEBFF;fontFamily=Inter;fontSize=17;spacingLeft=24;spacingTop=18;",
        vertex="1",
        parent="1",
    )
    usp.append(ET.Element("mxGeometry", {"x": "60", "y": "1595", "width": "2440", "height": "175", "as": "geometry"}))

    edge_pairs = [
        ("entry", "orientation"),
        ("orientation", "router"),
        ("entry", "context"),
        ("orientation", "guardrails"),
        ("router", "verify"),
        ("context", "guardrails"),
        ("guardrails", "verify"),
        ("guardrails", "handoff"),
        ("verify", "handoff"),
        ("main-agent", "report-auditor"),
        ("main-agent", "artifact-guardian"),
        ("report-auditor", "tc1-ops"),
        ("artifact-guardian", "judge-reviewer"),
        ("tc1-ops", "meetup-story"),
        ("judge-reviewer", "meetup-story"),
        ("configs", "loader"),
        ("loader", "benchmarks"),
        ("benchmarks", "artifacts"),
        ("artifacts", "sidecars"),
        ("sidecars", "analysis"),
    ]
    for idx, (src, dst) in enumerate(edge_pairs, start=1):
        cell = _mx_cell(
            root,
            f"edge-{idx}",
            value="",
            style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#64748B;strokeWidth=3;endArrow=block;endFill=1;",
            edge="1",
            parent="1",
            source=src,
            target=dst,
        )
        cell.append(ET.Element("mxGeometry", {"relative": "1", "as": "geometry"}))

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
