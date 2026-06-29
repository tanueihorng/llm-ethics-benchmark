"""Editorial-light theme for the fyp_quant dashboard.

Matches the FYP showcase-deck identity: warm paper, Newsreader serif headers,
Spline Sans body, IBM Plex Mono numerals, indigo primary + gold accent. Holds
three things, all Streamlit-free so they stay unit-testable:

- design tokens + the injected stylesheet (:func:`CSS` constant),
- a single Plotly styler (:func:`style_fig`) so every chart shares one identity,
- editorial HTML builders (hero, stat tiles, per-pair cards, label badge).

Token discipline: the stylesheet declares every colour/font/space as a CSS
custom property and references it by name; the Python colour constants below are
the matching token layer for the Plotly charts (Plotly needs literal values).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from dashboard import data as D

# --------------------------------------------------------------------------- #
# Tokens (the chart-layer mirror of the CSS :root block)                       #
# --------------------------------------------------------------------------- #
PAPER = "#faf7f1"        # warm off-white page
CARD = "#fffdf8"         # near-white card surface
INK = "#1b1b29"          # primary text
MUTED = "#6f6a7d"        # secondary text
INDIGO = "#4f46e5"       # primary accent
INDIGO_SOFT = "#a5b4fc"  # tint for fills/gridless emphasis
GOLD = "#c0892d"         # secondary accent
TERRA = "#b4452e"        # "concerning" / quantized direction
TEAL = "#2a8a7f"
PLUM = "#7c3aed"
GRID = "#e8e2d4"         # warm hairline gridlines
LINE = "#e2dccd"         # warm hairline borders
GREY = "#b3ab9b"         # n.s. / neutral bars

FONT_DISPLAY = "Newsreader, Georgia, 'Times New Roman', serif"
FONT_BODY = "'Spline Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
FONT_MONO = "'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, monospace"

#: Plotly categorical sequence, brand-ordered.
COLORWAY: Sequence[str] = [INDIGO, GOLD, TERRA, TEAL, PLUM, "#475569"]


# --------------------------------------------------------------------------- #
# Stylesheet                                                                   #
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Spline+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --paper: #faf7f1;
  --card: #fffdf8;
  --ink: #1b1b29;
  --muted: #6f6a7d;
  --indigo: #4f46e5;
  --gold: #c0892d;
  --line: #e2dccd;
  --line-soft: #efe9dc;
  --font-display: Newsreader, Georgia, 'Times New Roman', serif;
  --font-body: 'Spline Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-mono: 'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, monospace;
  --space-sm: 8px;
  --space-md: 16px;
  --space-lg: 28px;
  --radius: 12px;
}

/* ---- base ------------------------------------------------------------ */
.stApp { background: var(--paper); }
html, body, [class*="st-"], .stApp, .stMarkdown, p, span, div, label, input, button, select, textarea {
  font-family: var(--font-body);
  color: var(--ink);
}
.block-container { padding-top: 2.2rem; max-width: 1320px; }

/* hide Streamlit chrome for an app-grade feel */
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, header [data-testid="stStatusWidget"], footer { display: none !important; }
[data-testid="stHeader"] { background: transparent; }

/* ---- typography ------------------------------------------------------ */
h1, h2, h3, h4,
[data-testid="stHeading"] h1, [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {
  font-family: var(--font-display) !important;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--ink);
}
code, kbd, pre, .stCode, [data-testid="stMetricValue"] { font-family: var(--font-mono) !important; }

/* ---- hero ------------------------------------------------------------ */
.hero { margin: 0 0 1.4rem 0; }
.hero__eyebrow {
  font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--indigo); margin-bottom: 0.5rem;
}
.hero__title {
  font-family: var(--font-display); font-weight: 600; font-size: 2.5rem;
  line-height: 1.05; letter-spacing: -0.02em; color: var(--ink);
  margin: 0; overflow-wrap: anywhere;
}
.hero__rule { height: 3px; width: 64px; background: var(--gold); border-radius: 2px; margin: 0.85rem 0 0.7rem; }
.hero__dek { color: var(--muted); font-size: 1.0rem; max-width: 68ch; line-height: 1.5; }

/* ---- stat tiles ------------------------------------------------------ */
.stat-row { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: var(--space-md); margin: 0.4rem 0 1.4rem; }
.stat {
  background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 0.9rem 1.1rem;
}
.stat__label { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.stat__value { font-family: var(--font-mono); font-size: 1.9rem; font-weight: 600; color: var(--ink); line-height: 1.1; margin-top: 0.2rem; }
.stat__value .accent { color: var(--indigo); }

/* ---- per-pair card --------------------------------------------------- */
.pair-card {
  background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 1.05rem 1.25rem; margin-bottom: 0.9rem;
  transition: border-color .18s ease, box-shadow .18s ease;
}
.pair-card:hover { border-color: var(--indigo); box-shadow: 0 6px 22px -14px rgba(79,70,229,.45); }
.pair-card__head { display: flex; align-items: center; gap: 0.7rem; flex-wrap: wrap; }
.pair-card__id { font-family: var(--font-mono); font-size: 1.15rem; font-weight: 600; color: var(--ink); }
.pair-card__gloss { color: var(--muted); font-size: 0.9rem; margin-top: 0.35rem; }
.pair-card__stat { font-family: var(--font-mono); font-size: 0.78rem; color: var(--muted); margin-top: 0.3rem; }
.pair-card__deltas { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: var(--space-md); margin-top: 0.85rem; }
.delta { border-left: 2px solid var(--line-soft); padding-left: 0.7rem; }
.delta__label { font-size: 0.72rem; color: var(--muted); }
.delta__val { font-family: var(--font-mono); font-size: 1.35rem; font-weight: 600; color: var(--ink); line-height: 1.2; }
.delta__sig { font-size: 0.72rem; color: var(--muted); }
.delta__sig.is-sig { color: var(--terra, #b4452e); }

/* ---- label badge ----------------------------------------------------- */
.label-badge {
  display: inline-block; padding: 3px 11px; border-radius: 999px; color: #fff;
  font-family: var(--font-mono); font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.02em; white-space: nowrap;
}
.muted { color: var(--muted); font-size: 0.85rem; }

/* ---- sidebar --------------------------------------------------------- */
section[data-testid="stSidebar"] { background: #f4eee1; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
section[data-testid="stSidebar"] h1 { font-size: 1.5rem; }

/* nav radio → editorial list */
section[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
section[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: 6px 10px; border-radius: 8px; transition: background .15s ease; cursor: pointer;
  font-size: 0.96rem;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #ece4d3; }

/* ---- metrics (built-in) --------------------------------------------- */
[data-testid="stMetric"] {
  background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 0.7rem 0.9rem;
}
[data-testid="stMetricValue"] { color: var(--ink); font-weight: 600; }
[data-testid="stMetricLabel"] p { color: var(--muted); font-size: 0.78rem; }

/* ---- bordered containers (cards) ------------------------------------ */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--card); border-radius: var(--radius);
}

/* ---- buttons -------------------------------------------------------- */
.stButton > button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"], [data-testid="stDownloadButton"] button {
  border-radius: 9px; font-weight: 600; font-family: var(--font-body); border: 1px solid var(--line);
  transition: transform .12s ease, box-shadow .12s ease, background .15s ease;
}
[data-testid="stBaseButton-primary"] {
  background: var(--indigo); border-color: var(--indigo); color: #fff;
}
[data-testid="stBaseButton-primary"]:hover { box-shadow: 0 6px 18px -8px rgba(79,70,229,.6); transform: translateY(-1px); }
.stButton > button:active { transform: translateY(1px); }

/* ---- tabs ----------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-weight: 600; color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--indigo) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--indigo); }

/* ---- inputs / selects ----------------------------------------------- */
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
  border-radius: 9px !important; border-color: var(--line) !important; background: var(--card) !important;
}

/* ---- alerts (tint to brand) ----------------------------------------- */
.stAlert { border-radius: var(--radius); border: 1px solid var(--line); }

/* ---- dataframe ------------------------------------------------------ */
[data-testid="stDataFrame"] { border-radius: var(--radius); border: 1px solid var(--line); }

/* ---- mobile --------------------------------------------------------- */
@media (max-width: 768px) {
  .stat-row { grid-template-columns: 1fr 1fr; }
  .pair-card__deltas { grid-template-columns: 1fr; }
  .hero__title { font-size: 2rem; }
}
</style>
"""


# --------------------------------------------------------------------------- #
# Plotly styler — one identity for every chart                                 #
# --------------------------------------------------------------------------- #
def style_fig(fig: Any, height: Optional[int] = None) -> Any:
    """Applies the editorial theme to a Plotly figure (in place) and returns it."""
    axis = dict(
        gridcolor=GRID, zerolinecolor=GRID, linecolor=LINE,
        tickfont=dict(family=FONT_BODY, color=MUTED, size=12),
        title_font=dict(family=FONT_BODY, color=MUTED, size=13),
    )
    fig.update_layout(
        font=dict(family=FONT_BODY, color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=list(COLORWAY),
        margin=dict(l=10, r=18, t=30, b=10),
        legend=dict(font=dict(family=FONT_BODY, color=INK, size=12)),
        hoverlabel=dict(font_family=FONT_BODY, bgcolor=CARD, font_color=INK, bordercolor=LINE),
        xaxis=axis,
        yaxis=axis,
    )
    if height:
        fig.update_layout(height=height)
    return fig


# --------------------------------------------------------------------------- #
# Editorial HTML builders (pure string functions; testable)                    #
# --------------------------------------------------------------------------- #
def badge(label: Optional[str]) -> str:
    """Coloured pill for an interpretation label."""
    color = D.label_color(label)
    return f'<span class="label-badge" style="background:{color}">{label or "—"}</span>'


def hero_html(title: str, dek: Optional[str] = None, eyebrow: Optional[str] = None) -> str:
    """Editorial page header: optional mono eyebrow, serif title, gold rule, dek."""
    parts = ['<div class="hero">']
    if eyebrow:
        parts.append(f'<div class="hero__eyebrow">{eyebrow}</div>')
    parts.append(f'<h1 class="hero__title">{title}</h1>')
    parts.append('<div class="hero__rule"></div>')
    if dek:
        parts.append(f'<div class="hero__dek">{dek}</div>')
    parts.append("</div>")
    return "".join(parts)


def stat_tiles_html(items: Sequence[tuple]) -> str:
    """Row of stat tiles from ``[(label, value), ...]`` (value may be HTML)."""
    cells = "".join(
        f'<div class="stat"><div class="stat__label">{lab}</div>'
        f'<div class="stat__value">{val}</div></div>'
        for lab, val in items
    )
    return f'<div class="stat-row">{cells}</div>'


def _fmt_pp(val: Optional[float]) -> str:
    return "—" if val is None else f"{val * 100:+.1f} pp"


def _delta_block(label: str, val: Optional[float], sig: Optional[bool]) -> str:
    sig_cls = "delta__sig is-sig" if sig else "delta__sig"
    sig_txt = "✓ significant" if sig else ("· n.s." if val is not None else "")
    return (
        f'<div class="delta"><div class="delta__label">{label}</div>'
        f'<div class="delta__val">{_fmt_pp(val)}</div>'
        f'<div class="{sig_cls}">{sig_txt}</div></div>'
    )


def pair_card_html(row: Dict[str, Any], judge_primary: bool) -> str:
    """Full per-pair interpretation card (head + gloss + stat line + 3 deltas)."""
    label = row.get("interpretation_label")
    head = (
        f'<div class="pair-card__head"><div class="pair-card__id">{row.get("pair_id","—")}</div>'
        f"{badge(label)}</div>"
    )
    gloss = (
        f'<div class="pair-card__gloss">evidence: <b>{row.get("evidence_status","—")}</b> · '
        f"{D.label_gloss(label)}</div>"
    )
    stat = ""
    if judge_primary and row.get("harmbench_asr_fp16") is not None:
        bits: List[str] = []
        if row.get("harmbench_asr_nf4") is not None:
            bits.append(f"ASR {row['harmbench_asr_fp16']:.3f}→{row['harmbench_asr_nf4']:.3f}")
        p = row.get("harmbench_asr_p_value")
        if isinstance(p, (int, float)):
            bits.append(f"McNemar p={p:.3f}")
        q = row.get("harmbench_asr_bh_q")
        if isinstance(q, (int, float)):
            surv = "survives FDR" if row.get("harmbench_asr_bh_significant") else "n.s. after FDR"
            bits.append(f"q={q:.3f} ({surv})")
        if bits:
            stat = f'<div class="pair-card__stat">{" · ".join(bits)}</div>'
    asr_label = "Δ HarmBench ASR (judge)" if judge_primary else "Δ HarmBench ASR (v2)"
    deltas = (
        '<div class="pair-card__deltas">'
        + _delta_block(asr_label, row.get("harmbench_asr_delta"), row.get("harmbench_asr_delta_significant"))
        + _delta_block("Δ XSTest over-refusal", row.get("xstest_over_refusal_delta"), row.get("xstest_over_refusal_delta_significant"))
        + _delta_block("Δ MMLU accuracy", row.get("mmlu_accuracy_delta"), row.get("mmlu_accuracy_delta_significant"))
        + "</div>"
    )
    return f'<div class="pair-card">{head}{gloss}{stat}{deltas}</div>'
