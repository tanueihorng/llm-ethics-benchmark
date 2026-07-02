"""Editorial-light theme for the fyp_quant dashboard.

/* Hallmark · genre: editorial · macrostructure: journal-workbench (masthead +
 * ruled findings band + dossier cards) · design-system: project deck identity
 * (Newsreader / Spline Sans / IBM Plex Mono · warm paper · indigo + gold)
 * · designed-as-app · motion: hover/press/focus only, no reveals */

Matches the FYP showcase-deck identity and elevates it: a journal masthead with
a double keyline, a newspaper-style findings band framed by rules, dossier-style
per-pair cards with a verdict spine, and small-caps mono kickers throughout.
Holds three things, all Streamlit-free so they stay unit-testable:

- design tokens + the injected stylesheet (:data:`CSS` constant),
- a single Plotly styler (:func:`style_fig`) so every chart shares one identity,
- editorial HTML builders (masthead/hero, stat band, per-pair dossier, badge).

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
GRID = "#eee8db"         # warm hairline gridlines
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
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;0,6..72,700&family=Spline+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --paper: #faf7f1;
  --paper-deep: #f4eee1;
  --card: #fffdf8;
  --ink: #1b1b29;
  --muted: #6f6a7d;
  --indigo: #4f46e5;
  --indigo-ink: #3730a3;
  --gold: #c0892d;
  --terra: #b4452e;
  --line: #e2dccd;
  --line-soft: #efe9dc;
  --rule-heavy: #1b1b29;
  --focus: #4f46e5;
  --font-display: Newsreader, Georgia, 'Times New Roman', serif;
  --font-body: 'Spline Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-mono: 'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, monospace;
  --space-2xs: 4px; --space-xs: 8px; --space-sm: 12px; --space-md: 16px;
  --space-lg: 28px; --space-xl: 44px;
  --radius: 12px; --radius-sm: 9px; --radius-pill: 999px;
  --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
  --dur: 180ms;
}

/* ---- base ------------------------------------------------------------ */
.stApp { background: var(--paper); }
/* journal double keyline across the very top of the app */
.stApp::before {
  content: ""; position: fixed; top: 0; left: 0; right: 0; height: 6px; z-index: 999;
  background:
    linear-gradient(var(--indigo), var(--indigo)) top/100% 3px no-repeat,
    linear-gradient(var(--gold), var(--gold)) bottom/100% 1.5px no-repeat;
  pointer-events: none;
}
html, body, [class*="st-"], .stApp, .stMarkdown, p, span, div, label, input, button, select, textarea {
  font-family: var(--font-body);
  color: var(--ink);
}
.block-container { padding-top: 2.6rem; max-width: 1280px; }
::selection { background: color-mix(in oklab, var(--indigo) 18%, transparent); }

/* hide Streamlit chrome for an app-grade feel */
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, header [data-testid="stStatusWidget"], footer { display: none !important; }
[data-testid="stHeader"] { background: transparent; }

/* ---- typography ------------------------------------------------------ */
h1, h2, h3, h4,
[data-testid="stHeading"] h1, [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {
  font-family: var(--font-display) !important;
  font-style: normal;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--ink);
}
code, kbd, pre, .stCode, [data-testid="stMetricValue"] { font-family: var(--font-mono) !important; }
a { color: var(--indigo-ink); text-decoration-color: color-mix(in oklab, var(--indigo) 40%, transparent); }

/* ---- masthead / hero -------------------------------------------------- */
.hero { margin: 0 0 1.6rem 0; }
.hero__eyebrow {
  font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.22em;
  text-transform: uppercase; color: var(--indigo); margin-bottom: 0.55rem;
  display: flex; align-items: center; gap: 0.6rem;
}
.hero__eyebrow::after { content: ""; flex: 1; height: 1px; background: var(--line); }
.hero__title {
  font-family: var(--font-display); font-weight: 600; font-size: clamp(2rem, 4.2vw, 2.9rem);
  line-height: 1.04; letter-spacing: -0.022em; color: var(--ink);
  margin: 0; overflow-wrap: anywhere; min-width: 0;
}
.hero__rule {
  height: 4px; width: 72px; margin: 0.9rem 0 0.75rem; border-radius: 2px;
  background: linear-gradient(90deg, var(--gold) 0 60%, var(--indigo) 60% 100%);
}
.hero__dek { color: var(--muted); font-size: 1.02rem; max-width: 68ch; line-height: 1.55; }
.hero__dek b, .hero__dek strong { color: var(--ink); }

/* verdict strap under a masthead — the study's one-line reading */
.verdict {
  border-top: 2px solid var(--rule-heavy); border-bottom: 1px solid var(--line);
  padding: 0.9rem 0.2rem; margin: 0.4rem 0 1.2rem;
  font-family: var(--font-display); font-size: 1.28rem; line-height: 1.4; color: var(--ink);
}
.verdict .kicker {
  font-family: var(--font-mono); font-size: 0.7rem; letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--gold); display: block; margin-bottom: 0.35rem;
}
.verdict em { font-style: normal; color: var(--indigo-ink); font-weight: 600; }

/* ---- findings band (newspaper data strip) ----------------------------- */
.stat-row {
  display: grid; grid-template-columns: repeat(4, minmax(0,1fr));
  border-top: 2px solid var(--rule-heavy); border-bottom: 1px solid var(--line);
  margin: 0.4rem 0 1.5rem;
}
.stat {
  padding: 0.95rem 1.1rem 0.9rem;
  border-left: 1px solid var(--line);
}
.stat:first-child { border-left: none; padding-left: 0.2rem; }
.stat__label {
  font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--muted);
}
.stat__value {
  font-family: var(--font-mono); font-size: 2.05rem; font-weight: 600;
  color: var(--ink); line-height: 1.15; margin-top: 0.25rem; font-variant-numeric: tabular-nums;
}
.stat__value .accent { color: var(--indigo); }
.stat__value .unit { font-size: 0.85rem; color: var(--muted); font-weight: 500; margin-left: 2px; }

/* ---- section head (kicker stacked ABOVE the heading) ------------------ */
.section-head { margin: 1.9rem 0 0.9rem; }
.section-head__rules { height: 5px; border-top: 2px solid var(--rule-heavy); border-bottom: 1px solid var(--line); margin-bottom: 0.7rem; }
.section-head__kicker {
  font-family: var(--font-mono); font-size: 0.7rem; letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--indigo); margin-bottom: 0.25rem;
}
.section-head__title {
  font-family: var(--font-display); font-weight: 600; font-size: 1.55rem;
  letter-spacing: -0.015em; margin: 0; line-height: 1.15;
}
.section-head__note { color: var(--muted); font-size: 0.92rem; margin-top: 0.3rem; max-width: 75ch; }

/* ---- per-pair dossier card -------------------------------------------- */
.pair-card {
  position: relative;
  background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 1.05rem 1.25rem 1.05rem 1.45rem; margin-bottom: 0.9rem;
  transition: border-color var(--dur) var(--ease-out), box-shadow var(--dur) var(--ease-out),
              transform var(--dur) var(--ease-out);
}
.pair-card__spine {
  position: absolute; left: 0; top: 10px; bottom: 10px; width: 4px;
  border-radius: 2px; background: var(--line);
}
.pair-card:hover {
  border-color: color-mix(in oklab, var(--indigo) 45%, var(--line));
  box-shadow: 0 10px 30px -18px color-mix(in oklab, var(--indigo) 55%, transparent);
  transform: translateY(-1px);
}
.pair-card__head { display: flex; align-items: baseline; gap: 0.7rem; flex-wrap: wrap; }
.pair-card__id { font-family: var(--font-mono); font-size: 1.12rem; font-weight: 600; color: var(--ink); }
.pair-card__family {
  font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted);
}
.pair-card__gloss { color: var(--muted); font-size: 0.9rem; margin-top: 0.4rem; max-width: 78ch; }
.pair-card__gloss b { color: var(--ink); }
.pair-card__stat {
  font-family: var(--font-mono); font-size: 0.76rem; color: var(--muted); margin-top: 0.35rem;
  border-top: 1px dashed var(--line-soft); padding-top: 0.4rem;
}
.pair-card__deltas { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: var(--space-md); margin-top: 0.9rem; }
.delta { border-left: 2px solid var(--line-soft); padding-left: 0.75rem; }
.delta__label { font-family: var(--font-mono); font-size: 0.66rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
.delta__val { font-family: var(--font-mono); font-size: 1.4rem; font-weight: 600; color: var(--ink); line-height: 1.25; font-variant-numeric: tabular-nums; }
.delta__sig { font-size: 0.74rem; color: var(--muted); }
.delta__sig.is-sig { color: var(--terra); font-weight: 600; }

/* ---- verdict chip (tinted-outline, small-caps mono) -------------------- */
.label-badge {
  display: inline-block; padding: 3px 10px; border-radius: var(--radius-pill);
  font-family: var(--font-mono); font-size: 0.68rem; font-weight: 600;
  letter-spacing: 0.06em; text-transform: uppercase; white-space: nowrap;
  color: var(--ink); background: var(--line-soft); border: 1px solid var(--line);
}
.muted { color: var(--muted); font-size: 0.85rem; }

/* footnote block */
.footnote {
  border-top: 1px solid var(--line); margin-top: 1.6rem; padding-top: 0.7rem;
  color: var(--muted); font-size: 0.85rem; max-width: 90ch; line-height: 1.55;
}
.footnote b { color: var(--ink); }

/* ---- sidebar: contents rail ------------------------------------------- */
section[data-testid="stSidebar"] {
  background: var(--paper-deep);
  border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.3rem; }
section[data-testid="stSidebar"] h1 { font-size: 1.4rem; }
section[data-testid="stSidebar"] hr { border-color: var(--line); }

/* nav radio → contents list with an indigo spine on the active item */
section[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
section[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: 7px 12px; border-radius: var(--radius-sm); cursor: pointer;
  border-left: 3px solid transparent;
  transition: background var(--dur) var(--ease-out), border-color var(--dur) var(--ease-out);
  font-size: 0.95rem;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #ece4d3; }
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: #efe7d6; border-left-color: var(--indigo); font-weight: 600;
}

/* ---- metrics (built-in) ------------------------------------------------ */
[data-testid="stMetric"] {
  background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 0.7rem 0.9rem;
}
[data-testid="stMetricValue"] { color: var(--ink); font-weight: 600; }
[data-testid="stMetricLabel"] p { color: var(--muted); font-size: 0.78rem; }

/* ---- bordered containers ----------------------------------------------- */
[data-testid="stVerticalBlockBorderWrapper"] { background: var(--card); border-radius: var(--radius); }

/* ---- buttons — full state set ------------------------------------------ */
.stButton > button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"], [data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button {
  border-radius: var(--radius-sm); font-weight: 600; font-family: var(--font-body);
  border: 1px solid var(--line);
  transition: transform 120ms var(--ease-out), box-shadow 120ms var(--ease-out), background var(--dur) var(--ease-out);
}
[data-testid="stBaseButton-primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
  background: var(--indigo); border-color: var(--indigo); color: #fff;
}
[data-testid="stBaseButton-primary"]:hover { box-shadow: 0 8px 20px -10px color-mix(in oklab, var(--indigo) 70%, transparent); transform: translateY(-1px); }
.stButton > button:active { transform: translateY(1px); box-shadow: none; }
.stButton > button:focus-visible, [data-baseweb="select"] > div:focus-visible,
.stTextInput input:focus-visible, .stNumberInput input:focus-visible {
  outline: 2px solid var(--focus) !important; outline-offset: 2px;
}
.stButton > button:disabled { opacity: 0.55; transform: none; box-shadow: none; }

/* ---- tabs --------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-weight: 600; color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--indigo-ink) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--indigo); height: 2.5px; }

/* ---- inputs / selects ---------------------------------------------------- */
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input, .stTextArea textarea {
  border-radius: var(--radius-sm) !important; border-color: var(--line) !important; background: var(--card) !important;
}
[data-baseweb="select"] > div:hover, .stTextInput input:hover { border-color: color-mix(in oklab, var(--indigo) 45%, var(--line)) !important; }

/* ---- alerts / code / tables ---------------------------------------------- */
.stAlert { border-radius: var(--radius); border: 1px solid var(--line); }
.stCode, pre { border-radius: var(--radius-sm) !important; }
[data-testid="stDataFrame"] { border-radius: var(--radius); border: 1px solid var(--line); }

/* ---- reduced motion ------------------------------------------------------ */
@media (prefers-reduced-motion: reduce) {
  * { transition-duration: 1ms !important; animation: none !important; }
}

/* ---- mobile --------------------------------------------------------------- */
@media (max-width: 768px) {
  .stat-row { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .stat:nth-child(3) { border-left: none; padding-left: 0.2rem; }
  .pair-card__deltas { grid-template-columns: 1fr; }
  .hero__title { font-size: 1.9rem; }
  .verdict { font-size: 1.08rem; }
}
</style>
"""


# --------------------------------------------------------------------------- #
# Plotly styler — one identity for every chart                                 #
# --------------------------------------------------------------------------- #
def style_fig(fig: Any, height: Optional[int] = None) -> Any:
    """Applies the editorial theme to a Plotly figure (in place) and returns it."""
    axis = dict(
        gridcolor=GRID, zerolinecolor=LINE, linecolor=LINE,
        tickfont=dict(family=FONT_MONO, color=MUTED, size=11),
        title_font=dict(family=FONT_BODY, color=MUTED, size=13),
    )
    fig.update_layout(
        font=dict(family=FONT_BODY, color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=list(COLORWAY),
        margin=dict(l=10, r=18, t=34, b=10),
        legend=dict(font=dict(family=FONT_BODY, color=INK, size=12),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_family=FONT_MONO, bgcolor=CARD, font_color=INK, bordercolor=LINE),
        xaxis=axis,
        yaxis=axis,
    )
    # Serif chart titles — but only when the figure actually has one (setting
    # title_font on a titleless figure makes Plotly render "undefined").
    try:
        if fig.layout.title and fig.layout.title.text:
            fig.update_layout(title_font=dict(family=FONT_DISPLAY, color=INK, size=17))
    except Exception:
        pass
    if height:
        fig.update_layout(height=height)
    return fig


# --------------------------------------------------------------------------- #
# Editorial HTML builders (pure string functions; testable)                    #
# --------------------------------------------------------------------------- #
def _mix_hex(c1: str, c2: str, w: float) -> str:
    """Linear sRGB mix: ``w`` of ``c1`` + ``(1-w)`` of ``c2`` → hex.

    Computed in Python because Streamlit's markdown sanitizer strips CSS custom
    properties (and functions) from inline ``style`` attributes — only literal
    values on plain properties survive.
    """
    a = c1.lstrip("#"); b = c2.lstrip("#")
    out = "".join(
        f"{round(int(a[i:i+2],16)*w + int(b[i:i+2],16)*(1-w)):02x}" for i in (0, 2, 4)
    )
    return f"#{out}"


def badge(label: Optional[str]) -> str:
    """Verdict chip for an interpretation label (tinted, small-caps mono)."""
    color = D.label_color(label)
    bg = _mix_hex(color, CARD, 0.12)
    ink = _mix_hex(color, INK, 0.60)
    line = _mix_hex(color, LINE, 0.40)
    style = f"background:{bg};color:{ink};border-color:{line};"
    text = (label or "—").replace("_", " ")
    return f'<span class="label-badge" style="{style}">{text}</span>'


def hero_html(title: str, dek: Optional[str] = None, eyebrow: Optional[str] = None) -> str:
    """Editorial masthead: mono eyebrow with trailing rule, serif title, split rule, dek."""
    parts = ['<div class="hero">']
    if eyebrow:
        parts.append(f'<div class="hero__eyebrow"><span>{eyebrow}</span></div>')
    parts.append(f'<h1 class="hero__title">{title}</h1>')
    parts.append('<div class="hero__rule"></div>')
    if dek:
        parts.append(f'<div class="hero__dek">{dek}</div>')
    parts.append("</div>")
    return "".join(parts)


def verdict_html(kicker: str, statement: str) -> str:
    """One-line study verdict, framed by a heavy top rule (newspaper strap)."""
    return (
        f'<div class="verdict"><span class="kicker">{kicker}</span>{statement}</div>'
    )


def section_head_html(kicker: str, title: str, note: Optional[str] = None) -> str:
    """Double-rule section head; kicker stacked ABOVE the serif title."""
    note_html = f'<div class="section-head__note">{note}</div>' if note else ""
    return (
        '<div class="section-head"><div class="section-head__rules"></div>'
        f'<div class="section-head__kicker">{kicker}</div>'
        f'<h2 class="section-head__title">{title}</h2>{note_html}</div>'
    )


def footnote_html(text: str) -> str:
    """Hairline-ruled footnote for provenance / how-to-read notes."""
    return f'<div class="footnote">{text}</div>'


def stat_tiles_html(items: Sequence[tuple]) -> str:
    """Ruled findings band from ``[(label, value), ...]`` (value may be HTML)."""
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
    """Per-pair dossier card: verdict spine, head + gloss + stat line + 3 deltas."""
    label = row.get("interpretation_label")
    spine = D.label_color(label)
    family = str(row.get("pair_id", "")).split("_")[0]
    head = (
        f'<div class="pair-card__head"><div class="pair-card__id">{row.get("pair_id","—")}</div>'
        f'<div class="pair-card__family">{family}</div>'
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
    return (
        f'<div class="pair-card"><span class="pair-card__spine" style="background:{spine}"></span>'
        f"{head}{gloss}{stat}{deltas}</div>"
    )
