"""Tests for the dashboard's editorial theme builders (Streamlit-free)."""

from __future__ import annotations

import plotly.graph_objects as go

from dashboard import theme as T


def test_css_declares_brand_fonts_and_tokens() -> None:
    css = T.CSS
    assert "Newsreader" in css and "Spline Sans" in css and "IBM Plex Mono" in css
    assert "--paper" in css and "--indigo" in css and "--gold" in css


def test_hero_html_structure() -> None:
    html = T.hero_html("Title", dek="A dek.", eyebrow="EYEBROW")
    assert "hero__title" in html and "Title" in html
    assert "hero__eyebrow" in html and "EYEBROW" in html
    assert "hero__rule" in html and "A dek." in html
    # eyebrow + dek are optional
    bare = T.hero_html("Only")
    assert "hero__eyebrow" not in bare and "hero__dek" not in bare


def test_stat_tiles_html_count() -> None:
    html = T.stat_tiles_html([("Pairs", "5"), ("Contrasts", "20")])
    assert html.count("stat__value") == 2
    assert "Pairs" in html and "20" in html


def test_badge_uses_label_colour() -> None:
    # Streamlit's sanitizer strips inline CSS custom properties, so the badge
    # bakes LITERAL tints derived from the label colour (Codex round-7 era fix).
    html = T.badge("broad_degradation")
    assert T._mix_hex("#dc2626", T.INK, 0.60) in html   # text = darkened label red
    assert T._mix_hex("#dc2626", T.CARD, 0.12) in html  # bg = red-tinted paper
    assert "label-badge" in T.badge(None)


def test_mix_hex_endpoints() -> None:
    assert T._mix_hex("#ffffff", "#000000", 1.0) == "#ffffff"
    assert T._mix_hex("#ffffff", "#000000", 0.0) == "#000000"


def test_pair_card_judge_primary_renders_stat_line() -> None:
    row = {
        "pair_id": "qwen_2b", "interpretation_label": "broad_degradation",
        "evidence_status": "confirmed", "harmbench_asr_delta": 0.055,
        "harmbench_asr_delta_significant": True, "harmbench_asr_fp16": 0.135,
        "harmbench_asr_nf4": 0.19, "harmbench_asr_p_value": 0.0266,
        "harmbench_asr_bh_q": 0.133, "harmbench_asr_bh_significant": False,
        "xstest_over_refusal_delta": -0.024, "xstest_over_refusal_delta_significant": False,
        "mmlu_accuracy_delta": -0.087, "mmlu_accuracy_delta_significant": True,
    }
    html = T.pair_card_html(row, judge_primary=True)
    assert "qwen_2b" in html and "label-badge" in html
    assert "+5.5 pp" in html               # ΔASR formatted as pp
    assert "McNemar p=0.027" in html       # stat line present
    assert "n.s. after FDR" in html
    assert "(judge)" in html               # judge-primary label suffix


def test_pair_card_v2_has_no_stat_line() -> None:
    row = {
        "pair_id": "x", "interpretation_label": "robust_preservation", "evidence_status": "null",
        "harmbench_asr_delta": 0.0, "harmbench_asr_delta_significant": False,
        "xstest_over_refusal_delta": 0.0, "xstest_over_refusal_delta_significant": False,
        "mmlu_accuracy_delta": 0.0, "mmlu_accuracy_delta_significant": False,
    }
    html = T.pair_card_html(row, judge_primary=False)
    assert "pair-card__stat" not in html
    assert "(v2)" in html


def test_style_fig_applies_theme() -> None:
    fig = T.style_fig(go.Figure(go.Bar(x=[1, 2], y=[3, 4])), height=320)
    assert fig.layout.height == 320
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert tuple(fig.layout.colorway) == tuple(T.COLORWAY)
