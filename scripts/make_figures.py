#!/usr/bin/env python3
"""Generate the publication figure suite from committed analysis artifacts.

Every figure is derived deterministically from files under results/analysis/
(no new inference, no raw prompt/response text), so the figures are reproducible
and redaction-safe. Output: PNGs under docs/figures/, embedded into the report
and thesis docx by their build scripts.

Figures
  1. capability_anchor.png  -- the central contribution: each pair plotted in
     (ΔMMLU, ΔASR-judge) space with the interpretation-label regions drawn in.
  2. asr_forest.png         -- per-pair judge ΔASR with paired-bootstrap 95% CIs.
  3. precision_sweep.png    -- fp16 → INT8 → NF4 trajectories (ASR, MMLU, ARC):
     the capability cliff at four-bit and the Llama INT8 non-monotonic ASR.
  4. judge_vs_proxy.png     -- the scorer-validation story: judge ASR vs regex
     proxy per model, and Cohen's κ per model.
  5. category_asr.png       -- Qwen3-1.7B per-category HarmBench ASR, fp16 vs NF4.
  6. multiseed.png          -- Qwen3-1.7B judge ΔASR: greedy vs multi-seed mean±sd.

Run:  python scripts/make_figures.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
# Paths default to the 128-token study but can be redirected to the 512-token
# tree (or any other) via env vars, so the figure suite is reproducible against
# either results root without clobbering the committed figures:
#   FIG_ANALYSIS_DIR=results_512/analysis FIG_OUT_DIR=docs/figures_512 python scripts/make_figures.py
ANALYSIS = Path(os.environ.get("FIG_ANALYSIS_DIR") or REPO / "results" / "analysis")
FIGDIR = Path(os.environ.get("FIG_OUT_DIR") or REPO / "docs" / "figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# Consistent per-pair identity across all figures.
PAIRS = ["qwen_2b", "qwen_4b", "llama_3_2_3b", "mistral_7b", "phi4_mini"]
LABEL = {
    "qwen_2b": "Qwen3-1.7B", "qwen_4b": "Qwen3-4B", "llama_3_2_3b": "Llama-3.2-3B",
    "mistral_7b": "Mistral-7B", "phi4_mini": "Phi-4-mini",
}
COLOR = {
    "qwen_2b": "#2563eb", "qwen_4b": "#1e3a8a", "llama_3_2_3b": "#dc2626",
    "mistral_7b": "#d97706", "phi4_mini": "#059669",
}


def load(name):
    return json.load((ANALYSIS / name).open())


def pairwise_lookup():
    rows = load("pairwise_deltas.json")
    out = {}
    for r in rows:
        out[(r["pair_id"], r["benchmark"], r["metric"])] = r
    return out


JA = load("judge_agreement.json")
JUDGE_PAIR = {r["pair_id"]: r for r in JA["per_pair"]}
JUDGE_MODEL = JA["per_model"]
PW = pairwise_lookup()
SWEEP = load("precision_sweep.json")["per_pair"]


# ---------------------------------------------------------------------------
# Figure 1: capability-anchoring space (the money figure)
# ---------------------------------------------------------------------------
def fig_capability_anchor():
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    tsafe, tcap = 0.02, 0.02  # safety and capability thresholds used by the label rule

    # Region shading (projection onto ΔMMLU x ΔASR; the rule also uses ΔOR).
    xmin, xmax, ymin, ymax = -0.11, 0.05, -0.07, 0.09
    ax.axhspan(tsafe, ymax, xmin=0, xmax=1, color="#fee2e2", alpha=0.0)
    # Use rectangles via fill_between for clarity.
    ax.fill_between([xmin, -tcap], tsafe, ymax, color="#fecaca", alpha=0.45, zorder=0)   # broad_degradation
    ax.fill_between([-tcap, xmax], tsafe, ymax, color="#fed7aa", alpha=0.45, zorder=0)   # alignment_degradation
    ax.fill_between([xmin, -tcap], ymin, -tsafe, color="#e9d5ff", alpha=0.45, zorder=0)  # capability_collapse_masq
    ax.fill_between([-tcap, xmax], ymin, -tsafe, color="#bbf7d0", alpha=0.55, zorder=0)  # alignment_improvement
    ax.fill_between([xmin, xmax], -tsafe, tsafe, color="#e5e7eb", alpha=0.5, zorder=0)   # robust_preservation band

    ax.axhline(0, color="#374151", lw=0.8, zorder=1)
    ax.axvline(0, color="#374151", lw=0.8, zorder=1)
    for y in (tsafe, -tsafe):
        ax.axhline(y, color="#9ca3af", lw=0.6, ls="--", zorder=1)
    ax.axvline(-tcap, color="#9ca3af", lw=0.6, ls="--", zorder=1)

    # Region labels, placed in the corners so they never collide with data points.
    pad = 0.003
    txt = dict(fontsize=8.5, style="italic", color="#4b5563")
    ax.text(xmin + pad, ymax - pad, "broad\ndegradation", ha="left", va="top", **txt)
    ax.text(xmax - pad, ymax - pad, "alignment\ndegradation", ha="right", va="top", **txt)
    ax.text(xmin + pad, ymin + pad, "capability collapse\nmasq. as safety", ha="left", va="bottom", **txt)
    ax.text(xmax - pad, ymin + pad, "alignment\nimprovement", ha="right", va="bottom", **txt)
    ax.text(xmin + pad, 0.0, "robust preservation", ha="left", va="center",
            fontsize=8, color="#6b7280")

    for pid in PAIRS:
        jp = JUDGE_PAIR[pid]
        x = jp["mmlu_delta"]
        y = jp["judge_harm_delta"]
        # CI bars: x from pairwise MMLU, y from judge agreement.
        mm = PW.get((pid, "mmlu", "accuracy"))
        xerr = None
        if mm:
            xerr = [[x - mm["delta_ci_lower"]], [mm["delta_ci_upper"] - x]]
        yerr = [[y - jp["judge_harm_delta_ci_lower"]], [jp["judge_harm_delta_ci_upper"] - y]]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", ms=8, color=COLOR[pid],
                    ecolor=COLOR[pid], elinewidth=1.1, capsize=3, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.8)
        dy = 0.006 if pid != "phi4_mini" else -0.010
        ax.annotate(LABEL[pid], (x, y), textcoords="offset points",
                    xytext=(8, 6 if pid != "llama_3_2_3b" else -12),
                    fontsize=8.5, color=COLOR[pid], fontweight="bold")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Δ MMLU accuracy  (4-bit − fp16)   ← capability loss")
    ax.set_ylabel("Δ HarmBench ASR, judge  (4-bit − fp16)   ↑ more harmful")
    ax.set_title("Capability-anchored safety space (NF4 vs fp16)", fontweight="bold")
    fig.savefig(FIGDIR / "capability_anchor.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: ASR forest plot
# ---------------------------------------------------------------------------
def fig_asr_forest():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ys = list(range(len(PAIRS)))[::-1]
    for y, pid in zip(ys, PAIRS):
        jp = JUDGE_PAIR[pid]
        d = jp["judge_harm_delta"]
        lo, hi = jp["judge_harm_delta_ci_lower"], jp["judge_harm_delta_ci_upper"]
        sig = jp["judge_harm_delta_significant"]
        ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", ms=7,
                    color=COLOR[pid], capsize=3,
                    markerfacecolor=COLOR[pid] if sig else "white",
                    markeredgecolor=COLOR[pid], markeredgewidth=1.4)
    ax.axvline(0, color="#374151", lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[p] for p in PAIRS])
    ax.set_xlabel("Δ HarmBench ASR, judge (4-bit − fp16), with paired-bootstrap 95% CI")
    ax.set_title("Quantization effect on harmful compliance", fontweight="bold")
    ax.set_xlim(-0.14, 0.12)
    ax.text(0.02, -0.7, "filled marker = 95% CI excludes zero", fontsize=7.5, color="#6b7280")
    fig.savefig(FIGDIR / "asr_forest.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: precision sweep fp16 -> INT8 -> NF4
# ---------------------------------------------------------------------------
def fig_precision_sweep():
    precisions = ["fp16", "int8", "nf4"]
    xs = [0, 1, 2]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4))
    panels = [
        ("harmbench_asr_judge", "HarmBench ASR (judge)"),
        ("mmlu_accuracy", "MMLU accuracy"),
        ("arc_accuracy", "ARC accuracy"),
    ]
    for ax, (metric, title) in zip(axes, panels):
        for pid in PAIRS:
            m = SWEEP[pid]["metrics"].get(metric)
            if not m:
                continue
            ys = [m.get(p) for p in precisions]
            ax.plot(xs, ys, marker="o", ms=5, lw=1.6, color=COLOR[pid], label=LABEL[pid])
        ax.set_xticks(xs)
        ax.set_xticklabels(["fp16", "INT8", "NF4"])
        ax.set_title(title)
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    axes[0].set_ylabel("rate / accuracy")
    axes[1].legend(loc="lower left", fontsize=7.5, frameon=False, ncol=1)
    fig.suptitle("Precision sweep: capability cliff at four-bit; safety is method-specific",
                 fontweight="bold", y=1.02)
    fig.savefig(FIGDIR / "precision_sweep.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: judge vs regex proxy (scorer validation)
# ---------------------------------------------------------------------------
def fig_judge_vs_proxy():
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    a, b = axes

    # Panel A: judge ASR vs v2 proxy ASR per model.
    a.plot([0, 1], [0, 1], color="#9ca3af", lw=0.8, ls="--", zorder=0)
    seen = set()
    for alias, m in JUDGE_MODEL.items():
        pid = next((p for p in PAIRS if alias.startswith(p)), None)
        if pid is None:
            continue
        quant = alias.endswith("4bit") or alias.endswith("8bit")
        a.scatter(m["v2_asr"], m["judge_asr"], s=60, color=COLOR[pid],
                  marker="^" if quant else "o", edgecolor="white", linewidth=0.7, zorder=3)
    a.set_xlabel("regex proxy 'non-refusal' rate")
    a.set_ylabel("HarmBench classifier ASR (judge)")
    a.set_title("Regex over-counts harmful compliance", fontweight="bold")
    a.set_xlim(0, 1); a.set_ylim(0, 0.6)
    handles = [Patch(color=COLOR[p], label=LABEL[p]) for p in PAIRS]
    a.legend(handles=handles, loc="upper left", fontsize=7.5, frameon=False)
    a.text(0.55, 0.05, "points below diagonal\n= proxy > judge", fontsize=7.5, color="#6b7280")

    # Panel B: Cohen's kappa per model.
    aliases, kappas, colors = [], [], []
    for pid in PAIRS:
        for suffix in ("base", "4bit"):
            alias = f"{pid}_{suffix}"
            if alias in JUDGE_MODEL and JUDGE_MODEL[alias].get("cohens_kappa") is not None:
                aliases.append(LABEL[pid] + (" 4-bit" if suffix == "4bit" else " fp16"))
                kappas.append(JUDGE_MODEL[alias]["cohens_kappa"])
                colors.append(COLOR[pid])
    ys = list(range(len(aliases)))[::-1]
    b.barh(ys, kappas, color=colors, height=0.7)
    b.set_yticks(ys); b.set_yticklabels(aliases, fontsize=7.5)
    b.axvline(0.6, color="#374151", lw=0.7, ls=":")
    b.set_xlabel("Cohen's κ (judge vs regex proxy)")
    b.set_title("Agreement is family-dependent", fontweight="bold")
    b.set_xlim(0, 1)
    b.text(0.61, 0.2, "κ=0.6", fontsize=7, color="#374151")
    fig.savefig(FIGDIR / "judge_vs_proxy.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Qwen3-1.7B per-category HarmBench ASR
# ---------------------------------------------------------------------------
def fig_category_asr():
    cb = load("harmbench_category_breakdown.json")["per_model"]
    base, quant = cb["qwen_2b_base"], cb["qwen_2b_4bit"]
    cats = [c for c in base if c in quant]
    # order by 4-bit ASR descending
    cats.sort(key=lambda c: quant[c]["asr"], reverse=True)
    pretty = {
        "misinformation_disinformation": "misinformation", "harmful": "generic harmful",
        "cybercrime_intrusion": "cybercrime", "illegal": "illegal activity",
        "harassment_bullying": "harassment", "chemical_biological": "chem/bio",
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    import numpy as np
    x = np.arange(len(cats))
    w = 0.38
    ax.bar(x - w / 2, [base[c]["asr"] for c in cats], w, label="fp16", color="#93c5fd")
    ax.bar(x + w / 2, [quant[c]["asr"] for c in cats], w, label="NF4 4-bit", color="#1d4ed8")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty.get(c, c) for c in cats], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Attack Success Rate (judge)")
    ax.set_title("Qwen3-1.7B per-category harmful compliance", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGDIR / "category_asr.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: multi-seed decoding sensitivity for Qwen3-1.7B
# ---------------------------------------------------------------------------
def fig_multiseed():
    ms = {r["pair_id"]: r for r in load("sensitivity_multiseed.json")["per_pair"]}
    q = ms["qwen_2b"]["judge_delta"]
    greedy = ms["qwen_2b"]["greedy_judge_delta"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.axhline(0, color="#9ca3af", lw=0.7)
    ax.errorbar([0], [q["mean"]], yerr=[q["sd"]], fmt="o", ms=9, color="#2563eb",
                capsize=5, label=f"multi-seed mean ± sd (n={q['n_seeds']})")
    ax.scatter([0.35], [greedy], s=90, marker="D", color="#dc2626", zorder=4,
               label="greedy (headline)")
    ax.plot([0, 0], [q["min"], q["max"]], color="#2563eb", lw=1.0, alpha=0.5)
    ax.set_xlim(-0.4, 0.9)
    ax.set_xticks([])
    ax.set_ylabel("Δ HarmBench ASR, judge")
    ax.set_title("Qwen3-1.7B: headline vs stochastic decoding", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.savefig(FIGDIR / "multiseed.png")
    plt.close(fig)


def main():
    fig_capability_anchor()
    fig_asr_forest()
    fig_precision_sweep()
    fig_judge_vs_proxy()
    fig_category_asr()
    fig_multiseed()
    figs = sorted(p.name for p in FIGDIR.glob("*.png"))
    print(f"Wrote {len(figs)} figures to {FIGDIR}:")
    for f in figs:
        print(" -", f)


if __name__ == "__main__":
    main()
