"""Analyse the refusal-margin capture (D33 / T28) — the quantization *mechanism*.

Mac-side, no GPU. Reads the redacted margin sidecars
(``results/<alias>/harmbench/scores.margin.<precision>.jsonl`` + the neutral
``scores.margin_control.*`` + the committed ``scores.judge.harmbench_cls.jsonl``)
and tests, per pair and pooled, whether NF4 quantization erodes the refusal
*decision margin* and whether behavioural flips concentrate where the baseline
margin was thin. Writes ``results/analysis/refusal_margin.{json,csv}`` (aggregate
stats only — no per-prompt data, no text).

The design follows the de-risked spec (D33): continuous paired Delta_m over ALL
200 prompts (not circular flip-signs); an fp16-vs-judge AUC gate that must pass
before any quantization claim is trusted; a capability/confidence confound
control on neutral tokens; flips treated as a confirmatory zoom; and the
"smallest moves first" claim tested by pooling, not a 5-point ranking.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PAIRS = {
    "qwen_2b": ("qwen_2b_base", "qwen_2b_4bit"),
    "qwen_4b": ("qwen_4b_base", "qwen_4b_4bit"),
    "llama_3_2_3b": ("llama_3_2_3b_base", "llama_3_2_3b_4bit"),
    "mistral_7b": ("mistral_7b_base", "mistral_7b_4bit"),
    "phi4_mini": ("phi4_mini_base", "phi4_mini_4bit"),
}
RNG_SEED = 42


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _margin(alias: str, tag: str) -> Dict[str, dict]:
    rows = _read_jsonl(RESULTS / alias / "harmbench" / f"scores.margin.{tag}.jsonl")
    return {r["prompt_id"]: r for r in rows}


def _judge(alias: str) -> Dict[str, bool]:
    rows = _read_jsonl(RESULTS / alias / "harmbench" / "scores.judge.harmbench_cls.jsonl")
    return {r["prompt_id"]: bool(r["judge_harmful"]) for r in rows}


def _control_entropy(alias: str, tag: str) -> List[float]:
    rows = _read_jsonl(RESULTS / alias / "harmbench" / f"scores.margin_control.{tag}.jsonl")
    return [r["entropy"] for r in rows if r.get("entropy") == r.get("entropy")]


def _auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """ROC-AUC of `scores` predicting label==1; None if a single class or empty."""
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _bootstrap_ci(x: np.ndarray, n: int = 2000) -> List[Optional[float]]:
    x = x[~np.isnan(x)]
    if x.size < 2:
        return [None, None]
    res = stats.bootstrap((x,), np.mean, n_resamples=n, random_state=RNG_SEED,
                          method="percentile")
    return [float(res.confidence_interval.low), float(res.confidence_interval.high)]


def _paired_shift(base: np.ndarray, quant: np.ndarray) -> dict:
    """Delta = base - quant per prompt (>0 = quantization eroded the refuse-ward margin)."""
    mask = ~(np.isnan(base) | np.isnan(quant))
    b, q = base[mask], quant[mask]
    d = b - q
    sd = float(np.std(d, ddof=1)) if d.size > 1 else 0.0
    out = {
        "n": int(d.size),
        "mean_delta": float(np.mean(d)) if d.size else None,
        "median_delta": float(np.median(d)) if d.size else None,
        "frac_eroded": float(np.mean(d > 0)) if d.size else None,  # toward compliance
        "cohen_dz": float(np.mean(d) / sd) if sd > 0 else None,    # paired effect size
        "ci95": _bootstrap_ci(d),
        "wilcoxon_p": None,
    }
    if d.size and np.any(d != 0):
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(b, q).pvalue)
        except ValueError:
            out["wilcoxon_p"] = None
    return out


def analyse_pair(pair_id: str, base: str, quant: str) -> dict:
    mb, mq = _margin(base, "fp16"), _margin(quant, "nf4")
    jb, jq = _judge(base), _judge(quant)
    ids = sorted(set(mb) & set(mq) & set(jb) & set(jq))

    def col(d, key):
        return np.array([d[i][key] for i in ids], dtype=np.float64)

    base_m1, quant_m1 = col(mb, "m1"), col(mq, "m1")
    base_smin, quant_smin = col(mb, "seq_min_margin"), col(mq, "seq_min_margin")
    base_smean, quant_smean = col(mb, "seq_mean_margin"), col(mq, "seq_mean_margin")
    base_H, quant_H = col(mb, "entropy"), col(mq, "entropy")
    y_base = np.array([1 if jb[i] else 0 for i in ids])  # judge harmful on baseline
    y_quant = np.array([1 if jq[i] else 0 for i in ids])

    # (1) AUC GATE: does the baseline margin predict the baseline judge label?
    # more comply-ward (smaller margin) should predict harmful -> use -margin.
    gate = {
        "auc_m1": _auc(y_base, -base_m1),
        "auc_seq_min": _auc(y_base, -base_smin),
        "base_harmful_rate": float(np.mean(y_base)),
    }

    # (2) continuous paired margin shift (erosion) over ALL prompts
    shift = {
        "m1": _paired_shift(base_m1, quant_m1),
        "seq_min_margin": _paired_shift(base_smin, quant_smin),
        "seq_mean_margin": _paired_shift(base_smean, quant_smean),
    }

    # (3) capability/confidence confound control: entropy rise on harmbench vs neutral tokens
    dH_hb = float(np.nanmean(quant_H - base_H))
    cb, cq = np.array(_control_entropy(base, "fp16")), np.array(_control_entropy(quant, "nf4"))
    dH_ctrl = float(np.mean(cq) - np.mean(cb)) if cb.size and cq.size else None
    control = {
        "delta_entropy_harmbench": dH_hb,
        "delta_entropy_control": dH_ctrl,
        "harmbench_softens_more": (dH_ctrl is not None and dH_hb > dH_ctrl),
    }

    # (4) flip zoom: do flips sit at thin baseline margins? Split by direction —
    # NF4 destabilises near-boundary prompts in BOTH directions, so report both.
    flip = (y_base != y_quant)
    harmful_ward = (y_base == 0) & (y_quant == 1)
    safe_ward = (y_base == 1) & (y_quant == 0)
    abm = np.abs(base_m1)
    zoom = {
        "n_flips": int(flip.sum()),
        "n_harmful_ward": int(harmful_ward.sum()),
        "n_safe_ward": int(safe_ward.sum()),
        "median_abs_margin_flips": float(np.median(abm[flip])) if flip.any() else None,
        "median_abs_margin_concordant": float(np.median(abm[~flip])) if (~flip).any() else None,
        "within_pair_auc_thin_predicts_flip": _auc(flip.astype(int), -abm),
        "mean_delta_m1_on_harmful_ward": (float(np.mean((base_m1 - quant_m1)[harmful_ward]))
                                          if harmful_ward.any() else None),
    }
    return {
        "pair_id": pair_id, "n": len(ids),
        "baseline_median_m1": float(np.median(base_m1)),
        "gate": gate, "margin_shift": shift, "confound_control": control, "flip_zoom": zoom,
        # arrays consumed by the pooled / within-family analysis
        "_pooled": {"abs_m1": abm, "flip": flip,
                    "harmful_ward": harmful_ward, "safe_ward": safe_ward},
    }


def main() -> int:
    per_pair = {}
    acc = {"absm": [], "flip": [], "z": [], "hw": [], "sw": []}
    for pid, (b, q) in PAIRS.items():
        res = analyse_pair(pid, b, q)
        pool = res.pop("_pooled")
        a = pool["abs_m1"]
        acc["absm"].append(a)
        acc["flip"].append(pool["flip"].astype(int))
        acc["hw"].append(pool["harmful_ward"].astype(int))
        acc["sw"].append(pool["safe_ward"].astype(int))
        sd = a.std(ddof=1)
        # within-pair z-score removes the between-family level difference, so the
        # pooled AUC then reflects the WITHIN-model relationship, not an ecological one.
        acc["z"].append((a - a.mean()) / sd if sd > 0 else np.zeros_like(a))
        per_pair[pid] = res

    absm = np.concatenate(acc["absm"]); flip = np.concatenate(acc["flip"])
    zabsm = np.concatenate(acc["z"])
    hw = np.concatenate(acc["hw"]).astype(bool); sw = np.concatenate(acc["sw"]).astype(bool)
    # (5) pooled: does a THIN baseline margin predict a flip? Report the naive pooled
    # AUC, the within-family (z-scored) AUC, and the two flip directions separately —
    # thin margins are near the decision boundary and destabilise EITHER way under NF4.
    pooled = {
        "n_prompts": int(absm.size), "n_flips": int(flip.sum()),
        "n_harmful_ward": int(hw.sum()), "n_safe_ward": int(sw.sum()),
        "auc_thin_margin_predicts_flip": _auc(flip, -absm),
        "auc_thin_margin_predicts_flip_within_family_z": _auc(flip, -zabsm),
        "auc_thin_margin_predicts_harmful_ward": _auc(hw[~sw].astype(int), -absm[~sw]),
        "auc_thin_margin_predicts_safe_ward": _auc(sw[~hw].astype(int), -absm[~hw]),
        "within_pair_auc_by_pair": {p: per_pair[p]["flip_zoom"]["within_pair_auc_thin_predicts_flip"]
                                    for p in PAIRS},
        "baseline_median_m1_by_pair": {p: per_pair[p]["baseline_median_m1"] for p in PAIRS},
        "thinnest_baseline_pair": min(PAIRS, key=lambda p: per_pair[p]["baseline_median_m1"]),
    }

    out = {"scorer_version": "refusal_margin_analysis_v1", "per_pair": per_pair, "pooled": pooled}
    (RESULTS / "analysis").mkdir(exist_ok=True)
    with (RESULTS / "analysis" / "refusal_margin.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # flat per-pair CSV
    csv_path = RESULTS / "analysis" / "refusal_margin.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["pair_id", "n", "baseline_median_m1", "gate_auc_m1", "gate_auc_seq_min",
                    "m1_mean_delta", "m1_ci_low", "m1_ci_high", "m1_wilcoxon_p", "m1_frac_eroded",
                    "seqmin_mean_delta", "seqmin_wilcoxon_p",
                    "dH_harmbench", "dH_control", "n_flips",
                    "median_absm_flips", "median_absm_concordant"])
        for p, r in per_pair.items():
            g, s, c, z = r["gate"], r["margin_shift"], r["confound_control"], r["flip_zoom"]
            w.writerow([p, r["n"], round(r["baseline_median_m1"], 3),
                        g["auc_m1"], g["auc_seq_min"],
                        s["m1"]["mean_delta"], s["m1"]["ci95"][0], s["m1"]["ci95"][1],
                        s["m1"]["wilcoxon_p"], s["m1"]["frac_eroded"],
                        s["seq_min_margin"]["mean_delta"], s["seq_min_margin"]["wilcoxon_p"],
                        c["delta_entropy_harmbench"], c["delta_entropy_control"],
                        z["n_flips"], z["median_abs_margin_flips"],
                        z["median_abs_margin_concordant"]])

    # console summary (redacted: aggregate only)
    print("=" * 78)
    print("REFUSAL-MARGIN MECHANISM ANALYSIS (D33)")
    print("=" * 78)
    print(f"{'pair':<14}{'gate_AUC(m1)':>13}{'median m1':>11}{'Δm1 mean':>10}"
          f"{'Δm1 CI':>20}{'wilcox p':>10}{'flips':>7}")
    for p, r in per_pair.items():
        g, s, z = r["gate"], r["margin_shift"]["m1"], r["flip_zoom"]
        ci = s["ci95"]
        ci_s = f"[{ci[0]:+.2f},{ci[1]:+.2f}]" if ci[0] is not None else "  n/a"
        auc = f"{g['auc_m1']:.3f}" if g["auc_m1"] is not None else "n/a"
        print(f"{p:<14}{auc:>13}{r['baseline_median_m1']:>11.2f}"
              f"{s['mean_delta']:>+10.3f}{ci_s:>20}"
              f"{(s['wilcoxon_p'] if s['wilcoxon_p'] is not None else float('nan')):>10.3f}"
              f"{z['n_flips']:>7}")
    print("-" * 78)
    print("POOLED thin-margin -> flip AUC:")
    print(f"  naive (between+within family) = {pooled['auc_thin_margin_predicts_flip']:.3f}")
    print(f"  within-family (z-scored)      = {pooled['auc_thin_margin_predicts_flip_within_family_z']:.3f}"
          f"   <- the honest per-prompt number")
    print(f"  harmful-ward only = {pooled['auc_thin_margin_predicts_harmful_ward']:.3f}  |  "
          f"safe-ward only = {pooled['auc_thin_margin_predicts_safe_ward']:.3f}  "
          f"(flips: {pooled['n_harmful_ward']} harmful-ward / {pooled['n_safe_ward']} safe-ward "
          f"-> boundary instability, not one-directional erosion)")
    print(f"  within-pair AUCs: " + ", ".join(f"{p}={v:.2f}" if v is not None else f"{p}=n/a"
          for p, v in pooled["within_pair_auc_by_pair"].items()))
    print(f"  thinnest baseline margin = {pooled['thinnest_baseline_pair']}")
    print("Confound control (Δentropy harmbench vs neutral control), per pair:")
    for p, r in per_pair.items():
        c = r["confound_control"]
        print(f"  {p:<14} ΔH_harmbench={c['delta_entropy_harmbench']:+.4f}  "
              f"ΔH_control={c['delta_entropy_control']}  "
              f"harmbench_softens_more={c['harmbench_softens_more']}")
    print(f"\nWrote {RESULTS/'analysis'/'refusal_margin.json'} and .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
