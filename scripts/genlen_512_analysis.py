#!/usr/bin/env python3
"""Generation-length (128 vs 512) robustness analysis backing report §6.16.

Reproduces every number in §6.16 from the redacted judge sidecars + summaries:
  - judge-primary ΔASR @128 and @512 (classifier + gpt-4o), paired-bootstrap CI,
    exact McNemar, and BH-FDR across the 5 pairs at 512;
  - capability/over-refusal deltas (MMLU, ARC, XSTest) 128 vs 512 (from make analyze);
  - response-length distribution (proof the 512 budget took effect);
  - cross-judge agreement (classifier vs gpt-4o) at 512.

Inputs: results/ (128, committed sidecars) + results_512/ (512). XSTest/MMLU/ARC
deltas are read from each tree's analysis/pairwise_deltas.json (run make analyze
RESULTS_DIR=<tree> first). Output: results_512/analysis/genlen_robustness.json
(aggregates only; no raw text).
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PAIRS = [("qwen_2b","qwen_2b_base","qwen_2b_4bit"),("qwen_4b","qwen_4b_base","qwen_4b_4bit"),
         ("llama_3_2_3b","llama_3_2_3b_base","llama_3_2_3b_4bit"),("mistral_7b","mistral_7b_base","mistral_7b_4bit"),
         ("phi4_mini","phi4_mini_base","phi4_mini_4bit")]
CEIL_128 = 792  # observed max response length (chars) in the 128-token run

def _sidecar(root, alias, name):
    p = REPO / root / alias / "harmbench" / name
    return {str(r["prompt_id"]): (1 if r.get("judge_harmful") else 0)
            for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())}

def _kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a)/n, sum(b)/n; pe = pa*pb + (1-pa)*(1-pb)
    # Degenerate chance agreement (both raters constant) -> kappa is 0/0, undefined;
    # return NaN, not a spurious 1.0. Matches judge_agreement._cohens_kappa.
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else float("nan")

def _mcnemar(n01, n10):
    n = n01 + n10
    if n == 0: return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)

def _bootci(b, q, seed=42, nb=10000):
    rng = np.random.default_rng(seed); n = len(b); d = np.empty(nb)
    for i in range(nb):
        idx = rng.integers(0, n, n); d[i] = q[idx].mean() - b[idx].mean()
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def _bh_fdr(pvals, q=0.05):
    m = len(pvals); order = sorted(range(m), key=lambda i: pvals[i]); rej = [False]*m
    kmax = -1
    for rank, i in enumerate(order, 1):
        if pvals[i] <= rank / m * q: kmax = rank
    for rank, i in enumerate(order, 1):
        if rank <= kmax: rej[i] = True
    return rej

def asr_block(root, sidecar):
    rows = {}
    for pair, ba, qa in PAIRS:
        bm, qm = _sidecar(root, ba, sidecar), _sidecar(root, qa, sidecar)
        pids = sorted(set(bm) & set(qm))
        b = np.array([bm[p] for p in pids]); q = np.array([qm[p] for p in pids])
        n01 = int(((b == 0) & (q == 1)).sum()); n10 = int(((b == 1) & (q == 0)).sum())
        lo, hi = _bootci(b, q)
        rows[pair] = dict(n=len(pids), asr_base=round(float(b.mean()), 4), asr_quant=round(float(q.mean()), 4),
                          delta=round(float(q.mean() - b.mean()), 4), ci=[round(lo, 4), round(hi, 4)],
                          n01=n01, n10=n10, mcnemar_p=round(_mcnemar(n01, n10), 4),
                          sig_ci=bool(lo > 0 or hi < 0))
    rej = _bh_fdr([rows[p]["mcnemar_p"] for p, *_ in PAIRS])
    for (pair, *_), r in zip(PAIRS, rej):
        rows[pair]["bh_fdr_sig"] = bool(r)
    return rows

def cap_block():
    def deltas(root):
        d = json.load(open(REPO / root / "analysis" / "pairwise_deltas.json"))
        return {(r["pair_id"], r["benchmark"]): r for r in d}
    d128, d512 = deltas("results"), deltas("results_512")
    out = {}
    for bench in ("mmlu", "arc", "xstest"):
        out[bench] = {}
        for pair, *_ in PAIRS:
            a, b = d128.get((pair, bench)), d512.get((pair, bench))
            if a and b:
                out[bench][pair] = dict(d128=round(a["absolute_delta"], 4), sig128=a["delta_significant"],
                                        d512=round(b["absolute_delta"], 4), sig512=b["delta_significant"])
    return out

def length_block():
    def lens(root, alias):
        p = REPO / root / alias / "harmbench" / "raw.jsonl"
        if not p.exists(): return None
        return np.array([len(json.loads(l).get("response") or "") for l in p.read_text().splitlines() if l.strip()])
    per, all128, all512 = {}, [], []
    for pair, ba, qa in PAIRS:
        for alias in (ba, qa):
            l1, l5 = lens("results", alias), lens("results_512", alias)
            if l1 is not None: all128.append(l1)
            if l5 is not None: all512.append(l5)
            if l1 is not None and l5 is not None:
                per[alias] = dict(med128=int(np.median(l1)), max128=int(l1.max()),
                                  med512=int(np.median(l5)), max512=int(l5.max()),
                                  over_ceiling_512=int((l5 > CEIL_128).sum()), n=int(len(l5)))
    a1, a5 = np.concatenate(all128), np.concatenate(all512)
    return dict(per_model=per, ceiling_chars=CEIL_128,
                all_median_128=int(np.median(a1)), all_median_512=int(np.median(a5)),
                all_max_512=int(a5.max()), pct_512_over_ceiling=round(100 * float((a5 > CEIL_128).mean()), 1))

def kappa_block():
    out = {}
    for pair, ba, qa in PAIRS:
        for alias in (ba, qa):
            clf = _sidecar("results_512", alias, "scores.judge.harmbench_cls.jsonl")
            gpt = _sidecar("results_512", alias, "scores.judge.api_judge.jsonl")
            pids = sorted(set(clf) & set(gpt))
            out[alias] = round(_kappa([clf[p] for p in pids], [gpt[p] for p in pids]), 3)
    ks = list(out.values())
    return dict(per_model=out, kappa_min=min(ks), kappa_max=max(ks))

def prefix_truncation_block():
    """Direct evidence the 128-token budget truncated generations: under greedy
    decoding the two budgets share a decode path, so a truncated 128 response is
    a proper PREFIX of its 512 counterpart. Counts only (redaction-safe):
    truncated = 512 response strictly extends the 128 one; natural_stop = equal;
    mismatch = neither (decode divergence, e.g. detokenization/GPU nondeterminism)."""
    def resp(root, alias):
        p = REPO / root / alias / "harmbench" / "raw.jsonl"
        return {r["prompt_id"]: (r.get("response") or "").strip()
                for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())}
    per, T, N, M = {}, 0, 0, 0
    for pair, ba, qa in PAIRS:
        for alias in (ba, qa):
            d1, d5 = resp("results", alias), resp("results_512", alias)
            t = n = m = 0
            for pid in set(d1) & set(d5):
                s1, s5 = d1[pid], d5[pid]
                if s1 == s5: n += 1
                elif s5.startswith(s1) and len(s5) > len(s1): t += 1
                else: m += 1
            tot = t + n + m
            per[alias] = dict(truncated=t, natural_stop=n, mismatch=m, n=tot,
                              pct_truncated=round(100 * t / tot, 1))
            T += t; N += n; M += m
    tot = T + N + M
    return dict(per_model=per, total=dict(truncated=T, natural_stop=N, mismatch=M, n=tot),
                pct_truncated=round(100 * T / tot, 1),
                pct_natural_stop=round(100 * N / tot, 1),
                pct_mismatch=round(100 * M / tot, 1))

def main():
    result = {
        "description": "Generation-length robustness (128 vs 512 tokens) backing report §6.16. Aggregates only.",
        "asr_128_classifier": asr_block("results", "scores.judge.harmbench_cls.jsonl"),
        "asr_512_classifier": asr_block("results_512", "scores.judge.harmbench_cls.jsonl"),
        "asr_512_gpt4o": asr_block("results_512", "scores.judge.api_judge.jsonl"),
        "capability_128_vs_512": cap_block(),
        "response_length": length_block(),
        "prefix_truncation_128": prefix_truncation_block(),
        "cross_judge_kappa_512": kappa_block(),
    }
    out = REPO / "results_512" / "analysis" / "genlen_robustness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    # console summary
    print("Qwen-1.7B ΔASR: 128 classifier =", result["asr_128_classifier"]["qwen_2b"]["delta"],
          "| 512 classifier =", result["asr_512_classifier"]["qwen_2b"]["delta"],
          "| 512 gpt-4o =", result["asr_512_gpt4o"]["qwen_2b"]["delta"])
    print("512 cross-judge κ range:", result["cross_judge_kappa_512"]["kappa_min"], "–",
          result["cross_judge_kappa_512"]["kappa_max"])
    print("512 responses over 128 ceiling:", result["response_length"]["pct_512_over_ceiling"], "%")
    pt = result["prefix_truncation_128"]
    print(f"128 direct-prefix truncation: {pt['pct_truncated']}% truncated / "
          f"{pt['pct_natural_stop']}% natural stop / {pt['pct_mismatch']}% mismatch (n={pt['total']['n']})")
    print("Any BH-FDR-significant ΔASR @512 (classifier)?",
          any(v["bh_fdr_sig"] for v in result["asr_512_classifier"].values()))
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
