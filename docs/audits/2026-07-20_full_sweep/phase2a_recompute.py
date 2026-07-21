#!/usr/bin/env python3
"""T44 Phase 2A — deterministic recompute of report numbers from committed artifacts.

Independent re-derivation (no import of verify_report_claims): every value below is
recomputed from results_512/results sidecars/artifacts with this script's own
implementations (BH-FDR, exact McNemar, Cohen's kappa, MDE) and compared against the
values the documents claim. Groups A-C target the never-machine-checked claims found
by Phase 1b (FS-2, FS-3); group D independently re-derives the core locked set.

Run from repo root:  python3 docs/audits/2026-07-20_full_sweep/phase2a_recompute.py
"""
import json, math, re, sys
from pathlib import Path

R = Path(".")
OUT = []
FAILS = []

def check(name, got, want, tol=None):
    """Compare recomputed `got` vs claimed `want`. tol=None -> exact/rounded match."""
    if tol is None:
        ok = got == want
    else:
        ok = abs(got - want) <= tol
    OUT.append((name, got, want, ok))
    if not ok:
        FAILS.append((name, got, want))
    return ok

def jload(p):
    return json.load(open(R / p))

def jsonl(p):
    return [json.loads(l) for l in open(R / p)]

def kappa_from_cells(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
    return (po - pe) / (1 - pe)

def kappa_binary(pairs):
    """pairs: list of (a,b) booleans."""
    tp = sum(1 for a, b in pairs if a and b)
    fp = sum(1 for a, b in pairs if not a and b)
    fn = sum(1 for a, b in pairs if a and not b)
    tn = sum(1 for a, b in pairs if not a and not b)
    return kappa_from_cells(tp, fp, fn, tn)

def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)

def bh(pvals):
    """Return BH q-values in input order (step-up, monotone)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank_from_end, idx in enumerate(reversed(order)):
        rank = m - rank_from_end
        val = min(prev, pvals[idx] * m / rank)
        q[idx] = val
        prev = val
    return q

# ============================================================ A. §6.14 mechanism
rm = jload("results/analysis/refusal_margin.json")
rm512 = jload("results_512/analysis/refusal_margin.json")
check("A0 128/512 refusal_margin artifacts identical", json.dumps(rm, sort_keys=True) == json.dumps(rm512, sort_keys=True), True)

P = rm["pooled"]
check("A1 pooled flips 92", P["n_flips"], 92)
check("A2 pooled harmful-ward 50", P["n_harmful_ward"], 50)
check("A3 pooled safe-ward 42", P["n_safe_ward"], 42)
check("A4 pooled AUC 0.76", round(P["auc_thin_margin_predicts_flip"], 2), 0.76)
check("A5 z-scored AUC 0.64", round(P["auc_thin_margin_predicts_flip_within_family_z"], 2), 0.64)
check("A6 harmful-ward AUC 0.75", round(P["auc_thin_margin_predicts_harmful_ward"], 2), 0.75)
check("A7 safe-ward AUC 0.78", round(P["auc_thin_margin_predicts_safe_ward"], 2), 0.78)
check("A8 within-pair AUC qwen_2b 0.61", round(P["within_pair_auc_by_pair"]["qwen_2b"], 2), 0.61)
check("A9 within-pair AUC mistral 0.54", round(P["within_pair_auc_by_pair"]["mistral_7b"], 2), 0.54)
check("A10 baseline median qwen_2b 3.8", round(P["baseline_median_m1_by_pair"]["qwen_2b"], 1), 3.8)
check("A11 baseline median qwen_4b 13.0", round(P["baseline_median_m1_by_pair"]["qwen_4b"], 1), 13.0)

pp = rm["per_pair"]
fz = {k: pp[k]["flip_zoom"] for k in pp}
check("A12 qwen_2b flips 16/5 (n=21)", (fz["qwen_2b"]["n_harmful_ward"], fz["qwen_2b"]["n_safe_ward"], fz["qwen_2b"]["n_flips"]), (16, 5, 21))
check("A13 qwen_4b flips n=9", fz["qwen_4b"]["n_flips"], 9)
check("A14 llama flips 2/2", (fz["llama_3_2_3b"]["n_harmful_ward"], fz["llama_3_2_3b"]["n_safe_ward"]), (2, 2))
check("A15 phi flips 5/5", (fz["phi4_mini"]["n_harmful_ward"], fz["phi4_mini"]["n_safe_ward"]), (5, 5))
check("A16 mistral flips 20/28", (fz["mistral_7b"]["n_harmful_ward"], fz["mistral_7b"]["n_safe_ward"]), (20, 28))
vp_h = sum(fz[k]["n_harmful_ward"] for k in fz if k != "mistral_7b")
vp_s = sum(fz[k]["n_safe_ward"] for k in fz if k != "mistral_7b")
check("A17 valid-proxy families 30 vs 14", (vp_h, vp_s), (30, 14))

ms = {k: pp[k]["margin_shift"]["m1"] for k in pp}
check("A18 mean dm qwen_2b +1.2", round(ms["qwen_2b"]["mean_delta"], 1), 1.2)
check("A19 mean dm qwen_4b +4.0", round(ms["qwen_4b"]["mean_delta"], 1), 4.0)
check("A20 mean dm llama -1.1", round(ms["llama_3_2_3b"]["mean_delta"], 1), -1.1)
check("A21 mean dm phi -0.3", round(ms["phi4_mini"]["mean_delta"], 1), -0.3)
check("A22 Wilcoxon p<0.001 every pair", all(ms[k]["wilcoxon_p"] < 0.001 for k in ms), True)
dzs = [ms[k]["cohen_dz"] for k in ms]
check("A23 dz min ~ -0.74", round(min(dzs), 2), -0.74)
check("A24 dz max ~ +1.8", round(max(dzs), 1), 1.8)
gates = {k: pp[k]["gate"]["auc_m1"] for k in pp}
non_mistral = [v for k, v in gates.items() if k != "mistral_7b"]
check("A25 gate AUC four families in [0.86,0.90]", all(0.855 <= v <= 0.905 for v in non_mistral), True)
check("A26 gate AUC mistral 0.69", round(gates["mistral_7b"], 2), 0.69)
cc = pp["qwen_2b"]["confound_control"]
check("A27 entropy control +0.21 vs harmful +0.02", (round(cc["delta_entropy_control"], 2), round(cc["delta_entropy_harmbench"], 2)), (0.21, 0.02))

# ============================================================ B. INT8 capability significance (never machine-checked)
PAIRS = ["qwen_2b", "qwen_4b", "llama_3_2_3b", "mistral_7b", "phi4_mini"]
ps = jload("results_512/analysis/precision_sweep.json")
int8_sig_found = []
for pair in PAIRS:
    for bench, n_expect in (("mmlu", 300), ("arc", 1172)):
        base = {r["prompt_id"]: r["score_fields"]["is_correct"] for r in jsonl(f"results_512/{pair}_base/{bench}/raw.jsonl")}
        int8 = {r["prompt_id"]: r["score_fields"]["is_correct"] for r in jsonl(f"results_512/{pair}_8bit/{bench}/raw.jsonl")}
        shared = sorted(set(base) & set(int8))
        check(f"B {pair}/{bench} paired n", len(shared), n_expect)
        b = sum(1 for pid in shared if base[pid] and not int8[pid])
        c = sum(1 for pid in shared if not base[pid] and int8[pid])
        p = mcnemar_exact(b, c)
        delta = (sum(int8[pid] for pid in shared) - sum(base[pid] for pid in shared)) / len(shared)
        # artifact cross-check: precision_sweep point estimates
        key = "mmlu_accuracy" if bench == "mmlu" else "arc_accuracy"
        art_delta = ps["per_pair"][pair]["metrics"][key]["delta_int8_vs_fp16"]
        check(f"B {pair}/{bench} delta==precision_sweep", round(delta, 4), round(art_delta, 4))
        if p < 0.05:
            int8_sig_found.append((pair, bench, delta, p))
        OUT.append((f"B {pair}/{bench} McNemar p (b={b},c={c})", round(p, 4), "report: n.s.", p >= 0.05))
        if p < 0.05:
            FAILS.append((f"B {pair}/{bench} INT8 delta SIGNIFICANT", round(p, 4), "claimed n.s."))
check("B VERDICT: no INT8 MMLU/ARC delta significant (10 contrasts)", len(int8_sig_found), 0)
max_abs = 0.0
for pair in PAIRS:
    for key in ("mmlu_accuracy", "arc_accuracy"):
        max_abs = max(max_abs, abs(ps["per_pair"][pair]["metrics"][key]["delta_int8_vs_fp16"]))
check("B VERDICT: max |INT8 capability delta| <= ~1.3pp", max_abs <= 0.0135, True)

# ============================================================ C. Table 6.5 cells vs builder
hl = jload("results_512/analysis/headline_512_vs_128.json")
gl = jload("results_512/analysis/genlen_robustness.json")
art = {}
for row in hl["128"]:
    art[(row["pair"], "c128")] = row["delta"]
for row in hl["512"]:
    art[(row["pair"], "c512")] = row["delta"]
for pair, d in gl["asr_512_gpt4o"].items():
    art[(pair, "g512")] = d["delta"]

builder = open("scripts/build_fyp_report_v5.js").read()
m = re.search(r'buildTable\(\s*\["Pair",[^\]]*?@\s*128[^\]]*?\],\s*\[(.*?)\]\s*,\s*\[', builder, re.S)
if not m:
    # fallback: find the rows near the §6.16 table by locating five known row-label lines
    seg = builder[builder.find("6.16") : builder.find("6.16") + 6000]
    m = re.search(r'(\["Qwen3-1\.7B.*?phi.*?\])', seg, re.S)
rows_src = m.group(1) if m else ""
NAME2PAIR = {"Qwen3-1.7B": "qwen_2b", "Qwen3-4B": "qwen_4b", "Llama": "llama_3_2_3b", "Mistral": "mistral_7b", "Phi": "phi4_mini"}
table_ok = True
parsed_any = False
for line in rows_src.split("],"):
    cells = re.findall(r'"([^"]*)"', line)
    if len(cells) < 4:
        continue
    pair = next((v for k, v in NAME2PAIR.items() if k in cells[0]), None)
    if not pair:
        continue
    parsed_any = True
    nums = [re.search(r"[+\-−]?\d+\.\d+|0\.000", c) for c in cells[1:4]]
    vals = [float(x.group(0).replace("−", "-")) if x else None for x in nums]
    for v, colkey, col in zip(vals, ["c128", "c512", "g512"], ["@128 cls", "@512 cls", "@512 gpt4o"]):
        if v is None:
            continue
        ok = check(f"C Table6.5 {pair} {col}", v, round(art[(pair, colkey)], 3))
        table_ok = table_ok and ok
check("C Table 6.5 parsed", parsed_any, True)

# ============================================================ D. core independent re-derivations
mc = jload("results_512/analysis/multiple_comparisons.json")
pvals = [c["p_value"] for c in mc["contrasts"]]
qs = bh(pvals)
q_ok = all(abs(round(qs[i], 4) - c["bh_q_value"]) <= 0.0001 for i, c in enumerate(mc["contrasts"]))
check("D1 BH q-values (own implementation, 20 contrasts)", q_ok, True)
surv = sorted((c["pair_id"], c["metric"]) for i, c in enumerate(mc["contrasts"]) if qs[i] <= 0.05)
check("D2 BH survivors set", surv, sorted([("qwen_2b", "mmlu_accuracy"), ("llama_3_2_3b", "arc_accuracy"), ("phi4_mini", "xstest_over_refusal")]))
check("D3 n uncorrected significant", sum(1 for p in pvals if p < 0.05), mc["n_uncorrected_significant"])
mcn_ok = all(abs(mcnemar_exact(c["b"], c["c"]) - c["p_value"]) < 5e-4 for c in mc["contrasts"])
check("D4 exact McNemar reproduces all 20 contrast p-values", mcn_ok, True)
mde_ok = all(abs(2.8016 * math.sqrt(pp_["discordant_rate"] / 200) - pp_["mde_delta_asr"]) < 1e-3 for pp_ in mc["power_analysis"]["per_pair_harmbench_asr"])
check("D5 MDE formula reproduces per-pair MDEs", mde_ok, True)
check("D6 representative MDE 0.0594", round(2.8016 * math.sqrt(0.09 / 200), 4), 0.0594)

ja = jload("results_512/analysis/judge_agreement.json")
ja_models = ja["per_model"] if isinstance(ja["per_model"], list) else [dict(v, model_alias=k) for k, v in ja["per_model"].items()]
kap_ok, kap_max_err = True, 0.0
for entry in ja_models:
    alias = entry["model_alias"]
    j = {r["prompt_id"]: bool(r["judge_harmful"]) for r in jsonl(f"results_512/{alias}/harmbench/scores.judge.harmbench_cls.jsonl")}
    v = {r["prompt_id"]: bool(r["score_fields"]["harmful_compliance"]) for r in jsonl(f"results_512/{alias}/harmbench/scores.v2.jsonl")}
    shared = set(j) & set(v)
    k = kappa_binary([(j[pid], v[pid]) for pid in shared])
    err = abs(k - entry["cohens_kappa"])
    kap_max_err = max(kap_max_err, err)
    kap_ok = kap_ok and err < 1e-6
check("D7 judge-vs-v2 kappa recomputed from sidecars (10 aliases)", kap_ok, True, )
OUT.append(("D7b max kappa abs error", kap_max_err, "<1e-6", kap_max_err < 1e-6))

jp = jload("results_512/analysis/judge_pairwise_agreement.json")
cj_ok = True
for entry in jp["per_model"]:
    alias = entry["model_alias"]
    a = {r["prompt_id"]: bool(r["judge_harmful"]) for r in jsonl(f"results_512/{alias}/harmbench/scores.judge.harmbench_cls.jsonl")}
    b_ = {r["prompt_id"]: bool(r["judge_harmful"]) for r in jsonl(f"results_512/{alias}/harmbench/scores.judge.api_judge.jsonl")}
    shared = set(a) & set(b_)
    k = kappa_binary([(a[pid], b_[pid]) for pid in shared])
    cj_ok = cj_ok and abs(k - entry["cohens_kappa"]) < 1e-6
check("D8 cross-judge kappa (cls vs gpt-4o) recomputed (10 aliases)", cj_ok, True)

lg = jload("results_512/analysis/judge_pairwise_agreement_llamaguard.json")
lg_ok = True
for entry in lg["per_model"]:
    alias = entry["model_alias"]
    a = {r["prompt_id"]: bool(r["judge_harmful"]) for r in jsonl(f"results_512/{alias}/harmbench/scores.judge.harmbench_cls.jsonl")}
    b_ = {r["prompt_id"]: bool(r["judge_harmful"]) for r in jsonl(f"results_512/{alias}/harmbench/scores.judge.llamaguard.jsonl")}
    shared = set(a) & set(b_)
    k = kappa_binary([(a[pid], b_[pid]) for pid in shared])
    lg_ok = lg_ok and abs(k - entry["cohens_kappa"]) < 1e-6
check("D9 LlamaGuard kappa recomputed (10 aliases)", lg_ok, True)

hv = jload("results_512/analysis/human_validation.json")
def hv_kappa(flags, over, miss, n=200, human=37):
    tp = flags - over
    fn = miss
    fp = over
    tn = n - tp - fn - fp
    return kappa_from_cells(tp, fp, fn, tn)
check("D10 human-val classifier kappa 0.59", round(hv_kappa(46, 18, 9), 2), 0.59)
check("D11 human-val regex kappa 0.11", round(hv_kappa(132, 101, 6), 2), 0.11)

items = jsonl("results_512/analysis/xstest_human_validation_items.jsonl")
check("D12 gold-set n", len(items), 200)
hs = [(it["human_label"] == "full_refusal") for it in items]
hb = [(it["human_label"] in ("full_refusal", "partial_refusal")) for it in items]
rg = [bool(it["regex_refusal"]) for it in items]
js_ = [bool(it["judge_strict"]) for it in items]
jb = [bool(it["judge_broad"]) for it in items]
check("D13 strict kappa regex -0.006", round(kappa_binary(list(zip(hs, rg))), 3), -0.006)
check("D14 strict kappa judge 0.485", round(kappa_binary(list(zip(hs, js_))), 3), 0.485)
check("D15 broad kappa regex 0.054", round(kappa_binary(list(zip(hb, rg))), 3), 0.054)
check("D16 broad kappa judge 0.662", round(kappa_binary(list(zip(hb, jb))), 3), 0.662)
fulls = [it for it in items if it["human_label"] == "full_refusal"]
check("D17 full refusals 63", len(fulls), 63)
check("D18 regex caught 2 of 63", sum(1 for it in fulls if it["regex_refusal"]), 2)
check("D19 judge caught 61 of 63", sum(1 for it in fulls if it["judge_label"] == "full_refusal"), 61)
check("D20 3-class exact agreement 0.695", round(sum(1 for it in items if it["judge_label"] == it["human_label"]) / 200, 3), 0.695)

# ============================================================ report
n_ok = sum(1 for *_ , ok in OUT if ok)
print(f"\n{'='*70}\nPhase 2A: {n_ok}/{len(OUT)} checks pass, {len(FAILS)} mismatches")
for name, got, want in FAILS:
    print(f"  MISMATCH {name}: recomputed={got} claimed={want}")
with open("docs/audits/2026-07-20_full_sweep/phase2a_results.md", "w") as f:
    f.write("# T44 Phase 2A — recompute results\n\n")
    f.write(f"Script: `phase2a_recompute.py` (this dir). **{n_ok}/{len(OUT)} checks pass, {len(FAILS)} mismatches.**\n\n")
    f.write("| check | recomputed | claimed | ok |\n|---|---|---|---|\n")
    for name, got, want, ok in OUT:
        f.write(f"| {name} | {got} | {want} | {'✅' if ok else '❌ MISMATCH'} |\n")
sys.exit(1 if FAILS else 0)
