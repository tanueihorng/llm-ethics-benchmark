#!/usr/bin/env python3
"""Deterministic claim checker for the canonical FYP report (D43 follow-up).

Every load-bearing number in ``scripts/build_fyp_report_v5.js`` is asserted
twice:

  (a) the builder TEXT still contains the claim string, so a silent rewording
      or accidental deletion is caught; and
  (b) the value recomputed from the committed analysis artifacts equals the
      claimed value, so the report can never drift from the evidence.

Checks whose evidence is local-only (gitignored raw summaries, sensitivity
sidecars) are SKIPPED when absent (fresh clone / CI) rather than failed —
the same absence policy as the immutable-artifacts gate.

Run directly (``python scripts/verify_report_claims.py``) or via
``make verify-claims``; also wrapped by ``tests/test_report_claims.py`` so the
claim lock rides the normal pytest gate. Exit 0 = no FAIL.

Rationale: the 2026-07-02 adversarial audit found the canonical report
asserting retired 128-era kappa values as 512 results, and a follow-up sweep
found a prose/table mismatch (Mistral v2 proxy 0.835/0.900 vs the artifact's
0.825/0.890) that seven external review rounds had missed. Prose is not
self-verifying; this file makes the numeric claim surface machine-checked.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_fyp_report_v5.js"
A512 = ROOT / "results_512/analysis"
A128 = ROOT / "results/analysis"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def near(claimed: float, actual: float, places: int = 3) -> bool:
    """True when ``actual`` rounds to ``claimed`` at the claimed precision."""
    return abs(claimed - actual) <= 0.5 * 10 ** (-places) + 1e-12


class Checker:
    def __init__(self) -> None:
        self.text = BUILDER.read_text(encoding="utf-8")
        self.results: list[tuple[str, str, str]] = []

    def check(self, name: str, snippets: list[str], fn) -> None:
        missing = [s for s in snippets if s not in self.text]
        if missing:
            self.results.append(
                ("FAIL", name, f"report text missing: {missing[0][:90]!r}")
            )
            return
        try:
            ok, detail = fn()
        except FileNotFoundError as exc:
            self.results.append(("SKIP", name, f"local-only evidence absent: {exc}"))
            return
        except Exception as exc:  # noqa: BLE001 - a broken lookup is a failure
            self.results.append(("FAIL", name, f"checker error: {exc!r}"))
            return
        self.results.append(("PASS" if ok else "FAIL", name, detail))

    def report(self) -> int:
        width = max(len(n) for _, n, _ in self.results)
        fails = 0
        for status, name, detail in self.results:
            if status == "FAIL":
                fails += 1
            marker = {"PASS": "ok  ", "FAIL": "FAIL", "SKIP": "skip"}[status]
            print(f"{marker}  {name:<{width}}  {detail}")
        n = len(self.results)
        skips = sum(1 for s, _, _ in self.results if s == "SKIP")
        print("-" * 60)
        print(f"{n} checks: {n - fails - skips} pass, {fails} fail, {skips} skipped")
        return 1 if fails else 0


def run_checks(checker: Checker | None = None) -> Checker:
    c = checker or Checker()

    mc = _load(A512 / "multiple_comparisons.json")
    hl = _load(A512 / "headline_512_vs_128.json")
    ja = _load(A512 / "judge_agreement.json")["per_model"]
    jp = _load(A512 / "judge_pairwise_agreement.json")
    gl = _load(A512 / "genlen_robustness.json")
    sm = _load(A512 / "sensitivity_multiseed.json")
    pdl = _load(A512 / "pairwise_deltas.json")
    ps512 = _load(A512 / "precision_sweep.json")["per_pair"]
    ps128 = _load(A128 / "precision_sweep.json")["per_pair"]
    jp128 = _load(A128 / "judge_pairwise_agreement.json")
    ja128 = _load(A128 / "judge_agreement.json")["per_model"]

    hl512 = {r["pair"]: r for r in hl["512"]}
    hl128 = {r["pair"]: r for r in hl["128"]}
    jpp = {r["pair_id"]: r for r in jp["per_pair"]}
    jpm = {r["model_alias"]: r for r in jp["per_model"]}
    smp = {r["pair_id"]: r for r in sm["per_pair"]}
    jap = {r["pair_id"]: r for r in _load(A512 / "judge_agreement.json")["per_pair"]}
    pdix = {(r["pair_id"], r["benchmark"]): r for r in pdl}
    contrasts = {(r["pair_id"], r["metric"]): r for r in mc["contrasts"]}

    def surface_check(name: str, text: str, snippets: list[str], fn) -> None:
        """Apply the same text-plus-artifact contract to non-report surfaces."""
        missing = [s for s in snippets if s not in text]
        if missing:
            c.results.append(("FAIL", name, f"surface text missing: {missing[0][:90]!r}"))
            return
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001 - a broken lookup is a failure
            c.results.append(("FAIL", name, f"checker error: {exc!r}"))
            return
        c.results.append(("PASS" if ok else "FAIL", name, detail))

    # ---------------- BH-FDR family / headline -----------------------------
    c.check(
        "bh: exactly 3 survivors, none ASR",
        ["Not one HarmBench ASR contrast survives the correction"],
        lambda: (
            mc["n_bh_significant_q05"] == 3
            and not any("asr" in s["metric"] for s in mc["bh_survivors"]),
            f"n={mc['n_bh_significant_q05']}, metrics={[s['metric'] for s in mc['bh_survivors']]}",
        ),
    )
    c.check(
        "bh: survivor identities and deltas",
        ["Qwen-1.7B MMLU −0.090, Llama ARC −0.032, Phi over-refusal −0.048"],
        lambda: (
            {(s["pair_id"], s["metric"]) for s in mc["bh_survivors"]}
            == {("qwen_2b", "mmlu_accuracy"), ("llama_3_2_3b", "arc_accuracy"),
                ("phi4_mini", "xstest_over_refusal")}
            and near(-0.090, mc["bh_survivors"][0]["delta"])
            and near(-0.032, mc["bh_survivors"][1]["delta"])
            and near(-0.048, mc["bh_survivors"][2]["delta"]),
            str([(s["pair_id"], s["delta"]) for s in mc["bh_survivors"]]),
        ),
    )
    c.check(
        "bh: survivor q-values 0.008/0.008/0.012",
        [],
        lambda: (
            near(0.008, mc["bh_survivors"][0]["bh_q_value"])
            and near(0.008, mc["bh_survivors"][1]["bh_q_value"])
            and near(0.0122, mc["bh_survivors"][2]["bh_q_value"]),
            str([s["bh_q_value"] for s in mc["bh_survivors"]]),
        ),
    )
    c.check(
        "power: MDE ~0.06 at median discordant 0.09",
        [],
        lambda: (
            near(0.0594, mc["power_analysis"]["representative_mde_delta_asr_at_median_discordant_rate"], 4)
            and near(0.09, mc["power_analysis"]["median_discordant_rate"], 2),
            f"mde={mc['power_analysis']['representative_mde_delta_asr_at_median_discordant_rate']}",
        ),
    )

    # ---------------- per-pair judge ASR @512 -------------------------------
    def pair512(pair, base, quant, delta, lo, hi, p, places=3):
        r = hl512[pair]
        return (
            near(base, r["asr_base"]) and near(quant, r["asr_4bit"])
            and near(delta, r["delta"]) and near(lo, r["ci"][0])
            and near(hi, r["ci"][1]) and near(p, r["mcnemar_p"], places),
            f"{r['asr_base']}->{r['asr_4bit']} d={r['delta']} ci={r['ci']} p={r['mcnemar_p']}",
        )

    c.check("qwen_2b @512: 0.255->0.255, 0.000 [−0.055,+0.055], p=1.000",
            ['"0.255", "0.255", "0.000 [−0.055, +0.055]"'],
            lambda: pair512("qwen_2b", 0.255, 0.255, 0.0, -0.055, 0.055, 1.0))
    c.check("qwen_2b @512: symmetric 16/16 flips",
            ["sixteen prompts flip from refusal to compliance and sixteen"],
            lambda: (hl512["qwen_2b"]["n01"] == 16 and hl512["qwen_2b"]["n10"] == 16,
                     f"n01={hl512['qwen_2b']['n01']} n10={hl512['qwen_2b']['n10']}"))
    c.check("llama @512: 0.100->0.060, −0.040 [−0.075,−0.010], p=0.021",
            ["ΔASR = −0.040, CI [−0.075, −0.010]", "p = 0.021"],
            lambda: pair512("llama_3_2_3b", 0.100, 0.060, -0.040, -0.075, -0.010, 0.021))
    c.check("mistral @512: 0.585->0.565, −0.020 [−0.080,+0.040], p=0.627",
            ["0.585 at baseline and 0.565 under 4-bit", "McNemar p = 0.627"],
            lambda: pair512("mistral_7b", 0.585, 0.565, -0.020, -0.080, 0.040, 0.627))
    c.check("phi @512: 0.070->0.090, +0.020 [−0.015,+0.055], p=0.424",
            ["judge ASR 0.070 at baseline and 0.090 under 4-bit", "p = 0.424"],
            lambda: pair512("phi4_mini", 0.070, 0.090, 0.020, -0.015, 0.055, 0.424))
    # Phi's ΔASR sits exactly on the +0.02 tolerance; the boundary-deterministic
    # label rule (float noise rounded away) puts it at alignment_degradation with a
    # directional evidence status (CI includes zero). Pin both the artifact and the
    # report's boundary statement so the label can never silently revert to the
    # float-accidental robust_preservation.
    c.check("phi label @512 = alignment_degradation (directional), boundary-deterministic",
            ["Phi-4-mini +0.020 (alignment_degradation)", "directional"],
            lambda: (jap["phi4_mini"]["judge_label"] == "alignment_degradation"
                     and jap["phi4_mini"]["evidence_status"] == "directional",
                     f"phi judge_label={jap['phi4_mini']['judge_label']}, "
                     f"evidence={jap['phi4_mini']['evidence_status']}"))
    c.check("qwen_4b @512: 0.115->0.155, +0.040 [0.000,+0.080]",
            ['"0.115", "0.155", "+0.040 [0.000, +0.080]"'],
            lambda: pair512("qwen_4b", 0.115, 0.155, 0.040, 0.000, 0.080,
                            hl512["qwen_4b"]["mcnemar_p"]))
    c.check(
        "qwen baseline scaling: XSTest 0.052->0.028; MMLU 0.643->0.747",
        ["MMLU accuracy simultaneously rises from 0.643 to 0.747",
         "XSTest over-refusal falls from 0.052 to 0.028"],
        lambda: (
            near(0.052, pdix[("qwen_2b", "xstest")]["baseline_value"])
            and near(0.028, pdix[("qwen_4b", "xstest")]["baseline_value"])
            and near(0.643, pdix[("qwen_2b", "mmlu")]["baseline_value"])
            and near(0.747, pdix[("qwen_4b", "mmlu")]["baseline_value"]),
            "baseline levels match pairwise_deltas.json",
        ),
    )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    results_card = (ROOT / "docs/RESULTS_CARD.md").read_text(encoding="utf-8")
    surface_check(
        "outer surfaces: Llama/Mistral headline CIs match artifact",
        readme + "\n" + results_card,
        ["[−0.075, −0.010]", "[−0.080, +0.040]"],
        lambda: (
            readme.count("[−0.075, −0.010]") >= 1
            and results_card.count("[−0.075, −0.010]") >= 1
            and readme.count("[−0.080, +0.040]") >= 1
            and results_card.count("[−0.080, +0.040]") >= 1
            and "[−0.070, −0.010]" not in readme + results_card
            and "[−0.085, +0.040]" not in readme + results_card
            and near(-0.075, hl512["llama_3_2_3b"]["ci"][0])
            and near(-0.080, hl512["mistral_7b"]["ci"][0]),
            "README and RESULTS_CARD carry the current artifact CIs",
        ),
    )

    defense_deck = (ROOT / "docs/fyp-report-defense-deck-2026-07.html").read_text(encoding="utf-8")
    surface_check(
        "defense deck: qwen_4b BH q = .241",
        defense_deck,
        ["Qwen3-4B", '<span class="n">.241</span>'],
        lambda: (
            near(0.2406, contrasts[("qwen_4b", "harmbench_asr_judge")]["bh_q_value"], 4)
            and ".214</span>" not in defense_deck,
            "artifact q=" + str(contrasts[("qwen_4b", "harmbench_asr_judge")]["bh_q_value"]),
        ),
    )
    c.check(
        "128-era qwen_2b: +0.055, McNemar p=0.027, sig",
        ["+0.055 at 128 tokens"],
        lambda: (
            near(0.055, hl128["qwen_2b"]["delta"]) and near(0.027, hl128["qwen_2b"]["mcnemar_p"])
            and bool(hl128["qwen_2b"]["sig_ci"]),
            f"d={hl128['qwen_2b']['delta']} p={hl128['qwen_2b']['mcnemar_p']} sig={hl128['qwen_2b']['sig_ci']}",
        ),
    )

    # ---------------- second judge (gpt-4o) ---------------------------------
    c.check("gpt-4o deltas: qwen_2b +0.005, llama −0.035",
            ["gpt-4o gives Qwen 1.7B +0.005", "Llama −0.035"],
            lambda: (near(0.005, jpp["qwen_2b"]["api_judge_delta"])
                     and near(-0.035, jpp["llama_3_2_3b"]["api_judge_delta"]),
                     f"qwen={jpp['qwen_2b']['api_judge_delta']} llama={jpp['llama_3_2_3b']['api_judge_delta']}"))
    c.check(
        "cross-judge kappa range 0.68–0.95 @512",
        ["κ 0.68–0.95"],
        lambda: (
            near(0.68, min(m["cohens_kappa"] for m in jp["per_model"]), 2)
            and near(0.95, max(m["cohens_kappa"] for m in jp["per_model"]), 2),
            f"min={min(m['cohens_kappa'] for m in jp['per_model']):.3f} "
            f"max={max(m['cohens_kappa'] for m in jp['per_model']):.3f}",
        ),
    )
    c.check(
        "cross-judge kappa 0.60–0.95 across both budgets",
        ["κ 0.60–0.95 across all ten models, 0.68–0.95 at the 512-token reference budget"],
        lambda: (
            near(0.60, min(m["cohens_kappa"] for m in jp128["per_model"]), 2)
            and near(0.95, max(max(m["cohens_kappa"] for m in jp["per_model"]),
                               max(m["cohens_kappa"] for m in jp128["per_model"])), 2),
            f"128 min={min(m['cohens_kappa'] for m in jp128['per_model']):.3f}",
        ),
    )
    c.check("mistral cross-judge 0.83/0.78 @512",
            ["κ 0.83 (baseline) and 0.78 (4-bit)"],
            lambda: (near(0.83, jpm["mistral_7b_base"]["cohens_kappa"], 2)
                     and near(0.78, jpm["mistral_7b_4bit"]["cohens_kappa"], 2),
                     f"{jpm['mistral_7b_base']['cohens_kappa']:.3f}/{jpm['mistral_7b_4bit']['cohens_kappa']:.3f}"))
    jpm128 = {r["model_alias"]: r for r in jp128["per_model"]}
    c.check("phi cross-judge 0.68/0.83 @512 (0.79/0.95 scoped to 128)",
            ["κ 0.68 at baseline and 0.83 under 4-bit",
             "κ 0.79/0.95 belongs to the retired 128-token budget"],
            lambda: (near(0.68, jpm["phi4_mini_base"]["cohens_kappa"], 2)
                     and near(0.83, jpm["phi4_mini_4bit"]["cohens_kappa"], 2)
                     and near(0.79, jpm128["phi4_mini_base"]["cohens_kappa"], 2)
                     and near(0.95, jpm128["phi4_mini_4bit"]["cohens_kappa"], 2),
                     f"512={jpm['phi4_mini_base']['cohens_kappa']:.3f}/{jpm['phi4_mini_4bit']['cohens_kappa']:.3f} "
                     f"128={jpm128['phi4_mini_base']['cohens_kappa']:.3f}/{jpm128['phi4_mini_4bit']['cohens_kappa']:.3f}"))

    # ---------------- judge-vs-proxy kappa (Table 6.3 + prose) --------------
    KAPPA = {
        "qwen_2b_base": 0.36, "qwen_2b_4bit": 0.41,
        "qwen_4b_base": 0.49, "qwen_4b_4bit": 0.59,
        "llama_3_2_3b_base": 0.71, "llama_3_2_3b_4bit": 0.84,
        "mistral_7b_base": 0.29, "mistral_7b_4bit": 0.25,
        "phi4_mini_base": 0.67, "phi4_mini_4bit": 0.77,
    }
    c.check(
        "judge-vs-proxy kappa per model (all 10)",
        ['"0.36"', '"0.41"'],
        lambda: (
            all(near(v, ja[k]["cohens_kappa"], 2) for k, v in KAPPA.items()),
            str({k: round(ja[k]["cohens_kappa"], 3) for k in KAPPA}),
        ),
    )
    c.check(
        "mistral 512 kappa in prose: 0.29/0.25 (128-era 0.19/0.11 scoped)",
        ["Cohen's κ = 0.29 at baseline, 0.25 under 4-bit",
         "at the retired 128-token budget it had been lower still, 0.19/0.11"],
        lambda: (
            near(0.29, ja["mistral_7b_base"]["cohens_kappa"], 2)
            and near(0.25, ja["mistral_7b_4bit"]["cohens_kappa"], 2)
            and near(0.19, ja128["mistral_7b_base"]["cohens_kappa"], 2)
            and near(0.11, ja128["mistral_7b_4bit"]["cohens_kappa"], 2),
            f"512={ja['mistral_7b_base']['cohens_kappa']:.3f}/{ja['mistral_7b_4bit']['cohens_kappa']:.3f} "
            f"128={ja128['mistral_7b_base']['cohens_kappa']:.3f}/{ja128['mistral_7b_4bit']['cohens_kappa']:.3f}",
        ),
    )
    c.check(
        "no unscoped retired-128-era kappa left (0.19-at-baseline / 0.60-0.63 / as-low-as-0.11)",
        [],
        lambda: (
            all(("128" in line) for line in c.text.split("\n")
                if re.search(r"0\.19 at baseline|κ 0\.60–0\.63|κ as low as 0\.11|0\.25–0\.41", line)),  # retired 128-era values + a range wrong at both budgets
            "every retired-kappa mention is 128-scoped on its line",
        ),
    )
    c.check(
        "v2 proxy range 0.25–0.84; Qwen+Mistral 0.25–0.59; Llama+Phi 0.67–0.84",
        ["κ 0.25–0.59", "κ 0.67–0.84"],
        lambda: (
            near(0.25, min(m["cohens_kappa"] for m in ja.values()), 2)
            and near(0.84, max(m["cohens_kappa"] for m in ja.values()), 2)
            and near(0.59, max(ja[k]["cohens_kappa"] for k in KAPPA if "qwen" in k), 2)
            and near(0.67, min(ja[k]["cohens_kappa"] for k in KAPPA if "llama" in k or "phi" in k), 2),
            "ranges recomputed from judge_agreement.json",
        ),
    )
    c.check(
        "mistral v2 proxy 0.830 -> 0.890 (prose == table == artifact)",
        ["non-refusal rate (0.830) rising to 0.890", '"0.830", "0.890"'],
        lambda: (
            near(0.830, ja["mistral_7b_base"]["v2_asr"]) and near(0.890, ja["mistral_7b_4bit"]["v2_asr"]),
            f"{ja['mistral_7b_base']['v2_asr']}/{ja['mistral_7b_4bit']['v2_asr']}",
        ),
    )
    c.check(
        "qwen judge-vs-proxy over-count examples (0.255 vs 0.595; 0.115 vs 0.235)",
        ["0.255 vs 0.595", "0.115 vs 0.235"],
        lambda: (
            near(0.255, ja["qwen_2b_base"]["judge_asr"]) and near(0.595, ja["qwen_2b_base"]["v2_asr"])
            and near(0.115, ja["qwen_4b_base"]["judge_asr"]) and near(0.235, ja["qwen_4b_base"]["v2_asr"]),
            "judge/v2 per alias match",
        ),
    )

    # ---------------- generation-length (6.16) ------------------------------
    pt = gl["prefix_truncation_128"]
    rl = gl["response_length"]
    c.check(
        "truncation: 60.3% (1206/2000), 30.5% natural, 9.2% mismatch (184)",
        ["60.3 percent", "9.2 percent of the 2,000 paired generations (184"],
        lambda: (
            near(60.3, pt["pct_truncated"], 1) and pt["total"]["truncated"] == 1206
            and pt["total"]["n"] == 2000 and near(30.5, pt["pct_natural_stop"], 1)
            and near(9.2, pt["pct_mismatch"], 1) and pt["total"]["mismatch"] == 184,
            f"{pt['total']}",
        ),
    )
    c.check(
        "per-family truncation: mistral 93.5–98.0, phi 73.5–78.5, qwen 54.5–70.0, llama 3.0–4.0",
        ["93.5–98.0", "73.5–78.5", "54.5–70.0", "3.0–4.0"],
        lambda: (
            near(93.5, min(pt["per_model"][k]["pct_truncated"] for k in pt["per_model"] if "mistral" in k), 1)
            and near(98.0, max(pt["per_model"][k]["pct_truncated"] for k in pt["per_model"] if "mistral" in k), 1)
            and near(73.5, min(pt["per_model"][k]["pct_truncated"] for k in pt["per_model"] if "phi" in k), 1)
            and near(78.5, max(pt["per_model"][k]["pct_truncated"] for k in pt["per_model"] if "phi" in k), 1)
            and near(54.5, min(pt["per_model"][k]["pct_truncated"] for k in pt["per_model"] if "qwen" in k), 1)
            and near(70.0, max(pt["per_model"][k]["pct_truncated"] for k in pt["per_model"] if "qwen" in k), 1)
            and near(3.0, min(pt["per_model"][k]["pct_truncated"] for k in pt["per_model"] if "llama" in k), 1)
            and near(4.0, max(pt["per_model"][k]["pct_truncated"] for k in pt["per_model"] if "llama" in k), 1),
            "family ranges recomputed",
        ),
    )
    c.check(
        "median lengths 1,675 vs 567 chars; ceiling proxy 62%",
        ["median length 1,675 characters against 567"],
        lambda: (
            rl["all_median_512"] == 1675 and rl["all_median_128"] == 567
            and near(62.0, rl["pct_512_over_ceiling"], 1),
            f"{rl['all_median_128']}->{rl['all_median_512']} ceiling%={rl['pct_512_over_ceiling']}",
        ),
    )
    c.check(
        "mistral absolute ASR budget effect 0.585 vs 0.385",
        ["Mistral-7B classifier ASR is 0.585 at the reference budget against 0.385 at 128 tokens"],
        lambda: (
            near(0.585, hl512["mistral_7b"]["asr_base"]) and near(0.385, hl128["mistral_7b"]["asr_base"]),
            f"512={hl512['mistral_7b']['asr_base']} 128={hl128['mistral_7b']['asr_base']}",
        ),
    )

    # ---------------- multi-seed (6.6.1) ------------------------------------
    c.check(
        "multiseed qwen_2b: mean +0.013 [0.000,+0.035], 5 seeds, all non-negative, greedy in range",
        ["mean +0.013, range [0.000, +0.035]", "all five seed deltas are non-negative"],
        lambda: (
            near(0.013, smp["qwen_2b"]["judge_delta"]["mean"])
            and near(0.0, smp["qwen_2b"]["judge_delta"]["min"])
            and near(0.035, smp["qwen_2b"]["judge_delta"]["max"])
            and smp["qwen_2b"]["judge_delta"]["n_seeds"] == 5
            and bool(smp["qwen_2b"]["judge_delta"]["sign_consistent"])
            and smp["qwen_2b"]["judge_delta"]["min"] >= 0
            and bool(smp["qwen_2b"]["greedy_in_multiseed_range"]),
            str(smp["qwen_2b"]["judge_delta"]),
        ),
    )
    c.check(
        "multiseed qwen_4b mean +0.029; llama mean −0.024, no positive seed, greedy in range",
        ["mean +0.029", "mean −0.024 with no positive seed"],
        lambda: (
            near(0.029, smp["qwen_4b"]["judge_delta"]["mean"])
            and near(-0.024, smp["llama_3_2_3b"]["judge_delta"]["mean"])
            and smp["llama_3_2_3b"]["judge_delta"]["max"] < 0
            and bool(smp["llama_3_2_3b"]["greedy_in_multiseed_range"]),
            f"llama={smp['llama_3_2_3b']['judge_delta']}",
        ),
    )

    def _per_seed_ok():
        # 2026-07-08 audit: the §6.6.1 per-seed delta list and the
        # "k of 5 seeds individually significant" counts were stated in the
        # report but persisted nowhere; sensitivity_analysis.py now emits
        # judge_delta.per_seed + n_seeds_significant_p05 and this check pins
        # the report's claims to them.
        q2 = smp["qwen_2b"]["judge_delta"]
        q4 = smp["qwen_4b"]["judge_delta"]
        ll = smp["llama_3_2_3b"]["judge_delta"]
        q2_deltas = [round(s["delta"], 3) for s in q2["per_seed"]]
        ll_sig = [s for s in ll["per_seed"] if s["significant_p05"]]
        ok = (
            q2_deltas == [0.0, 0.01, 0.02, 0.0, 0.035]
            and q2["n_seeds_significant_p05"] == 0
            and q4["n_seeds_significant_p05"] == 1
            and ll["n_seeds_significant_p05"] == 2
            and all(s["delta"] < 0 for s in ll_sig)
        )
        return ok, (
            f"q2={q2_deltas} sig {q2['n_seeds_significant_p05']}/5; "
            f"q4 sig {q4['n_seeds_significant_p05']}/5; "
            f"llama sig {ll['n_seeds_significant_p05']}/5 (all decreases: "
            f"{all(s['delta'] < 0 for s in ll_sig)})"
        )

    c.check(
        "multiseed per-seed: delta list + 0/5, 1/5, 2/5 significance counts",
        [
            "0.000, +0.010, +0.020, 0.000, +0.035",
            "no seed is individually significant (0/5)",
            "1 of 5 seeds individually significant",
            "2 of 5 seeds individually significant, both decreases",
        ],
        _per_seed_ok,
    )

    # ---- XSTest qwen_2b over-refusal now cleanly non-significant (line-38 fix) ----
    # After removing the unanchored "unable to" refusal pattern (which over-counted
    # third-person content), qwen_2b over-refusal drops from the earlier "borderline"
    # state (bootstrap CI excluded 0 but McNemar n.s.) to non-significant on BOTH
    # criteria: ΔOR −0.024, CI [−0.048, 0.000] (touches 0), McNemar p = 0.109.
    def _qwen2b_or_ok():
        pd_x = pdix[("qwen_2b", "xstest")]
        mc_x = contrasts[("qwen_2b", "xstest_over_refusal")]
        ok = (
            pd_x["metric"] == "over_refusal_rate"
            and not bool(pd_x["delta_significant"])            # bootstrap CI now includes 0
            and near(-0.024, pd_x["absolute_delta"])
            and not mc_x["uncorrected_significant"]            # McNemar n.s.
            and near(0.109, mc_x["p_value"])
            and mc_x["b"] + mc_x["c"] == 10
        )
        return ok, (
            f"bootstrap sig={pd_x['delta_significant']} delta={pd_x['absolute_delta']}; "
            f"McNemar p={mc_x['p_value']:.4f} discordant={mc_x['b'] + mc_x['c']}"
        )

    c.check(
        "xstest qwen_2b: over-refusal non-significant (CI includes 0, McNemar p=0.109)",
        [
            '"−0.024 [−0.048, 0.000]"',
            "McNemar p = 0.109",
            "10 discordant prompts",   # artifact b+c=10; prose had drifted to 11 (P1 audit fix)
        ],
        _qwen2b_or_ok,
    )

    # ---- Strict-parser capability bracket (T38 / P1 audit fix). The report had
    # called ARC "immune to / not subject to" the 4-bit answer-format asymmetry
    # and used it to characterise the capability loss as MMLU-specific. The
    # committed strict-parser sensitivity refutes that: the smallest pair's 4-bit
    # ARC answers fall to the lenient fallback even more than its MMLU, so ARC's
    # near-zero primary delta is salvage, not immunity, and under a strict parser
    # ARC and MMLU fall comparably. Lock the bracket numbers to the artifact and
    # assert the retired "immune" phrasing is gone. -------------------------------
    pss = _load(A512 / "parser_strict_sensitivity.json")
    pss_pair = {(r["pair_id"], r["benchmark"]): r for r in pss["per_pair"]}
    pss_alias = {(r["model_alias"], r["benchmark"]): r for r in pss["per_alias"]}

    def _strict_bracket_ok():
        arc = pss_pair[("qwen_2b", "arc")]["strict_delta"]
        mmlu = pss_pair[("qwen_2b", "mmlu")]["strict_delta"]
        arc_fb = pss_alias[("qwen_2b_4bit", "arc")]["tier_usage"]["fallback_frac"]
        mmlu_fb = pss_alias[("qwen_2b_4bit", "mmlu")]["tier_usage"]["fallback_frac"]
        arc_base_fb = pss_alias[("qwen_2b_base", "arc")]["tier_usage"]["fallback_frac"]
        ok = (
            near(-0.343, arc["delta"], 3) and near(-0.375, arc["ci_lower"], 3)
            and near(-0.311, arc["ci_upper"], 3) and bool(arc["significant"])
            and near(-0.293, mmlu["delta"], 3) and near(-0.350, mmlu["ci_lower"], 3)
            and near(-0.237, mmlu["ci_upper"], 3)
            and near(0.523, arc_fb, 3) and near(0.487, mmlu_fb, 3)
            and near(0.025, arc_base_fb, 3)
            and "immune to this asymmetry" not in c.text
            and "not subject to this format asymmetry" not in c.text
        )
        return ok, (
            f"ARC strict {arc['delta']:.3f} [{arc['ci_lower']:.3f},{arc['ci_upper']:.3f}]; "
            f"MMLU strict {mmlu['delta']:.3f}; ARC 4bit fallback {arc_fb:.3f} vs base {arc_base_fb:.3f}; "
            "retired 'immune' phrasing absent"
        )

    c.check(
        "strict-parser bracket: ARC −0.343 / MMLU −0.293, ARC 52.3% fallback, not 'immune'",
        ["ARC falls −0.343", "MMLU falls −0.293", "52.3%"],
        _strict_bracket_ok,
    )

    def _llama_base_or_ok():
        v2 = _load(A512.parent / "llama_3_2_3b_base/xstest/summary.v2.json")
        return near(0.032, v2["metrics"]["over_refusal_rate"], 3), (
            f"llama base xstest v2 OR={v2['metrics']['over_refusal_rate']}"
        )

    c.check(
        "llama baseline XSTest over-refusal 0.032 (v2 scorer of record, not retired 0.036)",
        ["XSTest over-refusal is 0.032, low."],
        _llama_base_or_ok,
    )

    # ---------------- Appendix A reproduces tc1_512.yaml ---------------------
    def _appendix_a_ok():
        m = re.search(r"const tc1Yaml = `([\s\S]*?)`;", c.text)
        if not m:
            return False, "tc1Yaml template literal not found"
        appx = m.group(1)
        real = (ROOT / "configs/tc1_512.yaml").read_text(encoding="utf-8")

        def _kv(text, key):
            mm = re.search(rf"^\s*{key}:\s*(\S+)", text, re.M)
            return mm.group(1) if mm else None

        keys = ["max_new_tokens", "log_dir", "temperature", "seed"]
        diffs = [k for k in keys if _kv(appx, k) != _kv(real, k)]
        return not diffs, (
            "appendix matches tc1_512.yaml on " + ", ".join(keys)
            if not diffs
            else f"appendix/config mismatch on: {diffs}"
        )

    c.check(
        "appendix A: reproduced config matches configs/tc1_512.yaml",
        ["The full configs/tc1_512.yaml"],
        _appendix_a_ok,
    )

    # ---------------- human-label validation (§6.12 Result 5, T30) ----------
    def _human_val_ok():
        hv = _load(A512 / "human_validation.json")          # committed, redacted
        clf = hv["classifier_vs_human"]; rgx = hv["regex_vs_human"]
        n = hv["n_labeled"]
        tt = (ROOT / "scripts/build_fyp_thesis_v4.js").read_text(encoding="utf-8")
        ok = (
            near(0.59, clf["cohens_kappa"], 2) and near(0.11, rgx["cohens_kappa"], 2) and n == 200
            and "Cohen's κ = 0.59" in c.text and "κ = 0.11" in c.text   # report Result 5 κ
            and "κ 0.59 vs regex 0.11" in tt                            # thesis mirror
            # confusion counts (report Result 5 "human 37 < clf 46 < regex 132; 18/9 vs 101/6")
            and hv["human_harmful_count"] == 37
            and clf["n_flagged_harmful"] == 46 and clf["over_flag_vs_human"] == 18 and clf["missed_vs_human"] == 9
            and rgx["n_flagged_harmful"] == 132 and rgx["over_flag_vs_human"] == 101 and rgx["missed_vs_human"] == 6
        )
        return ok, (f"clf κ={clf['cohens_kappa']:.3f} flags {clf['n_flagged_harmful']} (over {clf['over_flag_vs_human']}/miss {clf['missed_vs_human']}); "
                    f"regex κ={rgx['cohens_kappa']:.3f} flags {rgx['n_flagged_harmful']} (over {rgx['over_flag_vs_human']}/miss {rgx['missed_vs_human']}); human {hv['human_harmful_count']}")

    c.check(
        "human validation: κ 0.59/0.11 + confusion counts (37/46/132; 18-9/101-6) == artifact",
        ["Cohen's κ = 0.59 (moderate agreement)", "κ = 0.11 (negligible)",
         "200 saved HarmBench generations",
         "37 of 200, against the classifier's 46 and the regex's 132",
         "101 over-flags against 6 misses"],
        _human_val_ok,
    )

    # ---------------- INT8 precision point (6.15) ---------------------------
    def _mcn(ps, pair, judge):
        blk = ps[pair]["harmbench_asr_int8_mcnemar"][judge]
        return blk.get("p_value", blk.get("p", blk.get("mcnemar_p")))

    c.check(
        "INT8 llama @512: clf +0.005 p=1.000; gpt-4o +0.010 p=0.688",
        ["classifier Δ+0.005, McNemar p = 1.000", "gpt-4o Δ+0.010, p = 0.688"],
        lambda: (
            near(0.005, ps512["llama_3_2_3b"]["metrics"]["harmbench_asr_judge"]["delta_int8_vs_fp16"])
            and near(1.0, _mcn(ps512, "llama_3_2_3b", "harmbench_cls"))
            and near(0.688, _mcn(ps512, "llama_3_2_3b", "api_judge")),
            f"d={ps512['llama_3_2_3b']['metrics']['harmbench_asr_judge']['delta_int8_vs_fp16']} "
            f"p_cls={_mcn(ps512, 'llama_3_2_3b', 'harmbench_cls')} p_api={_mcn(ps512, 'llama_3_2_3b', 'api_judge')}",
        ),
    )
    c.check(
        "INT8 llama @128: +0.040 both-judge (clf p=0.021, api p=0.008)",
        ["p = 0.021", "0.0078"] if "0.0078" in c.text else ["p = 0.021"],
        lambda: (
            near(0.040, ps128["llama_3_2_3b"]["metrics"]["harmbench_asr_judge"]["delta_int8_vs_fp16"])
            and near(0.021, _mcn(ps128, "llama_3_2_3b", "harmbench_cls"))
            and near(0.008, _mcn(ps128, "llama_3_2_3b", "api_judge")),
            f"d={ps128['llama_3_2_3b']['metrics']['harmbench_asr_judge']['delta_int8_vs_fp16']} "
            f"p_cls={_mcn(ps128, 'llama_3_2_3b', 'harmbench_cls')} p_api={_mcn(ps128, 'llama_3_2_3b', 'api_judge')}",
        ),
    )

    # Artifact-consistency guard for the precision-sweep XSTest over-refusal
    # column. Unlike the ASR-judge column (tracked judge sidecars, recomputable
    # in CI and pinned by the thesis Table 6.3 check below), this column reads the
    # gitignored summary.v2.json inputs, so it cannot be recomputed on a fresh
    # clone — and it drifted silently once the sweep script was switched to prefer
    # v2 while the committed column stayed v1 (2026-07-14; the follow-up T40/D47
    # deferred). It is not rendered in the report or thesis, so there is no prose
    # to pin; instead assert it against the authoritative v2 deltas in
    # pairwise_deltas.json (the source the §6.3 pair prose DOES cite, e.g.
    # "0.052 -> 0.028, deltaOR = -0.024"). This is the machine check the drift
    # lacked (D42); it would have fired on the stale column (e.g. Phi nf4 0.084 vs
    # the pairwise-implied 0.080). fp16/nf4 only — INT8 has no base-vs-4bit
    # pairwise counterpart.
    def _sweep_xstest_matches_pairwise():
        bad = []
        for pair in ("qwen_2b", "qwen_4b", "llama_3_2_3b", "mistral_7b", "phi4_mini"):
            s = ps512[pair]["metrics"]["xstest_over_refusal"]
            r = pdix[(pair, "xstest")]
            if not (near(s["fp16"], r["baseline_value"])
                    and near(s["nf4"], r["quantized_value"])
                    and near(s["delta_nf4_vs_fp16"], r["absolute_delta"])):
                bad.append(f"{pair}(fp16 {s['fp16']}/{r['baseline_value']}, "
                           f"nf4 {s['nf4']}/{r['quantized_value']})")
        return (not bad, "all 5 pairs agree" if not bad else "MISMATCH: " + "; ".join(bad))

    c.check(
        "sweep: XSTest over-refusal column == pairwise_deltas (v2, no drift)",
        [],  # artifact-vs-artifact invariant; the column is not rendered in prose
        _sweep_xstest_matches_pairwise,
    )

    # ---------------- capability / over-refusal (pairwise_deltas) -----------
    def pdelta(pair, bench, base=None, quant=None, delta=None, sig=None, places=3):
        r = pdix[(pair, bench)]
        ok = True
        if base is not None:
            ok &= near(base, r["baseline_value"], places)
        if quant is not None:
            ok &= near(quant, r["quantized_value"], places)
        if delta is not None:
            ok &= near(delta, r["absolute_delta"], places)
        if sig is not None:
            ok &= bool(r["delta_significant"]) == sig
        return ok, (f"{r['baseline_value']}->{r['quantized_value']} d={r['absolute_delta']:.4f} "
                    f"sig={r['delta_significant']}")

    c.check("qwen_2b MMLU 0.643->0.553 (−0.090, sig)",
            ["0.643 → 0.553"] if "0.643 → 0.553" in c.text else ["−0.090"],
            lambda: pdelta("qwen_2b", "mmlu", 0.643, 0.553, -0.090, True))
    c.check("llama MMLU 0.610->0.573 (−0.037, n.s.)",
            ["0.610 → 0.573"],
            lambda: pdelta("llama_3_2_3b", "mmlu", 0.610, 0.573, -0.037, False))
    c.check("llama ARC −0.032 (sig)", ["ΔARC = −0.032"],
            lambda: pdelta("llama_3_2_3b", "arc", delta=-0.032, sig=True))
    c.check("mistral MMLU −0.020 n.s.; ARC +0.009 n.s.",
            ["MMLU −2.0 pp, n.s.; ARC +0.9 pp, n.s."],
            lambda: (pdelta("mistral_7b", "mmlu", delta=-0.020, sig=False)[0]
                     and pdelta("mistral_7b", "arc", delta=0.009, sig=False)[0],
                     f"mmlu={pdix[('mistral_7b','mmlu')]['absolute_delta']:.4f} "
                     f"arc={pdix[('mistral_7b','arc')]['absolute_delta']:.4f}"))
    c.check("phi MMLU −2.7pp n.s.; ARC −1.5pp n.s.",
            ["MMLU −2.7 pp and ARC −1.5 pp (both n.s.)"],
            lambda: (pdelta("phi4_mini", "mmlu", delta=-0.027, sig=False)[0]
                     and pdelta("phi4_mini", "arc", delta=-0.015, sig=False)[0],
                     f"mmlu={pdix[('phi4_mini','mmlu')]['absolute_delta']:.4f} "
                     f"arc={pdix[('phi4_mini','arc')]['absolute_delta']:.4f}"))
    c.check("phi over-refusal −0.048 [−0.076,−0.020] (sig, a decrease)",
            ["ΔOR = −0.048 (CI [−0.076, −0.020]"],
            lambda: (
                pdelta("phi4_mini", "xstest", delta=-0.048, sig=True)[0]
                and near(-0.076, pdix[("phi4_mini", "xstest")]["delta_ci_lower"])
                and near(-0.020, pdix[("phi4_mini", "xstest")]["delta_ci_upper"]),
                f"ci=[{pdix[('phi4_mini','xstest')]['delta_ci_lower']},"
                f"{pdix[('phi4_mini','xstest')]['delta_ci_upper']}]"))
    c.check(
        "Table 6.1 v2-proxy CI cells (qwen_2b/qwen_4b/mistral)",
        ['"−0.025 [−0.070, +0.020]"', '"+0.070 [+0.030, +0.115]"', '"+0.060 [+0.015, +0.105]"'],
        lambda: (
            near(-0.025, pdix[("qwen_2b", "harmbench")]["absolute_delta"])
            and near(-0.070, pdix[("qwen_2b", "harmbench")]["delta_ci_lower"])
            and near(0.020, pdix[("qwen_2b", "harmbench")]["delta_ci_upper"])
            and near(0.070, pdix[("qwen_4b", "harmbench")]["absolute_delta"])
            and near(0.030, pdix[("qwen_4b", "harmbench")]["delta_ci_lower"])
            and near(0.115, pdix[("qwen_4b", "harmbench")]["delta_ci_upper"])
            and near(0.060, pdix[("mistral_7b", "harmbench")]["absolute_delta"])
            and near(0.015, pdix[("mistral_7b", "harmbench")]["delta_ci_lower"])
            and near(0.105, pdix[("mistral_7b", "harmbench")]["delta_ci_upper"]),
            "v2 deltas + CIs match pairwise_deltas.json",
        ),
    )
    c.check("qwen_4b ARC −1.6pp sig; OR 0.028->0.024 (−0.004, n.s.)",
            ["−1.6 pp", "0.028 → 0.024"],
            lambda: (pdelta("qwen_4b", "arc", delta=-0.016, sig=True)[0]
                     and pdelta("qwen_4b", "xstest", 0.028, 0.024, -0.004)[0],
                     f"arc={pdix[('qwen_4b','arc')]['absolute_delta']:.4f}"))
    c.check("llama OR 0.032->0.048 (+0.016, n.s.)",
            ["0.032 to 0.048"],
            lambda: pdelta("llama_3_2_3b", "xstest", 0.032, 0.048, 0.016, False))
    c.check("baseline MMLU anchors 0.643 / 0.747 / 0.610",
            ["0.643", "0.747", "0.610"],
            lambda: (near(0.643, pdix[("qwen_2b", "mmlu")]["baseline_value"])
                     and near(0.747, pdix[("qwen_4b", "mmlu")]["baseline_value"])
                     and near(0.610, pdix[("llama_3_2_3b", "mmlu")]["baseline_value"]),
                     "qwen_2b/qwen_4b/llama baseline MMLU"))

    # -------- XSTest refusal-judge sensitivity (§6.12 Result 6; T35/D45) -----
    xja = _load(A512 / "xstest_judge_agreement.json")
    xjp = {r["pair_id"]: r for r in xja["per_pair"]}
    xjm = xja["per_model"]
    phi_s = xjp["phi4_mini"]["judge_strict"]
    phi_b = xjp["phi4_mini"]["judge_broad"]

    c.check("Result 6: Phi judge ΔOR +0.016 strict (up) / −0.004 broad, both n.s. — regex −0.048 not reproduced",
            ["+0.016 (strict; direction reversed, CI [−0.028, +0.060], McNemar p = 0.597)",
             "−0.004 (broad; CI [−0.048, +0.036], McNemar p = 1.000)"],
            lambda: (
                near(0.016, phi_s["delta"]) and phi_s["direction"] == "up" and not phi_s["significant"]
                and near(-0.028, phi_s["ci_lower"]) and near(0.060, phi_s["ci_upper"])
                and near(0.597, phi_s["mcnemar_p_value"])
                and near(-0.004, phi_b["delta"]) and not phi_b["significant"]
                and near(-0.048, phi_b["ci_lower"]) and near(0.036, phi_b["ci_upper"])
                and near(1.000, phi_b["mcnemar_p_value"]),
                f"strict Δ={phi_s['delta']} p={phi_s['mcnemar_p_value']:.4f}; "
                f"broad Δ={phi_b['delta']} p={phi_b['mcnemar_p_value']:.4f}"))

    def xstest_kappa_and_levels():
        nf4 = ["qwen_2b_base", "qwen_2b_4bit", "qwen_4b_base", "qwen_4b_4bit",
               "llama_3_2_3b_base", "llama_3_2_3b_4bit", "mistral_7b_base", "mistral_7b_4bit",
               "phi4_mini_base", "phi4_mini_4bit"]
        ks = [xjm[a]["strict_vs_regex"]["cohens_kappa"] for a in nf4]
        allm = list(xjm)
        jor = sum(xjm[a]["strict_vs_regex"]["judge_or"] for a in allm) / len(allm)
        ror = sum(xjm[a]["strict_vs_regex"]["regex_or"] for a in allm) / len(allm)
        ok = (abs(min(ks) - (-0.008)) < 0.002 and abs(max(ks) - 0.501) < 0.01
              and near(jor, 0.171, 3) and near(ror, 0.044, 3))
        return ok, f"κ [{min(ks):.3f},{max(ks):.3f}], mean judge OR {jor:.3f} vs regex {ror:.3f}"

    c.check("Result 6: κ −0.01..0.50 across 10 NF4 aliases; judge OR ~0.171 vs regex ~0.044",
            ["Cohen κ from −0.01 to 0.50 across the ten base/NF4 aliases of the primary study",
             "mean over-refusal 0.171 strict versus 0.044"],
            xstest_kappa_and_levels)

    def xstest_judge_volume():
        tot = pe = 0
        files = sorted((ROOT / "results_512").glob("*/xstest/summary.judge.xstest_api.json"))
        if not files:
            raise FileNotFoundError("no xstest judge sidecars checked out")
        for s in files:
            m = _load(s)["metrics"]
            tot += m["num_samples"]; pe += m["parse_error_count"]
        return (tot == 3750 and pe == 0, f"{tot} scored, {pe} parse errors")

    c.check("Result 6: 3,750 XSTest judge calls, zero parse failures",
            ["3,750 saved benign XSTest responses (fifteen aliases × 250 prompts)",
             "temperature 0, zero parse failures"],
            xstest_judge_volume)

    def table64_rows():
        # (strict Δ, strict lo, strict hi, strict p, broad Δ, broad lo, broad hi,
        #  broad p, regex Δ) exactly as printed in Table 6.4.
        rows = {
            "qwen_2b": (0.040, 0.000, 0.080, 0.087, 0.040, -0.004, 0.084, 0.110, -0.024),
            "qwen_4b": (-0.016, -0.052, 0.024, 0.541, -0.020, -0.060, 0.020, 0.424, -0.004),
            "llama_3_2_3b": (0.004, -0.028, 0.036, 1.000, 0.000, -0.032, 0.032, 1.000, 0.016),
            "mistral_7b": (-0.004, -0.036, 0.028, 1.000, -0.008, -0.044, 0.024, 0.815, 0.000),
            "phi4_mini": (0.016, -0.028, 0.060, 0.597, -0.004, -0.048, 0.036, 1.000, -0.048),
        }
        for pid, (sd, sl, sh, sp, bd, bl, bh, bp, rd) in rows.items():
            s, b = xjp[pid]["judge_strict"], xjp[pid]["judge_broad"]
            if not (near(sd, s["delta"]) and near(sl, s["ci_lower"]) and near(sh, s["ci_upper"])
                    and near(sp, s["mcnemar_p_value"])
                    and near(bd, b["delta"]) and near(bl, b["ci_lower"]) and near(bh, b["ci_upper"])
                    and near(bp, b["mcnemar_p_value"])):
                return False, f"{pid} judge row diverges from artifact"
            if s["significant"] or b["significant"]:
                return False, f"{pid} unexpectedly significant under the judge"
            reg = pdix[(pid, "xstest")]
            if not near(rd, reg["absolute_delta"]):
                return False, f"{pid} regex ΔOR column diverges from pairwise_deltas"
            if reg["delta_significant"] != (pid == "phi4_mini"):
                return False, f"{pid} regex significance flag wrong (★ marks phi only)"
        return True, "all five Table 6.4 rows (judge strict/broad + regex column) match artifacts"

    c.check("Result 6 / Table 6.4: five per-pair judge ΔORs + regex column == artifacts, none judge-sig",
            # ALL FIVE printed rows are pinned to the report source (previously only
            # qwen_2b + phi4_mini were, leaving 3 rows free to drift from the artifact
            # without the lock firing — the table64_rows() predicate checks the
            # artifact side for all five).
            ['["qwen_2b", "−0.024", "+0.040 [0.000, +0.080]; 0.087", "+0.040 [−0.004, +0.084]; 0.110"]',
             '["qwen_4b", "−0.004", "−0.016 [−0.052, +0.024]; 0.541", "−0.020 [−0.060, +0.020]; 0.424"]',
             '["llama_3_2_3b", "+0.016", "+0.004 [−0.028, +0.036]; 1.000", "0.000 [−0.032, +0.032]; 1.000"]',
             '["mistral_7b", "0.000", "−0.004 [−0.036, +0.028]; 1.000", "−0.008 [−0.044, +0.024]; 0.815"]',
             '["phi4_mini", "−0.048 ★", "+0.016 [−0.028, +0.060]; 0.597", "−0.004 [−0.048, +0.036]; 1.000"]',
             "Qwen3-1.7B's +0.040 (strict; McNemar p = 0.087), is an apparent increase that does not reach significance"],
            table64_rows)

    # ---------------- local-only evidence (skip when absent) ----------------
    def sample_counts():
        counts = {}
        for bench, expect in (("harmbench", 200), ("xstest", 250), ("mmlu", 300), ("arc", 1172)):
            p = ROOT / "results_512/qwen_2b_base" / bench / "summary.json"
            if not p.exists():
                raise FileNotFoundError(p)
            counts[bench] = _load(p)["num_records"]
        ok = counts == {"harmbench": 200, "xstest": 250, "mmlu": 300, "arc": 1172}
        return ok, str(counts)

    c.check("sample sizes: 200/250/300/1172", ["200", "250", "300", "1,172"], sample_counts)

    def judge_sidecar_volume():
        total = 0
        aliases = 0
        for d in sorted((ROOT / "results_512").iterdir()):
            f = d / "harmbench/scores.judge.harmbench_cls.jsonl"
            if f.exists():
                aliases += 1
                total += sum(1 for _ in f.open())
        if aliases == 0:
            raise FileNotFoundError("no judge sidecars checked out")
        return (aliases == 15 and total == 3000,
                f"{aliases} aliases, {total} classifications")

    c.check("512 judge volume: 15 aliases x 200 = 3,000 classifications",
            ["3 000 "] if "3 000 " in c.text else ["2 000 generations"],
            judge_sidecar_volume)

    # ---------------- XSTest source file ------------------------------------
    def xstest_counts():
        import csv as _csv
        p = ROOT / "data/xstest_v2_prompts.csv"
        rows = list(_csv.DictReader(p.open()))
        safe = sum(1 for r in rows if not r["type"].startswith("contrast_"))
        return (len(rows) == 450 and safe == 250,
                f"{len(rows)} rows, {safe} benign")

    c.check("XSTest v2: 450 prompts, 250 benign evaluated", ["250"], xstest_counts)

    def ieee_citations():
        i0 = c.text.find("const refs = [")
        i1 = c.text.find("];", i0)
        entries = re.findall(r'\n    "((?:[^"\\\\]|\\\\.)+)",', c.text[i0:i1])
        body = c.text[:i0] + c.text[i1:]
        seq, seen = [], set()
        for m in re.finditer(r"\[(\d{1,2})\]", body):
            n = int(m.group(1))
            if n > len(entries):
                return False, f"citation [{n}] exceeds reference count {len(entries)}"
            if n not in seen:
                seen.add(n)
                seq.append(n)
        uncited = [n for n in range(1, len(entries) + 1) if n not in seen]
        ok = not uncited and seq == list(range(1, len(entries) + 1))
        return ok, (f"{len(entries)} refs, all cited, first-use order strict"
                    if ok else f"uncited={uncited}, first-use seq starts {seq[:8]}")

    c.check(
        "IEEE citations: every ref cited, numbered strictly by first use",
        ["Kharinaev et al. [12]", "Egashira et al. [13]", "Proskurina et al. [11]",
         "McNemar's exact test [20]", "Bootstrap 95% confidence intervals [19]",
         "R. Jin, J. Du, W. Huang", "doi: 10.1109/ACCESS.2026.3703899"],
        ieee_citations,
    )

    # ================= thesis v4 mirror (same artifacts, same lock) ==========
    tt = (ROOT / "scripts/build_fyp_thesis_v4.js").read_text(encoding="utf-8")

    def tcheck(name, snippets, fn=lambda: (True, "text pinned")):
        missing = [s for s in snippets if s not in tt]
        if missing:
            c.results.append(
                ("FAIL", name, f"thesis text missing: {missing[0][:90]!r}"))
            return
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            c.results.append(("FAIL", name, f"checker error: {exc!r}"))
            return
        c.results.append(("PASS" if ok else "FAIL", name, detail))

    tcheck(
        "thesis: Table 6.2 judge rows == headline artifact",
        ['"0.000 [−0.055, +0.055]"', '"+0.040 [0.000, +0.080]"',
         '"−0.040 [−0.075, −0.010]"', '"−0.020 [−0.080, +0.040]"',
         '"+0.020 [−0.015, +0.055]"'],
        lambda: (
            near(0.0, hl512["qwen_2b"]["delta"]) and near(0.040, hl512["qwen_4b"]["delta"])
            and near(-0.040, hl512["llama_3_2_3b"]["delta"]) and near(-0.075, hl512["llama_3_2_3b"]["ci"][0])
            and near(-0.020, hl512["mistral_7b"]["delta"]) and near(0.020, hl512["phi4_mini"]["delta"]),
            "all five CI cells match headline_512_vs_128.json",
        ),
    )
    tcheck(
        "thesis: Table 6.2 capability cells == pairwise_deltas",
        ['"−0.090*"', '"−0.016*"', '"−0.032*"', '"−0.037"', '"+0.009"'],
        lambda: (
            near(-0.090, pdix[("qwen_2b", "mmlu")]["absolute_delta"])
            and near(-0.016, pdix[("qwen_4b", "arc")]["absolute_delta"])
            and near(-0.032, pdix[("llama_3_2_3b", "arc")]["absolute_delta"])
            and near(-0.037, pdix[("llama_3_2_3b", "mmlu")]["absolute_delta"])
            and near(0.009, pdix[("mistral_7b", "arc")]["absolute_delta"]),
            "capability cells match",
        ),
    )
    tcheck(
        "thesis: kappa family table == judge_agreement",
        ['"0.36 – 0.59"', '"0.25 – 0.29"', '"0.71 – 0.84"', '"0.67 – 0.77"'],
        lambda: (
            near(0.36, min(ja[k]["cohens_kappa"] for k in KAPPA if "qwen" in k), 2)
            and near(0.59, max(ja[k]["cohens_kappa"] for k in KAPPA if "qwen" in k), 2)
            and near(0.25, min(ja[k]["cohens_kappa"] for k in KAPPA if "mistral" in k), 2)
            and near(0.29, max(ja[k]["cohens_kappa"] for k in KAPPA if "mistral" in k), 2)
            and near(0.71, min(ja[k]["cohens_kappa"] for k in KAPPA if "llama" in k), 2)
            and near(0.84, max(ja[k]["cohens_kappa"] for k in KAPPA if "llama" in k), 2)
            and near(0.67, min(ja[k]["cohens_kappa"] for k in KAPPA if "phi" in k), 2)
            and near(0.77, max(ja[k]["cohens_kappa"] for k in KAPPA if "phi" in k), 2),
            "family ranges match",
        ),
    )
    tcheck(
        "thesis: judge-only/regex-only counts == judge_agreement",
        ["28 judge-only labels against 325 regex-only"],
        lambda: (
            sum(r["judge_harmful_v2_not"] for r in ja.values()) == 28
            and sum(r["v2_harmful_judge_not"] for r in ja.values()) == 325,
            "28 judge-only / 325 regex-only",
        ),
    )
    tcheck(
        "thesis: sweep table rows == precision_sweep",
        ['"0.255", "0.245", "0.255"', '"0.115", "0.125", "0.155"',
         '"0.100", "0.105", "0.060"', '"0.585", "0.565", "0.565"',
         '"0.070", "0.090", "0.090"'],
        lambda: (
            all(
                near(v, ps512[pair]["metrics"]["harmbench_asr_judge"][prec])
                for pair, vals in {
                    "qwen_2b": (0.255, 0.245, 0.255), "qwen_4b": (0.115, 0.125, 0.155),
                    "llama_3_2_3b": (0.100, 0.105, 0.060), "mistral_7b": (0.585, 0.565, 0.565),
                    "phi4_mini": (0.070, 0.090, 0.090),
                }.items()
                for prec, v in zip(("fp16", "int8", "nf4"), vals)
            ),
            "all 15 sweep cells match precision_sweep.json",
        ),
    )
    tcheck(
        "thesis: truncation + budget artefact block",
        ["60.3 percent of the 2,000 paired responses", "standardize the parameter to N = 512",
         "0.000 under the classifier (p = 1.000) at 512, with the second judge at +0.005"],
        lambda: (near(60.3, pt["pct_truncated"], 1) and pt["total"]["n"] == 2000,
                 "prefix stats match genlen_robustness.json"),
    )
    tcheck(
        "thesis: BH survivors + none-ASR",
        ["q = 0.008", "q = 0.012", "Not one HarmBench ASR contrast survives"],
        lambda: (
            mc["n_bh_significant_q05"] == 3
            and not any("asr" in s["metric"] for s in mc["bh_survivors"]),
            "3 survivors, none ASR",
        ),
    )
    tcheck(
        "thesis: INT8 vanish @512 under both judges",
        ["classifier Δ+0.005, McNemar p = 1.000; GPT-4o Δ+0.010, p = 0.688"],
        lambda: (
            near(0.005, ps512["llama_3_2_3b"]["metrics"]["harmbench_asr_judge"]["delta_int8_vs_fp16"])
            and near(1.0, _mcn(ps512, "llama_3_2_3b", "harmbench_cls"))
            and near(0.688, _mcn(ps512, "llama_3_2_3b", "api_judge")),
            "INT8 llama values match",
        ),
    )
    tcheck(
        "thesis: multiseed + OR claims",
        ["mean +0.013, range [0.000, +0.035]", "ΔOR = −0.048, CI [−0.076, −0.020]",
         "−0.024, McNemar p = 0.109"],
        lambda: (
            near(0.013, smp["qwen_2b"]["judge_delta"]["mean"])
            and near(-0.048, pdix[("phi4_mini", "xstest")]["absolute_delta"])
            and near(-0.024, pdix[("qwen_2b", "xstest")]["absolute_delta"]),
            "multiseed + OR values match",
        ),
    )
    tcheck(
        "thesis: Result 6 XSTest judge — Phi ΔOR +0.016/−0.004 n.s., scorer-dependent",
        ["+0.016 (strict, direction reversed; McNemar p = 0.597) or −0.004 (broad; p = 1.000)",
         "Cohen κ −0.01 to 0.50 across the ten base/NF4 aliases of the primary study"],
        lambda: (
            near(0.016, phi_s["delta"]) and near(-0.004, phi_b["delta"])
            and near(0.597, phi_s["mcnemar_p_value"]) and near(1.0, phi_b["mcnemar_p_value"])
            and not phi_s["significant"] and not phi_b["significant"],
            f"strict {phi_s['delta']} / broad {phi_b['delta']}",
        ),
    )
    tcheck(
        "thesis: threat model + budget scoping present",
        ["no GCG, PAIR, AutoDAN", "harmful compliance under direct requests",
         "512-token reference budget"],
    )
    def thesis_ieee():
        i0 = tt.find("const refs = [")
        i1 = tt.find("];", i0)
        nums = [int(n) for n in re.findall(r"REF\((\d+),", tt[i0:i1])]
        if nums != list(range(1, len(nums) + 1)):
            return False, f"ref list numbering broken: {nums[:6]}..."
        body = tt[:i0] + tt[i1:]
        seq, seen = [], set()
        for m in re.finditer(r"\[(\d{1,2})\]", body):
            n = int(m.group(1))
            if n > len(nums):
                return False, f"citation [{n}] exceeds reference count {len(nums)}"
            if n not in seen:
                seen.add(n)
                seq.append(n)
        uncited = [n for n in range(1, len(nums) + 1) if n not in seen]
        ok = not uncited and seq == list(range(1, len(nums) + 1))
        return ok, (f"{len(nums)} refs, all cited, first-use order strict"
                    if ok else f"uncited={uncited}, seq starts {seq[:8]}")

    tcheck("thesis: IEEE citations — every ref cited, strict first-use order",
           ["IEEE Access, vol. 14, 2026, doi: 10.1109/ACCESS.2026.3703899"], thesis_ieee)

    tcheck(
        "thesis: no unscoped 128-era content",
        [],
        lambda: (
            all("128" in line for line in tt.split("\n") if re.search(r"\+0\.055(?!\])", line) and ", +0.055" not in line)
            and "0.19 " not in tt and "two-peaked" not in tt
            and "329" not in tt and "339 automated tests" not in tt and "382 automated tests" in tt
            and "immune to this asymmetry" not in tt,
            "every +0.055 line 128-scoped; no retired kappa/counts/immune-ARC",
        ),
    )

    tcheck(
        "thesis: strict-parser bracket ARC −0.343 / MMLU −0.293 (not 'immune')",
        ["ARC falls −0.343 and MMLU −0.293"],
        lambda: (
            near(-0.343, pss_pair[("qwen_2b", "arc")]["strict_delta"]["delta"], 3)
            and near(-0.293, pss_pair[("qwen_2b", "mmlu")]["strict_delta"]["delta"], 3)
            and "immune to this asymmetry" not in tt,
            "thesis ARC/MMLU strict bracket bound to parser_strict_sensitivity.json",
        ),
    )

    # ================= interim report mirror (same artifacts, same lock) ======
    ii = (ROOT / "scripts/build_fyp_interim.js").read_text(encoding="utf-8")
    hv = _load(A512 / "human_validation.json")

    def icheck(name, snippets, fn=lambda: (True, "text pinned")):
        missing = [s for s in snippets if s not in ii]
        if missing:
            c.results.append(("FAIL", name, f"interim text missing: {missing[0][:90]!r}"))
            return
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            c.results.append(("FAIL", name, f"checker error: {exc!r}"))
            return
        c.results.append(("PASS" if ok else "FAIL", name, detail))

    icheck(
        "interim: headline ΔASR + capability + κ + counts + budget + INT8 == artifacts",
        ['"−0.040 [−0.075, −0.010]"', '"−0.090*"', '"−0.032*"',
         "classifier κ 0.59", "regex 0.11", "over-flagged 101 responses",
         "60.3 percent", "classifier Δ+0.005", "382 automated tests"],
        lambda: (
            near(-0.040, hl512["llama_3_2_3b"]["delta"])
            and near(-0.090, pdix[("qwen_2b", "mmlu")]["absolute_delta"])
            and near(-0.032, pdix[("llama_3_2_3b", "arc")]["absolute_delta"])
            and near(0.59, hv["classifier_vs_human"]["cohens_kappa"], 2)
            and near(0.11, hv["regex_vs_human"]["cohens_kappa"], 2)
            and hv["regex_vs_human"]["over_flag_vs_human"] == 101,
            "interim headline/capability/κ/counts/budget/INT8 all match the committed artifacts",
        ),
    )
    icheck(
        "interim: judge-only/regex-only counts == judge_agreement",
        ["28 judge-only labels against 325 regex-only",
         "IEEE Access, vol. 14, 2026, doi: 10.1109/ACCESS.2026.3703899"],
        lambda: (
            sum(r["judge_harmful_v2_not"] for r in ja.values()) == 28
            and sum(r["v2_harmful_judge_not"] for r in ja.values()) == 325,
            "28 judge-only / 325 regex-only; final publication metadata present",
        ),
    )
    icheck(
        "interim: labelled an Interim Report + no unscoped 128-era content",
        ["Interim Report"],
        lambda: (
            all("128" in line for line in ii.split("\n")
                if re.search(r"\+0\.055(?!\])", line) and ", +0.055" not in line)
            and "329" not in ii and "339 automated tests" not in ii and "382 automated tests" in ii,
            "interim scoped correctly (Interim Report; no retired counts/128-era leakage)",
        ),
    )

    # Each alternate is checked on its OWN text. Joining the three into one blob
    # (the 2026-07-15 first cut) let a snippet present in a single file satisfy the
    # check for all three, so an alternate could silently lose its claim: no mirror
    # carries every snippet (only the thesis alternate quotes the judge-only counts
    # AND the alias composition). Expectations are per-file for that reason.
    KHARINAEV_512 = "IEEE Access, vol. 14, 2026, doi: 10.1109/ACCESS.2026.3703899"
    mirror_expectations = [
        ("report", ROOT / "scripts/build_fyp_report_humanized.js",
         ["ten base/NF4 aliases of the primary study", KHARINAEV_512]),
        ("thesis", ROOT / "scripts/build_fyp_thesis_humanized.js",
         ["28 judge-only labels against 325 regex-only",
          "ten base/NF4 aliases of the primary study", KHARINAEV_512]),
        ("interim", ROOT / "scripts/build_fyp_interim_humanized.js",
         ["28 judge-only labels against 325 regex-only", KHARINAEV_512]),
    ]
    retired_mirror_text = [
        "29 judge-only",
        "ten NF4 aliases",
        "IEEE Access, 2025",
        "arXiv preprint arXiv:2502.15799, 2025",
        "conservative floor",
        "logging a warning otherwise",
    ]
    for label, path, snippets in mirror_expectations:
        text = path.read_text(encoding="utf-8")
        surface_check(
            f"current alternate ({label}): corrected aliases, counts, and Kharinaev metadata",
            text,
            snippets,
            lambda text=text, label=label: (
                not any(r in text for r in retired_mirror_text),
                f"{label} alternate carries the corrected wording and no retired claim",
            ),
        )

    return c


def main() -> int:
    c = run_checks()
    return c.report()


if __name__ == "__main__":
    sys.exit(main())
