#!/usr/bin/env python3
"""Build the data payload for docs/fyp_showcase_latentspace.html ("Latent Space Cartography").

The deck renders every evaluated prompt as a real point in 3D space, so its data layer
must be derived from the primary artifacts rather than transcribed by hand. This script
reads results_512/ (the primary 512-token tree, D41) and injects a compact JSON payload
into the deck between the DATA:BEGIN / DATA:END markers.

Per-item outcomes are packed one character per prompt, base-8, bit 0 = fp16, bit 1 = int8,
bit 2 = nf4, where a set bit means "harmful" (HarmBench), "refusal" (XSTest) or "correct"
(MMLU / ARC). No prompt or response text ever enters the payload.

Usage:  python scripts/build_showcase_latentspace.py
"""
from __future__ import annotations

import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "results_512"
ANALYSIS = TREE / "analysis"
DECK = ROOT / "docs" / "fyp_showcase_latentspace.html"

PAIRS = [
    ("qwen_2b", "Qwen3-1.7B", "Qwen", "1.7 B", "Qwen/Qwen3-1.7B"),
    ("qwen_4b", "Qwen3-4B", "Qwen", "4.0 B", "Qwen/Qwen3-4B"),
    ("llama_3_2_3b", "Llama-3.2-3B", "Llama", "3.0 B", "meta-llama/Llama-3.2-3B-Instruct"),
    ("mistral_7b", "Mistral-7B", "Mistral", "7.2 B", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("phi4_mini", "Phi-4-mini", "Phi", "3.8 B", "microsoft/Phi-4-mini-instruct"),
]
ALIAS = {"fp16": "base", "int8": "8bit", "nf4": "4bit"}
PRECISIONS = ["fp16", "int8", "nf4"]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def score_fields(rec: dict) -> dict:
    """raw.jsonl stores score_fields as a dict or a repr string depending on writer."""
    sf = rec.get("score_fields")
    if isinstance(sf, dict):
        return sf
    if isinstance(sf, str):
        try:
            return ast.literal_eval(sf)
        except (ValueError, SyntaxError):
            return {}
    return {}


def truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def alias_dir(pair_id: str, precision: str) -> Path:
    return TREE / f"{pair_id}_{ALIAS[precision]}"


# --------------------------------------------------------------------------- outcomes


def harmbench_outcomes(pair_id: str, precision: str) -> dict[str, bool]:
    """Primary HarmBench scorer is the official classifier (D16)."""
    path = alias_dir(pair_id, precision) / "harmbench" / "scores.judge.harmbench_cls.jsonl"
    return {r["prompt_id"]: bool(r.get("judge_harmful")) for r in read_jsonl(path)}


def harmbench_regex_outcomes(pair_id: str, precision: str) -> dict[str, bool]:
    path = alias_dir(pair_id, precision) / "harmbench" / "scores.v2.jsonl"
    return {
        r["prompt_id"]: bool(r.get("score_fields", {}).get("harmful_compliance"))
        for r in read_jsonl(path)
    }


def xstest_outcomes(pair_id: str, precision: str) -> dict[str, bool]:
    """Pre-specified primary scorer for over-refusal is the v2 regex."""
    path = alias_dir(pair_id, precision) / "xstest" / "scores.v2.jsonl"
    return {
        r["prompt_id"]: bool(r.get("score_fields", {}).get("is_refusal"))
        for r in read_jsonl(path)
    }


def xstest_judge_outcomes(pair_id: str, precision: str) -> dict[str, bool]:
    path = alias_dir(pair_id, precision) / "xstest" / "scores.judge.xstest_api.jsonl"
    return {r["prompt_id"]: bool(r.get("judge_refusal_strict")) for r in read_jsonl(path)}


def capability_outcomes(pair_id: str, precision: str, benchmark: str) -> dict[str, bool]:
    path = alias_dir(pair_id, precision) / benchmark / "raw.jsonl"
    out = {}
    for r in read_jsonl(path):
        out[r["prompt_id"]] = truthy(score_fields(r).get("is_correct"))
    return out


# --------------------------------------------------------------------------- item axes


def harmbench_categories() -> dict[str, str]:
    cats = {}
    for r in read_jsonl(alias_dir("qwen_2b", "fp16") / "harmbench" / "raw.jsonl"):
        cats[r["prompt_id"]] = score_fields(r).get("category") or "unspecified"
    return cats


MMLU_SUBJECT_RE = re.compile(r"^(?P<subject>.+?)_\d+$")


def mmlu_subjects() -> dict[str, str]:
    subs = {}
    for r in read_jsonl(alias_dir("qwen_2b", "fp16") / "mmlu" / "raw.jsonl"):
        m = MMLU_SUBJECT_RE.match(r["prompt_id"])
        subs[r["prompt_id"]] = m.group("subject") if m else "unspecified"
    return subs


def xstest_types() -> dict[str, str]:
    """XSTest ships prompt-type metadata; safe and unsafe contrast items are distinct."""
    import csv

    types = {}
    path = ROOT / "data" / "xstest_v2_prompts.csv"
    if path.exists():
        with path.open() as fh:
            for row in csv.DictReader(fh):
                types[row["id"]] = row.get("type") or "unspecified"
    return types


# --------------------------------------------------------------------------- packing


def pack(ids: list[str], per_precision: dict[str, dict[str, bool]]) -> str:
    """One base-8 char per prompt: bit0 fp16, bit1 int8, bit2 nf4."""
    chars = []
    for pid in ids:
        code = 0
        for shift, precision in enumerate(PRECISIONS):
            if per_precision.get(precision, {}).get(pid):
                code |= 1 << shift
        chars.append(str(code))
    return "".join(chars)


def build_cloud() -> dict:
    """Every evaluated prompt, with its fp16/int8/nf4 outcome under each model pair."""
    hb_cats = harmbench_categories()
    mmlu_subs = mmlu_subjects()
    xs_types = xstest_types()

    specs = [
        ("harmbench", harmbench_outcomes, hb_cats, "classifier-judged harmful compliance"),
        ("xstest", xstest_outcomes, xs_types, "regex-scored refusal of a benign prompt"),
        ("mmlu", lambda p, q: capability_outcomes(p, q, "mmlu"), mmlu_subs, "correct answer"),
        ("arc", lambda p, q: capability_outcomes(p, q, "arc"), {}, "correct answer"),
    ]

    cloud = {}
    for name, getter, axis, set_bit_means in specs:
        # Item order is fixed by the fp16 Qwen3-1.7B run; every alias is checked against it.
        reference = getter("qwen_2b", "fp16")
        ids = sorted(reference)
        by_pair = {}
        for pair_id, *_ in PAIRS:
            per_precision = {p: getter(pair_id, p) for p in PRECISIONS}
            for precision, outcomes in per_precision.items():
                if outcomes and set(outcomes) != set(ids):
                    raise SystemExit(
                        f"prompt-id mismatch: {name}/{pair_id}/{precision} "
                        f"({len(outcomes)} items vs reference {len(ids)})"
                    )
            by_pair[pair_id] = pack(ids, per_precision)
        entry = {
            "n": len(ids),
            "setBitMeans": set_bit_means,
            # Prompt identifiers only. No prompt or response text ever enters the payload.
            "ids": ids,
            "codes": by_pair,
        }
        if axis:
            groups = sorted({axis.get(pid, "unspecified") for pid in ids})
            entry["groupNames"] = groups
            entry["groups"] = "".join(
                # single char index into groupNames; the axes here have < 36 groups
                "0123456789abcdefghijklmnopqrstuvwxyz"[groups.index(axis.get(pid, "unspecified"))]
                for pid in ids
            )
        cloud[name] = entry

    # Scorer-disagreement tracks: the same prompts under the retired/secondary scorer.
    cloud["harmbench"]["altCodes"] = {
        pair_id: pack(
            sorted(harmbench_outcomes("qwen_2b", "fp16")),
            {p: harmbench_regex_outcomes(pair_id, p) for p in PRECISIONS},
        )
        for pair_id, *_ in PAIRS
    }
    cloud["harmbench"]["altName"] = "v2 refusal regex (secondary proxy)"
    cloud["xstest"]["altCodes"] = {
        pair_id: pack(
            sorted(xstest_outcomes("qwen_2b", "fp16")),
            {p: xstest_judge_outcomes(pair_id, p) for p in PRECISIONS},
        )
        for pair_id, *_ in PAIRS
    }
    cloud["xstest"]["altName"] = "semantic judge, strict mapping (sensitivity)"
    return cloud


# --------------------------------------------------------------------------- analysis


def load(name: str):
    return json.loads((ANALYSIS / name).read_text())


def build_contrasts() -> list[dict]:
    """The 20 primary NF4 contrasts, with the CI attached from the matching artifact.

    HarmBench CIs come from judge_agreement.json because pairwise_deltas.json carries the
    v2 proxy ASR, not the classifier-primary ASR (D16).
    """
    mc = load("multiple_comparisons.json")
    sweep = load("precision_sweep.json")["per_pair"]
    judge_pairs = {r["pair_id"]: r for r in load("judge_agreement.json")["per_pair"]}
    deltas = load("pairwise_deltas.json")

    ci_by_key = {}
    for row in deltas:
        ci_by_key[(row["pair_id"], row["benchmark"])] = (
            row["delta_ci_lower"],
            row["delta_ci_upper"],
        )

    metric_to_sweep = {
        "harmbench_asr_judge": "harmbench_asr_judge",
        "xstest_over_refusal": "xstest_over_refusal",
        "mmlu_accuracy": "mmlu_accuracy",
        "arc_accuracy": "arc_accuracy",
    }
    metric_to_benchmark = {
        "xstest_over_refusal": "xstest",
        "mmlu_accuracy": "mmlu",
        "arc_accuracy": "arc",
    }

    out = []
    for c in mc["contrasts"]:
        metric = c["metric"]
        sweep_metric = sweep[c["pair_id"]]["metrics"][metric_to_sweep[metric]]
        if metric == "harmbench_asr_judge":
            jp = judge_pairs[c["pair_id"]]
            ci = (jp["judge_harm_delta_ci_lower"], jp["judge_harm_delta_ci_upper"])
        else:
            ci = ci_by_key[(c["pair_id"], metric_to_benchmark[metric])]
        out.append(
            {
                "pair": c["pair_id"],
                "metric": metric,
                "n": c["n"],
                "b": c["b"],
                "c": c["c"],
                "fp16": round(sweep_metric["fp16"], 6),
                "int8": round(sweep_metric["int8"], 6),
                "nf4": round(sweep_metric["nf4"], 6),
                "delta": round(c["delta"], 6),
                "ciLo": round(ci[0], 6),
                "ciHi": round(ci[1], 6),
                "p": c["p_value"],
                "q": c["bh_q_value"],
                "sigRaw": c["uncorrected_significant"],
                "sigBH": c["bh_significant_q05"],
            }
        )
    return out


def build_payload() -> dict:
    mc = load("multiple_comparisons.json")
    judge = load("judge_agreement.json")
    xs_judge = load("xstest_judge_agreement.json")
    human = load("human_validation.json")
    xs_human = load("xstest_human_validation.json")
    multiseed = load("sensitivity_multiseed.json")
    genlen = load("genlen_robustness.json")
    headline = load("headline_512_vs_128.json")
    equiv = load("capability_equivalence.json")
    llamaguard = load("judge_pairwise_agreement_llamaguard.json")
    cross = load("judge_pairwise_agreement.json")
    strict_family = load("multiple_comparisons_judge_strict.json")

    return {
        "meta": {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tree": "results_512",
            "sourceDocument": "docs/rewritten_fyp_interim_progress.docx",
            "builder": "scripts/build_showcase_latentspace.py",
        },
        "pairs": [
            {"id": pid, "label": label, "family": fam, "size": size, "modelId": mid}
            for pid, label, fam, size, mid in PAIRS
        ],
        "family": {
            "size": mc["family_size"],
            "alpha": mc["alpha"],
            "uncorrectedSignificant": mc["n_uncorrected_significant"],
            "bhSurvivors": mc["n_bh_significant_q05"],
            "survivors": mc["bh_survivors"],
            "power": mc["power_analysis"],
        },
        "contrasts": build_contrasts(),
        "cloud": build_cloud(),
        "precisionSweep": load("precision_sweep.json")["per_pair"],
        "scorers": {
            "classifierVsRegex": judge["per_model"],
            "classifierVsHuman": human["classifier_vs_human"],
            "regexVsHuman": human["regex_vs_human"],
            "humanHarmfulRate": human["human_harmful_rate"],
            "humanLabelled": human["n_labeled"],
            "xstestPerModel": xs_judge["per_model"],
            "xstestPerPair": xs_judge["per_pair"],
            "xstestHuman": {
                "labelCounts": xs_human["human_label_counts"],
                "strict": xs_human["strict"],
                "broad": xs_human["broad"],
                "nLabelled": xs_human["n_labeled"],
                "prereg": xs_human["prereg_version"],
            },
            "crossJudge": cross,
            "llamaGuard": llamaguard,
            "judgeStrictFamily": {
                "survivors": strict_family.get("bh_survivors", []),
                "familySize": strict_family.get("family_size"),
            },
        },
        "robustness": {
            "multiseed": multiseed["per_pair"],
            "genlen": genlen,
            "headline512vs128": headline,
            "equivalence": equiv,
        },
    }


def inject(payload: dict) -> None:
    html = DECK.read_text()
    blob = json.dumps(payload, separators=(",", ":"), sort_keys=False)
    begin, end = "/*DATA:BEGIN*/", "/*DATA:END*/"
    if begin not in html or end not in html:
        raise SystemExit(f"{DECK} is missing the DATA:BEGIN / DATA:END markers")
    head, rest = html.split(begin, 1)
    _, tail = rest.split(end, 1)
    DECK.write_text(f"{head}{begin}{blob}{end}{tail}")
    print(f"injected {len(blob):,} bytes of verified data into {DECK.relative_to(ROOT)}")


if __name__ == "__main__":
    data = build_payload()
    out = ROOT / "tmp"
    out.mkdir(exist_ok=True)
    (out / "latentspace_payload.json").write_text(json.dumps(data, indent=1))
    print(f"payload: {len(json.dumps(data, separators=(',', ':'))):,} bytes")
    for name, entry in data["cloud"].items():
        print(f"  cloud/{name}: {entry['n']} items x {len(entry['codes'])} pairs")
    if DECK.exists():
        inject(data)
