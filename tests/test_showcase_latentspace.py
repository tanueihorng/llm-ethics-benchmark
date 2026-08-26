"""Drift gate for docs/fyp_showcase_latentspace.html ("Latent Space Cartography").

The deck renders every number at runtime from a JSON payload that
scripts/build_showcase_latentspace.py injects between the DATA markers, so the usual
"does this string appear in the file" deck profile has nothing to match. These tests are
the replacement, and they are stronger: they re-derive the deck's own per-item codes from
the tracked scorer sidecars, and then re-compute every rate and discordant count from
those codes and check them against the analysis artifacts.

Coverage note (the denominator is derived twice on purpose, D61): HarmBench and XSTest
codes are checked item-by-item against the committed sidecars; MMLU and ARC come from
raw.jsonl, which is local-only, so those two axes are checked by recomputing their rates
from the deck payload and comparing with precision_sweep.json. Every one of the 20
primary contrasts is covered by at least one of the two routes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "docs" / "fyp_showcase_latentspace.html"
TREE = ROOT / "results_512"
ANALYSIS = TREE / "analysis"

PRECISIONS = ["fp16", "int8", "nf4"]
ALIAS = {"fp16": "base", "int8": "8bit", "nf4": "4bit"}
PAIR_IDS = ["qwen_2b", "qwen_4b", "llama_3_2_3b", "mistral_7b", "phi4_mini"]
METRIC_OF = {
    "harmbench": "harmbench_asr_judge",
    "xstest": "xstest_over_refusal",
    "mmlu": "mmlu_accuracy",
    "arc": "arc_accuracy",
}

pytestmark = pytest.mark.skipif(not DECK.exists(), reason="deck not present")


@pytest.fixture(scope="module")
def payload() -> dict:
    html = DECK.read_text()
    match = re.search(r"/\*DATA:BEGIN\*/(.*?)/\*DATA:END\*/", html, re.S)
    assert match, "deck is missing its DATA:BEGIN / DATA:END markers"
    blob = match.group(1).strip()
    assert blob and blob != "null", "deck data payload was never injected — run the builder"
    return json.loads(blob)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rate(codes: str, precision: str) -> float:
    bit = 1 << PRECISIONS.index(precision)
    return sum(1 for ch in codes if int(ch) & bit) / len(codes)


# --------------------------------------------------------------------------- shape


def test_payload_shape(payload):
    assert [p["id"] for p in payload["pairs"]] == PAIR_IDS
    assert set(payload["cloud"]) == set(METRIC_OF)
    for name, entry in payload["cloud"].items():
        assert len(entry["ids"]) == entry["n"], f"{name}: id count does not match n"
        assert len(set(entry["ids"])) == entry["n"], f"{name}: duplicate prompt ids"
        assert set(entry["codes"]) == set(PAIR_IDS), f"{name}: missing a pair"
        for pair_id, codes in entry["codes"].items():
            assert len(codes) == entry["n"], f"{name}/{pair_id}: code length != n"
            assert set(codes) <= set("01234567"), f"{name}/{pair_id}: code out of range"


def test_payload_carries_no_prompt_or_response_text(payload):
    """Redaction contract: identifiers, counts and booleans only, never benchmark text.

    The check is on key names rather than a substring sweep, because legitimate aggregate
    keys such as ``response_length`` (character counts, no text) would trip a blunt scan.
    """
    banned_keys = {"prompt_text", "response_text", "prompt", "response", "completion",
                   "generation", "score_fields", "text", "output"}

    def walk(node, path="payload"):
        if isinstance(node, dict):
            for key, value in node.items():
                assert key not in banned_keys, f"{path}.{key} would carry benchmark text"
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node[:50]):
                walk(value, f"{path}[{i}]")

    walk(payload)
    # Prompt ids are short tokens, not sentences.
    for name, entry in payload["cloud"].items():
        longest = max(entry["ids"], key=len)
        assert len(longest) <= 64, f"{name}: id {longest!r} looks like text, not an identifier"
        assert " " not in longest, f"{name}: id {longest!r} contains whitespace"


# ------------------------------------------------- route 1: item-level, vs sidecars


@pytest.mark.parametrize("pair_id", PAIR_IDS)
def test_harmbench_codes_match_the_committed_judge_sidecars(payload, pair_id):
    entry = payload["cloud"]["harmbench"]
    for shift, precision in enumerate(PRECISIONS):
        path = TREE / f"{pair_id}_{ALIAS[precision]}" / "harmbench" / "scores.judge.harmbench_cls.jsonl"
        if not path.exists():
            pytest.skip(f"{path.relative_to(ROOT)} not available")
        truth = {r["prompt_id"]: bool(r["judge_harmful"]) for r in _read_jsonl(path)}
        assert set(truth) == set(entry["ids"]), f"{pair_id}/{precision}: prompt-id set differs"
        for i, pid in enumerate(entry["ids"]):
            got = bool(int(entry["codes"][pair_id][i]) & (1 << shift))
            assert got is truth[pid], f"{pair_id}/{precision}/{pid}: deck says {got}, sidecar says {truth[pid]}"


@pytest.mark.parametrize("pair_id", PAIR_IDS)
def test_xstest_codes_match_the_committed_v2_sidecars(payload, pair_id):
    entry = payload["cloud"]["xstest"]
    for shift, precision in enumerate(PRECISIONS):
        path = TREE / f"{pair_id}_{ALIAS[precision]}" / "xstest" / "scores.v2.jsonl"
        if not path.exists():
            pytest.skip(f"{path.relative_to(ROOT)} not available")
        truth = {r["prompt_id"]: bool(r["score_fields"]["is_refusal"]) for r in _read_jsonl(path)}
        assert set(truth) == set(entry["ids"]), f"{pair_id}/{precision}: prompt-id set differs"
        for i, pid in enumerate(entry["ids"]):
            got = bool(int(entry["codes"][pair_id][i]) & (1 << shift))
            assert got is truth[pid], f"{pair_id}/{precision}/{pid}: deck says {got}, sidecar says {truth[pid]}"


# ------------------------------------ route 2: aggregate, recomputed from the payload


@pytest.mark.parametrize("bench", sorted(METRIC_OF))
@pytest.mark.parametrize("pair_id", PAIR_IDS)
def test_rates_recomputed_from_codes_match_the_precision_sweep(payload, pair_id, bench):
    """The deck's own points must add up to the published rate at all three precisions."""
    sweep = json.loads((ANALYSIS / "precision_sweep.json").read_text())
    published = sweep["per_pair"][pair_id]["metrics"][METRIC_OF[bench]]
    codes = payload["cloud"][bench]["codes"][pair_id]
    for precision in PRECISIONS:
        assert _rate(codes, precision) == pytest.approx(published[precision], abs=5e-4), (
            f"{pair_id}/{bench}/{precision}: deck points give {_rate(codes, precision):.4f}, "
            f"precision_sweep.json says {published[precision]:.4f}"
        )


def test_harmbench_discordant_counts_recomputed_from_codes(payload):
    """b/c are what McNemar is computed on; derive them a second way from the points."""
    mc = json.loads((ANALYSIS / "multiple_comparisons.json").read_text())
    contrasts = {c["pair_id"]: c for c in mc["contrasts"] if c["metric"] == "harmbench_asr_judge"}
    for pair_id, contrast in contrasts.items():
        codes = [int(ch) for ch in payload["cloud"]["harmbench"]["codes"][pair_id]]
        b = sum(1 for c in codes if not c & 1 and c & 4)   # safe under fp16, harmful under NF4
        c_ = sum(1 for c in codes if c & 1 and not c & 4)  # the reverse
        assert (b, c_) == (contrast["b"], contrast["c"]), (
            f"{pair_id}: deck points give b/c = {b}/{c_}, "
            f"multiple_comparisons.json says {contrast['b']}/{contrast['c']}"
        )


def test_contrast_table_matches_the_corrected_family(payload):
    mc = json.loads((ANALYSIS / "multiple_comparisons.json").read_text())
    assert len(payload["contrasts"]) == mc["family_size"] == 20
    published = {(c["pair_id"], c["metric"]): c for c in mc["contrasts"]}
    for row in payload["contrasts"]:
        want = published[(row["pair"], row["metric"])]
        assert row["delta"] == pytest.approx(want["delta"], abs=1e-9)
        assert row["p"] == pytest.approx(want["p_value"], abs=1e-9)
        assert row["q"] == pytest.approx(want["bh_q_value"], abs=1e-9)
        assert row["sigBH"] is want["bh_significant_q05"]
    survivors = {(s["pair_id"], s["metric"]) for s in payload["family"]["survivors"]}
    assert survivors == {(r["pair"], r["metric"]) for r in payload["contrasts"] if r["sigBH"]}
    assert payload["family"]["bhSurvivors"] == mc["n_bh_significant_q05"] == len(survivors)


def test_no_harmbench_contrast_survives_correction(payload):
    """The deck's headline claim, asserted against the deck's own data."""
    asr = [r for r in payload["contrasts"] if r["metric"] == "harmbench_asr_judge"]
    assert len(asr) == 5
    assert not any(r["sigBH"] for r in asr), "a HarmBench contrast now survives BH-FDR — the deck's copy is stale"
