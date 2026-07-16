"""Regression tests for the artifact-derived claim registry and its surfaces."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from claim_registry import build_registry, registry_is_fresh  # noqa: E402
from verify_claim_surfaces import (  # noqa: E402
    find_volatile_claims,
    load_manifest,
    run_checks,
    unregistered_surfaces,
    validate_deck_pairs,
)


def test_committed_registry_is_fresh() -> None:
    fresh, detail = registry_is_fresh(ROOT)
    assert fresh, detail


def test_registry_derives_load_bearing_survivors() -> None:
    registry = build_registry(ROOT)
    assert registry["sources"]["results_512/analysis/xstest_human_validation.json"] is None
    multiplicity = registry["claims"]["multiplicity"]
    assert multiplicity["asr_survivor_count"] == 0
    assert multiplicity["survivor_count"] == 3
    assert registry["render"]["report_table_6_1"][0] == [
        "qwen_2b",
        "HarmBench ASR (judge)",
        "0.255",
        "0.255",
        "0.000 [−0.055, +0.055]",
        "no",
    ]


def test_discovery_rejects_an_unregistered_deliverable(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "registered.html").write_text("ok", encoding="utf-8")
    (docs / "forgotten.html").write_text("drift", encoding="utf-8")
    manifest = {
        "discovery": ["docs/*.html"],
        "surfaces": [{"path": "docs/registered.html"}],
    }
    assert unregistered_surfaces(manifest, tmp_path) == {"docs/forgotten.html"}


def test_generated_deck_validation_is_semantic() -> None:
    registry = build_registry(ROOT)
    text = (ROOT / "docs/fyp_showcase.html").read_text(encoding="utf-8")
    ok, _ = validate_deck_pairs(text, registry)
    assert ok

    marker = "/* CLAIM_REGISTRY:PAIRS " + registry["registry_fingerprint"] + " */"
    start = text.index(marker)
    end = text.index("/* END_CLAIM_REGISTRY:PAIRS */", start)
    block = text[start:end]
    payload = block.split("const PAIRS = ", 1)[1].rsplit(";", 1)[0]
    pairs = json.loads(payload)
    pairs[0]["asrDelta"] = 0.123
    mutated = text[:start] + marker + "\nconst PAIRS = " + json.dumps(pairs) + ";\n" + text[end:]
    ok, detail = validate_deck_pairs(mutated, registry)
    assert not ok
    assert "qwen_2b.asrDelta" in detail


def test_volatile_claim_guard_targets_live_counts_and_git_state() -> None:
    bad = "382 automated tests\nverify-claims 81/81\nmain is at 30aad1e"
    assert len(find_volatile_claims(bad)) == 3
    assert find_volatile_claims("automated test suite; run the checks live") == []
    assert find_volatile_claims("2026-07-15 17:10: pytest reported 382 tests") == []


def test_all_registered_surfaces_satisfy_their_profiles() -> None:
    manifest = load_manifest()
    assert len(manifest["surfaces"]) >= 60
    failures = [item for item in run_checks() if item[0] == "FAIL"]
    assert failures == []
