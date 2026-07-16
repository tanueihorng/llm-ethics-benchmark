"""Regression tests for the artifact-derived claim registry and its surfaces."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from claim_registry import build_registry, registry_is_fresh  # noqa: E402
from verify_claim_surfaces import (  # noqa: E402
    _validate_primary_claims,
    find_volatile_claims,
    load_manifest,
    run_checks,
    unregistered_surfaces,
    validate_deck_pairs,
    validate_surface,
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


# --- Falsification tests for the semantic rendered-artifact checks ---------
# Each mutation of a minimal passing document must FAIL its specific claim:
# token presence alone must never satisfy a claim whose binding is wrong, and
# an ADDED contradictory sentence must fail even when the correct sentence is
# still present (the `contradiction:` half of each claim).

GOOD_UNITS = [
    "The study's primary configuration is HarmBench's 512-token reference budget.",
    "Exactly 3 primary contrasts survive BH-FDR: Qwen3-1.7B MMLU (-0.090, "
    "q = 0.008), Llama-3.2-3B ARC (-0.032, q = 0.008), Phi-4-mini "
    "over-refusal (-0.048, q = 0.012). No HarmBench ASR contrast survives.",
    "The only individually-significant ΔASR is Llama-3.2-3B's -0.040, a decrease.",
    "The Phi-4-mini over-refusal survivor (-0.048) does not replicate under an independent judge.",
    "Human grounding: classifier κ 0.59 vs regex κ 0.11 against human labels.",
    "The Qwen3-1.7B increase of +0.055 at 128 tokens was a truncation artefact.",
]


def _claim_status(units: list[str]) -> dict[str, bool]:
    registry = build_registry(ROOT)
    return {claim: ok for claim, ok, _ in _validate_primary_claims(units, registry)}


def test_minimal_document_passes_all_semantic_claims() -> None:
    status = _claim_status(GOOD_UNITS)
    assert status and all(status.values()), status


def test_correct_value_on_wrong_model_fails() -> None:
    units = list(GOOD_UNITS)
    units[1] = (
        "Exactly 3 primary contrasts survive BH-FDR: Mistral-7B MMLU (-0.090, "
        "q = 0.008), Llama-3.2-3B ARC (-0.032, q = 0.008), Phi-4-mini "
        "over-refusal (-0.048, q = 0.012). No HarmBench ASR contrast survives."
    )
    status = _claim_status(units)
    assert not status["survivor:qwen_2b:mmlu_accuracy"]
    assert status["survivor:llama_3_2_3b:arc_accuracy"]


def test_correct_delta_with_wrong_direction_fails() -> None:
    units = list(GOOD_UNITS)
    units[2] = "The only individually-significant ΔASR is Llama-3.2-3B's -0.040, a significant regression."
    status = _claim_status(units)
    assert not status["asr-direction:llama_3_2_3b"]


def test_survivor_without_scorer_dependence_caveat_fails() -> None:
    units = [u for u in GOOD_UNITS if "does not replicate" not in u]
    status = _claim_status(units)
    assert not status["or-survivor-caveat"]


def test_retired_128_value_without_scoping_fails() -> None:
    units = list(GOOD_UNITS)
    units[5] = "Qwen3-1.7B shows a ΔASR of +0.055, a significant safety regression."
    status = _claim_status(units)
    assert not status["retired-scope:+0.055"]


def test_ci_bounds_do_not_trip_the_retired_scope_rule() -> None:
    units = GOOD_UNITS + [
        "Qwen 1.7B ΔASR 0.000 (CI [-0.055, +0.055], not significant) at the reference budget."
    ]
    status = _claim_status(units)
    assert status["retired-scope:+0.055"]


def test_asr_survival_claim_against_zero_survivor_registry_fails() -> None:
    units = list(GOOD_UNITS)
    units[1] = (
        "Exactly 3 primary contrasts survive BH-FDR: Qwen3-1.7B MMLU (-0.090, "
        "q = 0.008), Llama-3.2-3B ARC (-0.032, q = 0.008), Phi-4-mini "
        "over-refusal (-0.048, q = 0.012). The HarmBench ASR contrast survives BH-FDR."
    )
    status = _claim_status(units)
    assert not status["no-asr-survivor"]


def test_docx_older_than_builder_fails(tmp_path: Path) -> None:
    import os

    (tmp_path / "docs").mkdir()
    docx = tmp_path / "docs/report.docx"
    builder = tmp_path / "builder.js"
    docx.write_bytes(b"stale")
    builder.write_text("fresh", encoding="utf-8")
    os.utime(docx, ns=(1_000, 1_000))
    os.utime(builder, ns=(2_000, 2_000))
    surface = {"path": "docs/report.docx", "source": "builder.js", "profiles": ["fresh_from_source"]}
    registry = build_registry(ROOT)
    results = validate_surface(surface, registry, tmp_path)
    assert results == [("fresh_from_source", False, "older than builder.js")]


def test_history_appendix_is_exempt_from_retired_scoping() -> None:
    units = GOOD_UNITS + [
        "Appendix G: Document Revision History",
        "2026-06-14: the then-headline +0.055 result was rolled in.",
    ]
    status = _claim_status(units)
    assert status["retired-scope:+0.055"]


# --- Additive contradictions: the correct sentence is still present --------


def test_added_asr_survival_sentence_fails_despite_correct_sentence() -> None:
    status = _claim_status(GOOD_UNITS + ["HarmBench ASR survives BH-FDR."])
    assert status["no-asr-survivor"]  # the correct sentence is still there
    assert not status["contradiction:asr-survival"]


def test_added_hedged_survivor_count_fails() -> None:
    status = _claim_status(GOOD_UNITS + ["At least three contrasts survive BH-FDR correction."])
    assert not status["contradiction:survivor-count"]


def test_added_wrong_exact_survivor_count_fails() -> None:
    status = _claim_status(GOOD_UNITS + ["Exactly four contrasts survive the BH-FDR correction."])
    assert not status["contradiction:survivor-count"]


def test_scorer_robust_assertion_fails_both_halves() -> None:
    without_caveat = [u for u in GOOD_UNITS if "does not replicate" not in u]
    status = _claim_status(without_caveat + ["The -0.048 survivor is scorer-robust under an independent judge."])
    assert not status["or-survivor-caveat"]
    assert not status["contradiction:or-survivor-caveat"]


def test_kappa_ownership_swap_fails() -> None:
    status = _claim_status(GOOD_UNITS + ["Agreement was classifier κ 0.11 vs regex κ 0.59."])
    assert not status["contradiction:kappa-ownership"]


def test_rival_primary_budget_fails() -> None:
    status = _claim_status(GOOD_UNITS + ["The 128-token budget is the primary configuration."])
    assert not status["contradiction:primary-budget"]


def test_adjacent_nonsignificance_on_significant_delta_fails() -> None:
    status = _claim_status(GOOD_UNITS + ["Llama-3.2-3B's ΔASR of -0.040 is a non-significant decrease."])
    assert not status["contradiction:asr-direction:llama_3_2_3b"]


def test_other_pairs_nonsignificance_in_enumeration_is_fine() -> None:
    status = _claim_status(GOOD_UNITS + [
        "Under the classifier the ΔASR values are: Llama 3.2 3B -0.040 (CI [-0.075, -0.010], "
        "significant, a decrease), Mistral-7B -0.020 (CI [-0.080, +0.040], not significant)."
    ])
    assert status["contradiction:asr-direction:llama_3_2_3b"]


def test_bh_nonsurvival_of_llama_delta_is_a_true_statement() -> None:
    status = _claim_status(GOOD_UNITS + [
        "Llama-3.2-3B's -0.040 decrease is not significant after BH-FDR correction."
    ])
    assert status["contradiction:asr-direction:llama_3_2_3b"]


# --- Round-3 falsifications (Codex re-review): identity, local negation,
# --- structured variants, and their legitimate near-misses -------------------


def test_survivor_delta_attributed_to_wrong_pair_fails() -> None:
    status = _claim_status(GOOD_UNITS + ["Mistral-7B MMLU -0.090 is a BH-FDR survivor."])
    assert not status["contradiction:survivor-identity:qwen_2b"]


def test_unrelated_negation_does_not_shadow_asr_survival() -> None:
    status = _claim_status(GOOD_UNITS + ["No MMLU contrast survives, while HarmBench ASR survives BH-FDR."])
    assert not status["contradiction:asr-survival"]


def test_negation_inside_matched_span_is_respected() -> None:
    # Real report sentence shape: the negation sits between marker and verb.
    status = _claim_status(GOOD_UNITS + [
        "The harmful-compliance change is not robust — no ASR contrast survives multiple-comparison correction."
    ])
    assert status["contradiction:asr-survival"]


def test_unrelated_negation_does_not_shadow_replication_claim() -> None:
    status = _claim_status(GOOD_UNITS + [
        "The -0.048 survivor does not depend on sample size and replicates under the independent judge."
    ])
    assert not status["contradiction:or-survivor-caveat"]


def test_kappa_value_first_attribution_swap_fails() -> None:
    status = _claim_status([u for u in GOOD_UNITS if "0.59" not in u] + [
        "Agreement was 0.59 for the regex and 0.11 for the classifier."
    ])
    assert not status["contradiction:kappa-ownership"]


def test_kappa_comparatives_are_not_swaps() -> None:
    # Real thesis/report shapes: the other scorer's name legitimately follows
    # a value in comparative constructions.
    status = _claim_status(GOOD_UNITS + [
        "The classifier agrees at Cohen's κ 0.59 versus the regex's 0.11.",
        "The audit finds the classifier substantially closer to the human (κ 0.59) than the regex (κ 0.11).",
    ])
    assert status["contradiction:kappa-ownership"]


def test_any_wrong_survivor_count_number_fails() -> None:
    status = _claim_status(GOOD_UNITS + ["Exactly five contrasts survive BH-FDR."])
    assert not status["contradiction:survivor-count"]
    status = _claim_status(GOOD_UNITS + ["Exactly 17 contrasts survive BH-FDR correction."])
    assert not status["contradiction:survivor-count"]


def test_count_must_anchor_to_surviving_contrasts_noun() -> None:
    units = [u.replace("Exactly 3 primary contrasts survive BH-FDR", "The BH-FDR survivors are")
             for u in GOOD_UNITS] + [
        "Exactly 3 models were tested; only two contrasts survive BH-FDR correction."
    ]
    status = _claim_status(units)
    assert not status["survivor-count"]           # "exactly 3" binds to models, not contrasts
    assert not status["contradiction:survivor-count"]  # "only two contrasts survive" is a wrong count


def test_waiver_fails_when_claim_family_present_with_wrong_values(tmp_path: Path) -> None:
    without_kappa = [u for u in GOOD_UNITS if "0.59" not in u] + [
        "Human grounding: classifier κ 0.50 versus regex κ 0.20 against human labels."
    ]
    waived = {"human-validation-kappa": "Scoped out — but the family is present, so this must fail."}
    failures = _docx_failures(tmp_path, without_kappa, waived=waived)
    assert failures == ["docx_primary.human-validation-kappa"]


# --- Round-4 falsifications (Codex re-re-review): parenthesised identity,
# --- clause-local negation, full count vocabulary, preposition variants ------


def test_parenthesised_wrong_attribution_fails() -> None:
    status = _claim_status(GOOD_UNITS + ["Mistral-7B MMLU (-0.090) is a BH-FDR survivor."])
    assert not status["contradiction:survivor-identity:qwen_2b"]


def test_enumeration_across_closing_paren_is_not_identity_theft() -> None:
    # Real report shape: the next list item's name follows the closing paren.
    status = _claim_status(GOOD_UNITS + [
        "NF4 significantly degrades capability on MMLU (Qwen 1.7B, -0.090) and on ARC "
        "for the Llama pair (-0.032), and both deltas survive BH-FDR correction."
    ])
    assert status["contradiction:survivor-identity:qwen_2b"]


def test_negation_does_not_cross_clause_boundaries() -> None:
    status = _claim_status(GOOD_UNITS + ["No MMLU; HarmBench ASR survives BH-FDR."])
    assert not status["contradiction:asr-survival"]
    status = _claim_status(GOOD_UNITS + ["The -0.048 effect is not small; it replicates under the independent judge."])
    assert not status["contradiction:or-survivor-caveat"]


def test_negation_does_not_cross_coordinating_conjunctions() -> None:
    status = _claim_status(GOOD_UNITS + [
        "The -0.048 survivor does not depend on sample size and replicates under the independent judge."
    ])
    assert not status["contradiction:or-survivor-caveat"]


def test_full_count_vocabulary() -> None:
    for lie in ("Exactly one contrast survives BH-FDR.",
                "Exactly eleven contrasts survive BH-FDR.",
                "Exactly twenty-one contrasts survive BH-FDR."):
        status = _claim_status(GOOD_UNITS + [lie])
        assert not status["contradiction:survivor-count"], lie


def test_wrong_noun_cannot_satisfy_the_positive_count() -> None:
    units = [u.replace("Exactly 3 primary contrasts survive BH-FDR", "The BH-FDR survivors are")
             for u in GOOD_UNITS] + [
        "Exactly three models survive screening under the BH-FDR protocol."
    ]
    assert not _claim_status(units)["survivor-count"]


def test_at_least_one_benchmark_is_not_a_survivor_count() -> None:
    status = _claim_status(GOOD_UNITS + [
        "Capability reaches significance on at least one benchmark for three pairs, "
        "and robustly survives multiplicity correction."
    ])
    assert status["contradiction:survivor-count"]


def test_kappa_with_the_attribution_swap_fails() -> None:
    status = _claim_status([u for u in GOOD_UNITS if "0.59" not in u] + [
        "Against human labels, agreement was 0.59 with the regex and 0.11 with the classifier."
    ])
    assert not status["contradiction:kappa-ownership"]


def test_waiver_family_catches_single_digit_and_split_sentences(tmp_path: Path) -> None:
    base = [u for u in GOOD_UNITS if "0.59" not in u]
    waived = {"human-validation-kappa": "scoped out"}
    failures = _docx_failures(
        tmp_path, base + ["The classifier kappa was 0.5 versus regex kappa 0.2 against human labels."],
        waived=waived)
    assert failures == ["docx_primary.human-validation-kappa"]
    failures = _docx_failures(
        tmp_path, base + ["The classifier kappa was 0.5 and the regex kappa 0.2. These agreements are against human labels."],
        waived=waived)
    assert failures == ["docx_primary.human-validation-kappa"]


# --- Integration fixtures: real DOCX / LaTeX through validate_surface ------

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _make_docx(path: Path, paragraphs: list[str], rows: list[list[str]] | None = None) -> None:
    import zipfile
    from xml.sax.saxutils import escape

    body = "".join(f"<w:p><w:r><w:t>{escape(p)}</w:t></w:r></w:p>" for p in paragraphs)
    if rows:
        body += "<w:tbl>" + "".join(
            "<w:tr>" + "".join(
                f"<w:tc><w:p><w:r><w:t>{escape(cell)}</w:t></w:r></w:p></w:tc>" for cell in row
            ) + "</w:tr>" for row in rows
        ) + "</w:tbl>"
    xml = f'<?xml version="1.0"?><w:document xmlns:w="{_W_NS}"><w:body>{body}</w:body></w:document>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)


def _docx_failures(tmp_path: Path, paragraphs: list[str], rows=None, waived=None) -> list[str]:
    registry = build_registry(ROOT)
    _make_docx(tmp_path / "doc.docx", paragraphs, rows)
    surface = {"path": "doc.docx", "profiles": ["docx_primary"]}
    if waived is not None:
        surface["waived_claims"] = waived
    return [name for name, ok, _ in validate_surface(surface, registry, tmp_path) if not ok]


def test_docx_fixture_with_all_claims_passes(tmp_path: Path) -> None:
    assert _docx_failures(tmp_path, GOOD_UNITS) == []


def test_docx_fixture_additive_contradiction_fails(tmp_path: Path) -> None:
    failures = _docx_failures(tmp_path, GOOD_UNITS + ["HarmBench ASR survives BH-FDR."])
    assert failures == ["docx_primary.contradiction:asr-survival"]


def test_docx_fixture_binds_claims_within_table_rows(tmp_path: Path) -> None:
    # The Qwen survivor claim appears ONLY split across two table rows: the
    # row is the binding unit, so this must NOT count as a binding.
    prose = [u for u in GOOD_UNITS if "Exactly 3" not in u] + [
        "Exactly 3 primary contrasts survive BH-FDR: Llama-3.2-3B ARC (-0.032, q = 0.008) "
        "and Phi-4-mini over-refusal (-0.048, q = 0.012) among them. No HarmBench ASR contrast survives.",
    ]
    rows = [["Qwen3-1.7B", "MMLU accuracy", "0.643"], ["other", "row", "-0.090"]]
    failures = _docx_failures(tmp_path, prose, rows)
    assert "docx_primary.survivor:qwen_2b:mmlu_accuracy" in failures
    rows_joined = [["Qwen3-1.7B", "MMLU accuracy", "-0.090"]]
    assert _docx_failures(tmp_path, prose, rows_joined) == []


def test_latex_fixture_rows_and_comments(tmp_path: Path) -> None:
    registry = build_registry(ROOT)
    prose = "\n\n".join(u for u in GOOD_UNITS if "Exactly 3" not in u)
    count_sentence = (
        "Exactly 3 primary contrasts survive BH-FDR: Llama-3.2-3B ARC ($-0.032$, q = 0.008) "
        "and Phi-4-mini over-refusal ($-0.048$, q = 0.012) among them. No HarmBench ASR contrast survives."
    )
    split_table = (
        "\\begin{tabular}{lll}\nQwen3-1.7B & MMLU accuracy \\\\\nfiller & $-0.090$ \\\\\n\\end{tabular}"
    )
    commented_lie = "% HarmBench ASR survives BH-FDR."
    tex = "\n\n".join([prose, count_sentence, split_table, commented_lie])
    (tmp_path / "doc.tex").write_text(tex, encoding="utf-8")
    surface = {"path": "doc.tex", "profiles": ["latex_primary"]}
    failures = [name for name, ok, _ in validate_surface(surface, registry, tmp_path) if not ok]
    # Cross-row binding rejected; the commented-out lie is ignored.
    assert failures == ["latex_primary.survivor:qwen_2b:mmlu_accuracy"]

    joined_table = "\\begin{tabular}{lll}\nQwen3-1.7B & MMLU accuracy & $-0.090$ \\\\\n\\end{tabular}"
    tex = "\n\n".join([prose, count_sentence, joined_table, commented_lie])
    (tmp_path / "doc.tex").write_text(tex, encoding="utf-8")
    failures = [name for name, ok, _ in validate_surface(surface, registry, tmp_path) if not ok]
    assert failures == []


# --- Waiver semantics -------------------------------------------------------


def test_waiver_excuses_only_an_absent_claim(tmp_path: Path) -> None:
    without_kappa = [u for u in GOOD_UNITS if "0.59" not in u]
    waived = {"human-validation-kappa": "This fixture is scoped without the human-grounding claim."}
    assert _docx_failures(tmp_path, without_kappa, waived=waived) == []


def test_stale_waiver_fails_once_the_claim_is_satisfied(tmp_path: Path) -> None:
    waived = {"human-validation-kappa": "No longer true: the claim is present."}
    failures = _docx_failures(tmp_path, GOOD_UNITS, waived=waived)
    assert failures == ["docx_primary.human-validation-kappa"]


def test_waiver_on_unknown_claim_fails(tmp_path: Path) -> None:
    waived = {"no-such-claim": "Typo or removed claim."}
    failures = _docx_failures(tmp_path, GOOD_UNITS, waived=waived)
    assert failures == ["docx_primary.waiver:no-such-claim"]


def test_contradiction_rules_cannot_be_waived(tmp_path: Path) -> None:
    waived = {"contradiction:asr-survival": "Attempting to silence a contradiction."}
    failures = _docx_failures(tmp_path, GOOD_UNITS + ["HarmBench ASR survives BH-FDR."], waived=waived)
    assert "docx_primary.waiver:contradiction:asr-survival" in failures
    assert "docx_primary.contradiction:asr-survival" in failures


def test_empty_waiver_rationale_fails(tmp_path: Path) -> None:
    without_kappa = [u for u in GOOD_UNITS if "0.59" not in u]
    failures = _docx_failures(tmp_path, without_kappa, waived={"human-validation-kappa": "  "})
    assert "docx_primary.waiver:human-validation-kappa" in failures
