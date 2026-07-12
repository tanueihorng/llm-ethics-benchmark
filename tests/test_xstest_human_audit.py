"""Tests for scripts/xstest_human_label_audit.py (T36 / D46).

Covers the stratified draw (determinism, disagreement fraction, 8-bit + excluded
prompt_ids never sampled), the agreement math (kappa / precision-recall), the
mechanical J/R/T/X outcome read-off, the blind HTML tool (scorer columns never
rendered), and the redacted score output (no raw text).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- synthetic results tree -------------------------------------------------
def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _build_tree(root: Path, aliases: list, n_prompts: int = 20) -> None:
    """Builds a minimal results_512-shaped xstest tree.

    For each alias, prompt i: regex is_refusal = (i % 2 == 0); the judge strict
    refusal alternates on a different parity so ~half disagree. Prompt "1" and
    "102" (excluded ids) and one non-benign prompt are seeded to check filtering.
    """
    for alias in aliases:
        xdir = root / alias / "xstest"
        raw, v2, judge = [], [], []
        for i in range(n_prompts):
            pid = str(i)
            benign = not (i == 7)  # prompt 7 is non-benign -> must be dropped
            raw.append({
                "benchmark": "xstest", "prompt_id": pid,
                "prompt_text": f"benign request {pid}", "response": f"response {pid}",
                "score_fields": {"is_refusal": (i % 2 == 0), "benign": benign},
                "model_alias": alias,
            })
            v2.append({"prompt_id": pid, "score_fields": {"is_refusal": (i % 2 == 0)},
                       "model_alias": alias})
            # judge strict refusal = (i % 3 == 0): differs from regex on many ids
            strict = (i % 3 == 0)
            label = "full_refusal" if strict else "full_compliance"
            judge.append({"prompt_id": pid, "model_alias": alias, "benchmark": "xstest",
                          "judge_label": label, "judge_refusal_strict": strict,
                          "judge_refusal_broad": strict, "status": "ok"})
        # inject the excluded ids so we can assert they never appear in a sheet
        for pid in ("1", "102"):
            raw.append({"benchmark": "xstest", "prompt_id": pid,
                        "prompt_text": f"excluded {pid}", "response": f"resp {pid}",
                        "score_fields": {"is_refusal": True, "benign": True},
                        "model_alias": alias})
            v2.append({"prompt_id": pid, "score_fields": {"is_refusal": True}, "model_alias": alias})
            judge.append({"prompt_id": pid, "model_alias": alias, "benchmark": "xstest",
                          "judge_label": "full_compliance", "judge_refusal_strict": False,
                          "judge_refusal_broad": False, "status": "ok"})
        _write_jsonl(xdir / "raw.jsonl", raw)
        _write_jsonl(xdir / "scores.v2.jsonl", v2)
        _write_jsonl(xdir / "scores.judge.xstest_api.jsonl", judge)


@pytest.fixture()
def mod_on_tree(tmp_path: Path):
    """Loads the module and points its globals at a synthetic tree + tmp sheet dir."""
    mod = _load_script("xstest_human_label_audit")
    root = tmp_path / "results_512"
    # cover the two priority pairs + one other pair, plus an 8-bit alias that must
    # never be sampled (it is not in PRIORITY/OTHER, so it should be ignored anyway;
    # we still build it to prove it is not drawn).
    aliases = (mod.PRIORITY_ALIASES + mod.OTHER_ALIASES + ["qwen_2b_8bit"])
    _build_tree(root, aliases, n_prompts=30)
    mod.RESULTS = root
    mod.ANALYSIS = root / "analysis"
    mod.SHEET_DIR = tmp_path / "human_labels"
    mod.SHEET = mod.SHEET_DIR / "xstest_label_sheet.csv"
    mod.ANNOTATE_HTML = mod.SHEET_DIR / "xstest_annotate.html"
    mod.LABELS_JSON = mod.SHEET_DIR / "xstest_labels.json"
    return mod


def _read_sheet_rows(mod) -> list:
    import csv
    with mod.SHEET.open() as fh:
        return list(csv.DictReader(fh))


# --- draw / stratification --------------------------------------------------
def test_sheet_is_deterministic(mod_on_tree):
    mod = mod_on_tree
    mod.make_sheet(60, 0.6)
    first = _read_sheet_rows(mod)
    mod.make_sheet(60, 0.6)
    second = _read_sheet_rows(mod)
    assert first == second and len(first) == 60


def test_disagreement_fraction_and_exclusions(mod_on_tree):
    mod = mod_on_tree
    mod.make_sheet(60, 0.6)
    rows = _read_sheet_rows(mod)
    # excluded prompt_ids never sampled
    assert not ({r["prompt_id"] for r in rows} & mod.EXCLUDE_PROMPT_IDS)
    # no 8-bit alias sampled
    assert not any("8bit" in r["model_alias"] for r in rows)
    # non-benign prompt (id "7") never sampled
    assert "7" not in {r["prompt_id"] for r in rows}
    # ~60% disagreement rows (regex != judge strict), by construction of the draw
    n_dis = sum(1 for r in rows
                if int(r["regex_refusal"]) != (1 if r["judge_label"] == "full_refusal" else 0))
    assert n_dis == round(60 * 0.6)


def test_priority_pairs_double_weighted(mod_on_tree):
    mod = mod_on_tree
    mod.make_sheet(100, 0.6)
    rows = _read_sheet_rows(mod)
    from collections import Counter
    counts = Counter(r["model_alias"] for r in rows)
    # a priority alias should be drawn at least as often as a non-priority one
    assert counts["phi4_mini_base"] >= counts["llama_3_2_3b_base"]
    assert counts["qwen_2b_base"] >= counts["mistral_7b_base"]


# --- agreement math ---------------------------------------------------------
def test_kappa_perfect_and_degenerate(mod_on_tree):
    mod = mod_on_tree
    assert mod._kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    # all-constant -> undefined -> None (not a spurious 1.0)
    assert mod._kappa([1, 1, 1], [1, 1, 1]) is None
    # independent-ish -> below 1
    k = mod._kappa([1, 1, 0, 0], [1, 0, 1, 0])
    assert k is not None and k < 0.5


def test_prf_confusion_counts(mod_on_tree):
    mod = mod_on_tree
    # pred vs truth: 2 tp, 1 fp, 1 fn
    prf = mod._prf([1, 1, 1, 0], [1, 1, 0, 1])
    assert prf["over_flag_vs_human"] == 1
    assert prf["missed_vs_human"] == 1
    assert prf["n_flagged_refusal"] == 3


# --- outcome read-off -------------------------------------------------------
def test_outcome_letters(mod_on_tree):
    mod = mod_on_tree
    assert mod._outcome_letter(200, 0.20, 0.60)["letter"] == "J"   # judge grounded
    assert mod._outcome_letter(200, 0.60, 0.20)["letter"] == "R"   # regex grounded
    assert mod._outcome_letter(200, 0.55, 0.58)["letter"] == "T"   # tie (small gap)
    assert mod._outcome_letter(200, 0.20, 0.30)["letter"] == "T"   # both weak
    assert mod._outcome_letter(150, 0.20, 0.60)["letter"] == "X"   # too few labeled
    # a decisive gap but both weak -> T (both-weak precedence)
    assert mod._outcome_letter(200, 0.10, 0.30)["letter"] == "T"


# --- end-to-end score -------------------------------------------------------
def test_score_math_and_redaction(mod_on_tree):
    mod = mod_on_tree
    # Build a tiny sheet by hand with known labels.
    mod.SHEET_DIR.mkdir(parents=True, exist_ok=True)
    import csv
    # distinctive raw text so the redaction assertion is meaningful (not tautological)
    RAWP = "ZZPROMPT_SECRET_TEXT"
    RAWR = "ZZRESPONSE_SECRET_TEXT"
    rows = [
        # regex, judge_label, human(1/2/3)
        ("a", "10", RAWP, RAWR, "1", "full_refusal", "3"),   # judge=refuse, human=refuse, regex=comply
        ("a", "11", RAWP, RAWR, "0", "full_refusal", "3"),
        ("a", "12", RAWP, RAWR, "0", "full_compliance", "1"),
        ("a", "13", RAWP, RAWR, "1", "full_compliance", "1"),  # regex says refuse, human comply
        ("a", "14", RAWP, RAWR, "0", "partial_refusal", "2"),
    ]
    with mod.SHEET.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model_alias", "prompt_id", "prompt", "response",
                    "regex_refusal", "judge_label", "human_label"])
        w.writerows(rows)
    mod.score()
    out = json.loads((mod.ANALYSIS / "xstest_human_validation.json").read_text())
    assert out["n_labeled"] == 5
    assert out["human_label_counts"] == {"full_compliance": 2, "partial_refusal": 1, "full_refusal": 2}
    # judge strict matches human strict on 4/5 (id 14: judge partial->strict 0, human 2->strict 0: match;
    # all strict: human=[1,1,0,0,0], judge_strict=[1,1,0,0,0] -> identical)
    assert out["strict"]["judge_vs_human"]["cohens_kappa"] == pytest.approx(1.0)
    # outcome present and mechanical
    assert out["outcome"]["letter"] in {"J", "R", "T", "X"}
    # redaction: the committed artifact must contain NEITHER the raw prompt nor
    # the raw response text that was present in the local sheet.
    blob = (mod.ANALYSIS / "xstest_human_validation.json").read_text()
    assert "ZZPROMPT_SECRET_TEXT" not in blob
    assert "ZZRESPONSE_SECRET_TEXT" not in blob
    assert "prompt_text" not in blob


def test_apply_labels_merges_and_scores(mod_on_tree):
    mod = mod_on_tree
    import csv
    mod.SHEET_DIR.mkdir(parents=True, exist_ok=True)
    with mod.SHEET.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model_alias", "prompt_id", "prompt", "response",
                    "regex_refusal", "judge_label", "human_label"])
        w.writerow(["a", "10", "p", "r", "1", "full_refusal", ""])
        w.writerow(["a", "11", "p", "r", "0", "full_compliance", ""])
    labels = {"fyp_xstest_labels": {"a||10": 3, "a||11": 1}}
    mod.LABELS_JSON.write_text(json.dumps(labels))
    mod.apply_labels(mod.LABELS_JSON)
    merged = _read_sheet_rows(mod)
    assert {r["prompt_id"]: r["human_label"] for r in merged} == {"10": "3", "11": "1"}
    assert (mod.ANALYSIS / "xstest_human_validation.json").exists()


# --- blind HTML -------------------------------------------------------------
def test_html_is_blind_and_offline(mod_on_tree):
    mod = mod_on_tree
    mod.make_sheet(20, 0.6)
    mod.make_html()
    html = mod.ANNOTATE_HTML.read_text()
    # the labelling UI renders only prompt/response
    assert '$("reqTxt").textContent = r.prompt' in html
    assert '$("respTxt").textContent = r.response' in html
    # the scorer fields are never written into a visible DOM node
    assert 'textContent = r.regex' not in html
    assert 'textContent = r.judge_label' not in html
    # HARDENED (Gate-1 review): blindness means the scorer labels are ABSENT FROM
    # THE PAYLOAD, not merely unrendered. A View-Source-visible judge label is not
    # blind. The embedded DATA rows carry only prompt/response/keys/human, and the
    # CSV export drops the scorer columns too.
    assert 'judge_label' not in html
    assert '"regex"' not in html
    assert 'regex_refusal' not in html
    # spot-check: the actual judge label strings from the synthetic tree are not
    # sitting in the page source anywhere.
    assert '"full_refusal"' not in html and '"full_compliance"' not in html
    # fully self-contained (no external resource load)
    for token in ('<script src=', '<link', '@import', 'fetch(', 'XMLHttpRequest', 'cdn'):
        assert token not in html
    # budget substituted from the 512 tree
    assert "512-token budget" in html


def test_sheet_presentation_is_shuffled(mod_on_tree):
    """Gate-1 review: the disagreement stratum must NOT be a contiguous prefix, or
    screen position decodes stratum (and, via the round-robin, alias) and the blind
    protocol leaks. The §2 draw membership is unchanged (fixed seed → reproducible);
    only the display order is shuffled (pre-reg §5.1 A2)."""
    mod = mod_on_tree
    mod.make_sheet(60, 0.6)
    rows = _read_sheet_rows(mod)

    def _is_dis(r):
        return int(r["regex_refusal"]) != (1 if r["judge_label"] == "full_refusal" else 0)

    n_dis = sum(1 for r in rows if _is_dis(r))
    dis_pos = [k for k, r in enumerate(rows) if _is_dis(r)]
    # unshuffled, disagreements would be exactly positions [0, n_dis); shuffled, not.
    assert dis_pos != list(range(n_dis)), "stratum is a contiguous prefix — sheet not shuffled"
    # determinism survives the shuffle (fixed seed): same draw twice → identical sheet
    mod.make_sheet(60, 0.6)
    assert _read_sheet_rows(mod) == rows
