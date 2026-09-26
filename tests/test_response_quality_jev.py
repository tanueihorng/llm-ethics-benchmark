import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_response_quality_jev", ROOT / "scripts" / "run_response_quality_jev.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cohen_kappa_perfect_and_chance_adjusted():
    assert MODULE.cohen_kappa(["A", "B", "tie"], ["A", "B", "tie"]) == 1.0
    assert MODULE.cohen_kappa([], []) is None


def test_redaction_rejects_raw_text_recursively():
    MODULE.assert_redacted({"answers": {"overall_quality": {"choice": "A"}}})
    try:
        MODULE.assert_redacted({"nested": [{"response_a": "secret"}]})
    except ValueError:
        pass
    else:
        raise AssertionError("raw response key was not rejected")


def test_lowest_confidence_selection_is_deterministic():
    rows = [
        {
            "item_id": f"rq-{index:03d}",
            "status": "ok",
            "answers": {"overall_quality": {"confidence": index / 100}},
        }
        for index in range(1, 61)
    ]
    assert MODULE.lowest_confidence_ids(rows) == [f"rq-{index:03d}" for index in range(1, 21)]
