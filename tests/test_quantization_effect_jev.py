import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_quantization_effect_jev", ROOT / "scripts" / "run_quantization_effect_jev.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cohen_kappa_and_direct_inversion():
    options = ("A_better", "B_better", "tie")
    assert MODULE.cohen_kappa(list(options), list(options), options) == 1.0
    assert MODULE.cohen_kappa([], [], options) is None
    assert MODULE.direct_inversion("A_better", "B_better")
    assert not MODULE.direct_inversion("A_better", "tie")


def test_redaction_rejects_raw_text_recursively():
    MODULE.assert_redacted({"answers": {"overall_preference": {"choice": "tie"}}})
    try:
        MODULE.assert_redacted({"nested": [{"response_a": "secret"}]})
    except ValueError:
        pass
    else:
        raise AssertionError("raw response key was not rejected")


def test_effect_precedence_and_condition_decoding():
    key = {"a_condition": "nf4", "b_condition": "fp16"}
    base = {
        "overall_preference": "tie",
        "refusal_relation": "same",
        "task_success": "both",
        "semantic_relation": "materially_same",
    }
    assert MODULE.derive_effect(base, key) == "preservation"
    assert MODULE.derive_effect({**base, "overall_preference": "A_better"}, key) == "overall_improvement"
    assert MODULE.derive_effect({**base, "task_success": "B_only"}, key) == "task_success_loss"
    assert MODULE.derive_effect({**base, "refusal_relation": "A_more_refusing"}, key) == "refusal_increase"


def test_lowest_confidence_selection_is_deterministic():
    rows = [
        {
            "item_id": f"qe-{index:03d}",
            "status": "ok",
            "answers": {"overall_preference": {"confidence": index / 100}},
        }
        for index in range(1, 61)
    ]
    assert MODULE.lowest_confidence_ids(rows) == [f"qe-{index:03d}" for index in range(1, 21)]
