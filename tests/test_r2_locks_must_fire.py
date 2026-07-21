"""T44 R2: must-fire proofs for the new claim locks (D42).

A gate that has never fired protects nothing. Each test perturbs exactly the
state a lock guards and asserts that specific lock FAILs, then the suite's
normal run proves the unperturbed state passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import verify_report_claims as vrc  # noqa: E402


def _status(checker, name_fragment: str) -> str:
    for status, name, _ in checker.results:
        if name_fragment in name:
            return status
    raise AssertionError(f"no check matching {name_fragment!r} ran")


def _run(perturb=None):
    c = vrc.Checker()
    if perturb:
        c.text = perturb(c.text)
    vrc.run_checks(c)
    return c


def test_dz_range_lock_fires_on_regressed_rounding() -> None:
    # The FS-7 defect class: "+1.7" silently reverting to the double-rounded "+1.8".
    c = _run(lambda t: t.replace("≈ −0.74 to +1.7", "≈ −0.74 to +1.8"))
    assert _status(c, "§6.14 margin shifts + dz range") == "FAIL"


def test_int8_lock_fires_when_verdict_loses_its_bound() -> None:
    # The FS-8 defect class: the capability verdict dropping its magnitude bound.
    c = _run(lambda t: t.replace("largest |Δ| 1.3 pp", "largest |Δ| 3 pp"))
    assert _status(c, "INT8 capability verdict") == "FAIL"


def test_instance_count_canary_fires_on_lost_occurrence() -> None:
    # The FS-6 defect class: one duplicate instance of a headline number drifting.
    c = _run(lambda t: t.replace("−0.090", "−0.091", 1))
    assert _status(c, "instance counts") == "FAIL"


def test_mechanism_lock_fires_on_artifact_mismatch(monkeypatch) -> None:
    # Artifact-side direction: the lock must also fail if the artifact stops
    # agreeing with the pinned prose (corrupt the pooled flip count in-memory).
    real_load = vrc._load

    def corrupted(path):
        data = real_load(path)
        if getattr(path, "name", "") == "refusal_margin.json":
            data["pooled"]["n_flips"] = 93
        return data

    monkeypatch.setattr(vrc, "_load", corrupted)
    c = vrc.Checker()
    vrc.run_checks(c)
    assert _status(c, "pooled flips + AUCs") == "FAIL"


def test_all_locks_quiet_on_clean_state() -> None:
    c = _run()
    for fragment in ("§6.14 margin shifts", "pooled flips", "INT8 capability verdict",
                     "Llama INT8 512", "interim surface", "thesis surface", "instance counts"):
        assert _status(c, fragment) == "PASS", fragment
