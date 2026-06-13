"""Tests for the judge-vs-v2 agreement statistics (Cohen's kappa)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str) -> ModuleType:
    """Imports a top-level script module by path (scripts/ is not a package)."""

    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ja = _load_script("judge_agreement")
_cohens_kappa = _ja._cohens_kappa


def test_kappa_known_value() -> None:
    """Cohen's kappa against a hand-computed value."""
    a = [True, True, False, False]
    b = [True, False, False, False]
    # po=0.75, pe=0.5 -> kappa = (0.75-0.5)/(1-0.5) = 0.5
    assert _cohens_kappa(a, b) == pytest.approx(0.5)


def test_kappa_perfect_non_degenerate() -> None:
    """Perfect agreement with a non-degenerate marginal is exactly 1.0."""
    a = [True, False, True, False]
    b = [True, False, True, False]
    assert _cohens_kappa(a, b) == pytest.approx(1.0)


def test_kappa_total_disagreement_is_zero() -> None:
    """All-True vs all-False is defined (pe=0) and equals 0.0, not None."""
    assert _cohens_kappa([True, True, True], [False, False, False]) == pytest.approx(0.0)


def test_kappa_degenerate_returns_none() -> None:
    """Both raters constant-and-identical -> chance agreement is total -> undefined.

    This is the trap a near-zero-ASR baseline (or a second judge) would hit: the
    prior code returned a spurious 1.0 ("perfect agreement") for a case carrying
    no agreement signal. The hardened version reports None (undefined).
    """
    assert _cohens_kappa([False] * 5, [False] * 5) is None
    assert _cohens_kappa([True] * 5, [True] * 5) is None


def test_kappa_empty_returns_none() -> None:
    assert _cohens_kappa([], []) is None
