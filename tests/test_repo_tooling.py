"""Regression checks for clean-clone report dispatch and package metadata."""
from __future__ import annotations

from pathlib import Path
import sys
import tomllib

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from report_tools.workflows import build_plan, execute_plan, load_workflows, missing_outputs, required_outputs


@pytest.mark.report  # needs the local-only builders (D94)
def test_registry_profile_resolves_and_includes_existing_ci_builders():
    registry = load_workflows(ROOT)
    plan = build_plan(ROOT, registry, profile="ci")
    # Independently preserve the previous CI build coverage, then add newer
    # rewrite variants. A renamed/deleted profile entry must not erase coverage.
    previous = {"build_rewritten_fyp_interim_v1.py", "build_fyp_report_v5.js",
                "build_fyp_report_v6.js", "build_fyp_thesis_v4.js",
                "build_fyp_interim.js", "build_fyp_report_humanized.js",
                "build_fyp_thesis_humanized.js", "build_fyp_interim_humanized.js"}
    assert previous <= {Path(command[1]).name for _, command in plan}
    assert "docs/rewritten_fyp_interim_v5.docx" in {path for path, _ in plan}
    expected_full_reports = {f"docs/rewritten_fyp_report_v{version}.docx" for version in (1, 2, 3, 4, 5)}
    assert expected_full_reports <= set(required_outputs(registry, "ci"))


def test_frozen_submission_cannot_be_dispatched():
    registry = load_workflows(ROOT)
    with pytest.raises(ValueError, match="Frozen submitted"):
        build_plan(ROOT, registry, target="submitted")


@pytest.mark.parametrize("mutation", ["unknown", "duplicate", "frozen"])
def test_invalid_profile_fails(tmp_path, mutation):
    registry = yaml.safe_load((ROOT / "configs/report_versions.yaml").read_text())
    paths = registry["build_profiles"]["ci"]
    paths.append({"unknown": "docs/missing.docx", "duplicate": paths[0],
                  "frozen": registry["entrypoints"]["submitted"]}[mutation])
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/report_versions.yaml").write_text(yaml.safe_dump(registry))
    with pytest.raises(ValueError):
        load_workflows(tmp_path)


def test_missing_required_output_is_not_a_skip(tmp_path):
    registry = load_workflows(ROOT)
    assert missing_outputs(tmp_path, registry, "ci") == required_outputs(registry, "ci")
    assert "docs/rewritten_fyp_report_v5.docx" in missing_outputs(tmp_path, registry, "ci")
    path = tmp_path / registry["build_profiles"]["ci"][0]
    path.parent.mkdir()
    path.write_bytes(b"fixture")
    assert path.relative_to(tmp_path).as_posix() not in missing_outputs(tmp_path, registry, "ci")


@pytest.mark.report  # needs the local-only builders (D94)
def test_missing_local_source_fails_before_any_builder_runs(tmp_path, monkeypatch):
    registry = load_workflows(ROOT)
    plan = build_plan(ROOT, registry, target="review")
    def unexpected(*args, **kwargs):
        pytest.fail("Builder ran before validating the local source")
    monkeypatch.setattr("report_tools.workflows.subprocess.run", unexpected)
    with pytest.raises(ValueError, match="preserved local input"):
        execute_plan(tmp_path, registry, plan)


@pytest.mark.report  # needs the local-only builders (D94)
def test_successful_process_without_output_is_rejected(tmp_path, monkeypatch):
    registry = load_workflows(ROOT)
    plan = build_plan(ROOT, registry, target="canonical")
    monkeypatch.setattr("report_tools.workflows.subprocess.run", lambda *a, **k: None)
    with pytest.raises(ValueError, match="without its registered output"):
        execute_plan(tmp_path, registry, plan)


def test_core_install_does_not_pull_gpu_or_report_dependencies():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = metadata["project"]["dependencies"]
    assert not any(name in dep for dep in dependencies for name in ("torch", "transformers", "datasets", "docx", "pytest"))
    assert metadata["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "ethical_benchmark.__version__"


def test_every_test_has_exactly_one_suite_marker(request):
    for item in request.session.items:
        markers = {marker.name for marker in item.iter_markers()} & {"core", "report"}
        assert len(markers) == 1, item.nodeid
