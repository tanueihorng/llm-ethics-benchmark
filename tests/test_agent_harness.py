"""Tests for the repo-native agent harness."""

from __future__ import annotations

from pathlib import Path
import subprocess

from ethical_benchmark.harness import (
    build_agent_start_packet,
    build_agent_status,
    format_agent_start_packet,
    load_policy,
    render_agent_dashboard,
    render_handoff,
    run_agent_checks,
    write_immutable_manifest,
)


SHARED_AGENT_BODY = """# AGENT FILE

Opening differs by agent.

---

## Project Overview

Shared body.
"""


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _init_harness_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "docs").mkdir()
    (repo / "docs/agent_tasks").mkdir()
    (repo / ".codex/agents").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "results/model/harmbench").mkdir(parents=True)

    (repo / "AGENTS.md").write_text(SHARED_AGENT_BODY.replace("AGENT FILE", "AGENTS.md"), encoding="utf-8")
    (repo / "CLAUDE.md").write_text(SHARED_AGENT_BODY.replace("AGENT FILE", "CLAUDE.md"), encoding="utf-8")
    (repo / "README.md").write_text("# README\n", encoding="utf-8")
    (repo / "docs/PROJECT_LOG.md").write_text(
        "\n".join([
            "# PROJECT_LOG.md",
            "",
            "| Field | Value |",
            "|---|---|",
            "| **Last updated** | 2026-06-08 (12:00 UTC+8) |",
            "| **Last updated by** | Codex |",
            "",
            "## 2. Open actions",
            "",
            "- [ ] **T99. Test open task.** Keep this task parseable.",
            "",
        ]),
        encoding="utf-8",
    )
    (repo / "docs/HANDOFF.md").write_text("# Handoff\nClean.\n", encoding="utf-8")
    (repo / "docs/agent_tasks/T99-test-task.md").write_text(
        "\n".join([
            "# Agent Task Packet: T99 Test Task",
            "",
            "## Objective",
            "",
            "Exercise agent-start packet rendering.",
            "",
        ]),
        encoding="utf-8",
    )
    (repo / ".codex/agents/fyp-test-agent.toml").write_text(
        "\n".join([
            'name = "fyp-test-agent"',
            'description = "Test subagent profile."',
            'developer_instructions = """Return findings only."""',
            "",
        ]),
        encoding="utf-8",
    )
    (repo / "configs/artifact_policy.yaml").write_text(
        "\n".join([
            "version: 1",
            "immutable_manifest: results/raw_artifact_manifest.sha256",
            "immutable_artifacts:",
            "  - results/*/*/raw.jsonl",
            "  - results/*/*/summary.json",
            "redaction:",
            "  scan_paths:",
            "    - docs/HANDOFF.md",
            "  patterns:",
            "    - 'SENSITIVE MODEL RESPONSE'",
            "    - '\"response\"\\s*:'",
            "stale_text:",
            "  scan_paths:",
            "    - docs/HANDOFF.md",
            "  patterns:",
            "    - 'RESULTS PENDING'",
            "report_worthy:",
            "  changed_file_patterns:",
            "    - scripts/build_fyp_report.js",
            "  required_changed_patterns:",
            "    - docs/FYP_Report_*.docx",
            "",
        ]),
        encoding="utf-8",
    )
    (repo / "results/model/harmbench/raw.jsonl").write_text('{"prompt_id":"p0","score_fields":{}}\n', encoding="utf-8")
    (repo / "results/model/harmbench/summary.json").write_text('{"metrics":{}}\n', encoding="utf-8")

    _run(["git", "init", "-q"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test"], repo)
    write_immutable_manifest(repo, load_policy(repo))
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-qm", "initial"], repo)
    return repo


def _result_by_name(results, name: str):
    return next(result for result in results if result.name == name)


def test_agent_harness_passes_on_clean_fixture(tmp_path: Path) -> None:
    repo = _init_harness_repo(tmp_path)

    results = run_agent_checks(repo, include_pytest=False, include_diff_check=False)

    assert all(result.ok for result in results)


def test_agent_harness_catches_agent_doc_drift(tmp_path: Path) -> None:
    repo = _init_harness_repo(tmp_path)
    (repo / "CLAUDE.md").write_text(SHARED_AGENT_BODY + "\nDrift.\n", encoding="utf-8")

    results = run_agent_checks(repo, include_pytest=False, include_diff_check=False)

    assert _result_by_name(results, "agent-doc-sync").status == "fail"


def test_agent_harness_requires_project_log_when_repo_changes(tmp_path: Path) -> None:
    repo = _init_harness_repo(tmp_path)
    (repo / "README.md").write_text("# README\nChanged.\n", encoding="utf-8")

    results = run_agent_checks(repo, include_pytest=False, include_diff_check=False)

    assert _result_by_name(results, "project-log-update").status == "fail"


def test_agent_harness_catches_stale_text(tmp_path: Path) -> None:
    repo = _init_harness_repo(tmp_path)
    (repo / "docs/HANDOFF.md").write_text("# Handoff\nRESULTS PENDING\n", encoding="utf-8")
    (repo / "docs/PROJECT_LOG.md").write_text((repo / "docs/PROJECT_LOG.md").read_text() + "\nlog update\n", encoding="utf-8")

    results = run_agent_checks(repo, include_pytest=False, include_diff_check=False)

    assert _result_by_name(results, "stale-text").status == "fail"


def test_agent_harness_catches_redaction_leak(tmp_path: Path) -> None:
    repo = _init_harness_repo(tmp_path)
    (repo / "docs/HANDOFF.md").write_text('# Handoff\n{"response": "secret"}\n', encoding="utf-8")
    (repo / "docs/PROJECT_LOG.md").write_text((repo / "docs/PROJECT_LOG.md").read_text() + "\nlog update\n", encoding="utf-8")

    results = run_agent_checks(repo, include_pytest=False, include_diff_check=False)

    assert _result_by_name(results, "redaction").status == "fail"


def test_agent_harness_catches_immutable_artifact_mutation(tmp_path: Path) -> None:
    repo = _init_harness_repo(tmp_path)
    (repo / "results/model/harmbench/raw.jsonl").write_text('{"prompt_id":"p0","changed":true}\n', encoding="utf-8")

    results = run_agent_checks(repo, include_pytest=False, include_diff_check=False)

    assert _result_by_name(results, "immutable-artifacts").status == "fail"


def test_agent_harness_immutable_tolerates_absent_artifacts(tmp_path: Path) -> None:
    """Absent immutable artifacts (gitignored / fresh CI checkout) must not fail.

    The raw artifacts are gitignored, so a fresh clone or CI checkout simply does
    not have them. Absence is not a mutation, so the check passes — it only fails
    on a present-but-changed or unmanifested artifact (see the mutation test
    above). Regression guard for the Agent Harness CI workflow, which previously
    failed by design on every push because the files it verifies cannot exist in
    CI.
    """
    repo = _init_harness_repo(tmp_path)
    (repo / "results/model/harmbench/raw.jsonl").unlink()
    (repo / "results/model/harmbench/summary.json").unlink()

    results = run_agent_checks(repo, include_pytest=False, include_diff_check=False)

    immutable = _result_by_name(results, "immutable-artifacts")
    assert immutable.ok
    assert immutable.status == "pass"


def test_agent_status_and_renderers_are_redaction_safe(tmp_path: Path) -> None:
    repo = _init_harness_repo(tmp_path)

    status = build_agent_status(repo)
    handoff = render_handoff(status)
    dashboard = render_agent_dashboard(status)

    assert status["project_log"]["open_actions"]
    assert "SENSITIVE MODEL RESPONSE" not in handoff
    assert '"response":' not in handoff
    assert "Test open task" in dashboard


def test_agent_start_packet_selects_task_and_subagent(tmp_path: Path) -> None:
    repo = _init_harness_repo(tmp_path)

    packet = build_agent_start_packet(repo, task="T99", agent="fyp-test-agent")
    rendered = format_agent_start_packet(packet)

    assert packet["selected_task"]["key"] == "T99-test-task"
    assert packet["selected_agent"]["name"] == "fyp-test-agent"
    assert "Context Isolation Prompt" in rendered
    assert "docs/agent_tasks/T99-test-task.md" in rendered
