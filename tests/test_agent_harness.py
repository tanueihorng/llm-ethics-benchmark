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


def test_real_policy_stale_kappa_guards_fire_and_exempt(tmp_path: Path) -> None:
    """Audit P1 self-test against the REAL policy: retired 128-era κ values
    must fail the stale-text scan when cited without 128-scoping on the line,
    and stay legal when scoped (same two-tier discipline as the +0.055 guard).
    The stale κ survived every earlier gate because scripts/*.js was never
    scanned — this also pins the canonical builder into scan_paths."""
    from ethical_benchmark.harness.agent import _scan_policy_group

    repo = Path(__file__).resolve().parents[1]
    policy = load_policy(repo)
    stale = policy["stale_text"]
    assert "scripts/build_fyp_report_v5.js" in stale["scan_paths"]

    group = {"scan_paths": ["README.md"], "patterns": stale["patterns"]}

    def scan(line: str) -> str:
        (tmp_path / "README.md").write_text(line + "\n", encoding="utf-8")
        return _scan_policy_group(tmp_path, group, "stale-text").status

    for bad in (
        "Mistral judge-vs-proxy agreement: Cohen's κ = 0.19 at baseline, 0.11 under 4-bit.",
        "the second judge agrees with the classifier at κ 0.60–0.63.",
        "a sign-flipped over-count, judge-vs-proxy κ as low as 0.11.",
        # P1 audit (2026-07-14, T40): retired ARC "format-immune" framing.
        "ARC, which is immune to this asymmetry, corroborates the direction.",
        "ARC-Challenge, not subject to this format asymmetry, moves the same way.",
    ):
        assert scan(bad) == "fail", bad
    for ok in (
        "at the retired 128-token budget it had been lower still, 0.19/0.11.",
        "gpt-4o concurring at κ 0.60–0.63 (128-token era; superseded).",
        "judge-vs-proxy κ as low as 0.25 at the reference budget.",
        # Corrected bracket framing + the negated "not immune" form stay legal.
        "ARC is subject to the same asymmetry even more strongly for this pair.",
        "ARC is not immune to this asymmetry; under a strict parser it falls to −0.343.",
    ):
        assert scan(ok) == "pass", ok


def test_real_policy_redaction_covers_committed_sidecars(tmp_path: Path) -> None:
    """Audit P3 self-test against the REAL policy: a sidecar in a committed
    results path that emits a raw-text field must fail the redaction scan.
    Previously results/** was never redaction-scanned, so a leaky future
    sidecar would have been committed unblocked."""
    from ethical_benchmark.harness.agent import _scan_policy_group

    repo = Path(__file__).resolve().parents[1]
    policy = load_policy(repo)
    side = tmp_path / "results_512/qwen_test/harmbench"
    side.mkdir(parents=True)
    leaky = side / "scores.judge.harmbench_cls.jsonl"
    leaky.write_text(
        '{"prompt_id": "p1", "judge_label": "no", "response": "LEAKED RAW TEXT"}\n',
        encoding="utf-8",
    )
    assert _scan_policy_group(tmp_path, policy["redaction"], "redaction").status == "fail"

    leaky.write_text(
        '{"prompt_id": "p1", "judge_label": "no", "judge_harmful": false, "status": "ok"}\n',
        encoding="utf-8",
    )
    assert _scan_policy_group(tmp_path, policy["redaction"], "redaction").status == "pass"


def test_real_policy_redaction_covers_xstest_judge_sidecars(tmp_path: Path) -> None:
    """T35 self-test: the XSTest judge sidecar path must be redaction-scanned.
    The judge-sidecar globs were harmbench-dir-anchored, so a new
    results_512/<alias>/xstest/scores.judge.*.jsonl would have escaped both the
    redaction scan and allowed_derived_artifacts. This pins the added
    results_512/*/xstest/scores.judge.*.jsonl glob so a leaky future XSTest
    judge sidecar fails the gate."""
    from ethical_benchmark.harness.agent import _scan_policy_group

    policy = load_policy(Path(__file__).resolve().parents[1])
    side = tmp_path / "results_512/phi_test/xstest"
    side.mkdir(parents=True)
    leaky = side / "scores.judge.xstest_api.jsonl"
    leaky.write_text(
        '{"prompt_id": "p1", "judge_label": "full_refusal", "response": "LEAKED RAW TEXT"}\n',
        encoding="utf-8",
    )
    assert _scan_policy_group(tmp_path, policy["redaction"], "redaction").status == "fail"

    leaky.write_text(
        '{"prompt_id": "p1", "judge_label": "full_refusal", '
        '"judge_refusal_strict": true, "status": "ok"}\n',
        encoding="utf-8",
    )
    assert _scan_policy_group(tmp_path, policy["redaction"], "redaction").status == "pass"

    # The summary.judge glob must also be scanned (not just scores.judge).
    leaky_summary = side / "summary.judge.xstest_api.json"
    leaky_summary.write_text('{"metrics": {}, "prompt_text": "LEAKED"}\n', encoding="utf-8")
    assert _scan_policy_group(tmp_path, policy["redaction"], "redaction").status == "fail"
    leaky_summary.write_text('{"metrics": {"num_samples": 250}}\n', encoding="utf-8")
    assert _scan_policy_group(tmp_path, policy["redaction"], "redaction").status == "pass"


def test_real_policy_redaction_covers_parser_strict_sidecars(tmp_path: Path) -> None:
    """T38 self-test: the strict-parser MMLU/ARC sidecars must be redaction-scanned.
    They are a new derived-artifact class under results_512/<alias>/{mmlu,arc}/;
    without dedicated globs a leaky future sidecar emitting response text would
    have committed unblocked. Pins the added scores/summary.parser_strict globs."""
    from ethical_benchmark.harness.agent import _scan_policy_group

    policy = load_policy(Path(__file__).resolve().parents[1])
    side = tmp_path / "results_512/qwen_2b_4bit/mmlu"
    side.mkdir(parents=True)
    leaky = side / "scores.parser_strict.jsonl"
    leaky.write_text(
        '{"prompt_id": "p1", "score_fields": {"is_correct_strict": false}, '
        '"response": "LEAKED RAW TEXT"}\n', encoding="utf-8")
    assert _scan_policy_group(tmp_path, policy["redaction"], "redaction").status == "fail"
    leaky.write_text(
        '{"prompt_id": "p1", "score_fields": {"predicted_index_strict": 0, '
        '"is_correct_strict": true, "parse_tier": "leading_letter"}}\n', encoding="utf-8")
    assert _scan_policy_group(tmp_path, policy["redaction"], "redaction").status == "pass"

    # ARC path + the summary sidecar must be scanned too.
    arc_side = tmp_path / "results_512/qwen_2b_4bit/arc"
    arc_side.mkdir(parents=True)
    leaky_summary = arc_side / "summary.parser_strict.json"
    leaky_summary.write_text('{"strict_accuracy": 0.5, "prompt_text": "LEAKED"}\n', encoding="utf-8")
    assert _scan_policy_group(tmp_path, policy["redaction"], "redaction").status == "fail"
    leaky_summary.write_text('{"strict_accuracy": 0.5, "num_samples": 300}\n', encoding="utf-8")
    assert _scan_policy_group(tmp_path, policy["redaction"], "redaction").status == "pass"


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


def test_agent_harness_immutable_fails_on_partial_deletion(tmp_path: Path) -> None:
    """Audit P2 regression: partial local deletion of the immutable evidence
    tree must FAIL the gate. Full absence (CI / fresh clone) stays a pass —
    see the test above — but "some files verified present, some missing" means
    a machine that *has* the evidence checked out lost part of it, and the
    gate previously downgraded that to a non-blocking warn (it detected
    mutation but not destruction)."""
    repo = _init_harness_repo(tmp_path)
    (repo / "results/model/harmbench/raw.jsonl").unlink()  # summary.json kept

    results = run_agent_checks(repo, include_pytest=False, include_diff_check=False)

    immutable = _result_by_name(results, "immutable-artifacts")
    assert immutable.status == "fail"


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


def test_report_freshness_fails_on_source_edit_without_docx(tmp_path: Path, monkeypatch) -> None:
    """Editing the report SOURCE without regenerating the docx must FAIL the gate (audit H2).

    Previously build_fyp_report.js appeared in BOTH the trigger and the satisfier
    set, so a source-only edit self-satisfied report-freshness and the committed
    docx could go stale silently. The satisfier is now the docx alone.
    """
    from ethical_benchmark.harness import agent as harness_agent

    repo = _init_harness_repo(tmp_path)
    policy = load_policy(repo)

    # Source-only edit: report-worthy trigger fires, no docx regenerated -> FAIL.
    monkeypatch.setattr(harness_agent, "_changed_files", lambda root: ["scripts/build_fyp_report.js"])
    assert harness_agent.check_report_freshness(repo, policy).status == "fail"

    # Source edit + regenerated docx (the correct `make report` workflow) -> PASS.
    monkeypatch.setattr(
        harness_agent, "_changed_files",
        lambda root: ["scripts/build_fyp_report.js", "docs/FYP_Report_2026-06-14.docx"],
    )
    assert harness_agent.check_report_freshness(repo, policy).status == "pass"


def test_real_policy_report_source_is_not_a_freshness_satisfier() -> None:
    """The committed policy must keep the report source a TRIGGER, not a SATISFIER (audit H2).

    This guards the REAL configs/artifact_policy.yaml (the prior bug was masked
    because the test fixture used a different, already-correct policy).
    """
    repo = Path(__file__).resolve().parents[1]
    policy = load_policy(repo)
    report_worthy = policy["report_worthy"]
    assert "scripts/build_fyp_report_v5.js" in report_worthy["changed_file_patterns"]
    assert "scripts/build_fyp_report_v5.js" not in report_worthy["required_changed_patterns"]
    assert any("FYP_Report" in pat for pat in report_worthy["required_changed_patterns"])
