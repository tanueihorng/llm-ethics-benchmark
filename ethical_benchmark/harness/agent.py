"""Repo-native agent harness checks, status, and generated handoffs.

This module is intentionally project-specific. It converts the FYP repo's
agent rules into reusable code so Codex, Claude, or a future session can ask
the repo what is true, what changed, and which safety rails are currently
passing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import yaml


DEFAULT_POLICY_PATH = Path("configs/artifact_policy.yaml")
LOCAL_TZ = ZoneInfo("Asia/Singapore")


@dataclass(frozen=True)
class CheckResult:
    """One harness check result."""

    name: str
    status: str
    message: str
    details: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in {"pass", "warn"}


def _run(
    args: Sequence[str],
    repo_root: Path,
    *,
    check: bool = False,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=check,
        timeout=timeout,
    )


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _now_local() -> datetime:
    return datetime.now(LOCAL_TZ)


def _ascii_safe(text: str) -> str:
    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2192": "->",
        "\u2248": "approx.",
        "\u03ba": "kappa",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def load_policy(repo_root: Path, policy_path: Path | None = None) -> dict[str, Any]:
    """Loads the machine-readable artifact policy."""

    path = repo_root / (policy_path or DEFAULT_POLICY_PATH)
    if not path.exists():
        return {
            "version": 1,
            "immutable_artifacts": [],
            "immutable_manifest": "results/raw_artifact_manifest.sha256",
            "stale_text": {"scan_paths": [], "patterns": []},
            "redaction": {"scan_paths": [], "patterns": []},
            "report_worthy": {"changed_file_patterns": [], "required_changed_patterns": []},
        }
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Artifact policy must be a mapping: {path}")
    return data


def _iter_paths(repo_root: Path, patterns: Iterable[str], excludes: Iterable[str] = ()) -> list[Path]:
    output: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in repo_root.glob(pattern):
            rel = _repo_rel(path, repo_root)
            if path.is_dir() or any(fnmatch(rel, item) for item in excludes):
                continue
            if rel not in seen:
                output.append(path)
                seen.add(rel)
    return sorted(output)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_entries(repo_root: Path, policy: dict[str, Any]) -> dict[str, str]:
    paths = _iter_paths(
        repo_root,
        policy.get("immutable_artifacts", []),
        policy.get("immutable_exclude", []),
    )
    return {_repo_rel(path, repo_root): _sha256(path) for path in paths}


def write_immutable_manifest(repo_root: Path, policy: dict[str, Any] | None = None) -> Path:
    """Writes the immutable raw-artifact hash manifest."""

    policy = policy or load_policy(repo_root)
    manifest_path = repo_root / policy.get("immutable_manifest", "results/raw_artifact_manifest.sha256")
    entries = _manifest_entries(repo_root, policy)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{digest}  {path}" for path, digest in sorted(entries.items()))
    manifest_path.write_text(body + ("\n" if body else ""), encoding="utf-8")
    return manifest_path


def _read_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        digest, _, rel = stripped.partition("  ")
        if digest and rel:
            entries[rel] = digest
    return entries


def _changed_files(repo_root: Path) -> list[str]:
    files: set[str] = set()
    commands = [
        ["git", "diff", "--name-only"],
        ["git", "diff", "--cached", "--name-only"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ]
    for cmd in commands:
        result = _run(cmd, repo_root)
        if result.returncode == 0:
            files.update(line.strip() for line in result.stdout.splitlines() if line.strip())
    return sorted(files)


def _tracked_changed_files(repo_root: Path) -> list[str]:
    files: set[str] = set()
    for cmd in (["git", "diff", "--name-only"], ["git", "diff", "--cached", "--name-only"]):
        result = _run(cmd, repo_root)
        if result.returncode == 0:
            files.update(line.strip() for line in result.stdout.splitlines() if line.strip())
    return sorted(files)


def _match_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch(path, pattern) for pattern in patterns)


def _project_overview_tail(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    marker = "## Project Overview"
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[idx:]


def _line_matches(path: Path, regexes: list[re.Pattern[str]]) -> list[str]:
    hits: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return hits
    for lineno, line in enumerate(lines, start=1):
        for regex in regexes:
            if regex.search(line):
                hits.append(f"{path.as_posix()}:{lineno}: {regex.pattern}")
                break
    return hits


def check_agent_docs_sync(repo_root: Path) -> CheckResult:
    agents = repo_root / "AGENTS.md"
    claude = repo_root / "CLAUDE.md"
    if not agents.exists() or not claude.exists():
        return CheckResult("agent-doc-sync", "fail", "AGENTS.md and CLAUDE.md must both exist.")
    if _project_overview_tail(agents) != _project_overview_tail(claude):
        return CheckResult(
            "agent-doc-sync",
            "fail",
            "AGENTS.md and CLAUDE.md diverge below Project Overview.",
            ["Run diff AGENTS.md CLAUDE.md and sync the shared body."],
        )
    return CheckResult("agent-doc-sync", "pass", "AGENTS.md and CLAUDE.md shared body is synchronized.")


def check_project_log_updated(repo_root: Path) -> CheckResult:
    changed = _changed_files(repo_root)
    if not changed:
        return CheckResult("project-log-update", "pass", "Working tree has no changed files.")
    if "docs/PROJECT_LOG.md" in changed:
        return CheckResult("project-log-update", "pass", "PROJECT_LOG.md is part of the current change.")
    return CheckResult(
        "project-log-update",
        "fail",
        "Repo changes are present but docs/PROJECT_LOG.md is not changed.",
        changed[:25],
    )


def check_immutable_artifacts(repo_root: Path, policy: dict[str, Any]) -> CheckResult:
    manifest_path = repo_root / policy.get("immutable_manifest", "results/raw_artifact_manifest.sha256")
    expected = _read_manifest(manifest_path)
    current = _manifest_entries(repo_root, policy)
    if not expected:
        return CheckResult(
            "immutable-artifacts",
            "fail",
            f"Immutable manifest is missing or empty: {_repo_rel(manifest_path, repo_root)}",
            ["Run python scripts/agent_check.py --write-immutable-manifest after verifying raw artifacts."],
        )

    missing = sorted(path for path in expected if path not in current)
    changed = sorted(path for path, digest in expected.items() if current.get(path) != digest and path in current)
    extra = sorted(path for path in current if path not in expected)

    # Real immutability violations are a *mutated* artifact (present but the hash
    # differs) or an *unmanifested* new raw artifact (present but not listed).
    # These always fail. An *absent* artifact is not a mutation: the immutable
    # files are gitignored, so a fresh checkout (CI, a clean clone) simply does
    # not have them — failing on absence would make the harness CI fail by
    # design. Absence is therefore tolerated; only present-and-wrong fails.
    if changed or extra:
        details = [f"changed: {path}" for path in changed]
        details += [f"unmanifested: {path}" for path in extra]
        details += [f"missing: {path}" for path in missing]
        return CheckResult(
            "immutable-artifacts",
            "fail",
            "Immutable raw artifacts do not match the hash manifest.",
            details[:40],
        )

    verified = len(expected) - len(missing)
    if missing and verified == 0:
        # No immutable artifacts checked out at all — the expected state in CI
        # and any fresh clone. Nothing to verify, nothing mutated.
        return CheckResult(
            "immutable-artifacts",
            "pass",
            f"Immutable artifacts not checked out ({len(missing)} gitignored files absent — "
            "expected in CI / fresh clone); nothing to verify.",
        )
    if missing:
        # Some present and verified, some absent — flag the absences (e.g. a
        # local deletion) without blocking the gate.
        return CheckResult(
            "immutable-artifacts",
            "warn",
            f"Immutable artifact manifest matches {verified} present files; "
            f"{len(missing)} manifest files are absent on disk.",
            [f"absent: {path}" for path in missing[:40]],
        )
    return CheckResult(
        "immutable-artifacts",
        "pass",
        f"Immutable artifact manifest matches {len(expected)} files.",
    )


def _scan_policy_group(repo_root: Path, group: dict[str, Any], name: str) -> CheckResult:
    patterns = [re.compile(item, re.IGNORECASE) for item in group.get("patterns", [])]
    if not patterns:
        return CheckResult(name, "pass", "No scan patterns configured.")
    paths = _iter_paths(repo_root, group.get("scan_paths", []), group.get("exclude", []))
    hits: list[str] = []
    for path in paths:
        hits.extend(_line_matches(path, patterns))
    if hits:
        return CheckResult(name, "fail", f"{name} scan found {len(hits)} hit(s).", hits[:40])
    return CheckResult(name, "pass", f"{name} scan clean across {len(paths)} files.")


def check_report_freshness(repo_root: Path, policy: dict[str, Any]) -> CheckResult:
    changed = _changed_files(repo_root)
    report_policy = policy.get("report_worthy", {})
    report_worthy = [
        path for path in changed
        if _match_any(path, report_policy.get("changed_file_patterns", []))
    ]
    if not report_worthy:
        return CheckResult("report-freshness", "pass", "No report-worthy changed files detected.")
    required = report_policy.get("required_changed_patterns", ["docs/FYP_Report_*.docx"])
    changed_required = [path for path in changed if _match_any(path, required)]
    if changed_required:
        return CheckResult(
            "report-freshness",
            "pass",
            "Report-worthy files changed and report artifact/source is also changed.",
            changed_required,
        )
    return CheckResult(
        "report-freshness",
        "fail",
        "Report-worthy files changed without a changed report artifact/source.",
        report_worthy,
    )


def check_git_diff_clean(repo_root: Path) -> CheckResult:
    result = _run(["git", "diff", "--check"], repo_root)
    if result.returncode != 0:
        return CheckResult("git-diff-check", "fail", "git diff --check failed.", result.stdout.splitlines())
    return CheckResult("git-diff-check", "pass", "git diff --check is clean.")


def check_pytest(repo_root: Path, pytest_args: Sequence[str] | None = None) -> CheckResult:
    args = ["pytest", *(pytest_args or ["-q"])]
    result = _run(args, repo_root, timeout=600)
    output = (result.stdout + result.stderr).splitlines()
    if result.returncode != 0:
        return CheckResult("pytest", "fail", f"{' '.join(args)} failed.", output[-40:])
    return CheckResult("pytest", "pass", f"{' '.join(args)} passed.", output[-10:])


def run_agent_checks(
    repo_root: Path,
    *,
    include_pytest: bool = True,
    include_diff_check: bool = True,
    pytest_args: Sequence[str] | None = None,
    policy: dict[str, Any] | None = None,
) -> list[CheckResult]:
    """Runs the configured agent harness checks."""

    policy = policy or load_policy(repo_root)
    results = [
        check_agent_docs_sync(repo_root),
        check_project_log_updated(repo_root),
        check_immutable_artifacts(repo_root, policy),
        _scan_policy_group(repo_root, policy.get("redaction", {}), "redaction"),
        _scan_policy_group(repo_root, policy.get("stale_text", {}), "stale-text"),
        check_report_freshness(repo_root, policy),
    ]
    if include_diff_check:
        results.append(check_git_diff_clean(repo_root))
    if include_pytest:
        results.append(check_pytest(repo_root, pytest_args))
    return results


def _git_status(repo_root: Path) -> dict[str, Any]:
    branch_line = _run(["git", "status", "-sb"], repo_root).stdout.splitlines()
    ahead = _run(["git", "log", "--oneline", "origin/main..HEAD"], repo_root)
    return {
        "status_line": branch_line[0] if branch_line else "unknown",
        "changed_files": _changed_files(repo_root),
        "tracked_changed_files": _tracked_changed_files(repo_root),
        "ahead_commits": [line for line in ahead.stdout.splitlines() if line.strip()] if ahead.returncode == 0 else [],
    }


def _project_log_status(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "docs/PROJECT_LOG.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    last_updated = ""
    last_by = ""
    for line in text.splitlines()[:20]:
        if line.startswith("| **Last updated** |"):
            last_updated = line.split("|")[2].strip()
        if line.startswith("| **Last updated by** |"):
            last_by = line.split("|")[2].strip()
    open_actions: list[str] = []
    for line in text.splitlines():
        if line.startswith("- [ ] **T"):
            open_actions.append(re.sub(r"\s+", " ", line).strip())
    return {
        "path": "docs/PROJECT_LOG.md",
        "last_updated": last_updated,
        "last_updated_by": last_by,
        "open_actions": open_actions,
    }


def _file_info(path: Path, repo_root: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": _repo_rel(path, repo_root), "exists": False}
    stat = path.stat()
    return {
        "path": _repo_rel(path, repo_root),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


def _judge_sidecar_status(repo_root: Path) -> dict[str, Any]:
    paths = sorted(repo_root.glob("results/*/harmbench/scores.judge.harmbench_cls.jsonl"))
    summaries = sorted(repo_root.glob("results/*/harmbench/summary.judge.harmbench_cls.json"))
    return {
        "scores_count": len(paths),
        "summary_count": len(summaries),
        "scores": [_repo_rel(path, repo_root) for path in paths],
    }


def _analysis_status(repo_root: Path) -> list[dict[str, Any]]:
    names = [
        "results/analysis/judge_agreement.json",
        "results/analysis/judge_agreement.csv",
        "results/analysis/pairwise_deltas.json",
        "results/analysis/pair_interpretations.csv",
        "results/analysis/quantization_analysis_summary.json",
    ]
    return [_file_info(repo_root / name, repo_root) for name in names]


def _heading(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _section_excerpt(text: str, heading: str) -> str:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().lower() == f"## {heading.lower()}":
            body: list[str] = []
            for item in lines[idx + 1:]:
                if item.startswith("## "):
                    break
                if item.strip():
                    body.append(item.rstrip())
            return "\n".join(body).strip()
    return ""


def _task_packets(repo_root: Path) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for path in sorted((repo_root / "docs/agent_tasks").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        key = path.stem
        aliases = [key]
        first_token = key.split("-", 1)[0]
        if first_token:
            aliases.append(first_token)
        packets.append({
            "key": key,
            "aliases": sorted(set(aliases)),
            "path": _repo_rel(path, repo_root),
            "title": _heading(text, key),
            "objective": _section_excerpt(text, "Objective"),
            "content": text,
        })
    return packets


def _agent_profiles(repo_root: Path) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for path in sorted((repo_root / ".codex/agents").glob("*.toml")):
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            data = {"name": path.stem, "description": f"TOML parse error: {exc}"}
        name = str(data.get("name") or path.stem)
        profiles.append({
            "name": name,
            "aliases": sorted({name, path.stem}),
            "path": _repo_rel(path, repo_root),
            "description": str(data.get("description", "")),
            "nickname_candidates": data.get("nickname_candidates", []),
        })
    return profiles


def _select_named(items: Sequence[dict[str, Any]], query: str | None) -> dict[str, Any] | None:
    if not query:
        return None
    needle = query.lower()
    for item in items:
        aliases = [str(alias).lower() for alias in item.get("aliases", [])]
        if needle in aliases:
            return item
    for item in items:
        haystack = " ".join([str(item.get("key", "")), str(item.get("name", "")), str(item.get("title", ""))]).lower()
        if needle in haystack:
            return item
    return None


def build_agent_start_packet(
    repo_root: Path,
    *,
    task: str | None = None,
    agent: str | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds a small startup packet for a task-specific agent session."""

    policy = policy or load_policy(repo_root)
    tasks = _task_packets(repo_root)
    agents = _agent_profiles(repo_root)
    selected_task = _select_named(tasks, task)
    selected_agent = _select_named(agents, agent)
    status = build_agent_status(repo_root, policy)
    return {
        "generated_at": status["generated_at"],
        "repo_root": str(repo_root),
        "status": status,
        "selected_task": selected_task,
        "selected_agent": selected_agent,
        "available_tasks": [{k: item[k] for k in ("key", "path", "title", "objective")} for item in tasks],
        "available_agents": [{k: item[k] for k in ("name", "path", "description")} for item in agents],
    }


def format_agent_start_packet(packet: dict[str, Any]) -> str:
    """Formats a compact startup packet for a human or agent."""

    status = packet["status"]
    task = packet.get("selected_task")
    agent = packet.get("selected_agent")
    lines = [
        "Agent Start Packet",
        f"- generated_at: {packet['generated_at']}",
        f"- repo_root: {packet['repo_root']}",
        f"- git: {status['git']['status_line']}",
        f"- project_log: {status['project_log']['last_updated']} by {status['project_log']['last_updated_by']}",
        f"- suggested_next_action: {status['suggested_next_action']}",
        "",
        "Startup Steps",
        "1. Read AGENTS.md.",
        "2. Read docs/PROJECT_LOG.md.",
        "3. Run python fyp_cli.py agent-status before editing.",
    ]
    if task:
        lines.append(f"4. Use task packet `{task['path']}`.")
    if agent:
        lines.append(f"5. For context isolation, spawn/use subagent `{agent['name']}`.")
    lines.extend(["", "Context Isolation Prompt", ""])
    if agent and task:
        lines.extend([
            f"Spawn/use the `{agent['name']}` subagent for `{task['key']}`.",
            "Give it only the task packet, the live status, and the files named in the packet.",
            "Wait for findings, then let the main agent decide edits and run checks.",
        ])
    elif agent:
        lines.extend([
            f"Spawn/use the `{agent['name']}` subagent for the narrow review.",
            "Wait for findings, then let the main agent decide edits and run checks.",
        ])
    else:
        lines.append("Choose one subagent only when the task benefits from isolated review.")

    if task:
        lines.extend(["", "Selected Task", "", task["content"].strip()])
    else:
        lines.extend(["", "Available Tasks", ""])
        for item in packet["available_tasks"]:
            objective = f" - {item['objective']}" if item["objective"] else ""
            lines.append(f"- `{item['key']}` ({item['path']}): {item['title']}{objective}")

    if agent:
        lines.extend([
            "",
            "Selected Subagent",
            f"- name: `{agent['name']}`",
            f"- path: `{agent['path']}`",
            f"- description: {agent['description']}",
        ])
    else:
        lines.extend(["", "Available Subagents", ""])
        for item in packet["available_agents"]:
            lines.append(f"- `{item['name']}` ({item['path']}): {item['description']}")

    lines.extend([
        "",
        "Finish Gate",
        "```bash",
        "python fyp_cli.py agent-status",
        "make agent-check",
        "```",
    ])
    return "\n".join(lines)


def _suggest_next_action(status: dict[str, Any]) -> str:
    if status["git"]["changed_files"]:
        if "docs/PROJECT_LOG.md" in status["git"]["changed_files"]:
            return "Run make agent-check, review the diff, then commit or hand off the current change."
        return "Finish the current change, update docs/PROJECT_LOG.md, then run make agent-check."
    open_actions = status["project_log"]["open_actions"]
    for preferred in ("T21", "T22", "T15", "T1", "T3"):
        for action in open_actions:
            if f"**{preferred}." in action:
                return f"Next project action from PROJECT_LOG.md: {action}"
    if open_actions:
        return f"Next project action from PROJECT_LOG.md: {open_actions[0]}"
    return "No open actions found; refresh PROJECT_LOG.md before starting new work."


def build_agent_status(repo_root: Path, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Builds a compact live status packet for agents."""

    policy = policy or load_policy(repo_root)
    status = {
        "generated_at": _now_local().strftime("%Y-%m-%d %H:%M:%S UTC+8"),
        "repo_root": str(repo_root),
        "git": _git_status(repo_root),
        "project_log": _project_log_status(repo_root),
        "report": _file_info(
            max(
                repo_root.glob("docs/FYP_Report_*.docx"),
                default=repo_root / "docs/FYP_Report_2026-06-26_v3.docx",
            ),
            repo_root,
        ),
        "handoff": _file_info(repo_root / "docs/HANDOFF.md", repo_root),
        "dashboard": _file_info(repo_root / "docs/AGENT_DASHBOARD.md", repo_root),
        "immutable_manifest": _file_info(
            repo_root / policy.get("immutable_manifest", "results/raw_artifact_manifest.sha256"),
            repo_root,
        ),
        "judge_sidecars": _judge_sidecar_status(repo_root),
        "analysis_artifacts": _analysis_status(repo_root),
    }
    status["suggested_next_action"] = _suggest_next_action(status)
    return status


def format_agent_status(status: dict[str, Any]) -> str:
    """Formats agent status for terminal output."""

    lines = [
        "Agent Status",
        f"- generated_at: {status['generated_at']}",
        f"- repo_root: {status['repo_root']}",
        f"- git: {status['git']['status_line']}",
        f"- changed_files: {len(status['git']['changed_files'])}",
        f"- project_log: {status['project_log']['last_updated']} by {status['project_log']['last_updated_by']}",
        f"- open_actions: {len(status['project_log']['open_actions'])}",
        f"- report: {status['report'].get('modified', 'missing')} ({status['report'].get('size_bytes', 0)} bytes)",
        f"- judge_sidecars: {status['judge_sidecars']['scores_count']} scores / {status['judge_sidecars']['summary_count']} summaries",
        f"- immutable_manifest: {status['immutable_manifest'].get('modified', 'missing')}",
        f"- suggested_next_action: {status['suggested_next_action']}",
    ]
    if status["project_log"]["open_actions"]:
        lines.append("")
        lines.append("Open Actions")
        lines.extend(f"- {item}" for item in status["project_log"]["open_actions"][:8])
    if status["git"]["changed_files"]:
        lines.append("")
        lines.append("Changed Files")
        lines.extend(f"- {item}" for item in status["git"]["changed_files"][:30])
    return "\n".join(lines)


def _check_summary_lines(results: Sequence[CheckResult]) -> list[str]:
    lines = []
    for result in results:
        marker = "PASS" if result.status == "pass" else "WARN" if result.status == "warn" else "FAIL"
        lines.append(f"- {marker}: {result.name} - {result.message}")
        lines.extend(f"  - {detail}" for detail in result.details[:6])
    return lines


def render_handoff(status: dict[str, Any], results: Sequence[CheckResult] | None = None) -> str:
    """Renders docs/HANDOFF.md content."""

    lines = [
        "# Handoff",
        "",
        f"Last refreshed: {status['generated_at']} by Codex harness.",
        "",
        "## Mode",
        "",
        "Fresh-session recovery for Codex, Claude Code, or another coding agent.",
        "",
        "## Objective",
        "",
        "Start from repo truth, verify live state, and continue the FYP without relying on chat memory.",
        "",
        "## Source Of Truth",
        "",
        "- Read `/Users/tanueihorng/fyp_quant/AGENTS.md`.",
        "- Read `/Users/tanueihorng/fyp_quant/docs/PROJECT_LOG.md`.",
        "- Run `python fyp_cli.py agent-status` for live state.",
        "- Run `make agent-check` before finishing changes.",
        "- Treat this file as a bridge, not as a replacement for `docs/PROJECT_LOG.md`.",
        "",
        "## Current State",
        "",
        f"- Git: `{status['git']['status_line']}`.",
        f"- PROJECT_LOG last updated: {status['project_log']['last_updated']} by {status['project_log']['last_updated_by']}.",
        f"- Judge sidecars: {status['judge_sidecars']['scores_count']} score files and {status['judge_sidecars']['summary_count']} summary files.",
        f"- Report artifact: {status['report'].get('path')} modified {status['report'].get('modified', 'missing')}.",
        f"- Immutable manifest: {status['immutable_manifest'].get('path')} modified {status['immutable_manifest'].get('modified', 'missing')}.",
        "",
        "## Verification To Run",
        "",
        "```bash",
        "python fyp_cli.py agent-status",
        "make agent-check",
        "python scripts/generate_handoff.py",
        "python scripts/generate_agent_dashboard.py",
        "```",
        "",
        "## Harness Check Summary",
        "",
    ]
    if results is None:
        lines.append("- Not run during this handoff generation.")
    else:
        lines.extend(_check_summary_lines(results))
    lines.extend([
        "",
        "## Risks / Things To Distrust",
        "",
        "- Stale prose can survive in docs, report appendices, and historical changelog rows.",
        "- Do not trust old v2-headline claims over `results/analysis/judge_agreement.{json,csv}`.",
        "- Do not mutate `raw.jsonl` or `summary.json`; new scoring must use derived sidecars.",
        "- Do not print raw HarmBench prompt or response text in handoffs, diagnostics, or chat.",
        "",
        "## Next Actions",
        "",
        f"1. {status['suggested_next_action']}",
        "2. If editing report-worthy content, edit `scripts/build_fyp_report.js` and run `make report`.",
        "3. Regenerate this handoff and the dashboard after meaningful harness or state changes.",
        "",
        "## Privacy / Artifact Guardrails",
        "",
        "- Use IDs, counts, labels, aggregate metrics, and redacted sidecars only.",
        "- Preserve `raw.jsonl` and `summary.json` as TC1-original artifacts.",
        "- Keep `docs/PROJECT_LOG.md` as the durable decision and changelog record.",
        "",
    ])
    return "\n".join(lines)


def render_agent_dashboard(status: dict[str, Any], results: Sequence[CheckResult] | None = None) -> str:
    """Renders a compact Markdown dashboard."""

    lines = [
        "# Agent Dashboard",
        "",
        f"Generated: {status['generated_at']}",
        "",
        "## Live State",
        "",
        f"- Git: `{status['git']['status_line']}`",
        f"- Changed files: {len(status['git']['changed_files'])}",
        f"- PROJECT_LOG: {status['project_log']['last_updated']} by {status['project_log']['last_updated_by']}",
        f"- Report: {status['report'].get('modified', 'missing')} ({status['report'].get('size_bytes', 0)} bytes)",
        f"- Handoff: {status['handoff'].get('modified', 'missing')}",
        f"- Immutable manifest: {status['immutable_manifest'].get('modified', 'missing')}",
        f"- Judge sidecars: {status['judge_sidecars']['scores_count']} score files",
        "",
        "## Suggested Next Action",
        "",
        status["suggested_next_action"],
        "",
        "## Open Actions",
        "",
    ]
    open_actions = status["project_log"]["open_actions"]
    lines.extend([_ascii_safe(item) for item in open_actions[:12]] if open_actions else ["- No open actions parsed."])
    lines.extend(["", "## Analysis Artifacts", ""])
    for artifact in status["analysis_artifacts"]:
        state = artifact.get("modified", "missing")
        lines.append(f"- `{artifact['path']}`: {state}")
    lines.extend(["", "## Harness Checks", ""])
    if results is None:
        lines.append("- Not run for this dashboard.")
    else:
        lines.extend(_check_summary_lines(results))
    lines.extend(["", "## Commands", "", "```bash", "python fyp_cli.py agent-status", "make agent-check", "python scripts/generate_handoff.py", "python scripts/generate_tc1_checklist.py", "```", ""])
    return "\n".join(lines)


def render_tc1_checklist(status: dict[str, Any]) -> str:
    """Renders a TC1-safe command checklist for agents."""

    return "\n".join([
        "# TC1 Agent Checklist",
        "",
        f"Generated: {status['generated_at']}",
        "",
        "## Boundary",
        "",
        "- Human handles Wi-Fi, VPN, MFA, and any interactive password step.",
        "- Agent handles repo checks, Git sync, sbatch submission, squeue/seff monitoring, and redacted artifact validation after login.",
        "- TC1 head node rule: no user compute code. Use `sbatch` for GPU work.",
        "",
        "## Pre-Login Local Checks",
        "",
        "```bash",
        "python fyp_cli.py agent-status",
        "make agent-check",
        "git status -sb",
        "```",
        "",
        "## TC1 Head Node Sequence",
        "",
        "```bash",
        "cd /tc1home/FYP/utan001/fyp_quant/repo",
        "git pull --ff-only",
        "squeue -u utan001",
        "```",
        "",
        "## If Re-Running Or Extending Judge Validation",
        "",
        "```bash",
        "python scripts/prefetch_tc1.py --config configs/tc1.yaml --judge",
        "sbatch slurm/judge_validation.sbatch",
        "squeue -u utan001",
        "seff <JOBID>",
        "```",
        "",
        "## Back On Mac After SCP",
        "",
        "```bash",
        "python scripts/judge_agreement.py",
        "make analyze",
        "make agent-check",
        "python scripts/generate_handoff.py",
        "```",
        "",
        "## Guardrails",
        "",
        "- Do not copy raw HarmBench prompt/response text into chat, logs, or handoffs.",
        "- Preserve `raw.jsonl` and `summary.json`; write new scoring as sidecars.",
        "- If report-worthy content changes, regenerate the report and log it in PROJECT_LOG.",
        "",
    ])


def dump_status_json(status: dict[str, Any]) -> str:
    return json.dumps(status, indent=2)
