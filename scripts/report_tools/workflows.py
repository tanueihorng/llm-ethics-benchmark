"""Resolve report purposes and build plans without importing prose builders."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import yaml


def load_workflows(root: Path) -> dict:
    registry = yaml.safe_load((root / "configs/report_versions.yaml").read_text())
    owners = {}
    for family in registry["families"]:
        for version in family["versions"]:
            if version.get("alias") == "base":
                continue
            path = version["path"]
            if path in owners:
                raise ValueError(f"Multiple owning entries for {path}")
            owners[path] = version
    for name, path in registry.get("entrypoints", {}).items():
        if path not in owners:
            raise ValueError(f"Unknown report entrypoint {name}: {path}")
    for name, paths in registry.get("build_profiles", {}).items():
        if not paths or len(paths) != len(set(paths)):
            raise ValueError(f"Empty or duplicate paths in build profile {name}")
        for path in paths:
            if path not in owners:
                raise ValueError(f"Unknown report in profile {name}: {path}")
            if path in registry.get("frozen_outputs", []):
                raise ValueError(f"Frozen output cannot be in a build profile: {path}")
    for path, outputs in registry.get("also_produces", {}).items():
        if path not in owners or any(output not in owners for output in outputs):
            raise ValueError(f"Unknown report in also_produces: {path}")
        if set(outputs) & set(registry.get("frozen_outputs", [])):
            raise ValueError(f"Builder would overwrite a frozen output: {path}")
    registry["owners"] = owners
    return registry


def build_plan(root: Path, registry: dict, *, profile: str | None = None,
               target: str | None = None) -> list[tuple[str, list[str]]]:
    if bool(profile) == bool(target):
        raise ValueError("Choose exactly one profile or target")
    if profile:
        if profile not in registry["build_profiles"]:
            raise ValueError(f"Unknown build profile: {profile}")
        paths = registry["build_profiles"][profile]
    else:
        path = registry.get("entrypoints", {}).get(target, target)
        paths = [path]
    plan = []
    for path in paths:
        if path not in registry["owners"]:
            raise ValueError(f"Unknown report target: {path}")
        if path in registry.get("frozen_outputs", []):
            raise ValueError(f"Frozen submitted copy: {path}; use a separate review variant")
        builder = registry["owners"][path].get("builder")
        if not builder or not (root / builder).is_file():
            raise ValueError(f"Missing registered builder for {path}: {builder}")
        suffix = Path(builder).suffix
        if suffix not in {".py", ".js"}:
            raise ValueError(f"Unsupported builder: {builder}")
        plan.append((path, [sys.executable if suffix == ".py" else "node", builder]))
    return plan


def execute_plan(root: Path, registry: dict, plan: list[tuple[str, list[str]]]) -> None:
    # Check every required local input before starting any build.
    planned = {path for path, _ in plan}
    for path, _ in plan:
        source = registry.get("local_inputs", {}).get(path)
        if source and source not in planned and not (root / source).is_file():
            raise ValueError(f"{path} requires the preserved local input {source}")
    for path, command in plan:
        print(f"Building {path}", flush=True)
        subprocess.run(command, cwd=root, check=True)
        if not (root / path).is_file():
            raise ValueError(f"Builder finished without its registered output: {path}")
        for output in registry.get("also_produces", {}).get(path, []):
            if not (root / output).is_file():
                raise ValueError(f"Builder finished without its declared additional output: {output}")


def required_outputs(registry: dict, profile: str) -> list[str]:
    if profile not in registry["build_profiles"]:
        raise ValueError(f"Unknown build profile: {profile}")
    paths = list(registry["build_profiles"][profile])
    for path in registry["build_profiles"][profile]:
        paths.extend(registry.get("also_produces", {}).get(path, []))
    return list(dict.fromkeys(paths))


def missing_outputs(root: Path, registry: dict, profile: str) -> list[str]:
    return [path for path in required_outputs(registry, profile) if not (root / path).is_file()]
