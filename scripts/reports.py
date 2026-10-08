"""One entry point for the registered report builders; plans are read-only."""
from __future__ import annotations

import argparse
from pathlib import Path

from report_tools.workflows import build_plan, execute_plan, load_workflows, missing_outputs, required_outputs

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Registered build profile, e.g. ci")
    parser.add_argument("--target", help="Purpose alias or registered document path")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--build", action="store_true", help="Execute the plan (writes report outputs)")
    action.add_argument("--check", action="store_true", help="Require every profile output to exist")
    args = parser.parse_args()
    try:
        registry = load_workflows(ROOT)
        if args.check:
            if not args.profile or args.target:
                parser.error("--check requires --profile and no --target")
            missing = missing_outputs(ROOT, registry, args.profile)
            if missing:
                raise ValueError("Missing required report outputs: " + ", ".join(missing))
            print(f"All {len(required_outputs(registry, args.profile))} required outputs exist.")
        elif args.profile or args.target:
            plan = build_plan(ROOT, registry, profile=args.profile, target=args.target)
            for path, command in plan:
                print(f"{path} <- {' '.join(command)}")
            if args.build:
                execute_plan(ROOT, registry, plan)
        elif args.build:
            parser.error("--build requires --profile or --target")
        else:
            for name, path in registry["entrypoints"].items():
                state = "present" if (ROOT / path).exists() else "absent locally"
                frozen = "; frozen" if path in registry.get("frozen_outputs", []) else ""
                print(f"{name}: {path} ({state}{frozen})")
            print("Build profiles: " + ", ".join(registry["build_profiles"]))
            print("CI covers rebuildable reports; frozen-input variants and historical archives remain local checks.")
        return 0
    except ValueError as exc:
        parser.exit(1, f"Report workflow error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
