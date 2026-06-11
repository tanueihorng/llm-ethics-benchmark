# Agent Task Packet: Codex Meetup Prep

## Objective

Prepare a repo-grounded explanation of how Codex is used in this FYP and why the harness is the differentiator.

## Likely Files

- `docs/CODEX_MEETUP_BRIEF.md`
- `docs/AGENT_DASHBOARD.md`
- `docs/HANDOFF.md`
- `docs/PROJECT_LOG.md`
- `README.md`

## Talking Points To Preserve

- Codex is used as a repo-native research harness, not just a code autocomplete tool.
- Memory is not magic; make the repo remember.
- The harness has orientation, constraints, actionability, verification, and recoverability.
- The strongest example is the scoring correction: raw outputs stayed immutable while v2 and judge sidecars carried revised scoring.
- The human handles identity/authentication; the agent handles repeatable post-login repo operations.

## Verification

```bash
python fyp_cli.py agent-status
make agent-check
```

## Done Criteria

- The brief names concrete repo files and commands.
- The story includes both the technical harness and the research-integrity benefit.
- The brief does not expose raw HarmBench prompt or response text.
