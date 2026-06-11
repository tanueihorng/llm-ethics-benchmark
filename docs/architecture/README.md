# fyp_quant Architecture Visuals

Presentation-grade repo architecture visuals:

## Integrated Agentic Stack

- `fyp_quant_integrated_agentic_stack.svg` - slide-ready all-in-one architecture poster.
- `fyp_quant_integrated_agentic_stack.drawio` - editable diagrams.net / draw.io source.
- `fyp_quant_integrated_agentic_stack.png` - rendered preview for quick sharing.

Use this as the main meetup diagram when you want everything in one view: folder
hierarchy, agent harness control plane, context-isolated subagents, repo-scoped
skills, harness checks, benchmark execution, and the protected evidence/report
contract.

## Top-Down Repo Hierarchy

- `fyp_quant_repo_hierarchy.svg` - slide-ready folder-first map.
- `fyp_quant_repo_hierarchy.drawio` - editable diagrams.net / draw.io source.
- `fyp_quant_repo_hierarchy.png` - rendered preview for quick sharing.

Use this when the audience needs to understand what is inside the repo and how
the folders relate to each other.

## Agentic System Flow

- `fyp_quant_agentic_architecture.svg` - slide-ready export.
- `fyp_quant_agentic_architecture.drawio` - editable diagrams.net / draw.io source.
- `fyp_quant_agentic_architecture.png` - rendered preview for quick sharing.

Use this when the audience already knows the repo shape and needs to understand
how the agent harness, benchmark pipeline, evidence, and report generation flow
together.

## Agent Harness Architecture

- `fyp_quant_agent_harness_architecture.svg` - slide-ready control-plane view.
- `fyp_quant_agent_harness_architecture.drawio` - editable diagrams.net / draw.io source.
- `fyp_quant_agent_harness_architecture.png` - rendered preview for quick sharing.

Use this when the audience specifically needs to understand the agent harness:
how a fresh agent is oriented, routed to a task packet, kept context-light,
isolated through subagents, checked by guardrails, and handed off to the next
session.

Regenerate all architecture visuals with:

```bash
make architecture-diagram
```

The system-flow diagram explains the repo in four columns:

1. Human and agent entry.
2. Repo memory and control plane.
3. Research execution pipeline.
4. Evidence and reporting.

The bottom band is the guardrail contract: immutable raw artifacts, derived
sidecars, redacted handoffs, `docs/PROJECT_LOG.md`, and `make agent-check`.

The harness architecture diagram explains the control plane in six steps:

1. Orientation kernel.
2. Task router.
3. Context isolation.
4. Guardrail contract.
5. Verification gate.
6. Recovery layer.

The integrated stack diagram combines those views into one poster:

1. Left side: folder hierarchy.
2. Middle: agent harness lifecycle and `make agent-check` gates.
3. Right side: custom subagents and repo-scoped skills.
4. Bottom: benchmark pipeline, immutable evidence, sidecars, and report output.
