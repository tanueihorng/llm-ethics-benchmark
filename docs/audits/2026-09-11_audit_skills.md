# Audit skills review — 2026-09-11

Scope: substantive review of the personal general-audit, reference-originality, iterative-verification, and FYP report-audit skills; read the four project skill entrypoints for routing context. Structural validation covered 154 immediate personal/project skill directories. This was not a behavioral audit of all 154 skills or installed plugin skills. Existing working-tree edits were preserved.

## Findings addressed

| Skill | Observed weakness | Correction |
| --- | --- | --- |
| audit-verify-explain-grade-5 | Fixed evidence hierarchy preferred tests regardless of claim; no operational coverage denominator | Match evidence to claim; independent inventory, omission/counterevidence checks, source entailment, final-artifact verification, and impact/confidence distinction |
| fyp-report-audit (project and personal copies) | Brief evidence rules did not operationalize citation support, extraction omissions, or rebuilt-artifact checks | Inventory actual document surfaces, separate citation identity from passage support, check denominators/statistical interpretation, retain redaction and scorer families, verify rebuilt output |
| iterate-until-verified | Independent approver required even when unavailable; component verification could be mistaken for integrated verification | Disclose self-review fallback unless independence is explicitly required; bind evidence to candidate version and recheck affected integrated gates |
| audit-reference-originality | Missing reference stopped independent work; rubric proposed creating original facts | Continue available checks with scoped gaps; use verified facts or labeled illustrative data; distinguish overlap from unauthorized reuse and historical presence from shipped exposure |

Changed personal files live under `/Users/tanueihorng/.codex/skills/`: the four named `SKILL.md` files plus `audit-reference-originality/references/audit-rubric.md`. Project copy: `.agents/skills/fyp-report-audit/SKILL.md`. Personal and project FYP audit entrypoints are byte-identical at validation time. Invocation metadata was preserved.

## Verification

- All revised skill entrypoints pass the bundled skill-creator structural validator. Structural validation does not prove audit quality.
- Independent subagent forward test used only a synthetic mini-report and the revised general/FYP audit skills. It identified an unsupported equivalence claim, correctly reconciled 12/40 after 10 exclusions from 50, treated the table denominator as ambiguous rather than automatically false, rejected prose-only 2/2 extraction as full coverage, and classified inaccessible citation text as an evidence gap. It disclosed that it had not run a script or inspected an original paper. This is one bounded behavioral test, not comprehensive evaluation.
- `python scripts/harness_eval.py`: 24 passed.
- `make agent-check`: 768 tests passed, one test failed. Focused rerun identified unregistered `docs/statistics_explainer.html` and `interim_progress_v2` older than its builder. The separate report-freshness gate flags `configs/claim_surfaces.yaml` and two progress builders already changed in the initial working tree. These paths were not edited in this task. No pre-edit baseline gate was run; attribution to existing state is supported by the unchanged paths, initial dirty tree, and historical record for the statistics-explainer issue, rather than a measured before/after comparison.
- `git diff --check` passed; FYP audit copies compare equal. Handoff and dashboard were regenerated.
- No report builder, scientific result, raw artifact, or scorer sidecar changed. `make report` was not run.

## Remaining structural compatibility flags

The bundled validator accepts 149 of 154 inspected directories. Five existing entries use syntax it rejects:

| Skill | Validator objection | Interpretation |
| --- | --- | --- |
| eli5 | Angle brackets in description (`/eli5 <topic>`) | Description-format flag; runtime loading not tested |
| hallmark | Top-level `version` | Metadata-schema compatibility flag |
| handoff | `argument-hint`, `disable-model-invocation` | Cross-tool metadata; preserve invocation intent pending a dedicated migration |
| html-diagram | `disable-model-invocation` | Cross-tool invocation metadata; not removed to silence validator |
| humanizer | `version`, `compatibility` | Cross-tool metadata; not proof that runtime discovery fails |

No runtime loading failure is claimed from these validator flags. Supporting scripts, demo assets, all non-audit skill bodies, and plugin-managed skills were not comprehensively reviewed or executed.
