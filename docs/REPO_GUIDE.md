# Repository guide

Start with `make help`, `make agent-status` and `make reports-list`. Current
decisions and full task details live in [PROJECT_LOG.md](PROJECT_LOG.md); search
by task ID. [AGENT_DASHBOARD.md](AGENT_DASHBOARD.md) is a generated compact view.
The project log and historical files remain in place.

This is the command and maintenance entry point. The existing
[FYP Repository Guide](FYP_REPO_GUIDE.md) retains the longer research workflow
and architectural explanation.

## Install for the work you need

Use separate virtual environments for analysis and GPU execution. Run setup
locally; TC1 setup follows [TC1_CLUSTER_RUNBOOK.md](TC1_CLUSTER_RUNBOOK.md).

| Work | Install inside its environment |
|---|---|
| Core library, configuration and repository tools | `python -m pip install -e .` |
| Historically tested analysis, tests and reports | `python -m pip install -r requirements-analysis.txt`, then `python -m pip install -e .` |
| GPU experiment environment | `python -m pip install -r requirements.txt`, then `python -m pip install -e .` |
| Report-only tools | `python -m pip install -e '.[reports]'`, then `npm ci` |
| Dashboard | `python -m pip install -e '.[dashboard]'` |
| Optional API judge | `python -m pip install -e '.[judge]'` |
| Development/package checks | `python -m pip install -e '.[dev]'` |

Run `python -m pip check` after installation. Core installation does not upgrade
the analysis environment to GPU dependency floors. Extras declare capability
dependencies; they are not historical environment locks. The existing analysis
pins are retained, including the locally verified PyArrow/fsspec versions.
The historical GPU environment was not fully captured; the GPU requirements
remain minimum versions, not a claim of exact reproduction.

On Linux CPU-only CI, install the CPU `torch==2.6.0` wheel first, as shown in
`.github/workflows/agent-harness.yml`, to avoid downloading CUDA libraries.
Node report dependencies are restored with `npm ci`. Report pagination also
needs Word/LibreOffice; Word remains the submission authority.

## Commands and verification

| Command | Purpose |
|---|---|
| `make` / `make help` | Read-only command overview |
| `make test-core` | Research-code and repository-tooling tests |
| `make test-reports` | Evidence, citation, report and presentation tests |
| `python -m pytest tests/ -q` | Both suites, without filtering |
| `make agent-check` | Full repository harness, including pytest |
| `make package-check` | Wheel contents plus isolated installed-version/import check |
| `make reports-list` | Purpose aliases and local artifact presence |
| `make reports-plan` | Preview the registered CI build plan |
| `make reports-ci` | Build and require the declared clean-clone report outputs |
| `make report` | Rebuild the canonical DOCX |
| `make report-versions` | Regenerate local version inventory/change records |

Reports and presentations are local-only (D94): the report and deck builders,
rendered documents, `docs/**/*.html` pages, `docs/viva/`, `docs/drafts/` and
`world3d/` are gitignored, so a clone has none of them. CI runs only the core
suite; the report suite runs locally through `make agent-check` or
`make test-reports`, and `make reports-ci` needs the local builders.
`FYP_REPORT_PROFILE=ci` still makes every declared output required. Ordinary
local checks require the complete registered inventory when local documents
exist. Missing artifacts outside the CI profile are not
proof of coverage: frozen-input variants, historical archives and private
LaTeX sources remain local checks. `make reports-ci` writes rebuildable report
outputs; run it in a separate checkout when preserving all local variants.

Package checks retain their staging directories under `tmp/package-checks/`.
They check every Python source module is in the wheel, then install it into an
isolated environment and compare runtime and distribution versions. Dependency
resolution is checked separately by CI's installation and `pip check`.

## Report purposes and sources

`configs/report_versions.yaml` is the report map. Its `entrypoints` identify
canonical, submitted and review copies; `build_profiles` select reproducible
build commands from the same version records. `python scripts/reports.py
--target review` previews the review command; adding `--build` executes it.
The dispatcher rejects the frozen submitted target. Existing legacy commands
remain available, so the preservation rule also applies when invoking a builder
directly.

The v3/v4 historical `selected` labels record voice/base choices, not two
current submission targets. The `submitted` alias identifies the selected
progress copy; `review` identifies the newer comparison copy. T71 still requires
the student's final-report base decision. No comparison draft is promoted by
this cleanup.

Shared orchestration lives in `scripts/report_tools/workflows.py`. Existing
prose builders and their text/citation invariance checks remain intact. Refactor
content templates only after selecting the active final-report base and proving
rendered equivalence; old overlays are part of the audit trail.

## Where things belong

| Area | Responsibility and lifecycle |
|---|---|
| `ethical_benchmark/` | Importable research library; benchmarks, models, pipeline, analysis, judges, cluster and harness modules |
| `fyp_cli.py`, root `run_*`/SLURM wrappers | Existing command compatibility; retain paths for callers |
| `scripts/reports.py`, `scripts/report_tools/workflows.py` | Current shared report dispatch |
| `scripts/build_*` | Registered report builders and presentation generators; resolve purpose before editing |
| `scripts/report_tools/` | Shared document tools plus historical editorial passes; see its README before running a pass |
| `scripts/verify_*`, `claim_registry.py`, `sync_claim_surfaces.py` | Evidence, citation and claim-surface checks |
| `scripts/*judge*`, `*analysis*`, `make_figures.py` | Judge extensions and derived analysis/figures; preserve original results |
| `configs/`, `slurm/` | Experiment configurations, policy and retained job provenance |
| `results_512/`, `results/`, sensitivity trees | Primary/comparison research evidence; never treat as caches |
| `docs/`, `docs/archive/`, `docs/audits/` | Current documentation, historical records and reproducible audits |
| `dashboard/` | Streamlit app, with its own optional dependencies |
| `world3d/` | Independent Three.js side app; own package manifest, build and tests; no research-pipeline dependency |
| `.agents/`, `.codex/`, `.claude/` | Task-specific agent skills, definitions and hooks |
| `.understand-anything/`, `.hallmark/` | Retained generated/design metadata; not authoritative research state |
| `tmp/`, caches, `node_modules/` | Local working/build material; no automatic deletion in this cleanup |

Directories and filenames were preserved to avoid breaking imports, historical
references, or uncommitted work. New shared tooling belongs in an existing
responsibility area rather than another versioned top-level script. Do not run
historical `pass*` editorial scripts against current builders: their anchors may
already have been consumed.
