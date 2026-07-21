# T44 Phase 6 — fresh-clone reproducibility results

Session S6, 2026-07-21. One cold agent cloned the repo into the session scratchpad
(`phase6_clone/`), followed the documented setup, and ran the full local pipeline. TC1/GPU
reproducibility explicitly out of scope. Orchestrator re-verified the two P1-class legs by
direct grep (13 builders carry the absolute path; claim_surfaces.yaml:63-65 lack the
local_optional marker that :146 proves exists).

## Headline: the RESULTS are portable; the TOOLING is not

**The scientific core reproduces from a fresh clone with zero failures** — committed
`results_512/analysis/*` + 60 redacted judge sidecars are self-contained, and
`verify_report_claims.py` passes 85/86 (1 documented skip) on the clone. 446/449 tests pass;
the citation gate skips the gitignored tex mirrors gracefully; the immutable-artifacts gate is
properly clone-aware ("340 gitignored files absent — expected").

## Defects found (filed)

- **FS-17 (P2)** — 13 of 14 docx builders hardcode `/Users/tanueihorng/fyp_quant/docs/...` as
  their output path. From the clone, `make report` LIVE-DIRTIED the parent repo's committed
  docx (restored; parent verified clean at `44b9d67`); on any other machine it would ENOENT.
  Only `build_agentic_report.js` does it right (`__dirname`).
- **FS-18 (P2)** — the repo's own gates go red on an untouched checkout: 3 email surfaces
  registered as required but gitignored, and 6 docx "fresh_from_source" checks compare mtimes
  (undefined after `git checkout`). A fresh agent following CLAUDE.md's "run make agent-check"
  meets a failing gate through no fault of its own.
- **FS-19 (P2)** — `make report` needs `npm install`, absent from the setup docs (a global
  docx@9.6.1 masked it here); the documented `.venv` doesn't exist (project actually runs on
  conda base); lower-bound-only pins resolve to untested major versions (transformers 5.x) with
  no lockfile.
- Status note — docx builds are content-deterministic but not byte-reproducible (timestamps in
  `docProps/core.xml` only; `word/document.xml` byte-identical across builds).

## Documented limitations (not defects)
- Tex mirrors are gitignored → cannot be compiled from a fresh clone (consistent with FS-16's
  theme; the citation gate skips them gracefully).
- Raw `raw.jsonl`/`summary.json` intentionally absent (harmful text policy) — correct.
- TC1 GPU path unexercised (no GPU host; stub backends only) — out of scope, as planned.

Full friction log in the Phase 6 agent transcript; clone left in scratchpad for inspection.
