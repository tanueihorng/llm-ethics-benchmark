# T44 — Full-Sweep Audit Plan (every dimension, both directions)

**Status:** PLAN ONLY (2026-07-20). Nothing here has been executed.
**Trigger:** user request after D54 ("next time i will need a more comprehensive full sweep audit in every way").
**Prime design lesson (D54):** an audit only sees what its enumeration basis iterates over. Six prior audits
missed the uncited-BH defect because every one enumerated *existing* citations (presence→fitness); the lone
verifier that ran the reverse direction (07-11 "25th verifier") was cut and its P3 never tracked. Therefore:

## Standing rules for every phase (non-negotiable)

1. **Declare the enumeration basis** in the phase deliverable: exactly what set is iterated over, and what
   is *structurally invisible* to it. A phase without a written blind-spot statement is incomplete.
2. **Run both directions** where they exist: presence→fitness (is what's there correct?) AND
   usage/absence→presence (is anything missing that should be there?).
3. **Cross-verify every finding against primary artifacts before acting** — past external audits produced
   hallucinated findings (~half of one Codex audit); only the 07-19 audit had zero. Never remediate unverified.
4. **Every finding gets tracked before session end** — into the phase ledger AND PROJECT_LOG §2/§4, including
   P3s. The 25th-verifier failure mode (caught → filed → lost) is the named enemy.
5. **Never cite a passing gate for a property it doesn't test** (D42). Each phase says what its checks do NOT cover.
6. **Cut verifiers must be logged as cut**, with an explicit list of which of their checks were dropped.

## Phase 0 — Freeze & baseline (cheap, first)
- Record HEAD SHA, run all gates (verify-claims, surfaces, pytest, agent-check), snapshot outputs to the ledger dir.
- Deliverable: `docs/audits/<date>_full_sweep/BASELINE.md`.
- Basis: the gate suite. Blind spot: everything the gates don't test (that's the rest of the plan).

## Phase 1 — Gate self-audit (do the locks test what they claim?)
- For each named check in verify-claims / surfaces / harness tests: perturb the guarded property in a sandbox
  copy and confirm the gate FAILS (must-fire), then restore. Sample ≥1 perturbation per check family.
- Reverse direction: **coverage map** — enumerate load-bearing claims in the six documents (numbers, labels,
  significance verdicts) and mark which are NOT machine-locked. Unlocked claims become Phase-2 recompute targets.
- Basis: the check registry + a claim inventory built from the builders. Blind spot: claims not recognisable as
  claims by the inventory pass (mitigated by Phase 8's hostile read).

## Phase 2 — Numbers & statistics (recompute, then judge appropriateness)
- Direction A (recompute): independently re-derive every reported number from the committed redacted sidecars
  (no reuse of `results_512/analysis` intermediates where avoidable): ASRs, deltas, McNemar exact p's,
  bootstrap CIs (seeded), κ values, BH q-values + survivor set, MDE/power, parser-sensitivity brackets.
- Direction B (appropriateness): a stats-examiner agent judges *choice* of method, not just execution:
  paired-binary → McNemar exact ok? BH family composition matches its pre-registration/notes? one-sided vs
  two-sided consistent? CI method vs test disagreements handled as documented? d_z the right effect size?
- Basis: the number inventory from Phase 1. Blind spot: numbers that exist only in gitignored artifacts.

## Phase 3 — Citations & scholarship (all four axes)
- 3a Fitness (presence→fitness): re-verify each citation site supports its sentence (07-19 method).
- 3b Completeness (usage→presence): the D54 gate + an agent pass for methods *used but never named*
  (a procedure applied in code/prose without its conventional name — lexicons are blind to these).
- 3c Conventions: named-but-unattributed conventions (T43 κ verbal bands is the known instance; look for others:
  effect-size adjectives, "significant" thresholds, benchmark-specific terms).
- 3d Quotes & datasets: verbatim quotes re-checked against sources; every dataset/model/tool has a citation.
- Basis: 3a iterates citation sites; 3b iterates a method lexicon + code-level scorer/test implementations;
  3c/3d iterate conventions and artifacts. Blind spot: 3b's unnamed-method arm is judgment-based — record misses.

## Phase 4 — Cross-document consistency
- Diff the load-bearing story across ALL surfaces: report v5, thesis v4, interim, 3 humanized, 2 LaTeX mirrors,
  both decks, README, RESULTS_CARD, dashboard data layer, CLAUDE/AGENTS instruction text, PROJECT_LOG §1.
- Both directions: (A) same claim stated differently anywhere? (B) claim present in one surface but *absent*
  where a reader would need it (e.g. a caveat disclosed in the report but missing from the thesis)?
- Basis: the shared-claim inventory. Blind spot: point-in-time snapshots (June deck etc.) are exempt but must
  carry their data-snapshot notice — verify the notice, don't re-base them.

## Phase 5 — Omission audit (the D54 direction, generalised)
- What SHOULD the documents contain that they don't? Checklist-driven: every limitation implied by the artifacts
  (power floors, single-annotator, deviation disclosures, API-egress, seed scope) is actually disclosed; every
  method implemented in code is described in Ch3/4; every negative/failed arm is reported; every pre-registration
  clause has its outcome stated; reproducibility checklist items all addressed.
- Basis: checklists + code-vs-prose diff. Blind spot: unknown-unknowns — partially mitigated by Phase 8.

## Phase 6 — Reproducibility (fresh-clone reality check)
- Clean worktree clone: pip install, pytest, make verify-claims (expect documented local-only SKIPs and nothing
  else), make report/thesis/interim byte-stability check, tectonic compile of both mirrors.
- Basis: the build path. Blind spot: TC1-side reproducibility (GPU runs) — out of scope, say so.

## Phase 7 — Audit-of-audits (close the loop that lost the 25th-verifier P3)
- Iterate EVERY ledger under docs/audits/ + audit-shaped PROJECT_LOG rows: for each recorded finding, verdict
  ∈ {remediated (link the row), tracked-open (link T-item), explicitly-waived (link decision), LOST}.
  Any LOST finding is itself a new P1 process finding.
- Basis: the audit ledgers. Blind spot: findings that were never written down anywhere.

## Phase 8 — Adversarial examiner panel (unknown-unknowns pass)
- 4 hostile lenses, each trying to REFUTE rather than verify: (1) stats examiner, (2) safety-ML domain examiner
  (construct validity, threat model, scorer choices), (3) reproducibility/engineering reviewer, (4) a "make the
  headline claim collapse" free hunter with no enumeration basis at all.
- Plus a completeness critic over the audit itself: "which modality did this sweep not run?"
- Basis: none by design — this is the phase that exists because enumeration bases have blind spots.

## Phase 9 — Synthesis, remediation, re-verify
- Merge findings → dedupe → severity-rank → cross-verified ledger → remediation plan (user approves scope)
  → execute → re-run Phases 0-gate + targeted re-checks on every touched surface → final scope-limits statement
  listing what the sweep did NOT cover.
- Deliverable: `docs/audits/<date>_full_sweep/` (ledger, per-phase reports, scope manifest, closure table),
  PROJECT_LOG §3 decision + §4 rows, all findings tracked per Standing Rule 4.

## Multi-session execution model (MANDATORY — the sweep runs across days, never one session)

The sweep is token-intensive, so it executes as **one-phase-(or sub-phase)-per-session over multiple days**.
No record may live only in a session's context: **the audit's memory is the committed ledger directory**, and
a session that dies mid-run loses at most its in-flight analysis, never a recorded finding.

**Durable state (all under `docs/audits/<date>_full_sweep/`, committed to git):**
- `STATE.md` — the sweep's single source of truth: baseline SHA, phase-status table
  (`pending / in-progress / done`, with date + agent + sessions used), a drift log (any repo commit landing
  mid-sweep, and which completed phases it touches), and an explicit NEXT-ACTION line for the next session.
- `FINDINGS.md` — append-only findings ledger. Each finding gets an ID (`FS-<n>`) **at discovery time** with:
  phase, severity, exact file:line evidence, status (`open / verified / refuted / remediated / waived`), and
  a tracking link once it enters PROJECT_LOG §2. Findings are written HERE FIRST, the moment they are found —
  never held in-context for an end-of-session write-up (the 25th-verifier loss is the named enemy).
- `phaseN_*.md` — one report per phase: enumeration basis, blind-spot statement, method, results, agent count.

**Per-session protocol:**
1. START: read `STATE.md` (before PROJECT_LOG orientation); confirm baseline SHA; log any drift since the last
   session in the drift log and mark which done-phases it invalidates (targeted re-check, not full re-run).
2. RUN: execute exactly the phase(s) STATE.md names as next. Append to `FINDINGS.md` as findings occur.
3. END (mandatory, even if the phase is unfinished): update `STATE.md` (status + NEXT-ACTION specific enough
   for a cold agent: files, commands, where the phase stopped), one PROJECT_LOG §4 row, a `todo.md` entry via
   the fyp-todo-capture pattern if in-flight context exists, then **commit the audit dir** (git is the
   no-record-lost guarantee; a never-committed ledger is one `rm -rf` from gone).
4. Workflow `resumeFromRunId` is same-session only — never rely on it across days; the files are the resume.

**Session plan (suggested; STATE.md is authoritative once the sweep starts):**
- S1: Phase 0 + Phase 1 · S2: Phase 2A (scripted recompute — cheap) · S3: Phase 2B + 3a
- S4: Phase 3b/3c/3d · S5: Phase 4 · S6: Phase 5 + 6 · S7: Phase 7 · S8: Phase 8 (4 lenses)
- S9: Phase 9 synthesis + remediation plan (user approves scope) · S10: remediation + re-verify + closure.
Sessions are natural pause points: the user can stop the sweep for days between any two with zero loss.

**How the user drives it (per session, when the usage limit allows):**
- To start or continue: say **"continue T44"** (or "run T44 phase N"). The agent MUST then read
  `docs/audits/<date>_full_sweep/STATE.md` first and execute the NEXT-ACTION line (or the named phase).
- To fit the remaining budget, pick by weight — order requirements permit it:
  - **Strictly ordered:** Phase 0 first · Phase 1 before 2 (its coverage map feeds the recompute list) ·
    Phase 8 after 1–7 · Phase 9 last.
  - **Order-flexible (run any time after Phase 0, alone if budget is tight):** Phase 2A (cheap, scripted),
    Phase 6 (cheap-medium, mostly deterministic), Phase 7 (medium, reads ledgers only), Phase 3c/3d (light).
  - **Heavy (want a fresh full-budget session):** Phase 2B, 3a/3b, 4, 5, 8.
- A phase may also be split mid-way: end-of-session protocol step 3 records exactly where it stopped, and the
  next "continue T44" resumes from that point. Partial phases are normal, not failures.
- To pause the whole sweep indefinitely: nothing to do — state is committed; "continue T44" weeks later works.

**Baseline & drift policy:** Phase 0 pins the baseline SHA. Prefer freezing document/content work during the
sweep; if a commit must land mid-sweep, it goes in the drift log and only its touched files get re-checked by
already-done phases. Phase 9 reconciles the final state against the baseline.

## Model policy (user decision 2026-07-20)

Fable-orchestrator / Opus-executor, tiered by seat — the executor pool dominates token spend (~90%), so the
orchestrator premium is cheap and the executor tiering is where the economy lives:
- **Orchestrator (main loop, every session): Fable 5** — verdict calls, refutation, synthesis; the only seat
  whose judgment leverages every finding. (Evidence: the 07-19 citation audit, the first with zero hallucinated
  findings, ran Fable-orchestrated with Opus verifiers.)
- **Heavy verifiers: Opus 4.8, effort xhigh** — Phase 2B stats-appropriateness, 3a/3b citation fitness+completeness,
  Phase 5 omissions, all four Phase 8 lenses.
- **Mechanical executors: Opus 4.8, effort low** — grep/quote sweeps, 3c/3d, Phase 6 fresh-clone, Phase 7 ledger reads.
- **Phase 2A: deterministic scripts, no model.**
Blanket-xhigh on all executors is explicitly NOT the policy (it would forfeit most of the savings).
Total scale ≈ the 07-15/07-16 sweeps (~100+ agents), but dosed at ~10–30 agents per session across ~10 sessions.

## Execution notes
- Within a session, run that phase's fan-out as a parallel multi-agent workflow (adversarial verify included);
  Phase 2A is deterministic scripting first, agents only for 2B. Phase 8 runs LAST before synthesis, on the
  post-Phase-1..7 state, so it hunts what the structured phases missed.
- Expected total scale: comparable to the 07-15/07-16 sweeps (≈100+ agents) spread across the sessions above —
  each session starts on an explicit user go, not by default.
- Prerequisites before S1: T43 decided (κ labels), any pending sends (T1) done or explicitly deferred so the
  sweep audits a stable state.
