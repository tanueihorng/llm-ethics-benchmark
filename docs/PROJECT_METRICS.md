# PROJECT_METRICS.md — Resource & Process Ledger

**What this is.** A measured record of what this project actually consumed: GPU time on TC1, API judge
spend, agent-session compute, and repository/process scale. Compiled 2026-07-25 from primary artifacts
(SLURM logs, judge sidecars, `raw.jsonl`, git history) and the 38 local Claude Code session transcripts.

**What this is not.** Not a results document — no experimental finding lives here. Nothing in this file
feeds `make report`, and no claim here is covered by `make verify-claims`. It is a *provenance record of
the process*, kept because the numbers are unusual enough to be worth citing in the final FYP write-up
(Chapter 9 / reflection / viva) even though they say nothing about quantization or safety.

**Source of truth for project state remains `docs/PROJECT_LOG.md`.** This file is a derived summary and
must never be treated as authoritative for status, decisions, or results.

---

## 0. Confidence key

Every figure below carries one tag. Do not cite an `ESTIMATED` figure without its stated assumption.

| Tag | Meaning |
|---|---|
| **MEASURED** | Read directly from a primary artifact (log timestamp, sidecar field, git object, transcript `usage` block). Reproducible by the command in Appendix A. |
| **DERIVED** | Arithmetic over MEASURED values only. No modelling assumption. |
| **ESTIMATED** | Involves a stated assumption (token heuristic, price list, coverage extrapolation). Assumption is named inline. |

---

## 1. TC1 GPU compute

Hardware: a single **Tesla V100-PCIE-32GB** (CCDS TC1, partition `UGGPU-TC1`).

### 1.1 Measured wall-clock

**41.10 GPU-hours** across 29 jobs whose `.err` logs carry parseable timestamps. — **MEASURED**

| Phase | Jobs | GPU-hours |
|---|---:|---:|
| INT8 matrix @512 (T29 / D35) | 5 | 15.82 |
| Primary matrix @512, fp16 + NF4 (T31 / D41) | 11 | 14.38 |
| Multi-seed sensitivity @512, 5 seeds (T39) | 6 | 9.37 |
| Early matrix @128 (partial logs retained) | 7 | 1.53 |
| **Total (timestamped jobs)** | **29** | **41.10** |

Per-job distribution — **MEASURED**:

| Statistic | Value |
|---|---|
| Longest single job | `phi4_mini_8bit__matrix` — **5 h 43 m** |
| Shortest (non-smoke) | `qwen_2b_base__harmbench` — 0.01 h |
| Median | 1.13 h |
| Mean | 1.42 h |
| Jobs exceeding the 6 h QoS wall | **0** |

INT8 is the slowest path: the five `*_8bit` jobs consumed 15.82 h against 14.38 h for eleven fp16/NF4
jobs — roughly **2.4× the wall-clock per alias** for bitsandbytes LLM.int8 vs NF4 on identical weights.
— **DERIVED**

### 1.2 Coverage gap (state this whenever citing 41.1)

38 distinct 5-digit job IDs in the TC1 range appear in `PROJECT_LOG.md`; only 37 `.err`/`.out` pairs were
SCP'd back, and the four judge-model jobs (`61047`, `61524`, `61217`, `61854`) emit no timestamped log
lines at all. Those four ran 18,000 classifications on a 13B classifier.

> **Honest headline: ~45–50 GPU-hours total, of which 41.1 h is directly measured.** — **ESTIMATED**
> (assumption: the four untimestamped judge jobs account for ~6–10 h, scaled from the classification
> volume and the observed throughput of the timestamped generation jobs.)

### 1.3 Scheduling reality

TC1's `QOSMaxGRESPerUser` permits **one running GPU job at a time** (a second may queue). Those ~45 GPU-hours
were therefore spread across roughly **six calendar weeks** of submit → wait → SCP cycles, not run in
parallel. Wall-clock elapsed ≫ GPU-hours consumed. — **MEASURED** (job start timestamps span 2026-05-27 → 2026-07-01)

### 1.4 Work produced on GPU

| Output | Count | Tag |
|---|---:|---|
| Model generations (prompt → response) | **69,660** | MEASURED |
| — @128 primary-era tree (`results/`) | 28,830 | MEASURED |
| — @512 primary tree (`results_512/`) | 28,830 | MEASURED |
| — multi-seed sensitivity (both budgets) | 12,000 | MEASURED |
| HarmBench-classifier judgements (13B, local) | 18,000 | MEASURED |
| LlamaGuard-3-8B judgements (T37) | 3,000 | MEASURED |
| **Total GPU-side judge classifications** | **21,000** | DERIVED |

Per-alias generation load: 4 benchmarks × (HarmBench 200 + XSTest 250 + MMLU 300 + ARC 1,172) = **1,922
generations per alias**, × 15 aliases = 28,830 per tree. — **DERIVED**

### 1.5 Notional compute cost

Institutional hardware — nothing was billed. For comparison only:

> ~45–50 GPU-hours × ~$2.50–3.00/GPU-hr (typical 2026 on-demand single-V100 rate) ≈ **$110–150**.
> — **ESTIMATED** (assumption: cloud on-demand V100 pricing; TC1 is free at point of use, so this figure
> exists solely to make the §3 comparison meaningful.)

---

## 2. External API judge spend (OpenAI, gpt-4o)

**9,750 API calls across three runs. Exactly one cost figure was ever recorded at the time.**

| Run | Calls | Cost | Tag |
|---|---:|---|---|
| XSTest 3-class refusal judge @512 (T35 / D45) | 3,750 | **$6.28** | **MEASURED** (billed; PROJECT_LOG 2026-07-12) |
| HarmBench 2nd judge @128 (T22 / D26) | 3,000 | ~$2.85 | ESTIMATED |
| HarmBench 2nd judge @512 (T31 / D41) | 3,000 | ~$4.45 | ESTIMATED |
| **Total** | **9,750** | **~$13.60** | DERIVED |

**How the two estimates were made.** Input token counts were reconstructed from the *actual* judged
content — every `behavior` + `generation` pair in the relevant `raw.jsonl` files — plus the fixed
`HARMBENCH_CLS_PROMPT` boilerplate, converted at ~4 characters/token, and priced at gpt-4o-2024-08-06
rates ($2.50 / $10.00 per MTok). The same estimator was run against the T35 XSTest workload as a
calibration check: it returned **$6.91 against a billed $6.28 (+10.0%)**, so the two HarmBench figures
were scaled down by that factor. — **ESTIMATED** (assumptions: 4 chars/token; gpt-4o-2024-08-06 list
pricing; the HarmBench 2nd-judge runs used the same dated snapshot family, which per finding **FS-20** is
*not* recoverable from artifacts — the sidecars record the floating alias `gpt-4o` with no snapshot date.)

Prior in-session guesses recorded at the time were "$2–5" and "$8–12"; the reconstruction lands inside
both windows.

**Parse reliability: 0 parse errors across all 9,750 calls (0.000%).** — **MEASURED** (from
`parse_error_count` in the 30 `summary.judge.api_judge.json` + 15 `summary.judge.xstest_api.json` sidecars.)

---

## 3. Agent-session compute

Scope: all 38 local Claude Code transcripts whose `cwd` is this repo (including two git worktrees and
`fyp_submission/`).

| | Value | Tag |
|---|---|---|
| Transcript files / distinct session IDs | 38 / 38 | MEASURED |
| On-disk size | 426 MB | MEASURED |
| Coverage window | 2026-06-13 → 2026-07-24 | MEASURED |
| Active days | 32 | MEASURED |
| Assistant messages | **19,329** | MEASURED |
| Typed user messages | 903 (1,872,313 chars) | MEASURED |
| Tool calls | 8,668 | MEASURED |
| **Total tokens** | **6,980,456,280 (6.98 B)** | MEASURED |

### 3.1 Token breakdown by model — **MEASURED**

| Model | Msgs | Input | Output | Cache write | Cache read |
|---|---:|---:|---:|---:|---:|
| `claude-opus-4-8` | 13,883 | 1,915,264 | 24,394,224 | 201,945,464 | 4,939,651,054 |
| `claude-fable-5` | 5,306 | 623,342 | 8,174,130 | 85,491,852 | 1,712,601,288 |
| `claude-opus-5` | 57 | 107 | 50,562 | 1,134,053 | 3,908,858 |
| `claude-opus-4-6` | 19 | 31 | 6,636 | 46,943 | 512,472 |
| **Total** | **19,329** | **2,538,744** | **32,625,552** | **288,618,312** | **6,656,673,672** |

### 3.2 API-equivalent cost

> **~$8,960** — **ESTIMATED** (assumptions: Fable 5 $10/$50 per MTok; Opus 4.x/5 $5/$25; cache **write**
> at 2× input, because the transcripts record `ephemeral_1h_input_tokens` — the 1-hour TTL — not the 5-minute
> 1.25× tier; cache **read** at 0.1× input.)

Nothing was billed — this ran on a subscription. The figure exists to make the work commensurable with
§1 and §2.

**Caching did most of the work on that bill.** 95.4% of all tokens were cache reads. Priced without
caching (all input at full rate) the same sessions come to **~$44,750** — caching removed roughly **80%**
of the notional cost. Read/write ratio: **23:1**. — **DERIVED**

### 3.3 Two undercounts — always state these

1. **Transcripts start 2026-06-13; the repo starts 2026-02-08.** ~40 of 162 commits (25%) predate any
   retained transcript, as does all Codex-side work.
2. **Subagent and workflow transcripts are not retained at all.** `~/.claude/tasks/` holds metadata only
   (548 KB, zero `.jsonl`). The 58 `Agent` calls and 69 `Workflow` invocations — T44 alone documented
   ~100+ agents across ten sessions — contribute **nothing** to the 6.98 B figure.

The true agent-side total is therefore materially higher than 6.98 B. Treat it as a **floor**, not a total.

### 3.4 Spend by project phase — **MEASURED** (output tokens as the work proxy)

| Phase | Active days | Assistant msgs | Output tokens | Share |
|---|---:|---:|---:|---:|
| Build: experiments + first report (06-13 → 07-02) | 13 | 8,334 | 15,148,511 | 46.4% |
| Validation layer: judges, human gold, multi-seed (07-03 → 07-14) | 11 | 7,234 | 12,480,202 | 38.3% |
| Audit + harden (07-15 → 07-22) | 6 | 3,667 | 4,884,107 | 15.0% |
| Post-close (07-23 →) | 2 | 94 | 112,732 | 0.3% |

T44 full-sweep alone (07-20 → 07-22): **1,490,378 output tokens = 4.6% of the project** in three days.

---

## 4. Repository & process scale — **MEASURED**

| | Value |
|---|---|
| Commits | 162 (first: 2026-02-08 `initialization commit`) |
| Contributors | 1 |
| Lines added / deleted across history | 296,640 / 13,901 |
| Tracked files | 1,030 |
| `summary.json` result files (all trees) | 180 (60 in the primary `results_512/` tree) |
| Test functions defined (`def test_`) | 417 across 35 files |
| Tests collected by pytest (parametrize-expanded) | 464 |
| `make verify-claims` checks | 93 |
| Claim-surface checks | 264 |
| Citation-completeness checks | 12–13 |
| `PROJECT_LOG.md` | 627,760 B / 621,964 chars / 942 lines |
| Formal audits run | 6 (`docs/audits/`, 2026-07-10 → 2026-07-20) |
| Decisions recorded (`§3 D-rows`) | 55 |
| Changelog rows (`§4`) | 279 |

Commits by month: Feb 1 · May 27 · Jun 51 · Jul 83. Busiest single day: **2026-05-27, 19 commits** — the
day the first TC1 matrix ran.

**22% of commits (36/162) are fix / remediation / stale-text corrections of earlier work.** Zero reverts.

---

## 5. Harness economics

The agentic scaffolding (`PROJECT_LOG.md`, `CLAUDE.md`/`AGENTS.md`, `todo.md`, `HANDOFF.md`,
`artifact_policy.yaml`, `verify_report_claims.py`, `test_agent_harness.py`, `docs/audits/`,
`docs/agent_tasks/`) is a large fraction of the repository. What follows is what can be measured about
whether it paid. **There is no control arm** — no version of this project ran without it — so no
efficiency claim here is causal.

### 5.1 Cost — **MEASURED**

| | Value |
|---|---|
| Commits touching harness files | 155 / 162 |
| Commits touching research code | 45 / 162 |
| Harness code | 2,104 lines (`verify_report_claims.py` 1,572 + `test_agent_harness.py` 532) |
| `artifact_policy.yaml` | 384 lines |
| `docs/audits/` + `docs/agent_tasks/` | 45 files, 772 KB |
| Audit + harden window | 15% of all output tokens in 6 days |

`PROJECT_LOG.md` grew **351,028 B → 621,936 B (+77%) between 2026-07-03 and 2026-07-22 — after the last
GPU job (2026-07-01), describing zero new experimental results.** — **MEASURED**

### 5.2 Context cost of `PROJECT_LOG.md` — **corrected figure**

> ⚠️ An earlier working estimate put the log at **26.9% of every cached prompt / ~$1,125**. That figure
> assumed the file is loaded in full each session. **The assumption is false and the figure is wrong by
> roughly 8×.** Do not cite it.

Measured behaviour across all 38 sessions — **MEASURED**:

| | Value |
|---|---|
| `Read` calls on `PROJECT_LOG.md` | 128 |
| — omitting offset/limit ("full" reads) | **5** |
| — windowed | 123 |
| `Grep`/`Bash`/`Glob` calls against it | **360** |
| Search-to-full-read ratio | **72 : 1** |
| Mean chars returned per `Read` | 9,104 (~2,276 tok) — **1.5% of the file** |
| Total content ever returned by `Read`, all sessions | 1,165,336 chars (~291K tok) |

> **Upper-bound share of cached context: 3.5%** (221 M of 6.38 B cache-read tokens), ≈ **$139**
> API-equivalent, ≈1.6% of §3.2. — **DERIVED** (upper bound: assumes every byte loaded stays in context
> and is re-read on every subsequent turn of that session.)

**The log is not a token problem.** It is a *retrieval* problem:

- The five "full" reads returned **32,839 / 38,871 / 39,733 / 41,893 chars** against files of 200–280 KB
  at those dates — i.e. **6–19% of the file**. The `Read` tool caps output below the file size.
- Consequence: `CLAUDE.md`'s opening instruction — *"Before doing anything else: read
  `docs/PROJECT_LOG.md`"* — **has never once been executed as written**, in 38 sessions.
- §4 changelog is **429,810 B = 69% of the file** (279 rows, mean 1,540 B / ~385 tokens, longest 4,634 B).
- Byte offsets vs the observed ~40 KB read ceiling: §1 Status at 1,249 ✅ · §2 Open actions at 25,693 ✅ ·
  **§3 Decisions at 55,068 ❌** · §4 changelog ❌. **All 55 D-rows sit past what any read returns** and are
  reachable only by grep — i.e. only by someone who already knows they exist.

That last point is the same failure mode **D54** records for citations (*"the uncited-BH defect survived
six audits because all of them enumerated from the reference list outward"*): a grep-only decision log
cannot surface a decision you did not know to search for. Unresolved as of 2026-07-25.

### 5.3 Evidence the harness fired — **MEASURED**

Documented instances of a gate catching a real error in flight:

- `agent-check`'s project-log-update gate fired correctly during T44 Phase 1a (ledger existed without its row).
- The freshness gate caught **5 stale docx mid-batch** after the builder-path edits (T44 R2).
- The stale-text tripwire caught its own author's comment wording (a bare `+0.055`).
- The FS-11 citation-lexicon extension fired live on the interim `.tex`, which then received its missing
  Connor 1987 entry.
- A revert-test showed the V7 lock **did not** fire — a gap that was then closed.
- One gate *prevented* damage: *"the new κ-family check failed first because the check was wrong, not the
  document — running it prevented 'correcting' a correct file."* (2026-07-15)
- The fresh-clone phase found defects nothing else could: **13 of 14 docx builders hardcoded
  `/Users/tanueihorng/fyp_quant/docs/...`** — from a fresh clone, `make report` silently wrote into the
  *original* repo (FS-17); and gates were red on an untouched checkout (FS-18).

### 5.4 The one near-experiment: ledger discipline (FS-24) — **MEASURED**

Same repo, same auditors, one rule changed:

| Era | Audits | Findings reaching a terminal disposition |
|---|---|---|
| Pre-ledger (2026-07-10, 07-11) | 2 | Only P1s got remediation rows; **P2/P3s had no tracking mechanism** |
| Post-ledger (07-15 onward) | 4 | **70/70** — jul15 26/26 · jul16 11/11 · claim-audit 25/25 · log-row 8/8 |

Named pre-ledger losses: **FS-21** (methods text says `torch.inference_mode`, code uses `torch.no_grad()`)
untracked **11 days**; **FS-22** (embedded HarmBench classifier prompt is the HF model-card fork, not the
GitHub-harness variant — undisclosed, so absolute ASRs are not literature-comparable) untracked **10 days**;
**FS-23** a ~9-item cluster residual. The uncited-BH defect (recovered by D54) was the same failure mode.

This is the strongest single piece of evidence in the dataset, and note its scope: it supports the
**append-only ledger rule (D55)**, not the harness as a whole.

### 5.5 Counter-evidence — **MEASURED**

1. **75% finder false-positive rate.** T44's own process lesson: *"refuters killed 47/62 raised findings
   ≈75% finder false-positive rate."* The verification layer earned its cost; the finding layer largely
   generated work for it.
2. **The largest audit changed nothing that matters.** T44: 25 findings — 23 remediated, 1 waived, 1
   refuted — **zero conclusion-touching.** No result, significance verdict, label or hedge changed.
3. **Fixes created new defects, repeatedly.** 2026-07-15: *"re-verified those fixes and found the same
   failure mode in them (68 agents; 24 findings) — TWO P1s where the V5 fix corrected each deck's data
   record but left the chart caption…"*. 2026-07-16: *"the **fourth consecutive round** where a fix landed
   at the named site and missed a differently-worded sibling, now including my own."*
4. **Gates have false positives too** (the κ-family check above).

### 5.6 Reading of the evidence

Stated as a reading, not a finding:

- The **cheap deterministic layer** (gates, locks, must-fire proofs, the ledger rule) has a documented
  record of firing on real defects at near-zero marginal cost, and §5.4 is the one place a harness change
  produced a measurable outcome difference.
- The **expensive probabilistic layer** (multi-agent audits) shows clearly declining returns across six
  audits: 07-10/07-11 found conclusion-relevant methods problems; 07-15/07-16 found real but
  non-conclusion-touching drift; T44 found 25 items, none conclusion-touching, at a 75% finder
  false-positive rate, for 15% of the project's output tokens.

---

## 6. Observations with narrative value

Kept because they are the parts a reader remembers. Each is sourced above.

1. **The GPU was never the expensive part.** ~45 GPU-hours of free institutional compute and **$13.60** of
   OpenAI credit produced every number in the thesis. The agent layer that wrote, audited and re-audited
   the *claims about* those numbers is worth ~$8,960 of compute — roughly **660× the out-of-pocket
   experiment cost** and ~60× the notional cloud cost of the GPU time. The measurement was cheap; the
   epistemics were expensive.

2. **The write-to-run ratio.** 296,640 lines added, 162 commits, 1,030 tracked files, 464 tests, six
   formal audits, a 622 KB decision log — all supporting **180 summary files**, 60 of which are primary.

3. **Near-perfect execution reliability.** 0 job failures among the 29 timestamped SLURM runs; 0 parse
   errors across **30,750** judge classifications (21,000 GPU-side + 9,750 API). The single documented
   crash — job `61836`, `KeyError: 'shape'`, dead at 1 m 35 s on a transformers chat-template
   incompatibility — happened in a **smoke** job, not a matrix job. That is exactly what the smoke tier
   exists for.

4. **Sequencing tells the real story.** Busiest git day: 2026-05-27, the day the first TC1 matrix ran.
   Busiest *token* days are all in July, after the last GPU job (2026-07-01). The experiment finished on
   1 July; the following three weeks were entirely claim-surface work.

5. **Bursty, not steady.** Three days carry 22% of all assistant messages (07-11: 1,678 · 06-26: 1,419 ·
   06-14: 1,367). The longest single session spanned **117.7 h** wall-clock across 2,157 assistant turns.

6. **The most expensive single lesson cost $6.28.** T35 bought exactly one finding: the FDR-surviving Phi
   over-refusal decrease (−0.048, regex) does not replicate under an independent gpt-4o judge (+0.016
   strict, n.s.). That $6.28 is why RQ2's headline is now stated scorer-invariantly instead of resting on
   a regex the study had already demoted for HarmBench ASR.

7. **A harness instruction can be unexecutable and nobody notices for 38 sessions.** See §5.2.

---

## Appendix A — How to reproduce each figure

All commands run from the repo root.

**§1 GPU-hours** — parse first/last `YYYY-MM-DD HH:MM:SS,mmm` timestamps in every `.err` under
`results/slurm_logs_tc1/`, `results/slurm_logs/`, `results_512/slurm_logs_tc1/`; per-file wall-clock is
`last − first`; sum. Files with no timestamped line contribute 0 and are the coverage gap in §1.2.

**§1.4 generation counts** — sum `num_records` (fallback `metrics.num_samples`) over
`results*/**/summary.json`. Judge volumes: same field over `summary.judge.<backend>.json`, grouped by
backend (`harmbench_cls`, `llamaguard`, `api_judge`, `xstest_api`).

**§2 API cost** — for each judged `raw.jsonl` record, take `len(prompt) + len(response)`, add the fixed
template length (HarmBench ≈ 1,150 chars; XSTest system rubric ≈ 1,180 chars), divide by 4 for tokens,
price at $2.50/MTok in and $10.00/MTok out. Calibrate against the billed $6.28 for T35 and apply the
resulting factor (0.909) to the HarmBench runs.

**§3 transcript stats** — stream every `*.jsonl` under
`~/.claude/projects/-Users-tanueihorng-fyp-quant*/`; for `type == "assistant"` accumulate
`message.usage.{input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens}`
grouped by `message.model`; count `tool_use` blocks by `name`; bucket by `timestamp[:10]`.

**§5.2 log-read behaviour** — in the same transcripts, collect `tool_use` blocks whose `input` mentions
`PROJECT_LOG`; classify `Read` calls by presence of `offset`/`limit`; join each `tool_use.id` to its
`tool_result` and measure returned `content` length. Upper-bound context share: per session,
`(log_tokens_loaded) × (assistant_messages_after_first_load)`, summed, divided by that session's actual
`cache_read_input_tokens` total.

**§4 / §5.1 repo scale** — `git rev-list --count HEAD`, `git ls-files | wc -l`,
`git log --format="" --numstat`, `git log --format="%H" -- <paths> | sort -u | wc -l` for per-area commit
counts, `git show <sha>:docs/PROJECT_LOG.md | wc -c` for the size trajectory.

---

## Appendix B — Known gaps

| Gap | Effect | What would close it |
|---|---|---|
| 4 judge jobs emit no timestamps | ~6–10 GPU-h unmeasured | `sacct -j <id> --format=Elapsed` on TC1 while the accounting DB still holds them |
| Transcripts start 2026-06-13 | ~25% of commits have no session record | Nothing — the data was never retained |
| Subagent/workflow transcripts not retained | 58 `Agent` + 69 `Workflow` runs contribute 0 tokens to §3 | Nothing retrospectively |
| gpt-4o snapshot not pinned for the HarmBench 2nd judge (FS-20) | §2 estimates assume a snapshot family | Unrecoverable; disclosed at report §6.12 Result 4 |
| `~4 chars/token` heuristic | §2 and §5.2 token counts ±~10% | Tokenizer-exact recount; the T35 calibration bounds the error |
| API-equivalent pricing (§3.2) | Cost figures move with the price list | Restate with the price list current at citation time |

---

*Compiled 2026-07-25 by Claude (Opus 5) at the student's request. Derived record — see
`docs/PROJECT_LOG.md` for authoritative project state.*
