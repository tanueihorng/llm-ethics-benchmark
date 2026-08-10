# Submission runbook — for a human, without an agent

**This is the only document in the repo written for you rather than for a coding agent.**
`PROJECT_LOG.md`, `HANDOFF.md`, `AGENT_DASHBOARD.md` and everything in `docs/agent_tasks/`
assume a machine reader with a grep loop. This one assumes you, at a laptop, in November,
with a deadline.

Written 2026-08-11. Every command was run before being written down. Where something is
assumed rather than verified, it says so.

---

## 1. The four dates

| Date | What | Task | Artifact to send |
|---|---|---|---|
| **Mon 24 Aug 2026** | Interim report → supervisor | T53 | see §3 |
| **Mon 19 Oct 2026** | Final report → supervisor **and examiners**. **This is the graded version.** | T15 | `docs/rewritten_fyp_report_v1.docx` |
| **Fri 13 Nov 2026** | Amended final + DR-NTU deposit. **No changes accepted after this.** | T54 | as above, amended |
| **4 / 7 / 8 Dec 2026** | Oral presentation, 20 min + 10 min Q&A | T55 | `docs/viva/FYP_Viva_Deck.pptx` |

Supervisor: Dr. Zhang Jiehuang — `jiehuang.zhang@ntu.edu.sg`.

> **EXAMINERS — not recorded anywhere in this repo.** The 19 Oct and 13 Nov submissions
> both go to examiners as well as the supervisor, and nobody has written down who they are
> or how to reach them. Find out well before October and fill this in:
>
> - Examiner name / email: `________________`
> - Second examiner (if any): `________________`
> - Confirmed submission channel (email? portal? both?): `________________`

---

## 2. Get the project running on a clean machine

```bash
python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements-analysis.txt
```

That is the machine you are almost certainly on: it computes the statistics, runs the
tests and builds the documents, and it **never loads a model**. The versions are exact
and are the ones everything here actually passes on.

Only if you are re-running experiments on a GPU do you want the other one:

```bash
pip install -r requirements.txt
```

> **Why there are two, in case an examiner asks (T60).** They describe different
> machines, and cannot be combined. `requirements.txt` floors transformers at 4.41
> because the study's Qwen3, Mistral and Phi-4-mini models cannot be loaded below that —
> verified: the 4.32.1 on this Mac supports Llama and none of the other three. The
> analysis environment pins 4.32.1 and is unaffected, because it only ever reads
> committed result files. Until 2026-08-11 this was a single file called
> `constraints-tested.txt` that tried to be both, and the install command in the README
> failed outright.

```bash
pip install -r requirements-dashboard.txt
```

Needed for the dashboard **and** for part of the test suite — `dashboard/data.py` is covered
by the main suite, so without this two test modules fail to import.

```bash
npm install
```

Only needed to build the older JavaScript documents (v5, v6, thesis, the humanized
alternates). The two documents you actually submit are built by Python and do not need it.

**LibreOffice** is optional. Only `scripts/report_tools/check_pagination.py` uses it, to
render a docx and confirm the front-matter page numbers are right. Without LibreOffice that
check *skips silently* — it does not fail. If you want it, install LibreOffice normally;
the script looks in `/Applications/LibreOffice.app` first.

---

## 3. The 24 Aug interim submission (T53)

There are two candidates and they are **not** equivalent.

| Candidate | Length | Contents page renders | Has "progress / remaining work" framing |
|---|---|---|---|
| `docs/rewritten_fyp_interim_v1.docx` | 57 pp | **Yes, anywhere** | **No** — same body as the final |
| `docs/FYP_Interim_2026-07-10.docx` | 38 pp | Word only (blank elsewhere until F9) | **Yes** — has a progress chapter |

**Two things to decide, and they pull in opposite directions.**

1. **Length.** The school guidance on file says an interim report is *about 50% of the
   final*. The final is ~57 pages, so the rewrite-interim is ~100% of it, not 50%. The
   older interim at 38 pp is closer but still above half.
2. **Framing.** The interim submission is described as *"stating project progress"*. The
   rewrite-interim is a pure retitle of the final report — it changes the cover line and the
   running header and nothing else — so it contains no progress narrative at all. The older
   interim has a "Progress and Remaining Work" chapter written for exactly this purpose.

**There is a third option, and it is probably the right one:** send the rewrite-interim and
put the progress framing in the covering email rather than in the document. A finished
report submitted at the interim stage is not a defect — being *ahead* is not a problem — but
an examiner who expects a progress statement should not have to hunt for one.

> The rewrite-interim also carries a real advantage: `docs/FYP_Interim_2026-07-10.docx`
> contains a claim the project itself verified as an over-claim — it calls the gpt-4o second
> judge "architecturally independent" four times, which cannot be asserted about a
> closed-weight model. That phrase was removed from the rewrite line and appears **0 times**
> in `rewritten_fyp_interim_v1.docx`. It is still present in the old interim, and in the
> thesis (3×) and v5 (2×). Tracked as **T56**.

**Whichever you send, do this:**

```bash
make rewrite-interim
```

Then open the result, check the cover says *Interim Report* and not *Final Year Project
Report*, and export to PDF (§5).

---

## 4. Rebuilding any document

```bash
make rewrite
```
→ `docs/rewritten_fyp_report_v1.docx` — **the submission candidate** (T52/D59).

```bash
make rewrite-interim
```
→ `docs/rewritten_fyp_interim_v1.docx`. Note this also rebuilds the report above as a side
effect, because it imports the report builder rather than copying its prose.

```bash
make report
```
→ `docs/FYP_Report_2026-07-01_v5.docx`. **This is NOT your submission document.** v5 is the
older frozen build, kept for comparison. It still names itself the report of record, and it
still carries 19 claim-to-source defects that were only ever fixed in v6 and the rewrite.
The name `make report` is historical; do not read it as "build the report I am submitting".

```bash
python scripts/build_viva_deck.py
```
→ `docs/viva/FYP_Viva_Deck.pptx`.

**The generated `.docx` files are not in git** (decision D60 — the repo was public and an
unsubmitted report sitting in a public repo can be returned as a similarity match against
your own submission). The *builders* are in git, so any document is one command away. The
`.pptx` deck is an exception and is committed, so you do not need to rebuild it to open it.

---

## 5. Exporting a PDF — read this before submitting

**The contents page is a Word field, not text.** Whether it renders depends on which
document you are exporting. Measured, not assumed:

| Document | Contents page in a non-Word renderer |
|---|---|
| `rewritten_fyp_report_v1.docx` | **Populated** — carries a cached result |
| `rewritten_fyp_interim_v1.docx` | **Populated** — carries a cached result |
| `FYP_Interim_2026-07-10.docx` | **Blank** |
| `FYP_Report_2026-07-01_v5.docx` | **Blank** |
| `FYP_Report_2026-07-25_v6.docx` | **Blank** |
| `FYP_Thesis_2026-07-02_v4.docx` | **Blank** |

So if you submit from the rewrite line, exporting from anything — Word, LibreOffice, Google
Docs, macOS Preview — gives a correct contents page. If you submit any of the other four,
**you must open it in Word first**, press `Ctrl+A` then `F9`, confirm the contents page
fills in, and export from there.

One catch even in Word: `F9` repaginates the Table of Contents but **not** the List of
Figures or List of Tables — those are static paragraphs. After any export, eyeball the
contents page against the figure and table lists and make sure the page numbers agree.

---

## 6. What the checks actually prove

```bash
make agent-check
```
The full gate: docs in sync, log updated, raw artifacts unmutated, no leaked prompt text, no
stale claims, and the whole test suite. Prints a live count — read it rather than quoting a
memorised one.

```bash
make verify-claims
```
Asserts every load-bearing number in the documents against the committed analysis files. This
is the one that catches a report claiming something the data does not say.

**What none of it proves.** These gates check that stated numbers match the artifacts. They
cannot check whether a *sentence* is a fair reading of its source. Three separate audits each
found real defects in text a green gate had just cleared. A passing run means "no known
mechanical defect", not "correct".

If a check fails, read what it says before changing anything. The failures are written to
name the fix. **Never make a check pass by weakening it** — that is a standing rule here
(D56), and every time it has been broken the defect came back later as a paraphrase.

---

## 7. Things that exist only on this laptop

If this machine dies, these are gone. They are not in git, by deliberate decision.

| What | Where | Why it matters |
|---|---|---|
| Supervisor email drafts | `docs/email_drZhang_july.md`, `..._august.md` | Written and ready to send; not recoverable |
| Whole submission tree (~19 MB) | `fyp_submission/` | LaTeX mirrors, Overleaf zips, compiled PDFs, deck copies |
| School guidance capture | `fyp_submission/report_latex/SCHOOL_GUIDELINES.md` | The page bands and presentation rules — the key parts are quoted in §8 so they survive |
| All generated `.docx` | `docs/*.docx` | Rebuildable from the tracked builders (§4) |

**Back these up now, outside the repo** — a cloud folder or an external drive.

> **Before you copy anything anywhere:** `results/*/raw.jsonl` and the two sheets under
> `human_labels/` contain **unredacted harmful prompts and model responses**. That is normal
> for this kind of study and they are correctly kept out of git, but do not sync them to a
> shared drive, and do not attach them to an email. Everything committed is redacted —
> IDs, counts and booleans only.

---

## 8. School rules, quoted so they survive

From the guidance capture (source: school/supervisor guidance you pasted on 2026-06-26, plus
Prof. Chng Eng Siong's thesis-writing notes). **This is a capture of a paste, not a retrieved
CCDS document — check anything load-bearing against the current official version.**

**Report**
- Final report length: **40–65 pages**. The rewrite is ~57 pp — inside the band.
- Interim report ≈ **50% of the final**. (See §3 — this is the open tension.)
- Must show: you defined the scope, did the work yourself, know the literature, built
  reasonable baselines, and proposed future work.

**Presentation**
- **Slide 1 must carry your name, the FYP title, and the year.**
- **Every slide must have a page number.**
- Cover, in this order: problem and motivation → who else solved it → how *you* solved it,
  with novelty and difficulty → results → conclusion and future work.
- Graded on: appreciation of the FYP, difficulty and contribution, technical ability, coding
  skill, knowledge.
- Their advice, which is good: record yourself rehearsing and watch it back.

The deck at `docs/viva/FYP_Viva_Deck.pptx` is built to that order, and the two mechanical
rules are enforced by `tests/test_viva_deck.py` so an edit cannot quietly break them.

> The capture says "20–23 min talk + ~5 min Q&A". The CCDS AY2026 timeline you supplied on
> 2026-08-10 says **20 min + 10 min Q&A**, and that is newer, so the deck is timed to 20:00.

---

## 9. December: the viva

1. `docs/viva/FYP_Viva_Deck.pptx` — 17 slides. **The speaker notes are the script**, with a
   time budget on every slide summing to 20:00. Open the notes view.
2. `docs/viva/VIVA_QA_BANK.md` — 46 questions with spoken answers, each pointing at an
   artifact. Read "the six answers you must know cold" and the cheat table last.
3. Rehearse against the per-slide budgets. If you run long, the notes say what to cut.
4. Do not memorise test counts or lock counts — they change with every commit. If asked,
   run `make agent-check` and read the live number aloud.

The single most useful habit in the room: **answer in one sentence first, then give the
evidence.** And where you do not know, say so and say what you would do to find out — the
bank marks those explicitly, and they score better than a confident guess.

---

## 10. If something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot find module 'docx'` | Node deps missing | `npm install` |
| `ModuleNotFoundError: docx` | python-docx missing | `pip install -r requirements.txt` |
| A document is missing from `docs/` | Generated files are not in git (D60) | Rebuild it — §4 |
| `verify_claim_surfaces` fails on a file you just added | `docs/*.md`, `docs/*.docx`, `docs/*.html` are discovery globs | Add an entry for it in `configs/claim_surfaces.yaml` |
| Pagination test skips | LibreOffice or PyMuPDF absent | Install them, or accept the skip |
| A gate fails and you cannot see why | — | Read the message; it names the fix. Do not weaken the check |

**Source of truth for everything else is `docs/PROJECT_LOG.md`** — §1 for current state, §2
for open tasks, §3 for why decisions were made, §4 for the full history. It is long; use
search rather than reading it top to bottom.
