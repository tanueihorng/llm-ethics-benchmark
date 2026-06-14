# Agent Task Packet: T21 Strengthen Results Discussion

## Objective

Make the judge-vs-v2 disagreement the analytical center of Chapter 6/7 without changing raw experiment artifacts.

## Likely Files

- `scripts/build_fyp_report.js`
- `docs/FYP_Report_2026-06-14.docx`
- `docs/PROJECT_LOG.md`
- `results/analysis/judge_agreement.{json,csv}`
- `results/analysis/pairwise_deltas.{json,csv}`

## Allowed Actions

- Read raw artifacts for aggregate computation only.
- Use IDs, counts, labels, and aggregate metrics.
- Add derived analysis outputs if needed.
- Regenerate the report with `make report`.

## Forbidden Actions

- Do not modify `results/*/*/raw.jsonl`.
- Do not modify `results/*/*/summary.json`.
- Do not copy raw HarmBench prompt or response text into docs, logs, or chat.
- Do not claim a second judge was run unless its sidecars exist.

## Verification

```bash
python fyp_cli.py agent-status
python scripts/judge_agreement.py
make report
make agent-check
```

## Done Criteria

- Chapter 6/7 explains why the scoring layer changed while raw outputs stayed valid.
- The Qwen 1.7B judge-primary broad-degradation result is stated clearly.
- `docs/PROJECT_LOG.md` has a new changelog row.
- The generated report is current.
