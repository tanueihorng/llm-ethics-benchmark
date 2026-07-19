#!/usr/bin/env python3
"""One-shot metadata correction of the committed LlamaGuard judge sidecars (T41).

Audit finding (T41 #6): the fifteen `*.judge.llamaguard.{jsonl,json}` sidecars were
written by the shared judge runner before it carried a backend-specific scorer tag,
so each one recorded `scorer_version = "judge_harmbench_cls_v1_2026-05-28"` (the
HarmBench-*classifier* tag) even though it was produced by the open-weight LlamaGuard
backend, and each summary labelled LlamaGuard's looser unsafe-content rate as
`attack_success_rate` with no construct note. The *numbers* are correct (the audit
independently reproduced every κ and ΔASR); only the provenance/construct labels
were wrong.

These are DERIVED redacted sidecars (never raw artifacts), so a documented
metadata-only correction is permitted. This script:
  * rewrites `scorer_version` -> "judge_llamaguard_v1_2026-07-13" (the tag the
    runner now emits for the llamaguard backend, ethical_benchmark/judges/validation.py);
  * adds `metric_construct = "llamaguard_unsafe_content_rate"` to each summary;
  * adds a dated `provenance_note` to each summary recording this correction;
  * changes NOTHING else — it asserts, per file, that every other key/value
    (judge_harmful labels, IDs, revision, all metrics/ASR) is byte-for-byte
    unchanged, and prints the per-file field-change counts as proof.

Idempotent: re-running after correction is a no-op (already-correct files report 0
value changes beyond the additive metadata).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLD_SCORER = "judge_harmbench_cls_v1_2026-05-28"
NEW_SCORER = "judge_llamaguard_v1_2026-07-13"
CONSTRUCT = "llamaguard_unsafe_content_rate"
NOTE = ("scorer_version corrected 2026-07-19 (T41): these sidecars were written by "
        "the shared runner before it carried a backend-specific tag and had inherited "
        "the HarmBench-classifier scorer_version; the LlamaGuard backend answers a "
        "looser 'is this response unsafe?' question (metric_construct). Numbers "
        "unchanged; metadata-only correction of a derived sidecar.")


def _correct_summary(path: Path) -> tuple[int, int]:
    """Returns (value_changes, keys_added). Raises if a non-metadata field would change."""
    obj = json.loads(path.read_text())
    before = json.dumps(obj, sort_keys=True)
    value_changes = 0
    keys_added = 0
    # The ONLY value we are allowed to change is scorer_version.
    if obj.get("scorer_version") == OLD_SCORER:
        obj["scorer_version"] = NEW_SCORER
        value_changes += 1
    elif obj.get("scorer_version") != NEW_SCORER:
        raise SystemExit(f"{path}: unexpected scorer_version {obj.get('scorer_version')!r}")
    if "metric_construct" not in obj:
        obj["metric_construct"] = CONSTRUCT
        keys_added += 1
    elif obj["metric_construct"] != CONSTRUCT:
        raise SystemExit(f"{path}: unexpected metric_construct {obj['metric_construct']!r}")
    if "provenance_note" not in obj:
        obj["provenance_note"] = NOTE
        keys_added += 1
    # Guard: nothing besides those three keys may have changed.
    check = json.loads(before)
    check["scorer_version"] = NEW_SCORER
    check["metric_construct"] = CONSTRUCT
    check["provenance_note"] = NOTE
    if json.dumps(check, sort_keys=True) != json.dumps(obj, sort_keys=True):
        raise SystemExit(f"{path}: correction would change a non-metadata field — aborting")
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    return value_changes, keys_added


def _correct_scores(path: Path) -> int:
    """Rewrites scorer_version on each JSONL record; asserts every other field is intact."""
    lines = path.read_text().splitlines()
    out = []
    changed = 0
    for ln in lines:
        if not ln.strip():
            continue
        rec = json.loads(ln)
        original = dict(rec)
        if rec.get("scorer_version") == OLD_SCORER:
            rec["scorer_version"] = NEW_SCORER
            changed += 1
        elif rec.get("scorer_version") != NEW_SCORER:
            raise SystemExit(f"{path}: unexpected scorer_version in a record")
        # every other field must be identical
        original.pop("scorer_version", None)
        after = dict(rec)
        after.pop("scorer_version", None)
        if original != after:
            raise SystemExit(f"{path}: a non-scorer_version field would change — aborting")
        out.append(json.dumps(rec, ensure_ascii=False))
    path.write_text("\n".join(out) + "\n")
    return changed


def main() -> None:
    summaries = sorted(glob.glob(str(REPO / "results_512/*/harmbench/summary.judge.llamaguard.json")))
    scores = sorted(glob.glob(str(REPO / "results_512/*/harmbench/scores.judge.llamaguard.jsonl")))
    assert len(summaries) == 15 and len(scores) == 15, f"expected 15+15, got {len(summaries)}+{len(scores)}"
    tot_v = tot_k = tot_lines = 0
    for p in summaries:
        v, k = _correct_summary(Path(p))
        tot_v += v; tot_k += k
        print(f"summary {Path(p).parent.parent.name:22s}: scorer_version changes={v}, keys_added={k}")
    for p in scores:
        c = _correct_scores(Path(p))
        tot_lines += c
        print(f"scores  {Path(p).parent.parent.name:22s}: {c} records retagged")
    print(f"\nDONE: {tot_v} summary scorer_version changes, {tot_k} metadata keys added, "
          f"{tot_lines} score records retagged across 15 aliases. No numeric field changed.")


if __name__ == "__main__":
    main()
