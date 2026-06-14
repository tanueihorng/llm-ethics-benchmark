"""Per-category HarmBench judge ASR + pairwise deltas (no GPU, redacted output).

T21(a): the headline safety claim (RQ1) reports a single aggregate HarmBench ASR
per model. This script decomposes that ASR by HarmBench semantic category so the
report can show *where* harmful compliance concentrates and how quantization
shifts it per category — strengthening the Results chapter without any new
inference.

It joins, per ``prompt_id``:
  * the semantic category from each model's ``results/<alias>/harmbench/raw.jsonl``
    (READ-ONLY, TC1-original/immutable; only ``score_fields.category`` is read —
    no prompt or response text enters the output), and
  * the PRIMARY (judge) label from
    ``results/<alias>/harmbench/scores.judge.harmbench_cls.jsonl``
    (``judge_harmful`` boolean; redacted sidecar).

Writes ``results/analysis/harmbench_category_breakdown.{json,csv}`` (categories,
counts, judge ASR per model, and per-pair ΔASR only).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

MODEL_ALIASES = [
    "qwen_2b_base",
    "qwen_2b_4bit",
    "qwen_4b_base",
    "qwen_4b_4bit",
    "llama_3_2_3b_base",
    "llama_3_2_3b_4bit",
    "mistral_7b_base",
    "mistral_7b_4bit",
    "phi4_mini_base",
    "phi4_mini_4bit",
]

# pair_id -> (baseline_alias, quantized_alias)
PAIRS = {
    "qwen_2b": ("qwen_2b_base", "qwen_2b_4bit"),
    "qwen_4b": ("qwen_4b_base", "qwen_4b_4bit"),
    "llama_3_2_3b": ("llama_3_2_3b_base", "llama_3_2_3b_4bit"),
    "mistral_7b": ("mistral_7b_base", "mistral_7b_4bit"),
    "phi4_mini": ("phi4_mini_base", "phi4_mini_4bit"),
}


def _prompt_categories(raw_path: Path) -> Dict[str, str]:
    """Returns ``{prompt_id: category}`` from a HarmBench raw.jsonl (labels only)."""

    out: Dict[str, str] = {}
    if not raw_path.exists():
        return out
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("prompt_id")
            sf = rec.get("score_fields") or {}
            category = sf.get("category")
            if pid is not None and category is not None:
                out[pid] = category
    return out


def _judge_labels(judge_path: Path) -> Dict[str, bool]:
    """Returns ``{prompt_id: judge_harmful}`` from a judge sidecar."""

    out: Dict[str, bool] = {}
    if not judge_path.exists():
        return out
    with judge_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("prompt_id")
            if pid is None:
                continue
            out[pid] = bool(rec.get("judge_harmful"))
    return out


def _category_counts(raw_path: Path, judge_path: Path) -> Dict[str, Dict[str, int]]:
    """Returns ``{category: {"harmful": int, "total": int}}`` (judge-scored)."""

    cats = _prompt_categories(raw_path)
    labels = _judge_labels(judge_path)
    out: Dict[str, Dict[str, int]] = defaultdict(lambda: {"harmful": 0, "total": 0})
    for pid, category in cats.items():
        if pid not in labels:
            continue  # only count prompts the judge actually scored
        out[category]["total"] += 1
        if labels[pid]:
            out[category]["harmful"] += 1
    return dict(out)


def _asr(cell: Dict[str, int]) -> Optional[float]:
    return cell["harmful"] / cell["total"] if cell["total"] else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-category HarmBench judge ASR breakdown (redacted)."
    )
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    per_model: Dict[str, Dict[str, Dict[str, int]]] = {}
    for alias in MODEL_ALIASES:
        hb = results_dir / alias / "harmbench"
        per_model[alias] = _category_counts(
            hb / "raw.jsonl", hb / "scores.judge.harmbench_cls.jsonl"
        )

    if not any(per_model.values()):
        print("No HarmBench raw.jsonl + judge sidecars found under", results_dir)
        return 1

    # Per-pair, per-category judge ASR deltas (4-bit minus baseline).
    per_pair: List[Dict[str, Any]] = []
    for pair_id, (base_alias, quant_alias) in PAIRS.items():
        base_counts = per_model.get(base_alias, {})
        quant_counts = per_model.get(quant_alias, {})
        categories = sorted(set(base_counts) | set(quant_counts))
        for category in categories:
            base_cell = base_counts.get(category, {"harmful": 0, "total": 0})
            quant_cell = quant_counts.get(category, {"harmful": 0, "total": 0})
            base_asr = _asr(base_cell)
            quant_asr = _asr(quant_cell)
            delta = (
                quant_asr - base_asr
                if base_asr is not None and quant_asr is not None
                else None
            )
            per_pair.append({
                "pair_id": pair_id,
                "category": category,
                "n": base_cell["total"],
                "baseline_asr": base_asr,
                "quantized_asr": quant_asr,
                "asr_delta": delta,
            })

    report = {
        "scorer": "judge.harmbench_cls",
        "per_model": {
            alias: {
                category: {
                    "harmful": cell["harmful"],
                    "total": cell["total"],
                    "asr": _asr(cell),
                }
                for category, cell in categories.items()
            }
            for alias, categories in per_model.items()
        },
        "per_pair_category_deltas": per_pair,
    }
    (analysis_dir / "harmbench_category_breakdown.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    csv_path = analysis_dir / "harmbench_category_breakdown.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["pair_id", "category", "n", "baseline_asr", "quantized_asr", "asr_delta"]
        )
        for row in per_pair:
            writer.writerow([
                row["pair_id"], row["category"], row["n"],
                row["baseline_asr"], row["quantized_asr"], row["asr_delta"],
            ])

    # Console (redacted): per-pair category ASR deltas.
    print("=" * 78)
    print("Per-category HarmBench judge ASR deltas (4-bit minus baseline)")
    print("=" * 78)
    for pair_id, (base_alias, quant_alias) in PAIRS.items():
        rows = [r for r in per_pair if r["pair_id"] == pair_id]
        if not any(r["baseline_asr"] is not None for r in rows):
            print(f"\n{pair_id}  (no judge-scored data)")
            continue
        risen = sum(
            1 for r in rows if isinstance(r["asr_delta"], float) and r["asr_delta"] > 0
        )
        print(f"\n{pair_id}  ({len(rows)} categories; {risen} rose under 4-bit)")
        print(f"  {'category':<30} {'n':>4} {'base':>6} {'4bit':>6} {'Δ':>8}")
        for r in sorted(
            rows, key=lambda x: (x["asr_delta"] is None, -(x["asr_delta"] or 0.0))
        ):
            def _f(x: Optional[float]) -> str:
                return f"{x:.3f}" if isinstance(x, float) else "  n/a"
            d = r["asr_delta"]
            d_s = f"{d:+.3f}" if isinstance(d, float) else "  n/a"
            print(f"  {r['category']:<30} {r['n']:>4} {_f(r['baseline_asr']):>6} "
                  f"{_f(r['quantized_asr']):>6} {d_s:>8}")
    print()
    print(f"Wrote {analysis_dir / 'harmbench_category_breakdown.json'}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
