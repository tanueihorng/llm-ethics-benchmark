"""Capture refusal-margin diagnostics for the quantization mechanism study (D33).

TC1 sbatch entry point (GPU node). For each model alias it loads the model
exactly as the benchmark pipeline does (reusing HFModelLoader + the config, so
the ``_4bit`` aliases get on-the-fly NF4 and Phi/Mistral get their flags), then
for every saved HarmBench prompt it runs ONE teacher-forced forward pass and
records scalar refusal-margin / confidence diagnostics. It performs no new
generation and never mutates raw.jsonl / summary.json / scores.judge.* — it
writes a NEW redacted sidecar ``scores.margin.<precision>.jsonl`` carrying only
``prompt_id`` + floats (no prompt or response text).

Metrics per HarmBench prompt (see ethical_benchmark/analysis/refusal_margin.py):
  * m1                : first-content-token refusal margin = logP(refusal set) - logP(compliance set)
  * entropy           : next-token entropy at the decision token (confidence diagnostic)
  * top1_top2_gap     : decisiveness at the decision token
  * seq_min/mean_margin: per-position refusal margin teacher-forced over the first
                         K committed-response tokens (captures late-resolved refusals,
                         e.g. Mistral's soft-preamble-then-comply)
  * seq_logprob       : mean per-token log-prob the model assigns to the committed response

It also captures the entropy / top1_top2_gap on a sample of NEUTRAL prompts
(MMLU, XSTest-benign) as the capability/confidence-confound control: the
analysis must show the refusal-margin shift exceeds this generic softening.

Typical TC1 usage (inside an sbatch job):
    python scripts/capture_refusal_margin.py --config configs/tc1.yaml --device cuda \
        --models qwen_2b_base qwen_2b_4bit ... phi4_mini_4bit

Run once per precision. The precision tag is derived from each model's
``quantized`` flag (fp16 vs nf4); pass --precision-tag to override (e.g. int8).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ethical_benchmark.analysis.refusal_margin import (  # noqa: E402
    build_token_sets,
    decision_entropy,
    refusal_margin,
    top1_top2_gap,
)
from ethical_benchmark.models.generation import (  # noqa: E402
    DecodingConfig,
    TextGenerator,
    set_global_seed,
)
from ethical_benchmark.models.loader import HFModelLoader, ModelSpec  # noqa: E402
from ethical_benchmark.quant.config_schema import load_quant_config  # noqa: E402

DEFAULT_MODELS = [
    "qwen_2b_base", "qwen_2b_4bit", "qwen_4b_base", "qwen_4b_4bit",
    "llama_3_2_3b_base", "llama_3_2_3b_4bit", "mistral_7b_base", "mistral_7b_4bit",
    "phi4_mini_base", "phi4_mini_4bit",
]
SEQ_WINDOW = 8  # teacher-force the first K committed-response tokens for the late-decision metrics


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _decision_logits(model, tokenizer, templated_prompt: str, max_input_tokens: int):
    """Returns the next-token logit vector at the first content position (numpy 1-D)."""

    torch = import_module("torch")
    inputs = tokenizer(
        templated_prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    return out.logits[0, -1, :].float().cpu().numpy(), inputs["input_ids"].shape[1]


def _sequence_margins(model, tokenizer, templated_prompt: str, response: str,
                      refusal_ids, comply_ids, max_input_tokens: int) -> Dict[str, float]:
    """Teacher-forces the committed response and returns late-decision diagnostics."""

    torch = import_module("torch")
    prompt_ids = tokenizer(templated_prompt, truncation=True,
                           max_length=max_input_tokens)["input_ids"]
    resp_ids = tokenizer(response or "", add_special_tokens=False)["input_ids"][:SEQ_WINDOW]
    if not resp_ids:
        return {"seq_min_margin": float("nan"), "seq_mean_margin": float("nan"),
                "seq_logprob": float("nan")}
    full = prompt_ids + resp_ids
    ids = torch.tensor([full], device=model.device)
    with torch.no_grad():
        out = model(input_ids=ids)
    logits = out.logits[0].float().cpu().numpy()  # [seq, vocab]
    margins: List[float] = []
    logps: List[float] = []
    start = len(prompt_ids) - 1  # logits[start] predicts the first response token
    for k, tok in enumerate(resp_ids):
        z = logits[start + k]
        margins.append(refusal_margin(z, refusal_ids, comply_ids))
        logz = z.max() + np.log(np.exp(z - z.max()).sum())
        logps.append(float(z[tok] - logz))
    margins_arr = np.array([m for m in margins if not np.isnan(m)], dtype=np.float64)
    return {
        "seq_min_margin": float(margins_arr.min()) if margins_arr.size else float("nan"),
        "seq_mean_margin": float(margins_arr.mean()) if margins_arr.size else float("nan"),
        "seq_logprob": float(np.mean(logps)) if logps else float("nan"),
    }


def _capture_model(alias: str, config, results_dir: Path, device: str,
                   precision_tag: Optional[str], control_n: int) -> Dict[str, Any]:
    entry = config.models[alias]
    decoding_cfg = DecodingConfig.from_dict(config.decoding.model_dump())
    set_global_seed(42)
    spec = ModelSpec(
        alias=alias, hf_id=entry.model_id, trust_remote_code=entry.trust_remote_code,
        revision=entry.revision, dtype=entry.dtype, quantized=entry.quantized,
        attn_implementation=entry.attn_implementation,
    )
    loader = HFModelLoader(device=device)
    model, tokenizer, runtime_device = loader.load(spec)
    generator = TextGenerator(model=model, tokenizer=tokenizer, device=runtime_device,
                              config=decoding_cfg)
    tag = precision_tag or ("nf4" if entry.quantized else "fp16")
    max_in = decoding_cfg.max_input_tokens

    sets = build_token_sets(tokenizer)
    R, C = sets["refusal_ids"], sets["compliance_ids"]

    # ---- HarmBench: full refusal-margin record per prompt ----
    hb_path = results_dir / alias / "harmbench" / "raw.jsonl"
    records: List[Dict[str, Any]] = []
    for row in _read_jsonl(hb_path):
        templated = generator._format_prompt(row["prompt_text"])
        z0, _ = _decision_logits(model, tokenizer, templated, max_in)
        rec = {
            "prompt_id": row["prompt_id"],
            "m1": refusal_margin(z0, R, C),
            "entropy": decision_entropy(z0),
            "top1_top2_gap": top1_top2_gap(z0),
        }
        rec.update(_sequence_margins(model, tokenizer, templated,
                                     row.get("response", ""), R, C, max_in))
        records.append(rec)
    margin_path = results_dir / alias / "harmbench" / f"scores.margin.{tag}.jsonl"
    with margin_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # ---- Neutral control: entropy / decisiveness only (confound control) ----
    control: List[Dict[str, Any]] = []
    for bench in ("mmlu", "xstest"):
        cpath = results_dir / alias / bench / "raw.jsonl"
        if not cpath.exists():
            continue
        for row in _read_jsonl(cpath)[:control_n]:
            templated = generator._format_prompt(row["prompt_text"])
            z0, _ = _decision_logits(model, tokenizer, templated, max_in)
            control.append({
                "prompt_id": row["prompt_id"], "benchmark": bench,
                "entropy": decision_entropy(z0), "top1_top2_gap": top1_top2_gap(z0),
            })
    control_path = results_dir / alias / "harmbench" / f"scores.margin_control.{tag}.jsonl"
    with control_path.open("w", encoding="utf-8") as f:
        for rec in control:
            f.write(json.dumps(rec) + "\n")

    # ---- redacted summary sidecar (token sets + metadata, no text) ----
    torch = import_module("torch")
    gpu = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    summary = {
        "model_alias": alias, "model_id": entry.model_id, "quantized": entry.quantized,
        "pair_id": entry.pair_id, "family": entry.family, "precision_tag": tag,
        "scorer_version": "refusal_margin_v1_2026-06-15", "seed": 42,
        "n_harmbench": len(records), "n_control": len(control),
        "seq_window": SEQ_WINDOW, "device": gpu,
        "token_sets": sets,
    }
    summary_path = results_dir / alias / "harmbench" / f"summary.margin.{tag}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # release GPU memory before the next model (mirrors the matrix runner)
    del model, generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture refusal-margin diagnostics (D33).")
    parser.add_argument("--config", default="configs/tc1.yaml")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision-tag", default=None,
                        help="Override the precision tag (default: fp16/nf4 from the "
                             "model's quantized flag). Use e.g. 'int8' for the INT8 point.")
    parser.add_argument("--control-n", type=int, default=50,
                        help="Neutral control prompts sampled per benchmark (mmlu, xstest).")
    args = parser.parse_args()

    config = load_quant_config(args.config)
    results_dir = Path(args.results_dir).resolve()

    print("=" * 72)
    print("Refusal-margin capture (D33) — teacher-forced forward passes, no generation.")
    print(f"{'model_alias':<22} {'precision':>9} {'n_hb':>6} {'n_ctrl':>7}")
    print("-" * 72)
    for alias in args.models:
        if alias not in config.models:
            print(f"WARN: unknown alias {alias}, skipping", file=sys.stderr)
            continue
        s = _capture_model(alias, config, results_dir, args.device,
                           args.precision_tag, args.control_n)
        print(f"{alias:<22} {s['precision_tag']:>9} {s['n_harmbench']:>6} {s['n_control']:>7}")
    print("-" * 72)
    print("Wrote scores.margin.<tag>.jsonl + scores.margin_control.<tag>.jsonl + "
          "summary.margin.<tag>.json per model (redacted: prompt_id + floats only).")
    print("Analyse on the Mac (no GPU) with scripts/refusal_margin_analysis.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
