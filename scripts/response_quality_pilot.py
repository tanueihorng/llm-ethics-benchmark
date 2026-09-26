#!/usr/bin/env python3
"""Create and manage the blinded paired XSTest response-quality pilot.

Raw prompt/response text is written only below gitignored ``human_labels/``.
Committed-compatible outputs contain IDs, strata and labels only.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_512"
LOCAL = ROOT / "human_labels"
ANALYSIS = RESULTS / "analysis"
SEED = 20260920
AXES = (
    "correctness", "task_fulfilment", "relevance", "completeness",
    "clarity", "safe_helpfulness", "overall_quality",
)
PAIRS = {
    "qwen_2b": ("qwen_2b_base", "qwen_2b_4bit"),
    "qwen_4b": ("qwen_4b_base", "qwen_4b_4bit"),
    "llama_3_2_3b": ("llama_3_2_3b_base", "llama_3_2_3b_4bit"),
    "mistral_7b": ("mistral_7b_base", "mistral_7b_4bit"),
    "phi4_mini": ("phi4_mini_base", "phi4_mini_4bit"),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["prompt_id"]): row for row in _read_jsonl(path)}


def make_sample() -> None:
    rng = random.Random(SEED)
    LOCAL.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    local_items: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for pair_id, (base_alias, nf4_alias) in PAIRS.items():
        base_dir = RESULTS / base_alias / "xstest"
        nf4_dir = RESULTS / nf4_alias / "xstest"
        base_raw = _by_id(base_dir / "raw.jsonl")
        nf4_raw = _by_id(nf4_dir / "raw.jsonl")
        base_api = _by_id(base_dir / "scores.judge.xstest_api.jsonl")
        nf4_api = _by_id(nf4_dir / "scores.judge.xstest_api.jsonl")
        base_jev = _by_id(base_dir / "scores.judge.xstest_jev.jsonl")
        nf4_jev = _by_id(nf4_dir / "scores.judge.xstest_jev.jsonl")
        base_v2 = _by_id(base_dir / "scores.v2.jsonl")
        nf4_v2 = _by_id(nf4_dir / "scores.v2.jsonl")
        shared = sorted(set(base_raw) & set(nf4_raw), key=lambda value: int(value))

        changed, stable = [], []
        for prompt_id in shared:
            api_change = base_api[prompt_id]["judge_label"] != nf4_api[prompt_id]["judge_label"]
            jev_change = base_jev[prompt_id]["judge_label"] != nf4_jev[prompt_id]["judge_label"]
            v2_change = bool(base_v2[prompt_id]["score_fields"]["is_refusal"]) != bool(
                nf4_v2[prompt_id]["score_fields"]["is_refusal"]
            )
            (changed if api_change or jev_change or v2_change else stable).append(prompt_id)
        selected = [(pid, "change_enriched") for pid in rng.sample(changed, 6)]
        selected += [(pid, "stable") for pid in rng.sample(stable, 6)]
        rng.shuffle(selected)

        for prompt_id, stratum in selected:
            a_is_base = bool(rng.getrandbits(1))
            a_alias, b_alias = (base_alias, nf4_alias) if a_is_base else (nf4_alias, base_alias)
            a_raw = base_raw[prompt_id] if a_is_base else nf4_raw[prompt_id]
            b_raw = nf4_raw[prompt_id] if a_is_base else base_raw[prompt_id]
            item_id = f"rq-{len(local_items) + 1:03d}"
            local_items.append({
                "item_id": item_id,
                "prompt": str(a_raw.get("prompt_text", "")),
                "response_a": str(a_raw.get("response", "")),
                "response_b": str(b_raw.get("response", "")),
            })
            key_rows.append({
                "item_id": item_id, "pair_id": pair_id, "prompt_id": prompt_id,
                "stratum": stratum, "a_alias": a_alias, "b_alias": b_alias,
                "a_condition": "fp16" if a_is_base else "nf4",
                "b_condition": "nf4" if a_is_base else "fp16",
            })
            manifest.append({
                "item_id": item_id, "pair_id": pair_id, "prompt_id": prompt_id,
                "stratum": stratum,
            })

    for path, rows in (
        (LOCAL / "response_quality_pilot_items.jsonl", local_items),
        (LOCAL / "response_quality_pilot_key.jsonl", key_rows),
        (ANALYSIS / "response_quality_pilot_manifest.jsonl", manifest),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Created {len(local_items)} blinded pairs; raw text remains local-only.")


def make_html() -> None:
    items = _read_jsonl(LOCAL / "response_quality_pilot_items.jsonl")
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    axis_json = json.dumps(AXES)
    output = LOCAL / "response_quality_pilot.html"
    document = f"""<!doctype html><meta charset='utf-8'>
<title>Blinded response-quality pilot</title>
<style>
body{{font:16px system-ui;max-width:1100px;margin:28px auto;padding:0 18px;color:#172033}}
.card{{border:1px solid #ccd3df;border-radius:12px;padding:20px;margin:20px 0}}
.responses{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.response{{white-space:pre-wrap;background:#f5f7fa;padding:14px;border-radius:8px}}
.prompt{{white-space:pre-wrap;background:#fff8dc;padding:12px;border-radius:8px}}
.axis{{display:grid;grid-template-columns:170px repeat(3,90px);align-items:center;margin:8px 0}}
button{{padding:7px;margin:2px}} button.sel{{background:#16213e;color:white}}
#status{{position:sticky;top:0;background:white;padding:10px;border-bottom:1px solid #ddd}}
</style>
<div id='status'></div><div id='root'></div>
<button onclick='download()'>Download labels JSON</button>
<script>
const items={payload}, axes={axis_json};
const labels=JSON.parse(localStorage.getItem('response_quality_pilot_labels')||'{{}}');
const pretty=s=>s.replaceAll('_',' ');
function choose(id,axis,value,button){{labels[id]??={{}};labels[id][axis]=value;
 localStorage.setItem('response_quality_pilot_labels',JSON.stringify(labels));
 button.parentElement.querySelectorAll('button').forEach(x=>x.classList.remove('sel'));
 button.classList.add('sel'); update();}}
function update(){{const done=items.filter(x=>axes.every(a=>labels[x.item_id]?.[a])).length;
 document.querySelector('#status').textContent=`Completed ${{done}} / ${{items.length}} pairs`;}}
document.querySelector('#root').innerHTML=items.map(x=>`<section class='card'>
<h2>${{x.item_id}}</h2><div class='prompt'><b>Prompt</b><br>${{escapeHtml(x.prompt)}}</div>
<div class='responses'><div><h3>A</h3><div class='response'>${{escapeHtml(x.response_a)}}</div></div>
<div><h3>B</h3><div class='response'>${{escapeHtml(x.response_b)}}</div></div></div>
${{axes.map(a=>`<div class='axis'><b>${{pretty(a)}}</b><span><button data-id='${{x.item_id}}' data-axis='${{a}}' data-value='A' onclick="choose('${{x.item_id}}','${{a}}','A',this)">A</button></span><span><button data-id='${{x.item_id}}' data-axis='${{a}}' data-value='B' onclick="choose('${{x.item_id}}','${{a}}','B',this)">B</button></span><span><button data-id='${{x.item_id}}' data-axis='${{a}}' data-value='tie' onclick="choose('${{x.item_id}}','${{a}}','tie',this)">Tie</button></span></div>`).join('')}}</section>`).join('');
document.querySelectorAll('button[data-id]').forEach(button=>{{
 if(labels[button.dataset.id]?.[button.dataset.axis]===button.dataset.value) button.classList.add('sel');
}});
function escapeHtml(s){{return s.replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#039;'}}[c]));}}
function download(){{const done=items.filter(x=>axes.every(a=>labels[x.item_id]?.[a])).length;
 if(done!==items.length){{alert(`Complete all pairs first (${{done}}/${{items.length}}).`);return;}}
 const blob=new Blob([JSON.stringify(labels,null,2)],{{type:'application/json'}});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='response_quality_pilot_labels.json';a.click();}}
update();
</script>"""
    output.write_text(document, encoding="utf-8")
    print(f"Wrote {output} ({len(items)} pairs; local-only raw text).")


def apply_labels(path: Path) -> None:
    labels = json.loads(path.read_text(encoding="utf-8"))
    manifest = _read_jsonl(ANALYSIS / "response_quality_pilot_manifest.jsonl")
    expected = {row["item_id"] for row in manifest}
    if set(labels) != expected:
        raise SystemExit("Label IDs do not exactly match the frozen sample.")
    rows = []
    for item in manifest:
        item_labels = labels[item["item_id"]]
        if set(item_labels) != set(AXES) or any(value not in {"A", "B", "tie"} for value in item_labels.values()):
            raise SystemExit(f"Incomplete or invalid labels for {item['item_id']}.")
        rows.append({**item, "human_labels_blinded": item_labels})
    out = ANALYSIS / "response_quality_pilot_human_labels.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {out} ({len(rows)} redacted labelled pairs).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Blinded paired response-quality pilot.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--make-sample", action="store_true")
    group.add_argument("--make-html", action="store_true")
    group.add_argument("--apply-labels", type=Path)
    args = parser.parse_args()
    if args.make_sample:
        make_sample()
    elif args.make_html:
        make_html()
    else:
        apply_labels(args.apply_labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
