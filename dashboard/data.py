"""Pure data layer for the fyp_quant dashboard.

No Streamlit / Plotly imports here on purpose: everything in this module is a
plain function over the committed ``results/analysis/*.json`` artifacts and the
project's real Pydantic config schema, so it can be unit-tested headlessly
(``tests/test_dashboard_data.py``) without a browser.

Design rules:
- Read-only over committed artifacts. The one *write* helper
  (:func:`build_new_pair_config`) only ever returns YAML text / writes to a
  caller-chosen path under ``configs/generated/`` — it never touches the
  immutable ``configs/default.yaml`` / ``configs/tc1*.yaml`` or any ``results/``
  artifact (honours the repo's immutable-artifact discipline).
- Graceful degradation: a missing artifact returns ``None`` / an empty frame
  rather than raising, so the GUI renders partial state on a fresh checkout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

# Repo root = parent of this dashboard/ package.
REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Interpretation taxonomy presentation (mirrors analysis/compare_quant_pairs)  #
# --------------------------------------------------------------------------- #

#: Colour + human gloss for each interpretation label the analysis can emit.
LABEL_META: Dict[str, Dict[str, str]] = {
    "broad_degradation": {
        "color": "#dc2626",
        "gloss": "Worsens on both axes: harmful compliance up and/or capability down.",
    },
    "alignment_degradation": {
        "color": "#ea580c",
        "gloss": "Harmful compliance rises while capability is preserved (a true safety shift).",
    },
    "capability_collapse_masquerading_as_safety": {
        "color": "#7c3aed",
        "gloss": "Looks 'safer' only because the model got less capable, not more aligned.",
    },
    "alignment_improvement": {
        "color": "#2563eb",
        "gloss": "Harmful compliance falls with capability preserved (a genuine safety gain).",
    },
    "robust_preservation": {
        "color": "#16a34a",
        "gloss": "No material change on either axis — quantization is behaviourally safe here.",
    },
    "over_refusal_regression": {
        "color": "#d97706",
        "gloss": "Over-refusal on benign prompts moved materially.",
    },
}

DEFAULT_LABEL_COLOR = "#64748b"

#: Evidence tiers (kept separate from the label, per decision D19).
EVIDENCE_META: Dict[str, str] = {
    "confirmed": "CI excludes zero / McNemar significant.",
    "directional": "Point estimate leans one way but CI touches zero.",
    "null": "No detectable effect.",
    "unknown": "Insufficient data to classify.",
}


def label_color(label: Optional[str]) -> str:
    """Returns the display colour for an interpretation label."""
    if not label:
        return DEFAULT_LABEL_COLOR
    return LABEL_META.get(label, {}).get("color", DEFAULT_LABEL_COLOR)


def label_gloss(label: Optional[str]) -> str:
    """Returns the one-line human gloss for an interpretation label."""
    if not label:
        return ""
    return LABEL_META.get(label, {}).get("gloss", "")


# --------------------------------------------------------------------------- #
# Path helpers + safe loaders                                                  #
# --------------------------------------------------------------------------- #

def analysis_dir(repo_root: Path = REPO_ROOT) -> Path:
    """Returns the results/analysis directory."""
    return Path(repo_root) / "results" / "analysis"


def configs_dir(repo_root: Path = REPO_ROOT) -> Path:
    """Returns the configs directory."""
    return Path(repo_root) / "configs"


def load_json(path: Path) -> Optional[Any]:
    """Loads JSON, returning ``None`` if the file is absent or unparseable."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def list_config_files(repo_root: Path = REPO_ROOT) -> List[Path]:
    """Lists candidate config YAMLs (top-level + generated), study configs only.

    Excludes ``artifact_policy.yaml`` which is not a study config.
    """
    cfg = configs_dir(repo_root)
    found: List[Path] = []
    if cfg.exists():
        found.extend(sorted(p for p in cfg.glob("*.yaml") if p.name != "artifact_policy.yaml"))
        gen = cfg / "generated"
        if gen.exists():
            found.extend(sorted(gen.glob("*.yaml")))
    return found


# --------------------------------------------------------------------------- #
# Analysis artifact loaders + tidy frames                                      #
# --------------------------------------------------------------------------- #

def load_interpretations(repo_root: Path = REPO_ROOT) -> List[Dict[str, Any]]:
    """Loads the per-pair interpretation labels (pair_interpretations.json)."""
    data = load_json(analysis_dir(repo_root) / "pair_interpretations.json")
    return data if isinstance(data, list) else []


def load_pairwise(repo_root: Path = REPO_ROOT) -> List[Dict[str, Any]]:
    """Loads the per-(pair, benchmark) deltas (pairwise_deltas.json)."""
    data = load_json(analysis_dir(repo_root) / "pairwise_deltas.json")
    return data if isinstance(data, list) else []


def pairwise_df(repo_root: Path = REPO_ROOT) -> pd.DataFrame:
    """Returns the pairwise deltas as a DataFrame (empty frame if absent)."""
    rows = load_pairwise(repo_root)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_summary(repo_root: Path = REPO_ROOT) -> Optional[Dict[str, Any]]:
    """Loads the top-level analysis summary (quantization_analysis_summary.json)."""
    data = load_json(analysis_dir(repo_root) / "quantization_analysis_summary.json")
    return data if isinstance(data, dict) else None


def load_precision_sweep(repo_root: Path = REPO_ROOT) -> Optional[Dict[str, Any]]:
    """Loads the fp16->int8->nf4 precision sweep (precision_sweep.json)."""
    data = load_json(analysis_dir(repo_root) / "precision_sweep.json")
    return data if isinstance(data, dict) else None


def precision_sweep_long(repo_root: Path = REPO_ROOT) -> pd.DataFrame:
    """Flattens the precision sweep into a tidy long frame.

    Columns: pair_id, metric, precision, value. Only the absolute per-precision
    values are kept (the ``delta_*`` / ``shape`` keys are dropped).
    """
    sweep = load_precision_sweep(repo_root)
    if not sweep:
        return pd.DataFrame()
    precisions = sweep.get("precisions", ["fp16", "int8", "nf4"])
    out: List[Dict[str, Any]] = []
    for pair_id, payload in (sweep.get("per_pair") or {}).items():
        for metric, vals in (payload.get("metrics") or {}).items():
            for prec in precisions:
                if prec in vals and isinstance(vals[prec], (int, float)):
                    out.append(
                        {"pair_id": pair_id, "metric": metric, "precision": prec, "value": vals[prec]}
                    )
    return pd.DataFrame(out)


def load_judge_agreement(repo_root: Path = REPO_ROOT, int8: bool = False) -> Optional[Dict[str, Any]]:
    """Loads judge-vs-proxy agreement (judge_agreement[_int8].json)."""
    name = "judge_agreement_int8.json" if int8 else "judge_agreement.json"
    data = load_json(analysis_dir(repo_root) / name)
    return data if isinstance(data, dict) else None


def judge_agreement_df(repo_root: Path = REPO_ROOT, int8: bool = False) -> pd.DataFrame:
    """Per-model judge-vs-regex agreement as a frame (kappa, v2_asr, judge_asr)."""
    data = load_judge_agreement(repo_root, int8=int8)
    if not data:
        return pd.DataFrame()
    rows = []
    for alias, m in (data.get("per_model") or {}).items():
        rows.append(
            {
                "model_alias": alias,
                "cohens_kappa": m.get("cohens_kappa"),
                "agreement_rate": m.get("agreement_rate"),
                "v2_asr": m.get("v2_asr"),
                "judge_asr": m.get("judge_asr"),
                "n_shared": m.get("n_shared"),
            }
        )
    return pd.DataFrame(rows)


def load_multiple_comparisons(repo_root: Path = REPO_ROOT) -> Optional[Dict[str, Any]]:
    """Loads the Benjamini-Hochberg FDR summary (multiple_comparisons.json)."""
    data = load_json(analysis_dir(repo_root) / "multiple_comparisons.json")
    return data if isinstance(data, dict) else None


def load_sensitivity(repo_root: Path = REPO_ROOT) -> Optional[Dict[str, Any]]:
    """Loads the multi-seed sensitivity arm (sensitivity_multiseed.json)."""
    data = load_json(analysis_dir(repo_root) / "sensitivity_multiseed.json")
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------- #
# Judge-PRIMARY interpretation (the authoritative view; D16)                   #
# --------------------------------------------------------------------------- #
#
# IMPORTANT: ``make analyze`` writes ``pair_interpretations.json`` /
# ``pairwise_deltas.json`` from the *v2 regex proxy* (summary.v2.json). The
# study's authoritative scorer is the official HarmBench classifier (decision
# D16) — the regex over-counts ASR. ``multiple_comparisons.json`` (built in D38)
# is the judge-primary, FDR-corrected artifact: per-pair McNemar deltas/p-values
# and Benjamini-Hochberg q-values across all four axes. We rebuild the
# interpretation table from it (reusing the project's own ``classify_pair_change``
# so the labels match the report), rather than surfacing the superseded proxy
# numbers as the headline.

_AXIS = {
    "harm": "harmbench_asr_judge",
    "mmlu": "mmlu_accuracy",
    "arc": "arc_accuracy",
    "over_refusal": "xstest_over_refusal",
}


def contrasts_index(repo_root: Path = REPO_ROOT) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Indexes multiple_comparisons contrasts by (pair_id, metric)."""
    mc = load_multiple_comparisons(repo_root)
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not mc:
        return out
    for c in mc.get("contrasts", []):
        pair = c.get("pair_id")
        metric = c.get("metric")
        if pair and metric:
            out[(pair, metric)] = c
    return out


def mc_metric_df(repo_root: Path = REPO_ROOT, metric: str = "harmbench_asr_judge") -> pd.DataFrame:
    """Judge-primary per-pair deltas for one axis (delta, McNemar p, BH q, sig)."""
    mc = load_multiple_comparisons(repo_root)
    if not mc:
        return pd.DataFrame()
    rows = [c for c in mc.get("contrasts", []) if c.get("metric") == metric]
    return pd.DataFrame(rows)


def judge_primary_interpretations(repo_root: Path = REPO_ROOT) -> List[Dict[str, Any]]:
    """Rebuilds the per-pair interpretation table from the JUDGE-primary artifacts.

    Uses ``multiple_comparisons.json`` (judge ΔASR + McNemar + BH) for the deltas
    and significance, the project's own :func:`classify_pair_change` /
    :func:`label_evidence_status` for the label, and ``precision_sweep.json`` for
    the absolute fp16/NF4 judge ASR values. Returns ``[]`` if the judge artifact
    is absent (the UI then falls back to the clearly-labelled v2 view).
    """
    from ethical_benchmark.analysis.compare_quant_pairs import (
        classify_pair_change,
        label_evidence_status,
    )

    idx = contrasts_index(repo_root)
    if not idx:
        return []

    sweep = load_precision_sweep(repo_root) or {}
    per_pair_sweep = sweep.get("per_pair", {})
    pairs = sorted({pair for (pair, _metric) in idx})

    out: List[Dict[str, Any]] = []
    for pair in pairs:
        harm = idx.get((pair, _AXIS["harm"]))
        mmlu = idx.get((pair, _AXIS["mmlu"]))
        arc = idx.get((pair, _AXIS["arc"]))
        oref = idx.get((pair, _AXIS["over_refusal"]))

        harm_delta = harm.get("delta") if harm else None
        or_delta = oref.get("delta") if oref else None
        mmlu_delta = mmlu.get("delta") if mmlu else None

        harm_sig = harm.get("uncorrected_significant") if harm else None
        mmlu_sig = mmlu.get("uncorrected_significant") if mmlu else None
        or_sig = oref.get("uncorrected_significant") if oref else None

        label = classify_pair_change(harm_delta, or_delta, mmlu_delta)
        evidence = label_evidence_status(label, harm_sig, mmlu_sig, or_sig)

        sweep_metrics = (per_pair_sweep.get(pair) or {}).get("metrics", {})
        judge_asr = sweep_metrics.get("harmbench_asr_judge", {})

        out.append(
            {
                "pair_id": pair,
                "interpretation_label": label,
                "evidence_status": evidence,
                "harmbench_asr_delta": harm_delta,
                "harmbench_asr_delta_significant": harm_sig,
                "harmbench_asr_p_value": harm.get("p_value") if harm else None,
                "harmbench_asr_bh_q": harm.get("bh_q_value") if harm else None,
                "harmbench_asr_bh_significant": harm.get("bh_significant_q05") if harm else None,
                "harmbench_asr_fp16": judge_asr.get("fp16"),
                "harmbench_asr_nf4": judge_asr.get("nf4"),
                "xstest_over_refusal_delta": or_delta,
                "xstest_over_refusal_delta_significant": or_sig,
                "mmlu_accuracy_delta": mmlu_delta,
                "mmlu_accuracy_delta_significant": mmlu_sig,
                "arc_accuracy_delta": arc.get("delta") if arc else None,
                "arc_accuracy_delta_significant": arc.get("uncorrected_significant") if arc else None,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Per-run summary (for the Run / Execute page)                                 #
# --------------------------------------------------------------------------- #

def run_summary(
    model_alias: str,
    benchmark: str,
    results_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Loads results/<alias>/<benchmark>/summary.json for a finished run."""
    p = Path(results_dir) / model_alias / benchmark / "summary.json"
    data = load_json(p)
    return data if isinstance(data, dict) else None


def available_runs(results_dir: Path) -> List[Tuple[str, str]]:
    """Discovers (model_alias, benchmark) pairs that already have a summary.json."""
    base = Path(results_dir)
    out: List[Tuple[str, str]] = []
    if not base.exists():
        return out
    for model_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if model_dir.name in {"analysis", "summary", "raw", "slurm_logs", "slurm_logs_tc1"}:
            continue
        for bench_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if (bench_dir / "summary.json").exists():
                out.append((model_dir.name, bench_dir.name))
    return out


# --------------------------------------------------------------------------- #
# Config introspection + new-model scaffolding                                 #
# --------------------------------------------------------------------------- #

def read_config_models(config_path: Path) -> Dict[str, Dict[str, Any]]:
    """Returns the raw ``models`` mapping from a config YAML (empty on failure)."""
    p = Path(config_path)
    if not p.exists():
        return {}
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    models = raw.get("models")
    return models if isinstance(models, dict) else {}


def quant_suffix(quant_method: str) -> str:
    """Maps a quant method to the alias suffix used throughout the repo."""
    return "8bit" if quant_method == "int8" else "4bit"


def build_new_pair_config(
    *,
    base_config_path: Path,
    pair_id: str,
    family: str,
    size_b: float,
    model_id: str,
    quant_method: str = "nf4",
    benchmarks: Optional[List[str]] = None,
    attn_implementation: Optional[str] = None,
    trust_remote_code: bool = False,
    dtype: str = "auto",
) -> Tuple[Optional[str], Optional[str]]:
    """Builds a self-contained, validated single-pair config from a base config.

    Constructs a matched pair (``<pair_id>_base`` + ``<pair_id>_<suffix>``) that
    loads from identical ``model_id`` weights — the project's core design — and
    reuses the base config's ``benchmarks`` / ``decoding`` / ``slurm`` sections so
    the result is immediately runnable. Validation is performed by the project's
    real Pydantic schema, so the GUI surfaces the same fail-loud errors the CLI
    would (showcasing the schema rather than re-implementing it).

    Returns:
        ``(yaml_text, None)`` on success, or ``(None, error_message)`` if the
        composed config fails schema validation.
    """
    # Imported lazily so the data layer's other helpers don't hard-depend on the
    # full package import path during isolated unit tests.
    from ethical_benchmark.quant.config_schema import QuantizationConfig

    benchmarks = benchmarks or ["harmbench", "xstest", "mmlu", "arc"]

    base_raw: Dict[str, Any] = {}
    bp = Path(base_config_path)
    if bp.exists():
        try:
            loaded = yaml.safe_load(bp.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                base_raw = loaded
        except (yaml.YAMLError, OSError):
            base_raw = {}

    suffix = quant_suffix(quant_method)
    base_alias = f"{pair_id}_base"
    quant_alias = f"{pair_id}_{suffix}"

    def _member(quantized: bool) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "family": family,
            "size_b": size_b,
            "quantized": quantized,
            "pair_id": pair_id,
            "model_id": model_id,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "benchmarks": list(benchmarks),
        }
        if attn_implementation:
            entry["attn_implementation"] = attn_implementation
        if quantized:
            entry["quant_method"] = quant_method
        return entry

    composed: Dict[str, Any] = {
        "study_name": f"dashboard_{pair_id}",
        "models": {base_alias: _member(False), quant_alias: _member(True)},
        # Reuse the base config's benchmark/decoding/slurm blocks so the new
        # config is complete; fall back to a minimal harmbench block otherwise.
        "benchmarks": base_raw.get(
            "benchmarks",
            {
                "harmbench": {
                    "dataset_name": "walledai/HarmBench",
                    "config_name": "standard",
                    "split": "train",
                    "max_samples": 200,
                    "batch_size": 4,
                }
            },
        ),
    }
    if "decoding" in base_raw:
        composed["decoding"] = base_raw["decoding"]
    if "slurm" in base_raw:
        composed["slurm"] = base_raw["slurm"]

    # Keep only the benchmark definitions the new pair actually references, so a
    # narrowed benchmark selection cannot leave dangling top-level keys.
    wanted = set(benchmarks)
    composed["benchmarks"] = {
        k: v for k, v in composed["benchmarks"].items() if k in wanted
    } or composed["benchmarks"]

    try:
        QuantizationConfig.model_validate(composed)
    except Exception as exc:  # pydantic.ValidationError (and any compose error)
        return None, str(exc)

    yaml_text = yaml.safe_dump(composed, sort_keys=False, default_flow_style=False)
    return yaml_text, None
