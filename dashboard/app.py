"""fyp_quant interactive dashboard (Streamlit).

A GUI over the quantization safety-vs-capability study that lets a user:
  1. **Overview**        — read the headline (per-pair labels, FDR survivors).
  2. **Results Explorer** — interactive charts over the committed analysis JSON.
  3. **Add a Model**      — fill a form -> a schema-validated, runnable config +
                            ready-to-submit TC1 sbatch (the "put in a new model" flow).
  4. **Run / Execute**    — launch a smoke/full run from the GUI with live logs.
  5. **Raw Results**      — browse any finished run's summary.json.

Launch:  ``make dashboard``  (or ``streamlit run dashboard/app.py``)

This is a read-only viewer over ``results/`` and an orchestration front-end over
the existing ``fyp_cli.py`` — it never mutates committed artifacts. New configs
are written under ``configs/generated/`` only.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow ``streamlit run dashboard/app.py`` from the repo root to import the package.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard import data as D  # noqa: E402
from dashboard import theme as T  # noqa: E402

st.set_page_config(
    page_title="fyp_quant · Quantization Safety Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Editorial-light theme (warm paper · Newsreader/Spline Sans/IBM Plex Mono · indigo+gold).
st.markdown(T.CSS, unsafe_allow_html=True)
badge = T.badge


# --------------------------------------------------------------------------- #
# Sidebar: global controls                                                     #
# --------------------------------------------------------------------------- #
st.sidebar.title("🛡️ fyp_quant")
st.sidebar.caption("Safety–capability trade-offs in quantized compact LLMs")

PAGES = ["Overview", "Results Explorer", "Add a Model", "Run / Execute", "Raw Results"]
page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")

config_files = D.list_config_files(REPO_ROOT)
config_labels = [str(p.relative_to(REPO_ROOT)) for p in config_files] or ["configs/default.yaml"]
sel_config = st.sidebar.selectbox("Config", config_labels, index=0)
config_path = REPO_ROOT / sel_config

results_dir_str = st.sidebar.text_input(
    "Results dir (browse)", value="results_512",
    help="Tree used for browsing. results_512 = the primary 512-token study (D41); "
         "results = the retained 128-token comparison (historical).",
)
results_dir = REPO_ROOT / results_dir_str
if results_dir_str.strip() == "results":
    st.sidebar.caption("⚠ `results/` is the retained **128-token** comparison — "
                       "the primary study is `results_512/`.")

st.sidebar.divider()
st.sidebar.caption(
    "Browsing is read-only over the results trees. Execution writes ONLY to a "
    "scratch tree (default `results_dev/`) — the canonical `results*/` evidence "
    "trees are write-protected in this UI. New models are written under "
    "`configs/generated/` and never overwrite committed configs."
)


# --------------------------------------------------------------------------- #
# Page: Overview                                                               #
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    st.markdown(
        T.hero_html(
            "Quantization Safety–Capability Study",
            dek="Matched-pair, judge-validated comparison of fp16 vs on-the-fly quantized "
            "compact LLMs (1.7–7.2B) across HarmBench (ASR), XSTest (over-refusal), and MMLU/ARC "
            "(capability) — at HarmBench's <b>512-token reference budget</b> (decision D41).",
            eyebrow="CCDS25-1136 · Judge-validated · 512-token primary",
        ),
        unsafe_allow_html=True,
    )

    summary = D.load_summary(REPO_ROOT)
    mc = D.load_multiple_comparisons(REPO_ROOT)

    # Authoritative view: judge-primary HarmBench ASR (D16), rebuilt from the
    # FDR artifact. Fall back to the v2 proxy table only if it's missing.
    interps = D.judge_primary_interpretations(REPO_ROOT)
    judge_primary = bool(interps)
    if not interps:
        interps = D.load_interpretations(REPO_ROOT)

    if not interps:
        st.warning(
            "No analysis artifacts found under `results/analysis/`. "
            "Run `make analyze` (or the **Run / Execute** page) to populate them."
        )
        return

    if judge_primary:
        st.markdown(
            T.verdict_html(
                "The reading",
                "At the reference budget, <em>no pair significantly raises harmful compliance</em> — "
                "the multiplicity-robust costs of 4-bit quantization are capability and over-refusal, "
                "not safety. The only significant ΔASR is a <em>decrease</em> (Llama-3.2-3B).",
            ),
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "Showing the **v2 regex-proxy** numbers (the judge artifact `multiple_comparisons.json` "
            "was not found). These over-count ASR — not the authoritative headline.",
            icon="⚠️",
        )

    tiles = [("Model pairs · families", f'{len(interps)}<span class="unit">/4</span>')]
    if mc:
        tiles.append(("ASR contrasts surviving FDR", '<span class="accent">0</span>'))
        tiles.append(("FDR survivors · q<0.05", f'{mc.get("n_bh_significant_q05", "—")}<span class="unit">cap/OR</span>'))
        tiles.append(("Uncorrected significant", str(mc.get("n_uncorrected_significant", "—"))))
    elif summary:
        tiles.append(("Pairwise contrasts", str(summary.get("pairwise_count", "—"))))
    st.markdown(T.stat_tiles_html(tiles), unsafe_allow_html=True)

    st.markdown(
        T.section_head_html(
            "01 · Findings",
            "Per-pair interpretation",
            note=("Judge-primary: the official HarmBench classifier (D16), McNemar exact significance, "
                  "Benjamini–Hochberg q over the 20-contrast family."
                  if judge_primary else "v2 regex proxy — NOT the authoritative headline."),
        ),
        unsafe_allow_html=True,
    )
    for row in interps:
        st.markdown(T.pair_card_html(row, judge_primary), unsafe_allow_html=True)

    if mc:
        st.markdown(T.section_head_html("02 · Multiplicity", "Benjamini–Hochberg correction"),
                    unsafe_allow_html=True)
        st.caption(mc.get("description", ""))
        survivors = mc.get("bh_survivors")
        if isinstance(survivors, list) and survivors:
            st.markdown("**Effects that survive FDR (q<0.05):**")
            for s in survivors:
                if isinstance(s, dict):
                    pair = s.get("pair_id") or s.get("contrast") or s.get("name") or "?"
                    metric = str(s.get("metric") or s.get("label") or "").replace("_", " ")
                    q = s.get("bh_q_value") or s.get("q_value") or s.get("q")
                    delta = s.get("delta")
                    bits = [f"**`{pair}`** — {metric}"]
                    if isinstance(delta, (int, float)):
                        bits.append(f"Δ {delta*100:+.1f} pp")
                    if isinstance(q, (int, float)):
                        bits.append(f"q = {q:.3f}")
                    st.markdown("- " + " · ".join(bits))
                else:
                    st.markdown(f"- {s}")
        st.info(
            "Headline reading: under correction, the robust effect of 4-bit quantization is "
            "**capability loss**, not a safety shift — observed safety changes are within noise at n≈200."
        )


# --------------------------------------------------------------------------- #
# Page: Results Explorer                                                       #
# --------------------------------------------------------------------------- #
def page_results() -> None:
    st.markdown(
        T.hero_html("Results Explorer",
                    dek="Judge-primary deltas with McNemar + FDR, the fp16→INT8→NF4 precision sweep, "
                        "scorer agreement, and multi-seed sensitivity."),
        unsafe_allow_html=True,
    )
    tabs = st.tabs(
        ["Judge ΔASR & capability", "Precision sweep", "Judge vs proxy", "Multi-seed sensitivity"]
    )

    # --- Tab 1: judge-primary deltas (from multiple_comparisons) ----------- #
    with tabs[0]:
        AXES = {
            "HarmBench ASR (judge)": "harmbench_asr_judge",
            "MMLU accuracy": "mmlu_accuracy",
            "ARC accuracy": "arc_accuracy",
            "XSTest over-refusal": "xstest_over_refusal",
        }
        choice = st.selectbox("Axis", list(AXES.keys()), index=0)
        mdf = D.mc_metric_df(REPO_ROOT, AXES[choice])
        if mdf.empty:
            st.warning("`multiple_comparisons.json` not found — run `scripts/multiple_comparisons.py`.")
        else:
            mdf = mdf.sort_values("delta")
            st.subheader(f"Δ {choice} (quantized − fp16) · McNemar + Benjamini–Hochberg FDR")
            # Colour: terracotta = nominally significant; ✦ marks FDR survival.
            colors = [T.TERRA if s else T.GREY for s in mdf["uncorrected_significant"]]
            text = []
            for _, r in mdf.iterrows():
                star = " ✓" if r.get("uncorrected_significant") else ""
                star += "✦" if r.get("bh_significant_q05") else ""
                text.append(f"{r['delta']:+.3f}{star}")
            figd = go.Figure(go.Bar(
                y=mdf["pair_id"], x=mdf["delta"], orientation="h",
                marker_color=colors, text=text, textposition="outside",
                customdata=mdf[["p_value", "bh_q_value"]],
                hovertemplate="%{y}<br>Δ=%{x:+.3f}<br>McNemar p=%{customdata[0]:.3f}<br>BH q=%{customdata[1]:.3f}<extra></extra>",
            ))
            figd.add_vline(x=0, line_dash="dash", line_color=T.MUTED)
            figd.update_layout(xaxis_title=f"Δ {choice}")
            st.plotly_chart(T.style_fig(figd, height=360), use_container_width=True)
            st.caption("✓ = nominally significant (McNemar p<0.05)  ·  ✦ = survives Benjamini–Hochberg FDR (q<0.05).")
            with st.expander("Raw contrasts (judge-primary)"):
                st.dataframe(mdf, use_container_width=True, hide_index=True)

        # v2 proxy demoted to a clearly-labelled secondary view.
        with st.expander("Secondary: v2 regex-proxy forest (over-counts ASR — not the headline)"):
            df = D.pairwise_df(REPO_ROOT)
            if df.empty:
                st.info("`pairwise_deltas.json` not found.")
            else:
                bench = st.selectbox("Benchmark (v2 proxy)", sorted(df["benchmark"].unique()),
                                     index=0, key="v2bench")
                sub = df[df["benchmark"] == bench].copy()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sub["baseline_value"], y=sub["pair_id"], mode="markers",
                    name="baseline (fp16)", marker=dict(size=12, color=T.INDIGO),
                    error_x=dict(type="data", array=sub["baseline_ci_upper"] - sub["baseline_value"],
                                 arrayminus=sub["baseline_value"] - sub["baseline_ci_lower"])))
                fig.add_trace(go.Scatter(
                    x=sub["quantized_value"], y=sub["pair_id"], mode="markers",
                    name="quantized", marker=dict(size=12, color=T.TERRA, symbol="diamond"),
                    error_x=dict(type="data", array=sub["quantized_ci_upper"] - sub["quantized_value"],
                                 arrayminus=sub["quantized_value"] - sub["quantized_ci_lower"])))
                fig.update_layout(xaxis_title=f"{sub['metric'].iloc[0]} (v2 proxy)", yaxis_title="",
                                  legend=dict(orientation="h", y=1.15))
                st.plotly_chart(T.style_fig(fig, height=340), use_container_width=True)
                st.caption(
                    "For HarmBench, these regex-proxy values disagree with the judge (e.g. Qwen-1.7B reads ~0.6 "
                    "here vs ~0.14 under the classifier) — the demonstration that *scorer choice changes the conclusion*."
                )

    # --- Tab 2: precision sweep -------------------------------------------- #
    with tabs[1]:
        sweep = D.precision_sweep_long(REPO_ROOT)
        if sweep.empty:
            st.warning("`precision_sweep.json` not found.")
        else:
            metrics = sorted(sweep["metric"].unique())
            default_metric = "harmbench_asr_judge" if "harmbench_asr_judge" in metrics else metrics[0]
            metric = st.selectbox("Metric", metrics, index=metrics.index(default_metric))
            sm = sweep[sweep["metric"] == metric].copy()
            order = ["fp16", "int8", "nf4"]
            sm["precision"] = pd.Categorical(sm["precision"], categories=order, ordered=True)
            sm = sm.sort_values(["pair_id", "precision"])
            st.subheader(f"{metric} across precisions (fp16 → INT8 → NF4)")
            fig = px.line(sm, x="precision", y="value", color="pair_id", markers=True,
                          color_discrete_sequence=list(T.COLORWAY))
            fig.update_traces(line=dict(width=2.5), marker=dict(size=9))
            st.plotly_chart(T.style_fig(fig, height=440), use_container_width=True)
            st.caption(
                "If the effect were bit-width-graded, lines would fall monotonically left→right. "
                "Capability metrics show a cliff at NF4; the safety axis is method-specific (two-peaked)."
            )

    # --- Tab 3: judge vs proxy -------------------------------------------- #
    with tabs[2]:
        ja = D.judge_agreement_df(REPO_ROOT)
        if ja.empty:
            st.warning("`judge_agreement.json` not found.")
        else:
            st.subheader("Scorer choice: regex proxy ASR vs validated judge ASR")
            melt = ja.melt(id_vars="model_alias", value_vars=["v2_asr", "judge_asr"],
                           var_name="scorer", value_name="ASR")
            melt["scorer"] = melt["scorer"].map({"v2_asr": "regex proxy (v2)", "judge_asr": "HarmBench classifier"})
            fig = px.bar(melt, x="model_alias", y="ASR", color="scorer", barmode="group",
                         color_discrete_map={"regex proxy (v2)": T.GOLD, "HarmBench classifier": T.INDIGO})
            fig.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(T.style_fig(fig, height=420), use_container_width=True)
            st.subheader("Agreement (Cohen's κ): regex vs judge")
            figk = px.bar(ja.sort_values("cohens_kappa"), x="cohens_kappa", y="model_alias", orientation="h")
            figk.update_traces(marker_color=T.INDIGO)
            figk.update_layout(xaxis_title="Cohen's κ")
            st.plotly_chart(T.style_fig(figk, height=420), use_container_width=True)
            st.caption("Low κ (Qwen) ⇒ the regex over-counts harm relative to the classifier; high κ (Llama) ⇒ they agree.")

    # --- Tab 4: multi-seed ------------------------------------------------- #
    with tabs[3]:
        sens = D.load_sensitivity(REPO_ROOT)
        if not sens:
            st.warning("`sensitivity_multiseed.json` not found.")
        else:
            rows = []
            for pp in sens.get("per_pair", []):
                jd = pp.get("judge_delta", {})
                rows.append({
                    "pair_id": pp.get("pair_id"),
                    "greedy_judge_delta": pp.get("greedy_judge_delta"),
                    "multiseed_mean": jd.get("mean"),
                    "multiseed_min": jd.get("min"),
                    "multiseed_max": jd.get("max"),
                    "sign_consistent": jd.get("sign_consistent"),
                })
            sdf = pd.DataFrame(rows)
            if not sdf.empty:
                st.subheader("Greedy headline vs multi-seed mean (judge ΔASR)")
                fig = go.Figure()
                fig.add_trace(go.Bar(name="greedy", x=sdf["pair_id"], y=sdf["greedy_judge_delta"], marker_color=T.TERRA))
                fig.add_trace(go.Bar(name="multi-seed mean", x=sdf["pair_id"], y=sdf["multiseed_mean"], marker_color=T.INDIGO))
                fig.update_layout(barmode="group")
                st.plotly_chart(T.style_fig(fig, height=420), use_container_width=True)
                st.dataframe(sdf, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Page: Add a Model                                                            #
# --------------------------------------------------------------------------- #
def page_add_model() -> None:
    st.markdown(
        T.hero_html("Add a Model",
                    dek="Define a new matched pair (baseline + quantized, loaded from identical weights). "
                        "Validated by the project's real Pydantic schema; emits a runnable config plus a "
                        "ready-to-submit sbatch. Committed configs are never modified."),
        unsafe_allow_html=True,
    )

    with st.form("add_model"):
        c1, c2 = st.columns(2)
        with c1:
            pair_id = st.text_input("pair_id (alias prefix)", value="gemma_2b",
                                    help="Members become <pair_id>_base and <pair_id>_4bit/_8bit").strip()
            family = st.text_input("family", value="gemma").strip()
            model_id = st.text_input("model_id (HF repo)", value="google/gemma-2-2b-it").strip()
            size_b = st.number_input("size_b (billions of params)", min_value=0.1, value=2.6, step=0.1)
        with c2:
            quant_method = st.selectbox("quant_method", ["nf4", "int8"], index=0)
            dtype = st.selectbox("dtype", ["auto", "float16", "bfloat16", "float32"], index=0)
            attn = st.selectbox("attn_implementation", ["(default)", "eager", "sdpa", "flash_attention_2"], index=0)
            trc = st.checkbox("trust_remote_code", value=False)
        benchmarks = st.multiselect("benchmarks", ["harmbench", "xstest", "mmlu", "arc"],
                                    default=["harmbench", "xstest", "mmlu", "arc"])
        out_name = st.text_input("Output config filename", value=f"{pair_id or 'new_pair'}.yaml").strip()
        submitted = st.form_submit_button("Validate & generate", type="primary")

    if not submitted:
        return

    if not all([pair_id, family, model_id]) or not benchmarks:
        st.error("pair_id, family, model_id and at least one benchmark are required.")
        return

    yaml_text, err = D.build_new_pair_config(
        base_config_path=config_path,
        pair_id=pair_id,
        family=family,
        size_b=float(size_b),
        model_id=model_id,
        quant_method=quant_method,
        benchmarks=benchmarks,
        attn_implementation=None if attn == "(default)" else attn,
        trust_remote_code=trc,
        dtype=dtype,
    )

    if err:
        st.error("Schema validation failed (the same check `fyp_cli.py` runs):")
        st.code(err)
        return

    st.success(f"Valid! Matched pair `{pair_id}_base` + `{pair_id}_{D.quant_suffix(quant_method)}` created.")
    st.code(yaml_text, language="yaml")

    gen_dir = (D.configs_dir(REPO_ROOT) / "generated").resolve()
    gen_dir.mkdir(parents=True, exist_ok=True)
    safe_name = D.safe_generated_config_name(out_name, fallback=f"{pair_id}.yaml")
    out_path = (gen_dir / safe_name).resolve()
    if gen_dir not in out_path.parents:
        st.error("Config filename escapes configs/generated/ — refused.")
        return

    colw = st.columns(2)
    with colw[0]:
        st.download_button("⬇️ Download config", yaml_text, file_name=out_path.name, mime="text/yaml")
    with colw[1]:
        if st.button("💾 Save to configs/generated/"):
            out_path.write_text(yaml_text, encoding="utf-8")
            st.success(f"Saved → `{out_path.relative_to(REPO_ROOT)}` (pick it in the sidebar to run it).")

    st.divider()
    st.subheader("Generate TC1 sbatch for this pair")
    st.caption("Writes per-model `*_matrix.sbatch` you can `sbatch` on the cluster.")
    if st.button("⚙️ Generate sbatch (cluster-generate)"):
        if not out_path.exists():
            out_path.write_text(yaml_text, encoding="utf-8")
        jobs_dir = REPO_ROOT / "slurm" / "jobs_generated"
        cmd = [
            sys.executable, str(REPO_ROOT / "fyp_cli.py"), "cluster-generate",
            "--config", str(out_path), "--jobs_dir", str(jobs_dir),
            "--group_by", "model", "--device", "cuda",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        if proc.returncode == 0:
            made = sorted(jobs_dir.glob(f"{pair_id}_*"))
            st.success(f"Generated {len(made)} sbatch file(s) under `slurm/jobs_generated/`.")
            for f in made:
                st.markdown(f"- `{f.relative_to(REPO_ROOT)}`")
        else:
            st.error("cluster-generate failed:")
            st.code((proc.stderr or proc.stdout)[-3000:])


# --------------------------------------------------------------------------- #
# Page: Run / Execute                                                          #
# --------------------------------------------------------------------------- #
def page_run() -> None:
    st.markdown(
        T.hero_html("Run / Execute",
                    dek="Launch a benchmark run from the GUI. On a machine without a GPU this is best used as a "
                        "small smoke run to prove the pipeline end-to-end; full runs (large n / big models) belong on TC1."),
        unsafe_allow_html=True,
    )

    models = D.read_config_models(config_path)
    if not models:
        st.warning(f"No models found in `{sel_config}`.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        alias = st.selectbox("Model", sorted(models.keys()))
        model_benches = models.get(alias, {}).get("benchmarks", ["harmbench"])
    with c2:
        benchmark = st.selectbox("Benchmark", sorted(model_benches) or ["harmbench"])
    with c3:
        mode = st.radio("Mode", ["Smoke (quick)", "Full run"], index=0)

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=1)
    with c5:
        n = st.number_input("max_samples", min_value=1, value=5 if mode.startswith("Smoke") else 200, step=1)
    with c6:
        batch = st.number_input("batch_size", min_value=1, value=2, step=1)
    with c7:
        seed = st.number_input("seed", min_value=0, value=42, step=1)

    st.warning(
        "Real model weights are downloaded on first run and generation can be slow on CPU. "
        "Gated models (Llama/Mistral/Phi) need a Hugging Face login. Keep `max_samples` small here.",
        icon="⚠️",
    )

    out_dir_str = st.text_input(
        "Execution output dir", value=D.DEFAULT_EXECUTION_DIR,
        help="Runs write here. The canonical evidence trees (results/, results_512/, "
             "results_sensitivity*/) are refused — they hold the study's immutable raw artifacts.",
    )
    if st.button("▶️ Run", type="primary"):
        try:
            exec_dir = D.resolve_execution_dir(out_dir_str, REPO_ROOT)
        except D.ProtectedResultsDirError as exc:
            st.error(str(exc))
            st.stop()
        sub = "smoke" if mode.startswith("Smoke") else "run"
        cmd = [
            sys.executable, str(REPO_ROOT / "fyp_cli.py"), sub,
            "--config", str(config_path), "--results_dir", str(exec_dir),
            "-m", alias, "-b", benchmark, "-n", str(int(n)),
            "--batch_size", str(int(batch)), "-s", str(int(seed)), "-d", device,
        ]
        st.code(" ".join(cmd), language="bash")

        log_box = st.empty()
        lines: list[str] = []
        with st.spinner("Running… (streaming logs)"):
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(REPO_ROOT),
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.append(line.rstrip())
                log_box.code("\n".join(lines[-400:]))
            proc.wait()

        if proc.returncode == 0:
            st.success("Run complete.")
            summ = D.run_summary(alias, benchmark, results_dir)
            if summ:
                st.subheader("Summary")
                st.json(summ)
            else:
                st.info("Run finished but no summary.json was found at the expected path.")
        else:
            st.error(f"Run failed (exit {proc.returncode}). See logs above.")


# --------------------------------------------------------------------------- #
# Page: Raw Results                                                            #
# --------------------------------------------------------------------------- #
def page_raw() -> None:
    st.markdown(
        T.hero_html("Raw Results", dek="Browse any finished run's summary.json."),
        unsafe_allow_html=True,
    )
    runs = D.available_runs(results_dir)
    if not runs:
        st.warning(f"No finished runs with a summary.json under `{results_dir_str}/`.")
        return
    aliases = sorted({a for a, _ in runs})
    alias = st.selectbox("Model", aliases)
    benches = sorted({b for a, b in runs if a == alias})
    benchmark = st.selectbox("Benchmark", benches)
    summ = D.run_summary(alias, benchmark, results_dir)
    if summ:
        flat = {k: v for k, v in summ.items() if isinstance(v, (int, float, str, bool))}
        if flat:
            st.dataframe(pd.DataFrame([flat]).T.rename(columns={0: "value"}), use_container_width=True)
        st.subheader("Full summary.json")
        st.json(summ)


# --------------------------------------------------------------------------- #
# Router                                                                       #
# --------------------------------------------------------------------------- #
ROUTES = {
    "Overview": page_overview,
    "Results Explorer": page_results,
    "Add a Model": page_add_model,
    "Run / Execute": page_run,
    "Raw Results": page_raw,
}
ROUTES[page]()
