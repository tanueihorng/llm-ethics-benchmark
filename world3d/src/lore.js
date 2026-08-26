// ── The FYP Codex ────────────────────────────────────────────────────────────
// Every number below was transcribed from the project's primary analysis
// artifacts (512-token primary tree, judge-primary HarmBench ASR,
// Benjamini–Hochberg FDR q = 0.05):
//   results_512/analysis/multiple_comparisons.json   (deltas, p, q, BH flags)
//   results_512/analysis/judge_agreement.json        (canonical per-pair labels)
//   results_512/analysis/pairwise_deltas.json        (baseline/quant values)
// Do not edit a number here without re-reading those files.

const LABEL_STYLE = {
  robust_preservation: { name: 'Robust Preservation', color: '#7fd8a0' },
  alignment_improvement: { name: 'Alignment Improvement', color: '#7fb8ff' },
  alignment_degradation: { name: 'Alignment Degradation', color: '#e8b07a' },
  capability_collapse_masquerading_as_safety: { name: 'Capability Collapse Masquerading as Safety', color: '#e88a8a' },
  broad_degradation: { name: 'Broad Degradation', color: '#b9a8d8' },
};

const pp = (d) => `${d > 0 ? '+' : d < 0 ? '−' : ''}${Math.abs(d * 100).toFixed(1)} pp`;

export const LORE = {
  mbs: {
    pair: 'qwen_2b', pairName: 'Qwen3-1.7B', modelId: 'Qwen/Qwen3-1.7B',
    deltas: { asr: 0.0, orr: -0.024, mmlu: -0.09, arc: -0.0094 },
    sig: { asr: false, orr: false, mmlu: true, arc: false },
    label: 'broad_degradation',
    story: 'The icon that never changed its light: quantization moved harmful compliance by exactly +0.0 pp. But the skyline\'s steepest capability cost hides in the towers — MMLU −9.0 pp, the study\'s largest confirmed drop.',
  },
  flyer: {
    pair: 'mistral_7b', pairName: 'Mistral-7B', modelId: 'mistralai/Mistral-7B-Instruct-v0.3',
    deltas: { asr: -0.02, orr: 0.0, mmlu: -0.02, arc: 0.0094 },
    sig: { asr: false, orr: false, mmlu: false, arc: false },
    label: 'alignment_improvement',
    story: 'From the top of the wheel everything looks calm: harm −2.0 pp, over-refusal exactly 0.0, capability ~flat. The gentlest reading in the study — though it stays shy of significance.',
  },
  merlion: {
    pair: 'llama_3_2_3b', pairName: 'Llama-3.2-3B', modelId: 'meta-llama/Llama-3.2-3B-Instruct',
    deltas: { asr: -0.04, orr: 0.016, mmlu: -0.0367, arc: -0.0316 },
    sig: { asr: false, orr: false, mmlu: false, arc: true },
    label: 'capability_collapse_masquerading_as_safety',
    story: 'The spring runs weaker than it looks: ARC −3.2 pp (FDR-confirmed) while harm fell too. What pours away is capability — a safer-seeming model that may simply be a weaker one. RQ2, written in water.',
  },
  gardens: {
    pair: 'phi4_mini', pairName: 'Phi-4-mini', modelId: 'microsoft/Phi-4-mini-instruct',
    deltas: { asr: 0.02, orr: -0.048, mmlu: -0.0267, arc: -0.0145 },
    sig: { asr: false, orr: true, mmlu: false, arc: false },
    label: 'alignment_degradation',
    story: 'The rhapsody depends on the lighting: the one FDR-surviving over-refusal contrast (−4.8 pp under the regex) flips to nothing under an independent judge. Trust the lens, not the glow.',
  },
  museum: {
    pair: 'qwen_4b', pairName: 'Qwen3-4B', modelId: 'Qwen/Qwen3-4B',
    deltas: { asr: 0.04, orr: -0.004, mmlu: -0.0033, arc: -0.0162 },
    sig: { asr: false, orr: false, mmlu: false, arc: false },
    label: 'alignment_degradation',
    story: 'Where art meets science: ASR +4.0 pp looks like a warning, but science says not yet proven (p = 0.096, fades under FDR). Capability held almost perfectly still.',
  },
};

export function codexHTML(key) {
  const L = LORE[key];
  if (!L) return '';
  const st = LABEL_STYLE[L.label];
  const cell = (name, d, sig) =>
    `<span class="cx-metric"><b>${name}</b> ${pp(d)}${sig ? '<i class="cx-sig" title="survives Benjamini–Hochberg FDR (q < 0.05)">†</i>' : ''}</span>`;
  return `
  <div class="codex">
    <div class="cx-head">FYP CODEX — ${L.pairName.toUpperCase()} · fp16 → NF4</div>
    <div class="cx-grid">
      ${cell('ΔASR', L.deltas.asr, L.sig.asr)}
      ${cell('ΔOver-refusal', L.deltas.orr, L.sig.orr)}
      ${cell('ΔMMLU', L.deltas.mmlu, L.sig.mmlu)}
      ${cell('ΔARC', L.deltas.arc, L.sig.arc)}
    </div>
    <div class="cx-verdict" style="color:${st.color};border-color:${st.color}44">${st.name}</div>
    <div class="cx-story">${L.story}</div>
    <div class="cx-src">${L.modelId} · results_512/analysis · judge-primary ASR · BH-FDR q=0.05 († survives)</div>
  </div>`;
}

export const CHRONICLE = `
  <h3>The Chronicle of the Quantized City</h3>
  <p>This city is the study, drawn at sky scale. Each landmark is one matched
  model pair (fp16 baseline vs NF4 4-bit); its card carries that pair's measured
  deltas from the 512-token primary analysis.</p>
  <table class="ch-table">
    <tr><th>Landmark</th><th>Pair</th><th>Verdict</th></tr>
    <tr><td>Marina Bay Sands</td><td>Qwen3-1.7B</td><td>Broad degradation (MMLU −9.0 pp†, ASR +0.0)</td></tr>
    <tr><td>Singapore Flyer</td><td>Mistral-7B</td><td>Alignment improvement, unconfirmed (ASR −2.0 pp)</td></tr>
    <tr><td>The Merlion</td><td>Llama-3.2-3B</td><td>Capability collapse masquerading as safety (ARC −3.2 pp†)</td></tr>
    <tr><td>Gardens by the Bay</td><td>Phi-4-mini</td><td>Alignment degradation, directional (ASR +2.0 pp); ORR survivor is scorer-dependent</td></tr>
    <tr><td>ArtScience Museum</td><td>Qwen3-4B</td><td>Alignment degradation, unconfirmed (ASR +4.0 pp, p=.096)</td></tr>
  </table>
  <p><b>What the city proves together.</b> Across all twenty primary contrasts
  (5 pairs × 4 metrics), <b>no harmful-compliance (ASR) change survives FDR
  correction</b> — the RQ1 null. The only FDR-confirmed shifts are two capability
  drops (Qwen-1.7B MMLU −9.0 pp, Llama ARC −3.2 pp) and one over-refusal
  <i>improvement</i> (Phi −4.8 pp) that an independent scorer cannot confirm.
  Where safety seemed to shift, capability explains it: that is RQ2, written in
  the Merlion's waterfall.</p>
  <p><b>Humility of the instruments.</b> At n=200 the ASR power floor is roughly
  ±5.9 pp (McNemar MDE at the median discordant rate) — nulls here are bounded by
  power, not proof of zero. Phi's over-refusal survivor flips sign under the
  judge-strict rescoring. An INT8 precision point was evaluated separately.
  † marks contrasts surviving Benjamini–Hochberg FDR (q &lt; 0.05).</p>
  <p class="ch-src">Sources (read, not recalled): results_512/analysis/multiple_comparisons.json ·
  judge_agreement.json · pairwise_deltas.json — judge-primary ASR (D16/D41).</p>
`;
