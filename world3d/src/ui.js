// DOM layer: dock controls, landmark card, tooltip, help modal, loader.
import { formatClock } from './timeOfDay.js';

export function initUI({ pois }) {
  const $ = (id) => document.getElementById(id);
  const els = {
    brand: $('brand'), dock: $('dock'), hint: $('hint'),
    card: $('poi-card'), title: $('card-title'), sub: $('card-sub'),
    body: $('card-body'), codex: $('card-codex'), keycap: $('card-key'), travel: $('card-travel'),
    closeCard: $('card-close'),
    slider: $('time-slider'), readout: $('time-readout'),
    play: $('btn-play'), quality: $('btn-quality'),
    help: $('help'), helpBtn: $('btn-help'), helpClose: $('help-close'),
    tooltip: $('tooltip'), ttName: $('tt-name'),
    loader: $('loader'), bar: $('loader-bar'), status: $('loader-status'),
    shardHud: $('shard-hud'), shardCount: $('shard-count'), shardTotal: $('shard-total'),
    toast: $('toast'),
  };

  let currentCard = -1;
  const handlers = {
    seek: () => {}, togglePlay: () => {}, setQuality: () => {},
    travel: () => {}, close: () => {},
  };

  // ── tooltip ──
  window.addEventListener('pointermove', (e) => {
    els.tooltip.style.left = `${e.clientX}px`;
    els.tooltip.style.top = `${e.clientY}px`;
  });

  function showTooltip(i) {
    if (i == null || i < 0 || document.body.classList.contains('ui-hidden')) {
      els.tooltip.classList.add('hidden');
      return;
    }
    els.ttName.textContent = pois[i].name;
    els.tooltip.classList.remove('hidden');
  }

  // ── landmark card ──
  function openCard(i, codexHTML = '') {
    const p = pois[i];
    if (!p) return;
    currentCard = i;
    els.title.textContent = p.name;
    els.sub.textContent = p.sub;
    els.body.textContent = p.body;
    els.codex.innerHTML = codexHTML;
    els.codex.style.display = codexHTML ? '' : 'none';
    els.keycap.textContent = String(i + 1);
    els.card.classList.remove('hidden');
  }
  function closeCard() {
    currentCard = -1;
    els.card.classList.add('hidden');
  }
  const cardOpen = () => currentCard >= 0;

  // ── dock ──
  els.slider.addEventListener('input', () => {
    handlers.seek(parseFloat(els.slider.value));
  });
  els.play.addEventListener('click', () => handlers.togglePlay());
  els.quality.addEventListener('change', () => handlers.setQuality(els.quality.value));
  els.travel.addEventListener('click', () => { if (currentCard >= 0) handlers.travel(currentCard); });
  els.closeCard.addEventListener('click', () => handlers.close());
  els.helpBtn.addEventListener('click', () => els.help.classList.toggle('hidden'));
  els.helpClose.addEventListener('click', () => els.help.classList.add('hidden'));

  function setPlayIcon(cycleRunning) {
    els.play.textContent = cycleRunning ? '❚❚' : '▶';
    els.play.title = cycleRunning ? 'Pause the day cycle (Space)' : 'Resume the day cycle (Space)';
  }
  function setTimeReadout(hour, label) {
    const txt = `${formatClock(hour)} · ${label}`;
    if (els.readout.textContent !== txt) els.readout.textContent = txt;
    const v = String(Math.round(hour * 20) / 20);
    if (els.slider.value !== v && document.activeElement !== els.slider) els.slider.value = v;
  }
  function setQualityValue(q) { els.quality.value = q; }

  // ── quest HUD + toast ──
  let lastShardText = '';
  function setShards(collected, total) {
    els.shardTotal.textContent = String(total);
    const txt = `${collected}/${total}`;
    if (txt !== lastShardText) {
      lastShardText = txt;
      els.shardCount.textContent = String(collected);
      if (collected === total) els.shardHud.classList.add('complete');
    }
  }

  let toastTimer = null;
  function showToast(text) {
    els.toast.textContent = text;
    els.toast.classList.remove('hidden');
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => els.toast.classList.add('hidden'), 4500);
  }

  // ── chronicle (FYP meaning layer, rendered once at boot) ──
  function setChronicle(html) { $('chronicle').innerHTML = html; }

  // ── loader ──
  function loaderProgress(frac, statusText) {
    els.bar.style.width = `${Math.round(frac * 100)}%`;
    if (statusText) els.status.textContent = statusText;
  }
  function hideLoader() {
    els.loader.classList.add('hidden');
    setTimeout(() => els.loader.remove(), 1100);
  }

  return {
    showTooltip, openCard, closeCard, cardOpen,
    setPlayIcon, setTimeReadout, setQualityValue,
    setShards, showToast, setChronicle,
    loaderProgress, hideLoader,
    _els: els,
    _handlers: handlers,
  };
}
