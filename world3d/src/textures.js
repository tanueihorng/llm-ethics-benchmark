// Procedural canvas textures — no network assets, all seeded & deterministic.
import * as THREE from 'three';
import { mulberry32 } from './noise.js';

export function radialTexture(size, stops) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  for (const [o, col] of stops) g.addColorStop(o, col);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function canvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

function tex(c, { repeat = true, srgb = true } = {}) {
  const t = new THREE.CanvasTexture(c);
  if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = 4;
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function shade(hex, v) {
  const n = parseInt(hex.slice(1), 16);
  const ch = (s) => Math.max(0, Math.min(255, ((n >> s) & 255) + v * 255));
  return `rgb(${ch(16)},${ch(8)},${ch(0)})`;
}

// ── building facade: day map + night emissive map (same window layout) ────
export function facadeTexture({
  cols = 8, rows = 24, base = '#9db0bc', glass = '#2e4250',
  glassVar = 0.16, litProb = 0.3, seed = 1, floorBand = true,
} = {}) {
  const W = 256, H = 256;
  const rng = mulberry32(seed * 7919 + 13);
  const day = canvas(W, H);
  const night = canvas(W, H);
  const d = day.getContext('2d');
  const n = night.getContext('2d');
  d.fillStyle = base;
  d.fillRect(0, 0, W, H);
  n.fillStyle = '#000';
  n.fillRect(0, 0, W, H);
  const cw = W / cols, ch = H / rows;
  for (let r = 0; r < rows; r++) {
    if (floorBand) {
      d.fillStyle = 'rgba(0,0,0,0.15)';
      d.fillRect(0, r * ch, W, ch * 0.18);
    }
    for (let c2 = 0; c2 < cols; c2++) {
      const x = c2 * cw + cw * 0.12;
      const y = r * ch + ch * 0.24;
      const w = cw * 0.76;
      const h = ch * 0.6;
      d.fillStyle = shade(glass, (rng() - 0.5) * 2 * glassVar);
      d.fillRect(x, y, w, h);
      if (rng() < 0.45) {
        const gr = d.createLinearGradient(x, y, x, y + h);
        gr.addColorStop(0, 'rgba(255,255,255,0.20)');
        gr.addColorStop(0.55, 'rgba(255,255,255,0)');
        d.fillStyle = gr;
        d.fillRect(x, y, w, h);
      }
      if (rng() < litProb) {
        n.globalAlpha = 0.45 + rng() * 0.55;
        n.fillStyle = rng() < 0.82 ? '#ffd9a0' : '#cfe4ff';
        n.fillRect(x, y, w, h);
        n.globalAlpha = 1;
      }
    }
  }
  return { map: tex(day), emissiveMap: tex(night, { srgb: false }), cols, rows };
}

// ── asphalt road with lane markings (repeats along its length) ────────────
export function roadTexture() {
  const W = 128, H = 256;
  const rng = mulberry32(5511);
  const c = canvas(W, H);
  const d = c.getContext('2d');
  d.fillStyle = '#2c2f33';
  d.fillRect(0, 0, W, H);
  for (let i = 0; i < 900; i++) {
    d.fillStyle = `rgba(${200 + rng() * 40},${200 + rng() * 40},${205 + rng() * 40},${rng() * 0.06})`;
    d.fillRect(rng() * W, rng() * H, 1.6, 1.6);
  }
  d.fillStyle = 'rgba(226,226,220,0.85)';
  d.fillRect(W * 0.06, 0, 3, H);          // edge lines
  d.fillRect(W * 0.94 - 3, 0, 3, H);
  d.fillStyle = 'rgba(214,196,110,0.9)';
  for (let y = 0; y < H; y += 64) d.fillRect(W / 2 - 2, y, 4, 34);  // centre dashes
  return tex(c);
}

// ── urban ground: concrete blocks, seams, grime ───────────────────────────
export function groundTexture() {
  const S = 512;
  const rng = mulberry32(9021);
  const c = canvas(S, S);
  const d = c.getContext('2d');
  d.fillStyle = '#686d61';
  d.fillRect(0, 0, S, S);
  for (let i = 0; i < 2600; i++) {
    d.fillStyle = `rgba(${40 + rng() * 60},${45 + rng() * 60},${40 + rng() * 50},${rng() * 0.10})`;
    d.fillRect(rng() * S, rng() * S, 2 + rng() * 5, 2 + rng() * 5);
  }
  d.strokeStyle = 'rgba(30,32,30,0.35)';
  d.lineWidth = 2;
  for (let i = 0; i <= 4; i++) {
    d.beginPath(); d.moveTo(i * S / 4, 0); d.lineTo(i * S / 4, S); d.stroke();
    d.beginPath(); d.moveTo(0, i * S / 4); d.lineTo(S, i * S / 4); d.stroke();
  }
  for (let i = 0; i < 26; i++) {
    d.fillStyle = `rgba(58,84,52,${0.10 + rng() * 0.16})`;
    d.beginPath();
    d.arc(rng() * S, rng() * S, 8 + rng() * 30, 0, Math.PI * 2);
    d.fill();
  }
  return tex(c);
}

// ── tileable water normal map (for the reflective bay) ────────────────────
export function waterNormals(size = 256) {
  const rng = mulberry32(772);
  const c = canvas(size, size);
  const d = c.getContext('2d');
  d.fillStyle = '#808080';
  d.fillRect(0, 0, size, size);
  for (let i = 0; i < 1400; i++) {
    const x = rng() * size, y = rng() * size, r = 4 + rng() * 22;
    const v = rng() < 0.5 ? 60 : 196;
    const g = d.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, `rgba(${v},${v},${v},${0.05 + rng() * 0.10})`);
    g.addColorStop(1, `rgba(${v},${v},${v},0)`);
    d.fillStyle = g;
    // draw 4 wrapped copies so the map tiles
    for (const [ox, oy] of [[0, 0], [size, 0], [0, size], [size, size]]) {
      d.beginPath();
      d.arc(x - ox, y - oy, r, 0, Math.PI * 2);
      d.fill();
    }
  }
  const img = d.getImageData(0, 0, size, size);
  const out = d.createImageData(size, size);
  const at = (x, y) => img.data[(((y + size) % size) * size + ((x + size) % size)) * 4];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (at(x + 1, y) - at(x - 1, y)) / 255;
      const dy = (at(x, y + 1) - at(x, y - 1)) / 255;
      const inv = 1 / Math.hypot(dx * 2.2, dy * 2.2, 1);
      const o = (y * size + x) * 4;
      out.data[o] = (-dx * 2.2 * inv * 0.5 + 0.5) * 255;
      out.data[o + 1] = (-dy * 2.2 * inv * 0.5 + 0.5) * 255;
      out.data[o + 2] = inv * 255;
      out.data[o + 3] = 255;
    }
  }
  d.putImageData(out, 0, 0);
  return tex(c, { srgb: false });
}

// ── supertree lattice: diagonal grid, glowing at night ────────────────────
export function latticeTexture() {
  const S = 128;
  const day = canvas(S, S);
  const night = canvas(S, S);
  const d = day.getContext('2d');
  const n = night.getContext('2d');
  d.fillStyle = '#5a4a3a';
  d.fillRect(0, 0, S, S);
  n.fillStyle = '#000';
  n.fillRect(0, 0, S, S);
  const line = (ctx, col, w) => {
    ctx.strokeStyle = col;
    ctx.lineWidth = w;
    for (let i = -S; i <= S * 2; i += 16) {
      ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i + S, S); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(i + S, 0); ctx.lineTo(i, S); ctx.stroke();
    }
  };
  line(d, 'rgba(28,22,16,0.9)', 3);
  line(d, 'rgba(120,100,80,0.5)', 1.2);
  line(n, 'rgba(53,224,160,0.95)', 2.4);
  return { map: tex(day), emissiveMap: tex(night, { srgb: false }) };
}
