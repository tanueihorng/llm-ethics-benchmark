// Deterministic noise utilities — no Math.random anywhere near world generation,
// so the layout is identical on every load (and the e2e assertions hold).

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function ihash(x, y, s) {
  let h = (x * 374761393 + y * 668265263 + s * 974634211) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

const smooth = (t) => t * t * (3 - 2 * t);

export function value2(x, y, seed = 0) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const u = smooth(x - xi), v = smooth(y - yi);
  const a = ihash(xi, yi, seed);
  const b = ihash(xi + 1, yi, seed);
  const c = ihash(xi, yi + 1, seed);
  const d = ihash(xi + 1, yi + 1, seed);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

export function fbm(x, y, seed = 0, oct = 4, lac = 2.02, gain = 0.5) {
  let f = 0, amp = 0.5, fr = 1, norm = 0;
  for (let i = 0; i < oct; i++) {
    f += amp * value2(x * fr, y * fr, seed + i * 101);
    norm += amp; fr *= lac; amp *= gain;
  }
  return f / norm;
}

// Ridged variant — produces sharp mountain crests.
export function ridge(x, y, seed = 0, oct = 4) {
  let f = 0, amp = 0.5, fr = 1, norm = 0;
  for (let i = 0; i < oct; i++) {
    const n = 1 - Math.abs(value2(x * fr, y * fr, seed + i * 77) * 2 - 1);
    f += amp * n * n; norm += amp; fr *= 2.07; amp *= 0.52;
  }
  return f / norm;
}

export const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
export const smoothstep = (a, b, v) => {
  const t = clamp01((v - a) / (b - a));
  return t * t * (3 - 2 * t);
};
export const lerp = (a, b, t) => a + (b - a) * t;
export const easeInOutCubic = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
