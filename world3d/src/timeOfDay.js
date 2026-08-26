// Hour-of-day → full lighting/colour environment for the world.
// Pure function of `hour`, so any given timestamp looks identical every run.
import * as THREE from 'three';
import { smoothstep } from './noise.js';

const C = (hex) => new THREE.Color(hex);

const P = {
  dayTop: C('#144fa8'), dayMid: C('#4a90dd'), dayHor: C('#d6e6f4'),
  duskTop: C('#2e2a5e'), duskMid: C('#b45a80'), duskHor: C('#ff9e5e'),
  nightTop: C('#060915'), nightMid: C('#0e1730'), nightHor: C('#1f2a44'),
  sunHigh: C('#fff3d2'), sunLow: C('#ff9448'), sunNight: C('#b9c8ff'),
  hemiSkyDay: C('#bdd9ff'), hemiGndDay: C('#6a7550'),
  hemiSkyNight: C('#232e4c'), hemiGndNight: C('#1a2030'),
  fillDay: C('#8fb4d8'), fillNight: C('#2c3a55'),
};

export function computeEnvironment(hour) {
  const h = ((hour % 24) + 24) % 24;

  // Solar altitude: −1 midnight … +1 noon. Azimuth drifts through the day.
  const alt = Math.sin(((h - 6) / 12) * Math.PI);
  const az = (h / 24) * Math.PI * 2 + Math.PI * 0.18;

  const daylight = smoothstep(-0.06, 0.30, alt);       // 0 night → 1 full day
  const night = 1 - smoothstep(-0.16, 0.08, alt);      // 0 day → 1 deep night
  const dusk = Math.exp(-Math.pow(alt / 0.17, 2));     // bell around the horizon

  const top = P.nightTop.clone().lerp(P.dayTop, daylight).lerp(P.duskTop, dusk * 0.72);
  const mid = P.nightMid.clone().lerp(P.dayMid, daylight).lerp(P.duskMid, dusk * 0.82);
  const hor = P.nightHor.clone().lerp(P.dayHor, daylight).lerp(P.duskHor, dusk * 0.9);

  // Sun / moon directions on the unit sphere.
  const el = Math.min(1.32, Math.max(-0.55, alt)) * 1.05;
  const sunDir = new THREE.Vector3(
    Math.cos(el) * Math.sin(az),
    Math.sin(el),
    Math.cos(el) * Math.cos(az),
  ).normalize();
  const moonEl = Math.min(1.2, Math.max(-0.4, -alt)) * 0.95;
  const moonAz = az + Math.PI;
  const moonDir = new THREE.Vector3(
    Math.cos(moonEl) * Math.sin(moonAz),
    Math.sin(moonEl),
    Math.cos(moonEl) * Math.cos(moonAz),
  ).normalize();

  const lowSun = smoothstep(0.55, 0.02, alt);
  const sunColor = P.sunHigh.clone().lerp(P.sunLow, lowSun * (1 - night));
  const moonColor = P.sunNight.clone();

  const env = {
    hour: h,
    alt,
    daylight,
    night,
    dusk,
    top, mid, hor,
    sunDir, moonDir,
    sunColor,
    sunIntensity: 3.0 * daylight,
    moonColor,
    moonIntensity: 0.7 * night,
    fillIntensity: 0.22 + 0.5 * daylight,
    fillColor: P.fillNight.clone().lerp(P.fillDay, daylight),
    hemiSky: P.hemiSkyNight.clone().lerp(P.hemiSkyDay, daylight),
    hemiGround: P.hemiGndNight.clone().lerp(P.hemiGndDay, daylight),
    hemiIntensity: 0.8 + 0.9 * daylight,
    fogColor: hor.clone().lerp(mid, 0.5),
    exposure: 1.03 + dusk * 0.05 - night * 0.12,
    label: hourLabel(h),
  };
  return env;
}

export function hourLabel(h) {
  const table = [
    [4.2, 'Deep Night'], [6.0, 'Late Night'], [7.0, 'Dawn'], [8.5, 'Sunrise'],
    [11.5, 'Morning'], [13.5, 'High Sun'], [16.5, 'Afternoon'],
    [18.2, 'Golden Hour'], [19.6, 'Dusk'], [21.2, 'Night'], [24.01, 'Deep Night'],
  ];
  for (const [limit, name] of table) if (h < limit) return name;
  return 'Night';
}

export function formatClock(h) {
  const hh = Math.floor(h) % 24;
  const mm = Math.floor((h - Math.floor(h)) * 60);
  return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`;
}
