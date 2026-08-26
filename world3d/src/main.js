// AETHERIA — bootstrap, render loop, quality control and the debug/test API.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

import { buildWorld } from './world.js';
import { buildProps } from './props.js';
import { buildLife } from './life.js';
import { buildLightShow } from './lights.js';
import { buildQuest } from './quest.js';
import { SkySystem } from './sky.js';
import { Sky } from 'three/addons/objects/Sky.js';
import { computeEnvironment } from './timeOfDay.js';
import { Interactor } from './interactions.js';
import { initUI } from './ui.js';
import { codexHTML, CHRONICLE } from './lore.js';
import { POIS } from './pois.js';

const nextFrame = () => new Promise((r) => requestAnimationFrame(r));
const params = new URLSearchParams(location.search);

async function main() {
  // ── renderer / scene / camera ────────────────────────────────────
  let quality = params.get('quality') === 'low' ? 'low' : 'high';
  const renderer = new THREE.WebGLRenderer({ antialias: quality === 'high', powerPreference: 'high-performance' });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(quality === 'high' ? Math.min(window.devicePixelRatio, 2) : 1);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  document.getElementById('app').appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x9db8c8, 900, 2600);

  const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.5, 6000);
  camera.position.set(430, 310, 760);   // intro flight starts far out

  const ui = initUI({ pois: POIS });
  ui.loaderProgress(0.05, 'waking the sky');
  await nextFrame();

  // ── lights ───────────────────────────────────────────────────────
  const sun = new THREE.DirectionalLight(0xffffff, 2.4);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = -360; sun.shadow.camera.right = 360;
  sun.shadow.camera.top = 360; sun.shadow.camera.bottom = -360;
  sun.shadow.camera.near = 100; sun.shadow.camera.far = 2000;
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.03;

  // ── image-based reflections from the sky, applied to glass only ────
  const pmrem = new THREE.PMREMGenerator(renderer);
  let envRT = null;
  let lastBakeHour = -99;
  function bakeEnvironment(hour) {
    if (quality !== 'high') {
      envRT?.dispose();
      envRT = null;
      scene.traverse((o) => {
        if (o.material && o.material.isMeshStandardMaterial) o.material.envMap = null;
      });
      return;
    }
    const tmp = new THREE.Scene();
    const s = new Sky();
    s.scale.setScalar(19000);
    const u = s.material.uniforms;
    const e = computeEnvironment(((hour % 24) + 24) % 24);
    u.sunPosition.value.copy(e.sunDir);
    u.turbidity.value = 2.2 + e.dusk * 6.5;
    u.rayleigh.value = 3.0 + e.dusk * 1.6;
    u.mieCoefficient.value = 0.0012;
    u.mieDirectionalG.value = 0.85;
    tmp.add(s);
    const rt = pmrem.fromScene(tmp, 0, 0.1, 19000);
    if (envRT) envRT.dispose();
    envRT = rt;
    // the Sky addon keeps a twilight halo even with the sun well below the
    // horizon — fade reflections out through the night so nothing washes tan
    const envScale = 1 - (e.night ?? 0) * 0.94;
    scene.traverse((o) => {
      const m = o.material;
      if (m && m.isMeshStandardMaterial) {
        m.envMap = rt.texture;
        m.envMapIntensity = (m.userData.glassEnv ?? 0.14) * envScale;
      }
    });
  }
  scene.add(sun, sun.target);
  const moonLight = new THREE.DirectionalLight(0xb9c8ff, 0.0);
  scene.add(moonLight);
  // cool fill from the anti-solar side keeps shadowed faces readable
  const fillLight = new THREE.DirectionalLight(0x8fb4d8, 0.4);
  scene.add(fillLight);
  const hemi = new THREE.HemisphereLight(0x9db8dd, 0x3a3f2f, 0.8);
  scene.add(hemi);

  // ── world (staged so the loader bar means something) ─────────────
  const world = buildWorld(scene, { reflections: quality === 'high' });
  const { updaters, nightReg } = world;
  ui.loaderProgress(0.35, 'laying the streets');
  await nextFrame();

  const sky = new SkySystem(scene);
  ui.loaderProgress(0.5, 'raising the skyline');
  await nextFrame();

  const { anchors } = buildProps(scene, { updaters, nightReg });
  buildLife(scene, { updaters, nightReg });
  const show = buildLightShow(scene, { anchors, updaters });
  const quest = buildQuest(scene, {
    updaters,
    onAllCollected: () => ui.showToast('✦ All 12 probes resolved — across the whole city, no confirmed safety shift. The null holds.'),
  });
  quest.bindCamera(camera);
  ui.loaderProgress(0.75, 'charting landmarks');
  await nextFrame();

  // ── landmark hit-spheres at fixed city coordinates ───────────────
  const hitMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false });
  const pois = POIS.map((def) => {
    const pos = new THREE.Vector3(...def.pos);
    const hitMeshes = (def.hits ?? [[...def.pos, 20]]).map(([x, y, z, r]) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(r, 10, 8), hitMat);
      m.position.set(x, y, z);
      scene.add(m);
      return m;
    });
    return { ...def, pos, hit: hitMeshes[0], hitMeshes };
  });

  // ── post-processing ──────────────────────────────────────────────
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight), 0.45, 0.4, 1.6,
  );
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  // ── interaction ──────────────────────────────────────────────────
  const interactor = new Interactor({ canvas: renderer.domElement, camera, renderer, pois });
  interactor.controls.target.set(0, 30, 40);

  // ── state ────────────────────────────────────────────────────────
  const state = {
    hour: params.has('hour') ? parseFloat(params.get('hour')) : 8.2,
    autoCycle: params.get('cycle') !== '0',
    speed: 24 / 150,   // one full day per 2.5 minutes
    quality,
  };

  // ── environment application ──────────────────────────────────────
  function applyEnv(env) {
    sun.position.copy(env.sunDir).multiplyScalar(850);
    sun.intensity = env.sunIntensity;
    sun.color.copy(env.sunColor);
    moonLight.position.copy(env.moonDir).multiplyScalar(850);
    moonLight.intensity = env.moonIntensity;
    fillLight.position.copy(env.sunDir).multiplyScalar(-140);
    fillLight.position.y = Math.abs(fillLight.position.y) * 0.4 + 30; // keep fill slightly above
    fillLight.intensity = env.fillIntensity;
    fillLight.color.copy(env.fillColor);
    hemi.color.copy(env.hemiSky);
    hemi.groundColor.copy(env.hemiGround);
    hemi.intensity = env.hemiIntensity;
    scene.fog.color.copy(env.fogColor);
    renderer.setClearColor(env.fogColor, 1);
    renderer.toneMappingExposure = env.exposure * 0.9;
    bloom.enabled = (env.night ?? 0) > 0.02 || (env.dusk ?? 0) > 0.25;
    for (const r of nightReg) r.mat[r.prop] = r.day + (r.night - r.day) * env.night;
  }

  // ── UI handlers ──────────────────────────────────────────────────
  const h = ui._handlers;
  h.seek = (v) => { state.hour = ((v % 24) + 24) % 24; state.autoCycle = false; ui.setPlayIcon(false); };
  h.togglePlay = () => {
    state.autoCycle = !state.autoCycle;
    ui.setPlayIcon(state.autoCycle);
  };
  h.travel = (i) => interactor.flyTo(i);
  h.close = () => ui.closeCard();
  h.setQuality = (q) => setQuality(q);

  interactor.onHover = (i) => ui.showTooltip(i);
  // each landmark card carries its model pair's real measured deltas (FYP codex)
  const codexFor = (i) => codexHTML(pois[i] && pois[i].id);
  ui.setChronicle(CHRONICLE);
  interactor.onSelect = (i) => {
    if (i >= 0) {
      ui.openCard(i, codexFor(i));
      interactor.flyTo(i);
    }
  };

  window.addEventListener('keydown', (e) => {
    const tag = (e.target && e.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (e.code === 'Space') { e.preventDefault(); h.togglePlay(); return; }
    if (e.key >= '1' && e.key <= String(pois.length)) {
      const i = parseInt(e.key, 10) - 1;
      ui.openCard(i, codexFor(i));
      interactor.flyTo(i);
      return;
    }
    if (e.key === 'h' || e.key === 'H') document.body.classList.toggle('ui-hidden');
    if (e.key === 'Escape') { ui.closeCard(); document.getElementById('help').classList.add('hidden'); }
  });

  function setQuality(q) {
    state.quality = q;
    quality = q;
    renderer.setPixelRatio(q === 'high' ? Math.min(window.devicePixelRatio, 2) : 1);
    sun.castShadow = q === 'high';
    lastBakeHour = -99;   // force an environment rebake (or clear) on next frame
    ui.setQualityValue(q);
  }
  setQuality(quality);
  ui.setPlayIcon(state.autoCycle);

  // ── resize ───────────────────────────────────────────────────────
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    composer.setSize(window.innerWidth, window.innerHeight);
    sky.resize(window.innerHeight);
  });

  // ── render helpers (also used by the test API) ───────────────────
  function doRender() {
    if (quality === 'high') composer.render();
    else renderer.render(scene, camera);
  }

  function captureStats() {
    doRender();
    const w = 220, hh = 124;
    const cv = document.createElement('canvas');
    cv.width = w; cv.height = hh;
    const ctx = cv.getContext('2d');
    ctx.drawImage(renderer.domElement, 0, 0, w, hh);
    const data = ctx.getImageData(0, 0, w, hh).data;
    const distinct = new Set();
    let lumSum = 0, black = 0;
    const n = w * hh;
    for (let i = 0; i < n; i++) {
      const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
      distinct.add(((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4));
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      lumSum += lum;
      if (lum < 8) black++;
    }
    return { distinct: distinct.size, meanLum: lumSum / n, blackRatio: black / n };
  }

  function captureDataURL() {
    doRender();
    return renderer.domElement.toDataURL('image/png');
  }

  // ── debug / e2e API ──────────────────────────────────────────────
  window.__WORLD__ = {
    version: '1.0.0',
    get ready() { return ready; },
    get quality() { return state.quality; },
    state,
    scene,
    camera,
    controls: interactor.controls,
    renderer,
    pois: pois.map(({ id, name, sub, pos }) => ({ id, name, sub, pos: pos.clone() })),
    setHour(v) { state.hour = ((parseFloat(v) % 24) + 24) % 24; },
    getHour() { return state.hour; },
    setAutoCycle(b) { state.autoCycle = !!b; ui.setPlayIcon(state.autoCycle); },
    flyTo(i) { interactor.flyTo(i); },
    openCard(i) { ui.openCard(i, codexHTML(pois[i] && pois[i].id)); },
    closeCard() { ui.closeCard(); },
    cardOpen() { return ui.cardOpen(); },
    isFlying() { return interactor.isFlying(); },
    setQuality(q) { setQuality(q); },
    lightShowIntensity() { return show.getIntensity(); },
    quest: {
      total: quest.total,
      collected: quest.collected,
      positions: quest.positions.map((p) => p.toArray()),
    },
    captureStats,
    captureDataURL,
  };

  // ── main loop ────────────────────────────────────────────────────
  const clock = new THREE.Clock();
  let t = 0;
  let ready = false;

  ui.loaderProgress(1, 'welcome to Aetheria');
  await nextFrame();
  await nextFrame();
  ui.hideLoader();
  // e2e runs pass ?intro=0 to skip the cinematic so tests start from a settled camera
  if (params.get('intro') !== '0') {
    interactor.startFlight(new THREE.Vector3(230, 160, 540), new THREE.Vector3(0, 30, 60), 2.4);
  }

  let lastReadout = '';
  renderer.setAnimationLoop(() => {
    const dt = Math.min(clock.getDelta(), 0.1);
    t += dt;
    if (state.autoCycle) {
      state.hour = (state.hour + dt * state.speed) % 24;
    }
    const env = computeEnvironment(state.hour);
    applyEnv(env);
    const hNow = ((state.hour % 24) + 24) % 24;
    if (Math.abs(hNow - lastBakeHour) > 1) {
      lastBakeHour = hNow;
      bakeEnvironment(hNow);
    }
    for (const u of updaters) u.update(t, dt, env);
    sky.setTime(env, t, renderer.getPixelRatio());
    interactor.update(dt);
    interactor.updateHover();

    const ro = `${Math.round(state.hour * 20)}:${env.label}`;
    if (ro !== lastReadout) { lastReadout = ro; ui.setTimeReadout(state.hour, env.label); }
    ui.setShards(quest.collected(), quest.total);

    doRender();
    if (!ready) { ready = true; }
  });
}

main().catch((err) => {
  console.error(err);
  const st = document.getElementById('loader-status');
  if (st) st.textContent = `failed to start: ${err.message}`;
});
