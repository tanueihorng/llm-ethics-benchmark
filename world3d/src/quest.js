// Evaluation-probe quest: 12 deterministic collectibles placed around
// Marina Bay. Fly close to collect — sparkle burst, HUD counter, and the
// study's conclusion as the completion toast.
import * as THREE from 'three';
import { mulberry32 } from './noise.js';
import { radialTexture } from './textures.js';

const PICKUP_RADIUS = 5.5;

const PROBE_SPOTS = [
  [35, 215, 205],     // above the MBS sky park
  [10, 26, 138],      // in front of the lotus
  [-88, 24, 88],      // over the Merlion's jet
  [250, 112, -55],    // at the Flyer's hub
  [230, 62, 190],     // atop the tallest supertree
  [172, 30, 232],     // above the conservatories
  [85, 22, 60],       // mid-Helix Bridge
  [-110, 34, 120],    // above the Esplanade
  [-180, 115, -150],  // among the CBD towers
  [-330, 45, -200],   // Fort Canning hilltop
  [0, 14, 0],         // centre of the bay
  [300, 26, 320],     // south-east shore
];

export function buildQuest(scene, { updaters, onAllCollected }) {
  const rng = mulberry32(99173);

  // ── visuals ──
  const shardGeo = new THREE.OctahedronGeometry(0.9, 0);
  shardGeo.scale(1, 1.5, 1);
  const shardMat = new THREE.MeshStandardMaterial({
    color: 0x8a6a2a, emissive: 0xffd27a, emissiveIntensity: 1.6,
    roughness: 0.3, metalness: 0.3, flatShading: true,
  });
  const glowTex = radialTexture(64, [
    [0, 'rgba(255,224,150,1)'], [0.4, 'rgba(255,200,110,0.45)'], [1, 'rgba(255,190,90,0)'],
  ]);

  const shards = PROBE_SPOTS.map((p, i) => {
    const pos = new THREE.Vector3(...p);
    const grp = new THREE.Group();
    grp.position.copy(pos);
    const core = new THREE.Mesh(shardGeo, shardMat);
    grp.add(core);
    const halo = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex, color: 0xffd27a, transparent: true, opacity: 0.5,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    halo.scale.setScalar(5.5);
    grp.add(halo);
    scene.add(grp);
    return { grp, core, halo, pos, phase: rng() * 10, collected: false };
  });

  // ── collect bursts (pooled particle pops) ──
  const BURSTS = 4, PER = 22;
  const burstPos = new Float32Array(BURSTS * PER * 3);
  const burstVel = new Float32Array(BURSTS * PER * 3);
  const burstLife = new Array(BURSTS).fill(0);
  const burstGeo = new THREE.BufferGeometry();
  burstGeo.setAttribute('position', new THREE.BufferAttribute(burstPos, 3));
  const burstMat = new THREE.PointsMaterial({
    map: glowTex, color: 0xffd98a, size: 2.2, transparent: true, opacity: 0.95,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const burstPoints = new THREE.Points(burstGeo, burstMat);
  burstPoints.frustumCulled = false;
  scene.add(burstPoints);
  const fireBurst = (at) => {
    const b = burstLife.findIndex((l) => l <= 0);
    if (b < 0) return;
    burstLife[b] = 0.85;
    for (let k = 0; k < PER; k++) {
      burstPos[b * PER * 3 + k * 3] = at.x;
      burstPos[b * PER * 3 + k * 3 + 1] = at.y;
      burstPos[b * PER * 3 + k * 3 + 2] = at.z;
      const th = rng() * Math.PI * 2, ph = Math.acos(2 * rng() - 1), sp = 6 + rng() * 9;
      burstVel[b * PER * 3 + k * 3] = Math.sin(ph) * Math.cos(th) * sp;
      burstVel[b * PER * 3 + k * 3 + 1] = Math.cos(ph) * sp + 4;
      burstVel[b * PER * 3 + k * 3 + 2] = Math.sin(ph) * Math.sin(th) * sp;
    }
  };

  let count = 0;
  const collected = () => count;

  let cameraRef = new THREE.Vector3();
  updaters.push({
    update(t, dt) {
      for (const s of shards) {
        if (s.collected) continue;
        s.grp.position.y = s.pos.y + Math.sin(t * 1.4 + s.phase) * 0.8;
        s.core.rotation.y = t * 1.3 + s.phase;
        s.halo.material.opacity = 0.38 + Math.sin(t * 2.2 + s.phase) * 0.14;
        if (s.grp.position.distanceTo(cameraRef.position ?? cameraRef) < PICKUP_RADIUS) {
          s.collected = true;
          s.grp.visible = false;
          count += 1;
          fireBurst(s.grp.position);
          if (count === shards.length && onAllCollected) onAllCollected();
        }
      }
      let anyBurst = false;
      for (let b = 0; b < BURSTS; b++) {
        if (burstLife[b] <= 0) continue;
        anyBurst = true;
        burstLife[b] -= dt;
        for (let k = 0; k < PER; k++) {
          const o = b * PER * 3 + k * 3;
          burstVel[o + 1] -= 14 * dt;
          burstPos[o] += burstVel[o] * dt;
          burstPos[o + 1] += burstVel[o + 1] * dt;
          burstPos[o + 2] += burstVel[o + 2] * dt;
        }
        if (burstLife[b] <= 0) {
          for (let k = 0; k < PER * 3; k++) burstPos[b * PER * 3 + k] = 0;
        }
      }
      if (anyBurst) burstGeo.attributes.position.needsUpdate = true;
      burstPoints.visible = anyBurst;
    },
  });

  // camera reference injected right after construction (main.js), before the
  // first updater tick — until then no pickup check can run.
  return {
    total: shards.length,
    collected,
    positions: shards.map((s) => s.pos.clone()),
    bindCamera(cam) { cameraRef = cam; },
  };
}
