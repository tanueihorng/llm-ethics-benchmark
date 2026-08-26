// Ambient life over Marina Bay: gulls by day, a circling airliner, tugboats
// on the bay and sunlight sparkle on the water.
import * as THREE from 'three';
import { mulberry32 } from './noise.js';

export function buildLife(scene, { updaters }) {
  const rng = mulberry32(424242);

  // ── gulls ─────────────────────────────────────────────────────────
  const birdMat = new THREE.MeshStandardMaterial({ color: 0xe8e8e8, roughness: 0.9, side: THREE.DoubleSide });
  const wingGeo = new THREE.PlaneGeometry(2.6, 0.9);
  const birds = [];
  for (let i = 0; i < 16; i++) {
    const b = new THREE.Group();
    const wl = new THREE.Mesh(wingGeo, birdMat);
    const wr = new THREE.Mesh(wingGeo, birdMat);
    wl.position.x = -1.25;
    wr.position.x = 1.25;
    b.add(wl, wr);
    scene.add(b);
    birds.push({
      g: b, wl, wr,
      r: 120 + rng() * 130,
      y: 45 + rng() * 55,
      speed: (0.05 + rng() * 0.05) * (rng() > 0.5 ? 1 : -1),
      phase: rng() * Math.PI * 2,
      flap: 5 + rng() * 3,
    });
  }
  updaters.push({
    update(t, dt, env) {
      const day = env.daylight ?? 1;
      for (const b of birds) {
        b.g.visible = day > 0.14;
        if (!b.g.visible) continue;
        const a = t * b.speed + b.phase;
        b.g.position.set(Math.cos(a) * b.r, b.y + Math.sin(t * 0.5 + b.phase) * 4, Math.sin(a) * b.r);
        b.g.rotation.y = -a + (b.speed > 0 ? 0 : Math.PI);
        const f = Math.sin(t * b.flap + b.phase) * 0.55;
        b.wl.rotation.z = f;
        b.wr.rotation.z = -f;
      }
    },
  });

  // ── airliner on approach ──────────────────────────────────────────
  const plane = new THREE.Group();
  {
    const white = new THREE.MeshStandardMaterial({ color: 0xdde3e8, roughness: 0.4, metalness: 0.2 });
    const fus = new THREE.Mesh(new THREE.CylinderGeometry(1.7, 1.7, 15, 8), white);
    fus.rotation.z = Math.PI / 2;
    plane.add(fus);
    const nose = new THREE.Mesh(new THREE.SphereGeometry(1.7, 8, 6), white);
    nose.position.x = 7.5;
    plane.add(nose);
    const wing = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.5, 17), white);
    wing.position.set(-0.5, 0, 0);
    plane.add(wing);
    const tail = new THREE.Mesh(new THREE.BoxGeometry(2.6, 3.4, 0.4), white);
    tail.position.set(-6.8, 1.9, 0);
    plane.add(tail);
    const strobe = new THREE.Mesh(
      new THREE.SphereGeometry(0.5, 6, 4),
      new THREE.MeshBasicMaterial({ color: 0xff4040 }),
    );
    strobe.position.set(-6.8, -0.6, 0);
    plane.add(strobe);
    plane.userData.strobe = strobe;
    scene.add(plane);
  }
  updaters.push({
    update(t) {
      const a = t * 0.021 + 1.5;
      plane.position.set(Math.cos(a) * 620, 265, Math.sin(a) * 620);
      plane.rotation.y = -a;
      plane.rotation.z = 0.06;
      plane.userData.strobe.visible = (t * 1.6) % 1 < 0.12;
    },
  });

  // ── tugboats ──────────────────────────────────────────────────────
  const boats = [];
  const hullColors = [0xc4553f, 0x3f6ac4, 0xd8a13f];
  for (let i = 0; i < 3; i++) {
    const b = new THREE.Group();
    const hull = new THREE.Mesh(
      new THREE.BoxGeometry(7, 1.7, 2.5),
      new THREE.MeshStandardMaterial({ color: hullColors[i], roughness: 0.7 }),
    );
    hull.position.y = 0.6;
    hull.castShadow = true;
    b.add(hull);
    const cab = new THREE.Mesh(
      new THREE.BoxGeometry(2.4, 1.4, 1.9),
      new THREE.MeshStandardMaterial({ color: 0xe8e4da, roughness: 0.7 }),
    );
    cab.position.set(-0.8, 2, 0);
    b.add(cab);
    scene.add(b);
    boats.push({ g: b, r: 42 + i * 32, cx: (i - 1) * 26, cz: i * 14 - 8, speed: 0.04 + i * 0.012, phase: i * 2.1 });
  }
  updaters.push({
    update(t) {
      for (const b of boats) {
        const a = t * b.speed + b.phase;
        b.g.position.set(b.cx + Math.cos(a) * b.r, 1.1, b.cz + Math.sin(a) * b.r);
        b.g.rotation.y = -a + Math.PI / 2;
        b.g.rotation.z = Math.sin(t * 1.4 + b.phase) * 0.03;
      }
    },
  });

  // ── water sparkle ─────────────────────────────────────────────────
  const N = 360;
  const pos = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const a = rng() * Math.PI * 2, r = Math.sqrt(rng()) * 124;
    pos[i * 3] = Math.cos(a) * r;
    pos[i * 3 + 1] = 1.1;
    pos[i * 3 + 2] = Math.sin(a) * r;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({
    color: 0xbfe8ff, size: 0.9, transparent: true, opacity: 0.3,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const sparkles = new THREE.Points(geo, mat);
  sparkles.frustumCulled = false;
  scene.add(sparkles);
  updaters.push({
    update(t, dt, env) {
      sparkles.visible = (env.daylight ?? 1) > 0.1;
      mat.opacity = 0.18 + 0.16 * Math.sin(t * 1.2) * (env.daylight ?? 1);
    },
  });
}
