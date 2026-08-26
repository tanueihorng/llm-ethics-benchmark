// Miniature Singapore — the landmarks: Marina Bay Sands, ArtScience Museum,
// Singapore Flyer, Gardens by the Bay, the Merlion, Esplanade, the Helix
// Bridge and the CBD skyline. All seeded & deterministic.
import * as THREE from 'three';
import { mulberry32 } from './noise.js';
import { radialTexture, facadeTexture, latticeTexture } from './textures.js';

const glass = (color, extra = {}) => new THREE.MeshStandardMaterial({
  color, roughness: 0.28, metalness: 0.55, userData: { glassEnv: 0.55 }, ...extra,
});

function box(w, h, d, mat, x, y, z, group) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y, z);
  m.castShadow = m.receiveShadow = true;
  group.add(m);
  return m;
}

export function buildProps(scene, { updaters, nightReg }) {
  const rng = mulberry32(8252026);
  const anchors = {};

  // shared facade texture sets (day map + night emissive map)
  const FACADES = [
    facadeTexture({ seed: 11, cols: 7, rows: 22, base: '#8fa4b2', glass: '#2c3e4a', litProb: 0.32 }),
    facadeTexture({ seed: 22, cols: 6, rows: 26, base: '#a3b2ba', glass: '#37474f', litProb: 0.26 }),
    facadeTexture({ seed: 33, cols: 8, rows: 20, base: '#b0aa9c', glass: '#3a4048', litProb: 0.30 }),
    facadeTexture({ seed: 44, cols: 5, rows: 30, base: '#95a8b8', glass: '#25333d', litProb: 0.36 }),
  ];

  // ═══════════════════════════ MARINA BAY SANDS ═══════════════════════════
  {
    const g = new THREE.Group();
    g.position.set(35, 0, 235);
    // glass-strip facade: day map + individually lit windows at night
    const mbsF = facadeTexture({
      cols: 5, rows: 14, base: '#c2ccd4', glass: '#41586a',
      glassVar: 0.2, litProb: 0.22, seed: 71,
    });
    mbsF.map.repeat.set(1, 3);
    mbsF.emissiveMap.repeat.set(1, 3);
    const towerMat = new THREE.MeshStandardMaterial({
      map: mbsF.map, emissiveMap: mbsF.emissiveMap,
      emissive: 0xffe2b0, emissiveIntensity: 0,
      roughness: 0.45, metalness: 0.22, userData: { glassEnv: 0.6 },
    });
    nightReg.push({ mat: towerMat, prop: 'emissiveIntensity', day: 0, night: 1.1 });
    // three curved, leaning towers (stacked tapering segments)
    const SEG = 5, SH = 37;
    for (const dx of [-60, 0, 60]) {
      for (let i = 0; i < SEG; i++) {
        const w = 26 - i * 1.7;
        const seg = new THREE.Mesh(new THREE.BoxGeometry(w, SH + 0.6, w), towerMat);
        seg.position.set(dx, i * SH + SH / 2, -Math.pow(i, 1.7) * 2.7);
        seg.castShadow = seg.receiveShadow = true;
        g.add(seg);
      }
      const crown = new THREE.Mesh(new THREE.BoxGeometry(15, 3.5, 15), towerMat);
      crown.position.set(dx, SEG * SH + 1, -Math.pow(SEG - 1, 1.7) * 2.7);
      g.add(crown);
    }
    // skypark: hull + cantilever bow + deck gardens + strips
    const parkMat = new THREE.MeshStandardMaterial({ color: 0xd9d4c8, roughness: 0.6 });
    const hull = new THREE.Mesh(new THREE.BoxGeometry(250, 7, 44), parkMat);
    hull.position.set(0, 192, 0);
    hull.castShadow = true;
    g.add(hull);
    const bow = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 11, 70, 4), parkMat);
    bow.rotation.z = Math.PI / 2;
    bow.rotation.y = Math.PI / 4;
    bow.position.set(-155, 193.5, 0);
    g.add(bow);
    // planted deck (the sky garden)
    const deckMat = new THREE.MeshStandardMaterial({ color: 0x4d6b40, roughness: 0.95 });
    const deck = new THREE.Mesh(new THREE.BoxGeometry(244, 0.8, 36), deckMat);
    deck.position.set(0, 196.2, 0);
    g.add(deck);
    const deckTreeGeo = new THREE.IcosahedronGeometry(1.6, 0);
    const deckTreeMat = new THREE.MeshStandardMaterial({ color: 0x3f5c38, roughness: 1, flatShading: true });
    const deckTrees = new THREE.InstancedMesh(deckTreeGeo, deckTreeMat, 26);
    const DM = new THREE.Matrix4();
    for (let i = 0; i < 26; i++) {
      const s = 0.7 + rng() * 0.7;
      DM.makeScale(s, s, s).setPosition(-115 + (i % 13) * 18 + rng() * 6, 197.6, (Math.floor(i / 13) - 0.5) * 22 + rng() * 10);
      deckTrees.setMatrixAt(i, DM);
    }
    g.add(deckTrees);
    const stripMat = new THREE.MeshStandardMaterial({
      color: 0x2a2a2a, emissive: 0xffd9a0, emissiveIntensity: 0,
    });
    nightReg.push({ mat: stripMat, prop: 'emissiveIntensity', day: 0, night: 2.4 });
    for (const dz of [-21.5, 21.5]) {
      const s = new THREE.Mesh(new THREE.BoxGeometry(248, 0.6, 1.4), stripMat);
      s.position.set(-5, 196, dz);
      g.add(s);
    }
    // podium
    const pod = new THREE.Mesh(new THREE.BoxGeometry(210, 10, 90), glass(0x74868f));
    pod.position.set(0, 5, 30);
    pod.receiveShadow = true;
    g.add(pod);
    scene.add(g);
    anchors.mbs = new THREE.Vector3(35, 95, 235);
    anchors.mbsSkypark = new THREE.Vector3(35 - 20, 199, 235);
  }

  // ═══════════════════════════ ARTSCIENCE MUSEUM ══════════════════════════
  {
    const g = new THREE.Group();
    g.position.set(10, 0, 150);
    const pod = new THREE.Mesh(new THREE.CylinderGeometry(26, 30, 6, 10), glass(0x74868f));
    pod.position.y = 3;
    pod.receiveShadow = true;
    g.add(pod);
    const petalMat = new THREE.MeshStandardMaterial({ color: 0xeae7df, roughness: 0.5 });
    petalMat.emissive = new THREE.Color(0x8a8578);
    petalMat.emissiveIntensity = 0;
    nightReg.push({ mat: petalMat, prop: 'emissiveIntensity', day: 0, night: 0.35 });
    const petalGeo = new THREE.SphereGeometry(1, 10, 8);
    petalGeo.scale(6, 4.6, 16);
    for (let i = 0; i < 10; i++) {
      const p = new THREE.Mesh(petalGeo, petalMat);
      const a = (i / 10) * Math.PI * 2;
      p.position.set(Math.sin(a) * 7.5, 9.5, Math.cos(a) * 7.5);
      p.rotation.y = a;
      p.rotation.x = 0.42;
      p.castShadow = true;
      g.add(p);
    }
    const core = new THREE.Mesh(new THREE.CylinderGeometry(9, 11, 12, 10), petalMat);
    core.position.y = 8;
    core.castShadow = true;
    g.add(core);
    scene.add(g);
    anchors.museum = new THREE.Vector3(10, 14, 150);
  }

  // ═══════════════════════════ SINGAPORE FLYER ════════════════════════════
  {
    const g = new THREE.Group();
    g.position.set(250, 96, -55);
    g.rotation.y = -2.35;                       // wheel faces the bay
    const frameMat = new THREE.MeshStandardMaterial({ color: 0xb8bfc6, roughness: 0.4, metalness: 0.6 });
    const rimMat = new THREE.MeshStandardMaterial({
      color: 0x9fb2bd, roughness: 0.35, metalness: 0.5,
      emissive: 0x6fd8ff, emissiveIntensity: 0,
    });
    nightReg.push({ mat: rimMat, prop: 'emissiveIntensity', day: 0, night: 1.4 });
    const spin = new THREE.Group();             // rotates about local Z (the axle)
    g.add(spin);
    const R = 82;
    spin.add(new THREE.Mesh(new THREE.TorusGeometry(R, 1.7, 8, 48), rimMat));
    spin.add(new THREE.Mesh(new THREE.TorusGeometry(R * 0.55, 1.1, 6, 40), frameMat));
    const spokeGeo = new THREE.CylinderGeometry(0.45, 0.45, R, 4);
    const spokes = new THREE.InstancedMesh(spokeGeo, frameMat, 32);
    const M = new THREE.Matrix4(), T = new THREE.Matrix4();
    for (let i = 0; i < 32; i++) {
      M.makeRotationZ((i / 32) * Math.PI * 2);
      T.makeTranslation(0, R / 2, 0);
      spokes.setMatrixAt(i, M.multiply(T));
    }
    spin.add(spokes);
    const capGeo = new THREE.SphereGeometry(2.6, 8, 6);
    const capMat = new THREE.MeshStandardMaterial({ color: 0xe3e8ec, roughness: 0.3, metalness: 0.3 });
    const caps = new THREE.InstancedMesh(capGeo, capMat, 24);
    for (let i = 0; i < 24; i++) {
      const a = (i / 24) * Math.PI * 2;
      M.makeTranslation(Math.cos(a) * R, Math.sin(a) * R, 0);
      caps.setMatrixAt(i, M);
    }
    caps.castShadow = true;
    spin.add(caps);
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(4, 4, 10, 10), frameMat);
    hub.rotation.x = Math.PI / 2;
    g.add(hub);
    // A-frame legs + base
    for (const dz of [-15, 15]) {
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(1.9, 2.6, 102, 8), frameMat);
      leg.position.set(0, -46, dz);
      leg.rotation.x = dz > 0 ? -0.3 : 0.3;
      leg.castShadow = true;
      g.add(leg);
    }
    const base = new THREE.Mesh(new THREE.BoxGeometry(34, 9, 52), glass(0x74868f));
    base.position.set(0, -92, 0);
    base.castShadow = base.receiveShadow = true;
    g.add(base);
    scene.add(g);
    anchors.flyer = new THREE.Vector3(250, 98, -55);
    updaters.push({ update(t) { spin.rotation.z = t * 0.05; } });
  }

  // ═══════════════════════════ GARDENS BY THE BAY ═════════════════════════
  {
    const g = new THREE.Group();
    g.position.set(230, 0, 190);
    // lattice trunk, glowing along its lattice at night (Garden Rhapsody)
    const lat = latticeTexture();
    lat.map.repeat.set(3, 6);
    lat.emissiveMap.repeat.set(3, 6);
    const trunkMat = new THREE.MeshStandardMaterial({
      map: lat.map, emissiveMap: lat.emissiveMap,
      emissive: 0x35e0a0, emissiveIntensity: 0, roughness: 0.85,
    });
    nightReg.push({ mat: trunkMat, prop: 'emissiveIntensity', day: 0, night: 1.2 });
    const canopyMat = new THREE.MeshStandardMaterial({
      color: 0x3f7d4f, roughness: 0.7, flatShading: true,
      emissive: 0x35e0a0, emissiveIntensity: 0,
    });
    nightReg.push({ mat: canopyMat, prop: 'emissiveIntensity', day: 0, night: 1.6 });
    const trees = [
      [0, 0, 48], [15, 9, 38], [-17, 7, 33], [7, -15, 28], [-9, -13, 24], [19, -5, 20],
    ];
    const tops = [];
    trees.forEach(([dx, dz, h], i) => {
      const trunk = new THREE.Mesh(new THREE.CylinderGeometry(1.7, 2.9, h, 8), trunkMat);
      trunk.position.set(dx, h / 2, dz);
      trunk.castShadow = true;
      g.add(trunk);
      const can = new THREE.Mesh(new THREE.CylinderGeometry(11 - i * 0.6, 3.5, 4.6, 8), canopyMat);
      can.position.set(dx, h + 1.6, dz);
      can.castShadow = true;
      g.add(can);
      const skirt = new THREE.Mesh(new THREE.CylinderGeometry(3.4, 8.5, 3.2, 8), canopyMat);
      skirt.position.set(dx, h - 2.4, dz);
      g.add(skirt);
      if (i < 2) tops.push(new THREE.Vector3(dx, h + 2, dz));
    });
    // OCBC skyway between the two tallest
    const way = new THREE.Mesh(
      new THREE.TubeGeometry(new THREE.CatmullRomCurve3([
        tops[0].clone().add(new THREE.Vector3(0, -1.5, 0)),
        tops[0].clone().add(tops[1]).multiplyScalar(0.5).add(new THREE.Vector3(0, 2.5, 0)),
        tops[1].clone().add(new THREE.Vector3(0, -1.5, 0)),
      ]), 16, 0.8, 6),
      new THREE.MeshStandardMaterial({ color: 0xc8ccd2, roughness: 0.5, metalness: 0.4 }),
    );
    way.castShadow = true;
    g.add(way);
    // conservatories: Flower Dome + Cloud Forest
    const domeMat = glass(0x9fc4d0, { transparent: true, opacity: 0.92 });
    domeMat.emissive = new THREE.Color(0x3a5a66);
    domeMat.emissiveIntensity = 0;
    nightReg.push({ mat: domeMat, prop: 'emissiveIntensity', day: 0, night: 0.55 });
    for (const [dx, dz, r] of [[-58, 42, 20], [-26, 54, 14]]) {
      const d = new THREE.Mesh(new THREE.SphereGeometry(r, 18, 9, 0, Math.PI * 2, 0, Math.PI / 2), domeMat);
      d.position.set(dx, 1.5, dz);
      d.castShadow = true;
      g.add(d);
      const rib = new THREE.Mesh(new THREE.TorusGeometry(r, 0.5, 5, 24, Math.PI), frameMatLite());
      rib.position.set(dx, 1.5, dz);
      g.add(rib);
    }
    function frameMatLite() {
      return new THREE.MeshStandardMaterial({ color: 0xa8b0b8, roughness: 0.5, metalness: 0.4 });
    }
    scene.add(g);
    anchors.gardens = new THREE.Vector3(230, 30, 190);
    anchors.supertreeTops = tops.map((v) => v.clone().add(new THREE.Vector3(230, 0, 190)));
  }

  // ═══════════════════════════ THE MERLION ════════════════════════════════
  {
    const g = new THREE.Group();
    g.position.set(-95, 0, 95);
    g.lookAt(0, 0, 0);                          // face the bay
    const stone = new THREE.MeshStandardMaterial({ color: 0xd9d2c2, roughness: 0.7 });
    const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(4.5, 5.5, 7, 10), stone);
    pedestal.position.y = 3.5;
    pedestal.castShadow = pedestal.receiveShadow = true;
    g.add(pedestal);
    const body = new THREE.Mesh(new THREE.CylinderGeometry(1.7, 2.9, 7.5, 9), stone);
    body.position.y = 10.5;
    body.castShadow = true;
    g.add(body);
    const head = new THREE.Mesh(new THREE.SphereGeometry(2.3, 10, 8), stone);
    head.position.set(0, 15.2, -0.6);
    head.scale.set(1, 0.92, 1.15);
    head.castShadow = true;
    g.add(head);
    const mane = new THREE.Mesh(new THREE.SphereGeometry(1.5, 8, 6), stone);
    mane.position.set(0, 14.2, 1.1);
    mane.scale.set(1.5, 1.1, 1);
    g.add(mane);
    const tail = new THREE.Mesh(new THREE.ConeGeometry(1.3, 3.4, 6), stone);
    tail.position.set(0, 12.6, 2.6);
    tail.rotation.x = 1.9;
    tail.castShadow = true;
    g.add(tail);
    scene.add(g);

    // the jet: a constant parabolic arc of water into the bay
    const N = 70;
    const pos = new Float32Array(N * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const jetMat = new THREE.PointsMaterial({
      color: 0xcfeefc, size: 1.1, transparent: true, opacity: 0.75,
      blending: THREE.AdditiveBlending, depthWrite: false,
    });
    const jet = new THREE.Points(geo, jetMat);
    jet.frustumCulled = false;
    scene.add(jet);
    const mouth = new THREE.Vector3(-95, 14.6, 95).addScaledVector(
      new THREE.Vector3(0, 0, 0).sub(new THREE.Vector3(-95, 0, 95)).normalize(), 3);
    const land = new THREE.Vector3().lerpVectors(mouth, new THREE.Vector3(0, 0.8, 0), 0.16);
    // splash ring where the jet lands
    const SN = 26;
    const spos = new Float32Array(SN * 3);
    const sgeo = new THREE.BufferGeometry();
    sgeo.setAttribute('position', new THREE.BufferAttribute(spos, 3));
    const splashMat = new THREE.PointsMaterial({
      color: 0xd8f2fc, size: 0.8, transparent: true, opacity: 0.65,
      blending: THREE.AdditiveBlending, depthWrite: false,
    });
    const splash = new THREE.Points(sgeo, splashMat);
    splash.frustumCulled = false;
    scene.add(splash);
    updaters.push({
      update(t) {
        for (let i = 0; i < N; i++) {
          const k = (t * 0.55 + i / N) % 1;
          const p = new THREE.Vector3().lerpVectors(mouth, land, k);
          p.y += 5.2 * k * (1 - k) * 2.2;
          pos[i * 3] = p.x + Math.sin(i * 12.9) * 0.3 * k;
          pos[i * 3 + 1] = p.y;
          pos[i * 3 + 2] = p.z + Math.cos(i * 7.7) * 0.3 * k;
        }
        geo.attributes.position.needsUpdate = true;
        for (let i = 0; i < SN; i++) {
          const k = (t * 0.9 + i / SN) % 1;
          const a = i * 2.399;
          const r = 0.6 + k * 3.2;
          spos[i * 3] = land.x + Math.cos(a) * r;
          spos[i * 3 + 1] = land.y + Math.sin(k * Math.PI) * 1.7 + 0.2;
          spos[i * 3 + 2] = land.z + Math.sin(a) * r;
        }
        sgeo.attributes.position.needsUpdate = true;
      },
    });
    anchors.merlion = new THREE.Vector3(-95, 12, 95);
  }

  // ═══════════════════════════ ESPLANADE (the durians) ════════════════════
  {
    const g = new THREE.Group();
    g.position.set(-110, 0, 120);
    const shellMat = glass(0x8aa4ae);
    shellMat.emissive = new THREE.Color(0xffc27a);
    shellMat.emissiveIntensity = 0;
    nightReg.push({ mat: shellMat, prop: 'emissiveIntensity', day: 0, night: 0.5 });
    const spikeGeo = new THREE.ConeGeometry(0.55, 2.4, 5);
    const spikeMat = new THREE.MeshStandardMaterial({ color: 0x93a8b0, roughness: 0.6 });
    [[0, 0, 17], [42, 6, 13]].forEach(([dx, dz, r], bi) => {
      const dome = new THREE.Mesh(new THREE.SphereGeometry(r, 14, 8), shellMat);
      dome.position.set(dx, r * 0.55, dz);
      dome.castShadow = true;
      g.add(dome);
      const spikes = new THREE.InstancedMesh(spikeGeo, spikeMat, 46);
      const M = new THREE.Matrix4(), Q = new THREE.Quaternion(), S = new THREE.Vector3();
      const up = new THREE.Vector3(0, 1, 0), nrm = new THREE.Vector3();
      for (let i = 0; i < 46; i++) {
        const u = rng() * 0.85 + 0.08;                 // upper band of the sphere
        const phi = Math.acos(1 - u);
        const th = rng() * Math.PI * 2;
        nrm.setFromSphericalCoords(1, phi, th);
        Q.setFromUnitVectors(up, nrm);
        S.set(1, 1, 1);
        M.compose(
          new THREE.Vector3(dx, r * 0.55, dz).add(nrm.multiplyScalar(r * 0.98)),
          Q, S,
        );
        spikes.setMatrixAt(i, M);
      }
      g.add(spikes);
      void bi;
    });
    scene.add(g);
    anchors.esplanade = new THREE.Vector3(-110, 20, 120);
  }

  // ═══════════════════════════ THE HELIX BRIDGE ═══════════════════════════
  {
    const curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(18, 7, 138),
      new THREE.Vector3(60, 15, 90),
      new THREE.Vector3(105, 13, 40),
      new THREE.Vector3(148, 7, -8),
    ]);
    const deckMat = new THREE.MeshStandardMaterial({ color: 0xd8dde2, roughness: 0.45, metalness: 0.35 });
    const deck = new THREE.Mesh(new THREE.TubeGeometry(curve, 28, 1.5, 6), deckMat);
    deck.castShadow = true;
    scene.add(deck);
    const ringGeo = new THREE.TorusGeometry(3.1, 0.28, 5, 14);
    const rings = new THREE.InstancedMesh(ringGeo, deckMat, 10);
    const M = new THREE.Matrix4(), Q = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0);
    for (let i = 0; i < 10; i++) {
      const p = curve.getPointAt(0.08 + (i / 10) * 0.84);
      const tan = curve.getTangentAt(0.08 + (i / 10) * 0.84);
      Q.setFromUnitVectors(new THREE.Vector3(0, 0, 1), tan);
      const tilt = new THREE.Quaternion().setFromAxisAngle(tan, (i % 2 ? 1 : -1) * 0.5);
      M.compose(p, Q.multiply(tilt), new THREE.Vector3(1, 1, 1));
      rings.setMatrixAt(i, M);
      void up;
    }
    scene.add(rings);
  }

  // ═══════════════════════════ CBD SKYLINE ════════════════════════════════
  {
    const g = new THREE.Group();
    const makeTower = (x, z, w, d, h, fi) => {
      const src = FACADES[fi % FACADES.length];
      const map = src.map.clone(); map.needsUpdate = true;
      const em = src.emissiveMap.clone(); em.needsUpdate = true;
      // one texture tile carries src.cols×src.rows windows ≈ 4 units each
      const rep = new THREE.Vector2(
        Math.max(1, Math.round(w / (4 * src.cols))),
        Math.max(1, Math.round(h / (4 * src.rows))),
      );
      map.repeat.copy(rep);
      em.repeat.copy(rep);
      const mat = new THREE.MeshStandardMaterial({
        map, emissiveMap: em, emissive: 0xffc98a, emissiveIntensity: 0,
        roughness: 0.55, metalness: 0.08, userData: { glassEnv: 0.45 },
      });
      nightReg.push({ mat, prop: 'emissiveIntensity', day: 0, night: 1.15 });
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
      m.position.set(x, h / 2, z);
      m.castShadow = m.receiveShadow = true;
      g.add(m);
      return m;
    };
    const towers = [];
    // three landmark towers (with setback crowns)
    towers.push([-150, -120, 26, 26, 200], [-215, -70, 24, 24, 172], [-120, -200, 22, 22, 150]);
    for (let i = 0; i < 15; i++) {
      const x = -90 - rng() * 250;
      const z = -45 - rng() * 230;
      const w = 16 + rng() * 14;
      const h = 50 + rng() * 80;
      towers.push([x, z, w, w * 0.9, h]);
    }
    towers.forEach(([x, z, w, d, h], i) => {
      makeTower(x, z, w, d, h, i);
      if (h > 120) {
        const crown = makeTower(x, z, w * 0.6, d * 0.6, h * 0.3, i + 1);
        crown.position.y = h + h * 0.15;
        const spire = new THREE.Mesh(
          new THREE.CylinderGeometry(0.3, 1.1, h * 0.16, 6),
          new THREE.MeshStandardMaterial({ color: 0xb8c0c8, roughness: 0.4, metalness: 0.6 }),
        );
        spire.position.set(x, h + h * 0.3 + h * 0.08, z);
        g.add(spire);
      }
    });
    scene.add(g);

    // warm city spill-light so the ground never goes pitch black
    const spill = [];
    for (const [px, py, pz, col, dist, max] of [
      [-180, 110, -150, 0xd9a86a, 380, 0.28],
      [35, 80, 235, 0xffd9a0, 300, 0.5],
      [230, 40, 190, 0x7fffd0, 260, 0.45],
    ]) {
      const l = new THREE.PointLight(col, 0, dist, 1.6);
      l.position.set(px, py, pz);
      scene.add(l);
      spill.push({ l, max });
    }
    updaters.push({
      update(t, dt, env) {
        for (const s of spill) s.l.intensity = s.max * (env.night ?? 0);
      },
    });
    anchors.cbd = new THREE.Vector3(-170, 120, -140);
  }

  // ═══════════════════════════ CITY FABRIC ════════════════════════════════
  // mid-rise districts on a street grid, filling the areas between landmarks
  {
    const g = new THREE.Group();
    const clear = [
      (x, z) => Math.hypot(x, z) < 140,                               // the bay
      (x, z) => Math.hypot(x - 230, z - 190) < 134,                   // Gardens park
      (x, z) => Math.hypot(x - 35, z - 250) < 98,                     // MBS block
      (x, z) => Math.hypot(x - 250, z + 55) < 64,                     // Flyer
      (x, z) => Math.hypot(x + 95, z - 95) < 44,                      // Merlion plaza
      (x, z) => Math.hypot(x + 110, z - 120) < 50,                    // Esplanade
      (x, z) => Math.hypot(x + 10, z - 150) < 42,                     // Museum
      (x, z) => Math.hypot(x + 330, z + 200) < 118,                   // Fort Canning
      (x, z) => Math.abs(z - 620) < 26 || Math.abs(x - 620) < 26,     // roads
      (x, z) => Math.abs((x + z) * Math.SQRT1_2 - 678) < 26,          // diag road
    ];
    const districts = [
      // [cx, cz, cols, rows] — Marina Centre, beach road, south shore, outskirts
      [190, -120, 7, 5],
      [400, 120, 6, 5],
      [-40, 330, 7, 4],
      [420, -260, 5, 4],
      [-320, 60, 6, 4],
      [-200, -330, 5, 4],
      [80, 480, 5, 4],
      [-460, 230, 4, 4],
    ];
    let idx = 0;
    for (const [cx, cz, cols, rows] of districts) {
      for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
          const x = cx + (i - (cols - 1) / 2) * 30 + (rng() - 0.5) * 6;
          const z = cz + (j - (rows - 1) / 2) * 30 + (rng() - 0.5) * 6;
          if (clear.some((f) => f(x, z))) continue;
          const w = 13 + rng() * 8;
          const d = 13 + rng() * 8;
          const h = 8 + rng() * 20 + (rng() < 0.15 ? 16 : 0);
          const src = FACADES[idx % 4];
          const map = src.map.clone(); map.needsUpdate = true;
          const em = src.emissiveMap.clone(); em.needsUpdate = true;
          const rep = new THREE.Vector2(
            Math.max(1, Math.round(w / (4 * src.cols))),
            Math.max(1, Math.round(h / (4 * src.rows))),
          );
          map.repeat.copy(rep);
          em.repeat.copy(rep);
          const mat = new THREE.MeshStandardMaterial({
            map, emissiveMap: em, emissive: 0xffc98a, emissiveIntensity: 0,
            roughness: 0.6, metalness: 0.05, userData: { glassEnv: 0.3 },
          });
          nightReg.push({ mat, prop: 'emissiveIntensity', day: 0, night: 0.95 });
          const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
          m.position.set(x, h / 2, z);
          m.castShadow = m.receiveShadow = true;
          g.add(m);
          if (rng() < 0.3) {   // rooftop unit / water tank
            const roof = new THREE.Mesh(
              new THREE.BoxGeometry(w * 0.3, 1.8, d * 0.3),
              new THREE.MeshStandardMaterial({ color: 0x9aa2a8, roughness: 0.8 }),
            );
            roof.position.set(x, h + 0.9, z);
            g.add(roof);
          }
          idx += 1;
        }
      }
    }
    scene.add(g);
  }

  // ═══════════════════════════ SUBURB RING ════════════════════════════════
  // instanced low-rise sprawl out to the horizon (one draw call)
  {
    const src = facadeTexture({ seed: 77, cols: 6, rows: 14, base: '#a8b0b6', glass: '#424e58', litProb: 0.24 });
    src.map.repeat.set(1, 1);
    src.emissiveMap.repeat.set(1, 1);
    const mat = new THREE.MeshStandardMaterial({
      map: src.map, emissiveMap: src.emissiveMap, emissive: 0xffc98a, emissiveIntensity: 0,
      roughness: 0.7, metalness: 0.04,
    });
    nightReg.push({ mat, prop: 'emissiveIntensity', day: 0, night: 0.8 });
    const N = 300;
    const geo = new THREE.BoxGeometry(1, 1, 1);
    geo.translate(0, 0.5, 0);
    const inst = new THREE.InstancedMesh(geo, mat, N);
    const M = new THREE.Matrix4();
    let placed = 0;
    for (let i = 0; i < N; i++) {
      const a = (i / N) * Math.PI * 2 + rng() * 0.5;
      const r = 560 + rng() * 480;
      const x = Math.cos(a) * r;
      const z = Math.sin(a) * r;
      const w = 7 + rng() * 9;
      const h = 3.5 + rng() * 6.5;
      M.makeScale(w, h, w * (0.8 + rng() * 0.4)).setPosition(x, 0, z);
      inst.setMatrixAt(i, M);
      placed += 1;
    }
    inst.castShadow = false;
    inst.receiveShadow = true;
    scene.add(inst);
  }

  return { anchors };
}
