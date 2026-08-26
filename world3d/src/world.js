// Miniature Singapore — the land layer: Marina Bay (reflective Water when the
// quality budget allows), promenade, textured roads, parks, Fort Canning and
// organic instanced trees. Everything seeded & deterministic.
import * as THREE from 'three';
import { Water } from 'three/addons/objects/Water.js';
import { mulberry32 } from './noise.js';
import { groundTexture, roadTexture, waterNormals } from './textures.js';

export const BAY_R = 130;          // Marina Bay radius
const GROUND = 2800;

export function buildWorld(scene, { reflections = false } = {}) {
  const updaters = [];
  const nightReg = [];
  const rng = mulberry32(20260825);
  const g = new THREE.Group();
  scene.add(g);

  // ── ground: urban concrete ────────────────────────────────────────
  const gtex = groundTexture();
  gtex.repeat.set(46, 46);
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(GROUND, GROUND),
    new THREE.MeshStandardMaterial({ map: gtex, roughness: 0.96 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  g.add(ground);

  // ── the bay ───────────────────────────────────────────────────────
  const waterMat = new THREE.MeshStandardMaterial({
    color: 0x1a5a66, roughness: 0.3, metalness: 0.25,
    emissive: 0x14323c, emissiveIntensity: 0,
  });
  nightReg.push({ mat: waterMat, prop: 'emissiveIntensity', day: 0, night: 0.9 });
  if (reflections) {
    const water = new Water(new THREE.CircleGeometry(BAY_R, 64), {
      textureWidth: 512,
      textureHeight: 512,
      waterNormals: waterNormals(256),
      sunDirection: new THREE.Vector3(0, 1, 0),
      sunColor: 0xffffff,
      waterColor: 0x0d4a56,
      distortionScale: 2.4,
      fog: true,
    });
    water.rotation.x = -Math.PI / 2;
    water.position.y = 0.5;
    g.add(water);
    updaters.push({
      update(t, dt, env) {
        water.material.uniforms.time.value += dt * 0.55;
        water.material.uniforms.sunDirection.value.copy(env.sunDir).normalize();
        water.material.uniforms.waterColor.value.setHex(0x0d4a56).lerp(
          new THREE.Color(0x14262e), env.night ?? 0,
        );
      },
    });
  } else {
    const water = new THREE.Mesh(new THREE.CircleGeometry(BAY_R, 56), waterMat);
    water.rotation.x = -Math.PI / 2;
    water.position.y = 0.5;
    g.add(water);
  }

  // promenade ring around the bay
  const prom = new THREE.Mesh(
    new THREE.RingGeometry(BAY_R - 1, BAY_R + 22, 56),
    new THREE.MeshStandardMaterial({ color: 0x8d8f88, roughness: 0.95 }),
  );
  prom.rotation.x = -Math.PI / 2;
  prom.position.y = 0.55;
  prom.receiveShadow = true;
  g.add(prom);

  // ── parks ─────────────────────────────────────────────────────────
  const park = (x, z, r) => {
    const m = new THREE.Mesh(
      new THREE.CircleGeometry(r, 30),
      new THREE.MeshStandardMaterial({ color: 0x3e6a3f, roughness: 1 }),
    );
    m.rotation.x = -Math.PI / 2;
    m.position.set(x, 0.45, z);
    m.receiveShadow = true;
    g.add(m);
  };
  park(230, 190, 118);    // Gardens by the Bay
  park(-330, -200, 96);   // Fort Canning

  // Fort Canning hill
  const hill = new THREE.Mesh(
    new THREE.CylinderGeometry(58, 98, 17, 22),
    new THREE.MeshStandardMaterial({ color: 0x47663e, roughness: 1, flatShading: true }),
  );
  hill.position.set(-330, 8.5, -200);
  hill.castShadow = hill.receiveShadow = true;
  g.add(hill);

  // ── roads: textured planes ────────────────────────────────────────
  const rtex = roadTexture();
  rtex.repeat.set(1, 26);
  const roadMat = new THREE.MeshStandardMaterial({ map: rtex, roughness: 0.92 });
  const road = (x, z, len, rot) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(13, len), roadMat);
    m.rotation.x = -Math.PI / 2;
    m.rotation.z = rot;
    m.position.set(x, 0.42, z);
    m.receiveShadow = true;
    g.add(m);
  };
  road(0, 620, 520, 0);
  road(620, 0, 520, Math.PI / 2);
  road(-480, 480, 640, Math.PI / 4);
  road(-560, -120, 520, Math.PI / 2);

  // ── trees (instanced: trunks + organic canopies) ──────────────────
  const spots = [];
  for (let i = 0; i < 100; i++) {   // Gardens parkland
    const a = rng() * Math.PI * 2, r = Math.sqrt(rng()) * 105;
    spots.push([230 + Math.cos(a) * r, 0.4, 190 + Math.sin(a) * r, 0.8 + rng() * 0.7]);
  }
  for (let i = 0; i < 80; i++) {    // Fort Canning top
    const a = rng() * Math.PI * 2, r = Math.sqrt(rng()) * 52;
    spots.push([-330 + Math.cos(a) * r, 17, -200 + Math.sin(a) * r, 0.7 + rng() * 0.6]);
  }
  for (let i = 0; i < 60; i++) {    // scattered outskirts
    const a = rng() * Math.PI * 2, r = 420 + rng() * 800;
    spots.push([Math.cos(a) * r, 0, Math.sin(a) * r, 0.9 + rng() * 0.9]);
  }
  const trunkGeo = new THREE.CylinderGeometry(0.45, 0.7, 4.2, 5);
  const canopyGeo = new THREE.IcosahedronGeometry(3.1, 1);
  {
    // organic blob: deterministic per-vertex jitter
    const p = canopyGeo.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const h = Math.abs(Math.sin(i * 127.1 + 311.7)) % 1;
      const s = 1 + (h - 0.5) * 0.42;
      p.setXYZ(i, p.getX(i) * s, p.getY(i) * s * 0.86, p.getZ(i) * s);
    }
    canopyGeo.computeVertexNormals();
  }
  const trunkMat = new THREE.MeshStandardMaterial({ color: 0x5a4634, roughness: 1, flatShading: true });
  const canopyMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9 });
  const trunks = new THREE.InstancedMesh(trunkGeo, trunkMat, spots.length);
  const canopies = new THREE.InstancedMesh(canopyGeo, canopyMat, spots.length);
  const M = new THREE.Matrix4(), C = new THREE.Color();
  spots.forEach(([x, y, z, s], i) => {
    M.makeScale(s, s, s).setPosition(x, y + 2.1 * s, z);
    trunks.setMatrixAt(i, M);
    M.makeScale(s, s * 0.82, s).setPosition(x, y + 5.4 * s, z);
    canopies.setMatrixAt(i, M);
    C.setHSL(0.29 + rng() * 0.07, 0.45, 0.24 + rng() * 0.12);
    canopies.setColorAt(i, C);
  });
  trunks.castShadow = canopies.castShadow = true;
  g.add(trunks, canopies);

  return { updaters, nightReg };
}
