// The Marina Bay night light show — rotating beams from the Marina Bay Sands
// sky park and the two tallest supertrees, night-gated. Replaces the old
// aurora system; same debug API shape (getIntensity).
import * as THREE from 'three';

const COLORS = [0x9fd8ff, 0x7fffd0, 0xffe08a, 0xd0a8ff, 0xff9f9f, 0x9fffdf, 0x9fb8ff];

function beam(len, color) {
  const geo = new THREE.CylinderGeometry(0.5, 4.5, len, 6, 1, true);
  geo.translate(0, -len / 2, 0);          // pivot at the source end
  const mat = new THREE.MeshBasicMaterial({
    color, transparent: true, opacity: 0,
    blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide,
    fog: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  return mesh;
}

export function buildLightShow(scene, { anchors, updaters }) {
  const sources = [
    { at: anchors.mbsSkypark, n: 5, len: 420, ySpeed: 0.16 },
    ...(anchors.supertreeTops || []).map((at) => ({ at, n: 1, len: 260, ySpeed: 0.3 })),
  ];
  const beams = [];
  let bi = 0;
  for (const src of sources) {
    const pivot = new THREE.Group();
    pivot.position.copy(src.at);
    scene.add(pivot);
    for (let k = 0; k < src.n; k++) {
      const b = beam(src.len, COLORS[bi % COLORS.length]);
      bi += 1;
      b.rotation.z = 0.55 + (k - (src.n - 1) / 2) * 0.16;
      pivot.add(b);
      beams.push({
        pivot, mesh: b,
        phase: (bi * 2.399) % (Math.PI * 2),
        ySpeed: src.ySpeed * (0.7 + (bi % 3) * 0.25),
      });
    }
  }
  let intensity = 0;
  updaters.push({
    update(t, dt, env) {
      intensity = Math.pow(env.night, 1.5);
      for (const b of beams) {
        b.pivot.rotation.y = t * b.ySpeed + b.phase;
        b.pivot.rotation.x = Math.sin(t * 0.35 + b.phase) * 0.22;
        b.mesh.material.opacity = intensity * (0.16 + 0.12 * Math.sin(t * 1.8 + b.phase));
      }
    },
  });
  return { getIntensity: () => intensity };
}
