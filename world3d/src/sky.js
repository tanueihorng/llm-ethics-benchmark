// Sky dome (hour-tuned gradient shader), star field, moon billboards.
// The atmospheric Sky addon is used only for PMREM environment bakes (main.js).
import * as THREE from 'three';
import { mulberry32 } from './noise.js';
import { radialTexture } from './textures.js';

const SKY_VERT = /* glsl */`
varying vec3 vDir;
void main() {
  vDir = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const SKY_FRAG = /* glsl */`
uniform vec3 uTop;
uniform vec3 uMid;
uniform vec3 uHorizon;
uniform vec3 uSunDir;
uniform vec3 uSunTint;
uniform float uDusk;
varying vec3 vDir;
void main() {
  vec3 d = normalize(vDir);
  float h = clamp(d.y, -1.0, 1.0);
  float t = pow(1.0 - max(h, 0.0), 2.1);
  float below = smoothstep(0.0, -0.4, h);

  vec3 col = mix(uTop, uMid, smoothstep(0.0, 0.42, t));
  col = mix(col, uHorizon, smoothstep(0.34, 0.96, t));

  float sunAmt = max(dot(d, normalize(uSunDir)), 0.0);
  float band = exp(-pow(max(h, 0.0) * 3.4, 2.0));
  col += uSunTint * pow(sunAmt, 10.0) * band * (0.35 + uDusk * 0.95);
  col += uSunTint * pow(sunAmt, 90.0) * 0.5;
  col *= (1.0 - below * 0.74);

  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

const STAR_VERT = /* glsl */`
attribute float aSize;
attribute float aSeed;
uniform float uTime;
uniform float uScale;
varying float vTwinkle;
varying float vSeed;
void main() {
  vSeed = aSeed;
  vTwinkle = 0.72 + 0.45 * sin(uTime * (0.4 + fract(aSeed) * 1.4) + aSeed * 91.0);
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = aSize * uScale * vTwinkle;
  gl_Position = projectionMatrix * mv;
}`;

const STAR_FRAG = /* glsl */`
uniform float uOpacity;
varying float vTwinkle;
varying float vSeed;
void main() {
  vec2 uv = gl_PointCoord - 0.5;
  float dist = length(uv);
  float a = smoothstep(0.5, 0.08, dist) * uOpacity * vTwinkle;
  vec3 tint = mix(vec3(0.78, 0.84, 1.0), vec3(1.0, 0.93, 0.78), fract(vSeed * 7.31));
  gl_FragColor = vec4(tint, a);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

function celestialSprite(tex, scale, color, opacity) {
  const mat = new THREE.SpriteMaterial({
    map: tex, color, transparent: true, opacity,
    blending: THREE.AdditiveBlending, depthWrite: false, fog: false,
  });
  const spr = new THREE.Sprite(mat);
  spr.scale.setScalar(scale);
  return spr;
}

export class SkySystem {
  constructor(scene) {
    this.scene = scene;

    // ── gradient dome (visible sky) ──────────────────────────────────
    const domeMat = new THREE.ShaderMaterial({
      vertexShader: SKY_VERT,
      fragmentShader: SKY_FRAG,
      side: THREE.BackSide,
      depthWrite: false,
      fog: false,
      uniforms: {
        uTop: { value: new THREE.Color('#070b1c') },
        uMid: { value: new THREE.Color('#111b3a') },
        uHorizon: { value: new THREE.Color('#28334f') },
        uSunDir: { value: new THREE.Vector3(0, 1, 0) },
        uSunTint: { value: new THREE.Color('#ffd9a0') },
        uDusk: { value: 0 },
      },
    });
    this.domeUniforms = domeMat.uniforms;
    this.dome = new THREE.Mesh(new THREE.SphereGeometry(3800, 40, 24), domeMat);
    this.dome.renderOrder = -10;
    scene.add(this.dome);

    // ── stars ────────────────────────────────────────────────────────
    const rng = mulberry32(1207);
    const N = 1500;
    const pos = new Float32Array(N * 3);
    const size = new Float32Array(N);
    const seed = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const v = rng() * 2 - 1;
      const phi = rng() * Math.PI * 2;
      const r = Math.sqrt(1 - v * v);
      const R = 3500;
      pos[i * 3] = Math.cos(phi) * r * R;
      pos[i * 3 + 1] = Math.abs(v) * R * 0.94 + 40;
      pos[i * 3 + 2] = Math.sin(phi) * r * R;
      size[i] = 3.5 + rng() * 7;
      seed[i] = rng() * 100;
    }
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    sg.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    sg.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
    this.starUniforms = {
      uTime: { value: 0 },
      uScale: { value: window.innerHeight / 900 },
      uOpacity: { value: 0 },
    };
    const starMat = new THREE.ShaderMaterial({
      vertexShader: STAR_VERT, fragmentShader: STAR_FRAG,
      uniforms: this.starUniforms, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending, fog: false,
    });
    this.stars = new THREE.Points(sg, starMat);
    this.stars.renderOrder = -9;
    scene.add(this.stars);

    // ── moon ─────────────────────────────────────────────────────────
    const moonTex = radialTexture(128, [
      [0, 'rgba(228,238,255,1)'], [0.3, 'rgba(190,208,255,0.85)'],
      [0.55, 'rgba(150,180,255,0.2)'], [1, 'rgba(150,180,255,0)'],
    ]);
    this.moon = celestialSprite(moonTex, 200, 0xffffff, 0.9);
    this.moonHalo = celestialSprite(moonTex, 640, 0x9fb6ff, 0.22);
    for (const s of [this.moon, this.moonHalo]) {
      s.renderOrder = -8;
      scene.add(s);
    }
  }

  setTime(env, timeSec, pixelRatio = 1) {
    const u = this.domeUniforms;
    u.uTop.value.copy(env.top);
    u.uMid.value.copy(env.mid);
    u.uHorizon.value.copy(env.hor);
    u.uSunDir.value.copy(env.sunDir);
    u.uSunTint.value.copy(env.sunColor);
    u.uDusk.value = env.dusk;

    this.starUniforms.uTime.value = timeSec;
    this.starUniforms.uOpacity.value = Math.pow(env.night, 1.4);
    this.stars.visible = env.night > 0.02;

    const D = 3300;
    this.moon.position.copy(env.moonDir).multiplyScalar(D);
    this.moonHalo.position.copy(this.moon.position);
    this.moon.material.opacity = 0.9 * env.night;
    this.moonHalo.material.opacity = 0.22 * env.night;

    this.starUniforms.uScale.value = (window.innerHeight / 900) * pixelRatio;
  }

  resize(height) {
    this.starUniforms.uScale.value = height / 900;
  }
}
