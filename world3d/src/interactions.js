// Pointer interaction: raycast landmark picking, hover tooltip callbacks,
// cinematic camera flights between landmarks.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { easeInOutCubic } from './noise.js';

export class Interactor {
  constructor({ canvas, camera, renderer, pois }) {
    this.camera = camera;
    this.canvas = canvas;
    this.pois = pois;               // [{id,name,pos,viewDist,viewElev,hit}]
    this.enabled = true;

    this.controls = new OrbitControls(camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.minDistance = 40;
    this.controls.maxDistance = 950;
    this.controls.maxPolarAngle = Math.PI * 0.485;
    this.controls.target.set(0, 30, 40);

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2(-10, -10);

    this.hovered = null;
    this.flight = null;
    this.onHover = () => {};
    this.onSelect = () => {};

    this._downAt = null;
    canvas.addEventListener('pointermove', (e) => {
      const r = canvas.getBoundingClientRect();
      this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1,
                       -((e.clientY - r.top) / r.height) * 2 + 1);
      if (this._downAt && (Math.abs(e.clientX - this._downAt.x) > 5 ||
                           Math.abs(e.clientY - this._downAt.y) > 5)) {
        this._downAt = null; // became a drag
      }
    });
    canvas.addEventListener('pointerdown', (e) => {
      this._downAt = { x: e.clientX, y: e.clientY };
      this._cancelFlightIfUserMoves = false;
    });
    canvas.addEventListener('pointerup', (e) => {
      if (!this._downAt) return;
      const dx = e.clientX - this._downAt.x, dy = e.clientY - this._downAt.y;
      this._downAt = null;
      if (Math.hypot(dx, dy) > 5 || !this.enabled) return;
      const hit = this.pick();
      if (hit >= 0) this.onSelect(hit);
    });
  }

  pick() {
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const targets = this.pois.flatMap((p) => p.hitMeshes ?? [p.hit]);
    const hits = this.raycaster.intersectObjects(targets, false);
    if (!hits.length) return -1;
    const obj = hits[0].object;
    return this.pois.findIndex((p) => (p.hitMeshes ?? [p.hit]).includes(obj));
  }

  updateHover() {
    if (!this.enabled || this.flight) { this.setHover(null); return; }
    const i = this.pick();
    this.setHover(i >= 0 ? i : null);
  }

  setHover(i) {
    const id = i === null ? null : this.pois[i].id;
    if (id === this.hovered) return;
    this.hovered = id;
    this.canvas.style.cursor = i === null ? 'grab' : 'pointer';
    this.onHover(i);
  }

  flyTo(i, { duration = 1.8 } = {}) {
    const poi = this.pois[i];
    if (!poi) return;
    const dir = new THREE.Vector3().subVectors(poi.pos, this.controls.target);
    dir.y = 0;
    if (dir.lengthSq() < 0.01) dir.set(0.7, 0, 0.7);
    dir.normalize();
    const endTarget = poi.pos.clone();
    const endPos = poi.pos.clone()
      .addScaledVector(dir, poi.viewDist)
      .add(new THREE.Vector3(0, poi.viewElev, 0));
    this.startFlight(endPos, endTarget, duration);
  }

  startFlight(endPos, endTarget, duration = 1.8) {
    this.flight = {
      t: 0,
      duration,
      startPos: this.camera.position.clone(),
      endPos,
      startTarget: this.controls.target.clone(),
      endTarget,
    };
    this.controls.enabled = false;
    this.setHover(null);
  }

  isFlying() { return !!this.flight; }

  update(dt) {
    if (this.flight) {
      const f = this.flight;
      f.t += dt;
      const k = easeInOutCubic(Math.min(1, f.t / f.duration));
      this.camera.position.lerpVectors(f.startPos, f.endPos, k);
      this.controls.target.lerpVectors(f.startTarget, f.endTarget, k);
      if (f.t >= f.duration) {
        this.flight = null;
        this.controls.enabled = true;
      }
    }
    this.controls.update();
  }

  cancelFlight() {
    if (this.flight) {
      this.flight = null;
      this.controls.enabled = true;
    }
  }
}
