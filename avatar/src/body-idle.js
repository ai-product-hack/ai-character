// Спокойные аддитивные движения тела. Внутренний формат намеренно простой:
// Mixamo-клип один раз ретаргетится в градусы локальных поворотов, после чего
// в браузере не нужны ни FBXLoader, ни AnimationMixer, ни лишние аллокации.

import * as THREE from 'three';
import { BONE_ZONES, BONE_LAYERS, bodyOwns } from './bone-zones.js';

const DEG = Math.PI / 180;

function finite(v) { return Number.isFinite(v); }

/** Проверить и скомпилировать JSON-клип; чужие кости отбрасываются на импорте. */
export function compileBodyClip(raw) {
  if (!raw || raw.schema !== 'avatar-body-clip@1') throw new Error('body clip: bad schema');
  const durationMs = Number(raw.durationMs);
  if (!finite(durationMs) || durationMs < 500) throw new Error('body clip: bad duration');

  const tracks = [];
  const droppedBones = [];
  for (const [bone, rows] of Object.entries(raw.tracks || {})) {
    if (!bodyOwns(bone)) { droppedBones.push(bone); continue; }
    if (!Array.isArray(rows) || rows.length < 2) continue;
    const data = new Float32Array(rows.length * 4);
    let previous = -1;
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      if (!Array.isArray(row) || row.length !== 4 || !row.every(finite)) {
        throw new Error(`body clip: bad key in ${bone}`);
      }
      if (row[0] < previous || row[0] < 0 || row[0] > durationMs) {
        throw new Error(`body clip: bad time in ${bone}`);
      }
      previous = row[0];
      data.set(row, i * 4);
    }
    tracks.push({ bone, data, count: rows.length });
  }
  if (!tracks.length) throw new Error('body clip: no owned tracks');
  return Object.freeze({
    name: raw.name || 'idle', durationMs, tracks,
    droppedBones: Object.freeze(droppedBones), source: raw.source || null,
  });
}

/** Линейная выборка одного трека в заранее выделенный вектор. */
export function sampleBodyTrack(track, timeMs, durationMs, out) {
  const data = track.data;
  const t = ((timeMs % durationMs) + durationMs) % durationMs;
  let lo = 0, hi = track.count - 1;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1;
    if (data[mid * 4] <= t) lo = mid; else hi = mid;
  }
  let a = lo, b = hi;
  // После последнего ключа интерполируем к первому через границу цикла.
  let ta = data[a * 4], tb = data[b * 4], sampleT = t;
  if (t >= data[(track.count - 1) * 4]) {
    a = track.count - 1; b = 0;
    ta = data[a * 4]; tb = durationMs + data[0];
  } else if (t < data[0]) {
    a = track.count - 1; b = 0;
    ta = data[a * 4] - durationMs; tb = data[0];
  }
  if (sampleT < ta) sampleT += durationMs;
  const k = tb > ta ? (sampleT - ta) / (tb - ta) : 0;
  const ai = a * 4, bi = b * 4;
  out[0] = data[ai + 1] + (data[bi + 1] - data[ai + 1]) * k;
  out[1] = data[ai + 2] + (data[bi + 2] - data[ai + 2]) * k;
  out[2] = data[ai + 3] + (data[bi + 3] - data[ai + 3]) * k;
  return out;
}

function seeded(seed) {
  let x = seed | 0;
  return () => {
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    return (x >>> 0) / 4294967296;
  };
}

export class BodyIdle {
  constructor(model, cfg = {}) {
    this.model = model;
    this.cfg = cfg.bodyIdle || {};
    this.motionEnabled = this.cfg.enabled !== false;
    this.layers = [];
    this.timeMs = 0;
    this.bones = new Map();
    this.rest = new Map();
    for (const name of BONE_ZONES[BONE_LAYERS.BODY]) {
      const bone = model.bones[name];
      if (!bone) continue;
      this.bones.set(name, bone);
      this.rest.set(name, bone.quaternion.clone());
    }
    this._sum = new Map();
    for (const name of this.bones.keys()) this._sum.set(name, new Float32Array(3));
    this._sample = new Float32Array(3);
    this._e = new THREE.Euler();
    this._q = new THREE.Quaternion();
  }

  async loadClips(fetcher = fetch) {
    const specs = this.cfg.clips || [];
    const rnd = seeded(this.cfg.seed ?? 731927);
    const loaded = [];
    for (const spec of specs) {
      try {
        const base = (this.cfg.basePath || '/avatar/body-clips').replace(/\/$/, '');
        const response = await fetcher(`${base}/${spec.file}`);
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        const clip = compileBodyClip(await response.json());
        loaded.push({
          clip,
          weight: Math.max(0, Number(spec.weight) || 0),
          phaseMs: rnd() * clip.durationMs,
        });
        if (clip.droppedBones.length) {
          console.warn(`avatar: body clip ${clip.name} ignored bones:`, clip.droppedBones);
        }
      } catch (error) {
        console.warn(`avatar: body clip ${spec.file} unavailable:`, error);
      }
    }
    this.layers = loaded;
    return this;
  }

  /** Dev freeze: сохраняет текущую позу, но останавливает фазы клипов. */
  setMotionEnabled(on) { this.motionEnabled = !!on; }

  update(dt) {
    if (this.motionEnabled) this.timeMs += dt * 1000;
    for (const sum of this._sum.values()) sum.fill(0);

    for (const layer of this.layers) {
      const clipTime = this.timeMs + layer.phaseMs;
      for (const track of layer.clip.tracks) {
        const sum = this._sum.get(track.bone);
        if (!sum) continue;
        sampleBodyTrack(track, clipTime, layer.clip.durationMs, this._sample);
        sum[0] += this._sample[0] * layer.weight;
        sum[1] += this._sample[1] * layer.weight;
        sum[2] += this._sample[2] * layer.weight;
      }
    }

    for (const [name, sum] of this._sum) {
      const bone = this.bones.get(name);
      this._e.set(sum[0] * DEG, sum[1] * DEG, sum[2] * DEG);
      bone.quaternion.copy(this.rest.get(name)).multiply(this._q.setFromEuler(this._e));
    }
  }

  debug() {
    return {
      clips: this.layers.map((layer) => layer.clip.name),
      enabled: this.motionEnabled,
      ownedBones: [...this.bones.keys()],
    };
  }
}

