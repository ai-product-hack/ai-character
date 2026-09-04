import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { BONE_LAYERS, BONE_ZONES, boneOwner } from '../src/bone-zones.js';
import { BodyIdle, compileBodyClip, sampleBodyTrack } from '../src/body-idle.js';

const here = dirname(fileURLToPath(import.meta.url));
const cfg = JSON.parse(readFileSync(resolve(here, '../behavior.config.json'), 'utf8'));

function model() {
  const bones = {};
  for (const name of [...BONE_ZONES.body, ...BONE_ZONES.micro]) {
    const bone = new THREE.Bone(); bone.name = name; bones[name] = bone;
  }
  return { bones };
}

const raw = {
  schema: 'avatar-body-clip@1', name: 'test', durationMs: 1000,
  tracks: {
    Hips: [[0, 0, 0, 0], [500, 10, 0, 0], [1000, 0, 0, 0]],
    Head: [[0, 90, 0, 0], [1000, 90, 0, 0]],
  },
};

test('bone zones do not overlap and protect Neck/Head', () => {
  const all = [...BONE_ZONES.body, ...BONE_ZONES.micro];
  assert.equal(new Set(all).size, all.length);
  assert.equal(boneOwner('Head'), BONE_LAYERS.MICRO);
  assert.equal(boneOwner('Neck'), BONE_LAYERS.MICRO);
  assert.equal(boneOwner('Hips'), BONE_LAYERS.BODY);
});

test('body clip drops protected tracks at import', () => {
  const clip = compileBodyClip(raw);
  assert.deepEqual(clip.tracks.map((t) => t.bone), ['Hips']);
  assert.deepEqual(clip.droppedBones, ['Head']);
});

test('track sampling interpolates and loops without a seam', () => {
  const clip = compileBodyClip(raw);
  const out = new Float32Array(3);
  sampleBodyTrack(clip.tracks[0], 250, 1000, out);
  assert.ok(Math.abs(out[0] - 5) < 1e-6);
  sampleBodyTrack(clip.tracks[0], 1001, 1000, out);
  assert.ok(Math.abs(out[0] - 0.02) < 1e-4);
});

test('three calm clips load with independent randomized phases', async () => {
  const idle = new BodyIdle(model(), cfg);
  const fetcher = async (url) => {
    const file = resolve(here, '..', url.replace('/avatar/', ''));
    return { ok: true, json: async () => JSON.parse(readFileSync(file, 'utf8')) };
  };
  await idle.loadClips(fetcher);
  assert.equal(idle.layers.length, 3);
  assert.equal(new Set(idle.layers.map((l) => Math.round(l.phaseMs))).size, 3);
  assert.ok(idle.layers.every((l) => l.weight > 0 && l.weight <= 0.35));
});

test('body layer changes owned bones but never Head or Neck', async () => {
  const m = model();
  const idle = new BodyIdle(m, { bodyIdle: {
    enabled: true, seed: 1, basePath: '/x', clips: [{ file: 'a.json', weight: 0.3 }],
  } });
  await idle.loadClips(async () => ({ ok: true, json: async () => raw }));
  const head = m.bones.Head.quaternion.clone();
  const neck = m.bones.Neck.quaternion.clone();
  idle.update(0.25);
  assert.ok(m.bones.Hips.quaternion.angleTo(new THREE.Quaternion()) > 0.001);
  assert.ok(m.bones.Head.quaternion.equals(head));
  assert.ok(m.bones.Neck.quaternion.equals(neck));
});

test('dev freeze keeps the current additive pose', async () => {
  const m = model();
  const idle = new BodyIdle(m, { bodyIdle: {
    enabled: true, seed: 2, basePath: '/x', clips: [{ file: 'a.json', weight: 0.3 }],
  } });
  await idle.loadClips(async () => ({ ok: true, json: async () => raw }));
  idle.update(0.2);
  idle.setMotionEnabled(false);
  const frozen = m.bones.Hips.quaternion.clone();
  idle.update(2);
  assert.ok(m.bones.Hips.quaternion.angleTo(frozen) < 1e-9);
});

