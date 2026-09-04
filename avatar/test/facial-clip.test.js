import { test } from 'node:test';
import assert from 'node:assert/strict';
import { compileFacialClip, sampleFacialClip } from '../src/facial-clip.js';

const morphs = {
  slotOf(name) { return new Map([['browInnerUp', 0], ['jawOpen', 1], ['eyeBlinkLeft', 2]]).get(name) ?? -1; },
};

test('runtime clip keeps only emotion-owned channels', () => {
  const clip = compileFacialClip({
    version: 1, kind: 'face-mocap', name: 'x', durationMs: 1000,
    channels: ['browInnerUp', 'jawOpen', 'unknown'],
    frames: [{ tMs: 0, weights: [0, 1, 1] }, { tMs: 1000, weights: [1, 0, 0] }],
  }, morphs);
  assert.deepEqual(clip.channels.map((c) => c.name), ['browInnerUp']);
  assert.deepEqual(clip.filtered, ['jawOpen', 'unknown']);
});

test('loop seam has no jump at wrap', () => {
  const clip = compileFacialClip({
    version: 1, kind: 'face-mocap', name: 'x', durationMs: 1000,
    channels: ['browInnerUp'],
    frames: [{ tMs: 0, weights: [0.1] }, { tMs: 500, weights: [0.9] }, { tMs: 1000, weights: [0.7] }],
  }, morphs);
  const before = sampleFacialClip(clip, 999, 300)[0];
  const after = sampleFacialClip(clip, 1001, 300)[0];
  assert.ok(Math.abs(before - after) < 0.02, `${before} -> ${after}`);
});
