import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeClip, sampleClip, trimClip, validateClip } from './clip.js';

const clip = makeClip('x', [
  { tMs: 100, weights: [0, 1] },
  { tMs: 200, weights: [1, 0] },
  { tMs: 300, weights: [0, 1] },
], ['a', 'b']);

test('makeClip normalizes timestamps', () => {
  assert.equal(clip.durationMs, 200);
  assert.deepEqual(clip.frames.map((f) => f.tMs), [0, 100, 200]);
  assert.equal(validateClip(clip), null);
});

test('sampling interpolates and loop seam is continuous', () => {
  assert.deepEqual(sampleClip(clip, 50, false, 0).weights, [0.5, 0.5]);
  const before = sampleClip(clip, 199, true, 80).weights[0];
  const after = sampleClip(clip, 201, true, 80).weights[0];
  assert.ok(Math.abs(before - after) < 0.05);
});

test('trim inserts interpolated boundary frames', () => {
  const out = trimClip(clip, 40, 160);
  assert.equal(out.durationMs, 120);
  assert.equal(out.frames[0].tMs, 0);
  assert.equal(out.frames.at(-1).tMs, 120);
  assert.ok(Math.abs(out.frames[0].weights[0] - 0.4) < 1e-6);
});
