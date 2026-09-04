import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const look = JSON.parse(readFileSync(new URL('../look.config.json', import.meta.url)));

test('front portrait uses two weak cool back rims', () => {
  assert.equal(look.lights.rim, undefined, 'old single rim must be removed');
  const rims = [look.lights.rimLeft, look.lights.rimRight];
  assert.ok(rims.every(Boolean));
  assert.ok(rims.every((rim) => rim.azimuthDeg > 120 && rim.azimuthDeg < 240));
  assert.ok(rims.every((rim) => rim.intensity < 15));
  assert.notEqual(rims[0].intensity, rims[1].intensity, 'perfect symmetry reads as synthetic');
});

test('camera remains the approved frontal camera', () => {
  assert.equal(look.camera.focalLengthMm, 45);
  assert.equal(look.camera.azimuthDeg, 6);
  assert.equal(look.camera.elevationDeg, 4);
  assert.equal(look.camera.frameHeightM, 0.46);
});

test('expensive post path has a measured pixel budget and no redundant DOF', () => {
  assert.equal(look.post.maxPixelRatio, 1.2);
  assert.equal(look.post.dof.enabled, false);
  assert.equal(look.renderer.maxPixelRatio, 1.5);
});
