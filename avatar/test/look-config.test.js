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

test('catchlights reveal the iris instead of masking gaze', () => {
  assert.ok(look.eyes.catchlight.sizeRel <= 0.04,
    'catchlight must stay small enough for pupil motion to remain readable');
  assert.ok(look.eyes.catchlight.intensity <= 1,
    'untone-mapped catchlight must not clip to a flat white disc');
});

test('expensive post path has a measured pixel budget and no redundant DOF', () => {
  assert.equal(look.post.maxPixelRatio, 1.2);
  assert.equal(look.post.dof.enabled, false);
  assert.equal(look.renderer.maxPixelRatio, 1.5);
});

test('portrait target uses the actual midpoint of both eye bones', async () => {
  const THREE = await import('three');
  const { AvatarModel } = await import('../src/model.js');
  const root = new THREE.Object3D();
  const left = new THREE.Bone(), right = new THREE.Bone();
  left.position.set(0.0358, 1.694, 0.0721);
  right.position.set(-0.0352, 1.693, 0.0735);
  root.add(left, right);
  const model = { root, bones: { LeftEye:left, RightEye:right }, cfg:look };
  const target = AvatarModel.prototype.frameTarget.call(model);
  assert.ok(Math.abs(target.x - 0.0003) < 1e-8);
  assert.ok(Math.abs(target.y - 1.6935) < 1e-8);
});
