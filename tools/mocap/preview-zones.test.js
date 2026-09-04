import test from 'node:test';
import assert from 'node:assert/strict';
import { LAYERS } from '../../avatar/src/zones.js';
import { previewLayerFor } from './preview-zones.js';

test('mocap preview shows captured gaze without giving it to emotion runtime', () => {
  assert.equal(previewLayerFor('eyeLookOutLeft'), LAYERS.IDLE);
  assert.equal(previewLayerFor('eyeLookDownRight'), LAYERS.IDLE);
  assert.equal(previewLayerFor('browInnerUp'), LAYERS.EMOTION);
  assert.equal(previewLayerFor('jawOpen'), null);
});

