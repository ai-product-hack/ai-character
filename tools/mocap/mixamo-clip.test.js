import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../avatar/node_modules/three/build/three.module.js';
import { animationToBodyClip, mixamoBoneName } from './mixamo-clip.js';

test('Mixamo track names normalize and protected bones are excluded', () => {
  assert.equal(mixamoBoneName('mixamorig:Hips.quaternion'), 'Hips');
  assert.equal(mixamoBoneName('mixamorigSpine1.quaternion'), 'Spine1');
  assert.equal(mixamoBoneName('mixamorigHead.position'), null);

  const id = new THREE.Quaternion();
  const turn = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, Math.PI / 6, 0));
  const animation = {
    duration: 1,
    tracks: [
      { name: 'mixamorigHips.quaternion', ValueTypeName: 'quaternion',
        times: new Float32Array([0, 1]), values: new Float32Array([...id, ...turn]) },
      { name: 'mixamorigHead.quaternion', ValueTypeName: 'quaternion',
        times: new Float32Array([0, 1]), values: new Float32Array([...id, ...turn]) },
    ],
  };
  const clip = animationToBodyClip(animation, 'idle', 'idle.fbx');
  assert.deepEqual(Object.keys(clip.tracks), ['Hips']);
  assert.ok(Math.abs(clip.tracks.Hips[1][2] - 30) < 0.01);
});
