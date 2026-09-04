import * as THREE from '../../avatar/node_modules/three/build/three.module.js';

const BODY_BONES = new Set(['Hips', 'Spine', 'Spine1', 'LeftArm', 'RightArm']);
const RAD_TO_DEG = 180 / Math.PI;

export function mixamoBoneName(trackName) {
  if (!trackName.endsWith('.quaternion')) return null;
  let name = trackName.slice(0, -'.quaternion'.length).split(/[/.]/).at(-1);
  name = name.replace(/^.*\[/, '').replace(/\]$/, '');
  return name.replace(/^mixamorig:?/i, '') || null;
}

/** Превратить QuaternionKeyframeTrack из FBXLoader в аддитивные градусы. */
export function animationToBodyClip(animation, name, sourceFile = '') {
  if (!animation?.tracks?.length) throw new Error('В FBX нет анимационных треков');
  const tracks = {};
  const q0 = new THREE.Quaternion();
  const inv = new THREE.Quaternion();
  const q = new THREE.Quaternion();
  const delta = new THREE.Quaternion();
  const e = new THREE.Euler(0, 0, 0, 'XYZ');
  let durationMs = Math.max(500, Math.round((animation.duration || 0) * 1000));

  for (const track of animation.tracks) {
    const bone = mixamoBoneName(track.name);
    if (!BODY_BONES.has(bone) || track.ValueTypeName !== 'quaternion') continue;
    const times = track.times, values = track.values;
    if (!times || times.length < 2 || values.length !== times.length * 4) continue;
    q0.fromArray(values, 0); inv.copy(q0).invert();
    const originMs = times[0] * 1000;
    const rows = [];
    for (let i = 0; i < times.length; i++) {
      q.fromArray(values, i * 4);
      delta.copy(inv).multiply(q).normalize();
      e.setFromQuaternion(delta, 'XYZ');
      const tMs = Math.round(times[i] * 1000 - originMs);
      rows.push([tMs,
        +(e.x * RAD_TO_DEG).toFixed(4),
        +(e.y * RAD_TO_DEG).toFixed(4),
        +(e.z * RAD_TO_DEG).toFixed(4)]);
      durationMs = Math.max(durationMs, tMs);
    }
    tracks[bone] = rows;
  }
  if (!Object.keys(tracks).length) {
    throw new Error('Не найдены Hips/Spine/Spine1/LeftArm/RightArm quaternion-треки Mixamo');
  }
  return {
    schema: 'avatar-body-clip@1', name, durationMs,
    source: { kind: 'mixamo-fbx', file: sourceFile, convertedAt: new Date().toISOString() },
    tracks,
  };
}
