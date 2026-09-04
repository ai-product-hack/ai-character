// Явное владение костями. Два анимационных слоя никогда не записывают один
// и тот же локальный поворот: их движения складываются естественно через
// иерархию скелета, а не зависят от порядка вызовов в Avatar.frame().

export const BONE_LAYERS = Object.freeze({
  BODY: 'body',
  MICRO: 'micro',
});

export const BONE_ZONES = Object.freeze({
  [BONE_LAYERS.BODY]: Object.freeze([
    'Hips', 'Spine', 'Spine1', 'LeftArm', 'RightArm',
  ]),
  [BONE_LAYERS.MICRO]: Object.freeze([
    'Spine2', 'LeftShoulder', 'RightShoulder', 'Neck', 'Head',
    'LeftEye', 'RightEye',
  ]),
});

const OWNER = new Map();
for (const [layer, names] of Object.entries(BONE_ZONES)) {
  for (const name of names) {
    if (OWNER.has(name)) throw new Error(`bone zone conflict: ${name}`);
    OWNER.set(name, layer);
  }
}

export function boneOwner(name) { return OWNER.get(name) || null; }
export function bodyOwns(name) { return boneOwner(name) === BONE_LAYERS.BODY; }

