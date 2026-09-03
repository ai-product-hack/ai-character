// Кто каким морфом владеет.
//
// Единственное место в модуле, где это записано. Правило из задания: конфликт
// за один морф разрешается приоритетом владельца, а не последней записью —
// поэтому владение объявлено явной таблицей, а не выводится из порядка вызовов.
//
// Разбиение по зонам (см. avatar/MODEL_REPORT.md, п. 2):
//   viseme  — механика рта: 15 висем Oculus, челюсть, губы, язык
//   emotion — выразительная мимика: брови, веки, улыбка, щёки, нос
//   idle    — микроповедение: моргание и морфы взгляда
//
// Морфы взгляда eyeLook* отданы idle, хотя взгляд мы ведём поворотом костей
// LeftEye/RightEye: владение всё равно закреплено, чтобы никакой другой слой
// не начал в них писать «просто потому что они свободны».

export const LAYERS = Object.freeze({
  VISEME: 'viseme',
  EMOTION: 'emotion',
  IDLE: 'idle',
});

// Порядок в массиве значения не имеет — веса пишутся по имени, а не по индексу.
// Индексы морфов у мешей разные (jawOpen — 49-й в Head_Mesh и 16-й в Teeth_Mesh),
// поэтому индексная адресация здесь была бы гарантированным багом.
export const ZONE_MORPHS = Object.freeze({
  [LAYERS.VISEME]: Object.freeze([
    // 15 висем Oculus — база артикуляции
    'viseme_sil', 'viseme_PP', 'viseme_FF', 'viseme_TH', 'viseme_DD',
    'viseme_kk', 'viseme_CH', 'viseme_SS', 'viseme_nn', 'viseme_RR',
    'viseme_aa', 'viseme_E', 'viseme_I', 'viseme_O', 'viseme_U',
    // ARKit-механика рта — ей дожимаются русские висемы, которых в наборе
    // Oculus нет (`ы` через mouthStretch* и т.д., см. MODEL_REPORT п. 3)
    'jawOpen', 'jawForward', 'jawLeft', 'jawRight',
    'mouthClose', 'mouthFunnel', 'mouthPucker',
    'mouthStretchLeft', 'mouthStretchRight',
    'mouthRollLower', 'mouthRollUpper',
    'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight',
    'mouthPressLeft', 'mouthPressRight',
    'mouthShrugLower', 'mouthShrugUpper',
    'mouthLeft', 'mouthRight',
    'tongueOut',
  ]),
  [LAYERS.EMOTION]: Object.freeze([
    'browDownLeft', 'browDownRight', 'browInnerUp',
    'browOuterUpLeft', 'browOuterUpRight',
    'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight',
    'mouthSmileLeft', 'mouthSmileRight',
    'mouthFrownLeft', 'mouthFrownRight',
    'mouthDimpleLeft', 'mouthDimpleRight',
    'cheekSquintLeft', 'cheekSquintRight',
    'noseSneerLeft', 'noseSneerRight',
    'cheekPuff',
  ]),
  [LAYERS.IDLE]: Object.freeze([
    'eyeBlinkLeft', 'eyeBlinkRight',
    'eyeLookDownLeft', 'eyeLookDownRight',
    'eyeLookUpLeft', 'eyeLookUpRight',
    'eyeLookInLeft', 'eyeLookInRight',
    'eyeLookOutLeft', 'eyeLookOutRight',
  ]),
});

// Агрегаты Avaturn: симметричные обёртки над парами L/R и дубли уже имеющихся
// морфов. Не используются намеренно — они пересекаются по зонам с ARKit-морфами
// и дали бы ровно тот конфликт за один морф, который зоны и разводят.
// Перечислены явно, чтобы writer отличал «сознательно не трогаем» от «опечатка».
export const UNUSED_MORPHS = Object.freeze([
  'mouthOpen',      // дубль jawOpen
  'mouthSmile',     // дубль mouthSmileLeft/Right
  'eyesClosed',     // дубль eyeBlinkLeft/Right
  'eyesLookUp',     // дубль eyeLookUpLeft/Right
  'eyesLookDown',   // дубль eyeLookDownLeft/Right
]);

/** Морф -> слой-владелец. Собирается один раз при импорте. */
export const OWNER = (() => {
  const map = new Map();
  for (const [layer, names] of Object.entries(ZONE_MORPHS)) {
    for (const name of names) {
      const prev = map.get(name);
      if (prev) {
        throw new Error(
          `zones.js: морф «${name}» объявлен и в «${prev}», и в «${layer}». ` +
          `У морфа должен быть ровно один владелец.`);
      }
      map.set(name, layer);
    }
  }
  for (const name of UNUSED_MORPHS) {
    if (map.has(name)) {
      throw new Error(`zones.js: морф «${name}» одновременно в UNUSED_MORPHS и в зоне «${map.get(name)}».`);
    }
  }
  return map;
})();
