// Тесты автомата состояний и слоя эмоции.
//
// Проверяется то, что названо в приёмке: все четыре состояния переключаются
// плавно и визуально различимы, эмоции видны и не конфликтуют с артикуляцией,
// перебивание отыгрывает реакцию.

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import * as THREE from 'three';

import { EmotionLayer } from '../src/emotion.js';
import { StateMachine, STATES } from '../src/states.js';
import { Microbehavior } from '../src/behavior.js';
import { VisemeLayer } from '../src/viseme.js';
import { MorphWriter } from '../src/morphs.js';
import { ManualClock } from '../src/clock.js';
import { LAYERS } from '../src/zones.js';

const here = dirname(fileURLToPath(import.meta.url));
const expr = JSON.parse(readFileSync(resolve(here, '../expression.config.json'), 'utf8'));
const behCfg = JSON.parse(readFileSync(resolve(here, '../behavior.config.json'), 'utf8'));
const visCfg = JSON.parse(readFileSync(resolve(here, '../visemes.json'), 'utf8'));
const report = JSON.parse(readFileSync(resolve(here, '../../bench/results/r5_glb_inspect.json'), 'utf8'))[0];

function makeRig() {
  const bones = {};
  const root = new THREE.Object3D();
  const mk = (name, pos, parent) => {
    const b = new THREE.Bone();
    b.name = name; b.position.copy(pos);
    (parent || root).add(b); bones[name] = b; return b;
  };
  const spine2 = mk('Spine2', new THREE.Vector3(0, 1.2957, 0.0112));
  const neck = mk('Neck', new THREE.Vector3(0, 0.1908, 0), spine2);
  const head = mk('Head', new THREE.Vector3(0, 0.1289, 0), neck);
  mk('LeftEye', new THREE.Vector3(0.0358, 0.0792, 0.0721), head);
  mk('RightEye', new THREE.Vector3(-0.0352, 0.0781, 0.0735), head);
  mk('LeftShoulder', new THREE.Vector3(0.0542, 0.1707, 0.0037), spine2);
  mk('RightShoulder', new THREE.Vector3(-0.0542, 0.1707, 0.0037), spine2);
  root.updateMatrixWorld(true);
  return { root, bones };
}

function setup() {
  const { root, bones } = makeRig();
  const meshes = Object.entries(report.meshes)
    .filter(([, i]) => i.count > 0)
    .map(([name, info]) => {
      const dict = {};
      info.names.forEach((n, i) => { dict[n] = i; });
      return { name, morphTargetDictionary: dict, morphTargetInfluences: new Array(info.names.length).fill(0) };
    });
  const morphs = new MorphWriter(meshes, { strict: true });
  const model = {
    root, bones, morphs, meshes,
    cfg: { model: { neckBone: 'Neck', headBone: 'Head', chestBone: 'Spine2',
                    eyeBones: ['LeftEye', 'RightEye'],
                    shoulderBones: ['LeftShoulder', 'RightShoulder'] } },
  };
  const behavior = new Microbehavior(model, structuredClone(behCfg));
  const emotion = new EmotionLayer(morphs, structuredClone(expr));
  const states = new StateMachine(behavior, emotion, emotion.cfg);
  const visemes = new VisemeLayer(morphs, structuredClone(visCfg));
  const clock = new ManualClock(0.024);
  visemes.attachClock(clock);
  return { model, morphs, behavior, emotion, states, visemes, clock, meshes };
}

/** Прогнать кадры в том же порядке, что и настоящий кадр аватара. */
function run(ctx, seconds, onFrame) {
  const dt = 1 / 60;
  for (let i = 0; i < Math.round(seconds / dt); i++) {
    ctx.clock.advance(dt);
    const fast = ctx.states.update(dt);
    ctx.morphs.begin();
    ctx.visemes.update(dt * ctx.emotion.articulationRate);
    ctx.emotion.update(dt, fast);
    ctx.behavior.update(dt, i * dt);
    ctx.morphs.commit();
    if (onFrame) onFrame(i * dt, i);
  }
}

/** Снимок выразительных морфов. */
function pose(morphs) {
  const out = {};
  for (const n of ['browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'browDownLeft',
                   'browDownRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft',
                   'eyeWideRight', 'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft']) {
    const v = morphs.get(n);
    if (v > 0.005) out[n] = v;
  }
  return out;
}

const dist = (a, b) => {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  let s = 0;
  for (const k of keys) s += ((a[k] || 0) - (b[k] || 0)) ** 2;
  return Math.sqrt(s);
};

describe('матрица поз и модель согласованы', () => {
  test('все морфы состояний и эмоций есть в модели', () => {
    const ctx = setup();
    assert.deepEqual(ctx.emotion.validate(), []);
  });

  test('позы не выходят за пределы разрешённых слою зон', () => {
    const ctx = setup();
    ctx.morphs.begin();
    const all = [];
    for (const s of Object.values(expr.states)) {
      if (!s || typeof s !== 'object') continue;
      all.push(s.pose, s.impatience && s.impatience.pose);
    }
    for (const e of Object.values(expr.emotions)) {
      if (e && typeof e === 'object') all.push(e.pose);
    }
    for (const p of all) {
      for (const morph of Object.keys(p || {})) {
        if (morph.startsWith('_')) continue;
        assert.equal(ctx.morphs.write(LAYERS.EMOTION, morph, 0.1), true,
          `морф ${morph} не разрешён слою эмоции`);
      }
    }
  });
});

describe('состояния визуально различимы', () => {
  test('четыре состояния дают четыре разные позы', () => {
    const poses = {};
    for (const st of STATES) {
      const ctx = setup();
      ctx.states.set(st);
      run(ctx, 1.0);
      poses[st] = pose(ctx.morphs);
    }
    // Каждая пара состояний должна заметно отличаться.
    const names = Object.keys(poses);
    for (let i = 0; i < names.length; i++) {
      for (let j = i + 1; j < names.length; j++) {
        const d = dist(poses[names[i]], poses[names[j]]);
        assert.ok(d > 0.05,
          `${names[i]} и ${names[j]} почти неразличимы: расстояние ${d.toFixed(3)}`);
      }
    }
  });

  test('thinking уводит взгляд вверх-влево', () => {
    const ctx = setup();
    ctx.states.set('thinking');
    run(ctx, 2.0);
    assert.ok(ctx.behavior.gazeBias.yaw < -10, 'взгляд должен уйти влево');
    assert.ok(ctx.behavior.gazeBias.pitch > 5, 'и вверх');
    // Сам взгляд должен оказаться в той области, а не метаться к собеседнику.
    let inRegion = 0, total = 0;
    run(ctx, 6.0, () => {
      total++;
      if (ctx.behavior.gaze.yaw < -5 && ctx.behavior.gaze.pitch > 2) inRegion++;
    });
    assert.ok(inRegion / total > 0.7,
      `взгляд в нужной области только ${(100 * inRegion / total).toFixed(0)}% времени`);
  });

  test('в thinking моргания редеют', () => {
    const listening = setup();
    listening.states.set('listening');
    let blinksL = 0, prev = 'idle';
    run(listening, 60, () => {
      if (listening.behavior.blink.phase === 'close' && prev !== 'close') blinksL++;
      prev = listening.behavior.blink.phase;
    });

    const thinking = setup();
    thinking.states.set('thinking');
    let blinksT = 0; prev = 'idle';
    run(thinking, 60, () => {
      if (thinking.behavior.blink.phase === 'close' && prev !== 'close') blinksT++;
      prev = thinking.behavior.blink.phase;
    });
    assert.ok(blinksT < blinksL * 0.8,
      `в thinking ${blinksT} морганий против ${blinksL} в listening — не редеют`);
  });

  test('listening наклоняет голову', () => {
    const ctx = setup();
    ctx.states.set('listening');
    run(ctx, 0.5);
    assert.ok(Math.abs(ctx.behavior.headTiltDeg) > 1, 'наклон головы должен быть заметен');
  });
});

describe('нетерпение в listening', () => {
  test('до порога нетерпения нет', () => {
    const ctx = setup();
    ctx.states.set('listening');
    run(ctx, expr.states.listening.impatience.afterSec - 1);
    assert.equal(ctx.states.impatience, 0);
  });

  test('после порога нарастает плавно, а не скачком', () => {
    const ctx = setup();
    ctx.states.set('listening');
    const imp = expr.states.listening.impatience;
    const seen = [];
    run(ctx, imp.afterSec + imp.rampSec + 1, () => seen.push(ctx.states.impatience));
    assert.equal(seen.at(-1), 1, 'нетерпение должно дойти до предела');
    // Плавность: между соседними кадрами не должно быть скачка.
    let maxStep = 0;
    for (let i = 1; i < seen.length; i++) maxStep = Math.max(maxStep, seen[i] - seen[i - 1]);
    assert.ok(maxStep < 0.02, `скачок нетерпения ${maxStep.toFixed(3)} за кадр`);
  });

  test('нетерпение уводит взгляд в сторону и поднимает бровь', () => {
    const ctx = setup();
    ctx.states.set('listening');
    const imp = expr.states.listening.impatience;
    run(ctx, imp.afterSec + imp.rampSec + 1);
    assert.ok(Math.abs(ctx.behavior.gazeBias.yaw) > 8, 'взгляд должен уйти в сторону');
    assert.ok(ctx.morphs.get('browOuterUpLeft') > 0.15, 'бровь должна подняться');
  });

  test('смена состояния обнуляет нетерпение', () => {
    const ctx = setup();
    ctx.states.set('listening');
    run(ctx, expr.states.listening.impatience.afterSec + 3);
    assert.ok(ctx.states.impatience > 0);
    ctx.states.set('thinking');
    assert.equal(ctx.states.impatience, 0);
  });

  test('повторная установка того же состояния не сбрасывает накопленное', () => {
    const ctx = setup();
    ctx.states.set('listening');
    run(ctx, expr.states.listening.impatience.afterSec + 3);
    const before = ctx.states.impatience;
    ctx.states.set('listening');            // как если бы вызвали ещё раз
    assert.equal(ctx.states.impatience, before);
  });

  test('набор текста откладывает порог, а не только гасит накопленное', () => {
    const ctx = setup();
    ctx.states.set('listening');
    const imp = expr.states.listening.impatience;
    // Печатает раз в секунду дольше порога — нетерпение не должно появиться.
    for (let i = 0; i < imp.afterSec + 4; i++) {
      ctx.states.noteActivity();
      run(ctx, 1);
      assert.equal(ctx.states.impatience, 0, `нетерпение на ${i}-й секунде набора`);
    }
  });

  test('нетерпение спадает после набора и снова растёт на тишине', () => {
    const ctx = setup();
    ctx.states.set('listening');
    const imp = expr.states.listening.impatience;
    run(ctx, imp.afterSec + imp.rampSec + 1);
    assert.equal(ctx.states.impatience, 1);
    ctx.states.noteActivity();
    run(ctx, 0.1);
    assert.equal(ctx.states.impatience, 0, 'набор должен снять нетерпение');
    run(ctx, imp.afterSec + imp.rampSec + 1);
    assert.equal(ctx.states.impatience, 1, 'на новой тишине нарастает снова');
  });

  test('набор не сбрасывает состояние и не трогает часы состояния', () => {
    const ctx = setup();
    ctx.states.set('listening');
    run(ctx, 3);
    const since = ctx.states.sinceEnter;
    ctx.states.noteActivity();
    assert.equal(ctx.states.state, 'listening');
    assert.equal(ctx.states.sinceEnter, since);
  });
});

describe('перебивание', () => {
  test('взгляд возвращается на собеседника мгновенно', () => {
    const ctx = setup();
    ctx.states.set('listening');
    run(ctx, 5);
    ctx.behavior.gaze.yaw = 12; ctx.behavior.gaze.pitch = -6;
    ctx.states.set('interrupted');
    // Снап происходит в момент установки состояния, а не за время перехода.
    assert.equal(ctx.behavior.gaze.yaw, 0);
    assert.equal(ctx.behavior.gaze.pitch, 0);
  });

  test('брови взлетают быстрее обычного перехода', () => {
    const ctx = setup();
    ctx.states.set('speaking');
    run(ctx, 0.5);
    ctx.states.set('interrupted');
    const target = expr.states.interrupted.pose.browOuterUpLeft;
    let reachedAt = null;
    run(ctx, 0.5, (t) => {
      if (reachedAt === null && ctx.morphs.get('browOuterUpLeft') > target * 0.8) reachedAt = t * 1000;
    });
    // Первая версия применяла enterMs один кадр, и брови выходили на 80%
    // за 273 мс вместо обещанных 90.
    assert.ok(reachedAt !== null && reachedAt < 160,
      `брови вышли на 80% за ${reachedAt === null ? '>500' : reachedAt.toFixed(0)} мс`);
  });

  test('через holdMs автомат сам уходит в listening', () => {
    const ctx = setup();
    ctx.states.set('interrupted');
    const hold = expr.states.interrupted.holdMs;
    run(ctx, hold / 1000 - 0.1);
    assert.equal(ctx.states.state, 'interrupted');
    run(ctx, 0.2);
    assert.equal(ctx.states.state, 'listening');
  });

  test('рот закрывается, пока брови идут вверх', () => {
    const ctx = setup();
    ctx.clock.anchor(0);
    ctx.states.set('speaking');
    ctx.visemes.playGeneration('g', [
      { pts_ms: 0, viseme: 'AA' }, { pts_ms: 3000, viseme: 'AA' },
    ]);
    run(ctx, 0.5);
    assert.ok(ctx.morphs.get('jawOpen') > 0.1, 'до перебивания рот открыт');

    ctx.visemes.cancel('g', visCfg.timing.interruptReleaseMs);
    ctx.states.set('interrupted');
    let mouthClosedAt = null, browAt = null;
    run(ctx, 0.6, (t) => {
      if (mouthClosedAt === null && ctx.morphs.get('jawOpen') < 0.02) mouthClosedAt = t * 1000;
      if (browAt === null && ctx.morphs.get('browOuterUpLeft') > 0.3) browAt = t * 1000;
    });
    assert.ok(mouthClosedAt !== null && mouthClosedAt < 250, `рот закрылся за ${mouthClosedAt} мс`);
    assert.ok(browAt !== null && browAt < 250, `брови поднялись за ${browAt} мс`);
  });
});

describe('эмоции', () => {
  test('пять эмоций дают пять разных поз', () => {
    const poses = {};
    for (const em of Object.keys(expr.emotions)) {
      if (em.startsWith('_')) continue;
      const ctx = setup();
      ctx.emotion.setEmotion(em, 1);
      ctx.states.apply();
      run(ctx, 1.0);
      poses[em] = pose(ctx.morphs);
    }
    const names = Object.keys(poses);
    assert.equal(names.length, 5);
    for (let i = 0; i < names.length; i++) {
      for (let j = i + 1; j < names.length; j++) {
        const d = dist(poses[names[i]], poses[names[j]]);
        assert.ok(d > 0.08, `${names[i]} и ${names[j]} похожи: ${d.toFixed(3)}`);
      }
    }
  });

  test('интенсивность масштабирует позу', () => {
    const half = setup();
    half.emotion.setEmotion('pressing', 0.5);
    run(half, 1.0);
    const full = setup();
    full.emotion.setEmotion('pressing', 1.0);
    run(full, 1.0);
    const a = half.morphs.get('browDownLeft'), b = full.morphs.get('browDownLeft');
    assert.ok(Math.abs(a / b - 0.5) < 0.1, `${a.toFixed(3)} против ${b.toFixed(3)} — не масштабируется`);
  });

  test('эмоция не трогает морфы рта, которыми владеют висемы', () => {
    const ctx = setup();
    ctx.clock.anchor(0);
    ctx.emotion.setEmotion('warming', 1);
    ctx.visemes.playGeneration('g', [
      { pts_ms: 0, viseme: 'MBP' }, { pts_ms: 3000, viseme: 'MBP' },
    ]);
    run(ctx, 1.0);
    // Губы сомкнуты висемой, а улыбка эмоции живёт на других морфах.
    assert.ok(ctx.morphs.get('mouthClose') > 0.3, 'висема должна держать губы сомкнутыми');
    assert.ok(ctx.morphs.get('mouthSmileLeft') > 0.1, 'улыбка при этом видна');
    assert.equal(ctx.morphs.stats.trespassWrites, 0, 'ни один слой не полез в чужую зону');
  });

  test('эмоция меняет скорость артикуляции, но не форму рта', () => {
    const fast = setup();
    fast.emotion.setEmotion('pressing', 1);
    const slow = setup();
    slow.emotion.setEmotion('skeptical', 1);
    assert.ok(fast.emotion.articulationRate > 1, 'напор ускоряет речь');
    assert.ok(slow.emotion.articulationRate < 1, 'скепсис замедляет');

    // При этом целевые веса висемы одинаковы: скорость — это время, не форма.
    for (const ctx of [fast, slow]) {
      ctx.clock.anchor(0);
      ctx.visemes.playGeneration('g', [
        { pts_ms: 0, viseme: 'AA' }, { pts_ms: 5000, viseme: 'AA' },
      ]);
      run(ctx, 2.0);
    }
    const a = fast.morphs.get('viseme_aa'), b = slow.morphs.get('viseme_aa');
    assert.ok(Math.abs(a - b) < 0.02, `форма рта разошлась: ${a.toFixed(3)} против ${b.toFixed(3)}`);
  });

  test('прищур эмоции и моргание не дерутся: моргание закрывает глаз полностью', () => {
    const ctx = setup();
    ctx.emotion.setEmotion('skeptical', 1);
    run(ctx, 1.0);
    const squint = ctx.morphs.get('eyeBlinkLeft');
    assert.ok(squint > 0.1 && squint < 0.4, `прищур ${squint.toFixed(3)} вне ожидаемого`);

    // Моргание поверх прищура — по правилу MAX глаз должен закрыться целиком.
    let maxBlink = 0;
    ctx.behavior.blink.next = 0.05;
    run(ctx, 2.0, () => { maxBlink = Math.max(maxBlink, ctx.morphs.get('eyeBlinkLeft')); });
    assert.ok(maxBlink > 0.95, `глаз закрылся только на ${maxBlink.toFixed(3)}`);

    // И отпустил обратно на уровень эмоции, а не в ноль.
    run(ctx, 1.0);
    assert.ok(Math.abs(ctx.morphs.get('eyeBlinkLeft') - squint) < 0.05,
      'после моргания прищур должен вернуться');
  });

  test('множители моргания состояния и эмоции перемножаются', () => {
    const ctx = setup();
    ctx.states.set('thinking');
    ctx.emotion.setEmotion('pressing', 1);
    ctx.states.apply();
    const expected = expr.states.thinking.blinkScale * expr.emotions.pressing.blinkScale;
    assert.ok(Math.abs(ctx.behavior.blinkScale - expected) < 1e-6,
      `${ctx.behavior.blinkScale} против ожидаемого ${expected}`);
  });
});

describe('переходы плавные', () => {
  test('смена состояния не даёт скачка морфов', () => {
    const ctx = setup();
    ctx.states.set('speaking');
    run(ctx, 1.0);
    const watch = ['browInnerUp', 'browDownLeft', 'browOuterUpLeft'];
    let prev = watch.map((m) => ctx.morphs.get(m));
    let maxStep = 0;
    ctx.states.set('thinking');
    run(ctx, 1.0, () => {
      const now = watch.map((m) => ctx.morphs.get(m));
      for (let i = 0; i < now.length; i++) maxStep = Math.max(maxStep, Math.abs(now[i] - prev[i]));
      prev = now;
    });
    // За кадр при переходе 200 мс изменение не может превышать ~8% от полного.
    assert.ok(maxStep < 0.08, `скачок морфа ${maxStep.toFixed(3)} за кадр`);
  });

  test('переход занимает заявленное время, а не мгновенный', () => {
    const ctx = setup();
    ctx.states.set('speaking');
    run(ctx, 1.0);
    ctx.states.set('thinking');
    const target = expr.states.thinking.pose.browInnerUp;
    let at80 = null;
    run(ctx, 1.5, (t) => {
      if (at80 === null && ctx.morphs.get('browInnerUp') > target * 0.8) at80 = t * 1000;
    });
    assert.ok(at80 > 100, `переход прошёл за ${at80} мс — слишком резко`);
    assert.ok(at80 < 600, `переход занял ${at80} мс — слишком вяло`);
  });
});

describe('бюджет кадра', () => {
  test('слой эмоции и автомат не аллоцируют', (t) => {
    if (typeof global.gc !== 'function') { t.skip('нужен --expose-gc'); return; }
    const ctx = setup();
    ctx.states.set('listening');
    ctx.emotion.setEmotion('skeptical', 1);
    run(ctx, 5);
    global.gc();
    const before = process.memoryUsage().heapUsed;
    run(ctx, 90);
    global.gc();
    const grew = process.memoryUsage().heapUsed - before;
    assert.ok(grew < 192 * 1024, `куча выросла на ${grew} байт за 90 с`);
  });
});
