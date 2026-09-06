// Тесты слоя микроповедения.
//
// Проверяется то, что отличает живое лицо от куклы и что легко сломать
// незаметно: асимметрия времён моргания, баллистичность саккад, компенсация
// движения головы взглядом, отсутствие аллокаций в кадре.
//
// three.js здесь настоящий — Microbehavior работает с кватернионами и
// матрицами, и подменять их заглушками значило бы тестировать заглушки.

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import * as THREE from 'three';

import { Microbehavior } from '../src/behavior.js';
import { MorphWriter } from '../src/morphs.js';
import { Noise1D, Lognormal } from '../src/noise.js';

const here = dirname(fileURLToPath(import.meta.url));
const cfg = JSON.parse(readFileSync(resolve(here, '../behavior.config.json'), 'utf8'));
const report = JSON.parse(readFileSync(resolve(here, '../../bench/results/r5_glb_inspect.json'), 'utf8'))[0];

/**
 * Скелет модели, воспроизведённый по замеренным мировым позициям костей:
 * Spine2 -> Neck -> Head -> {LeftEye, RightEye}, плюс плечи.
 */
function makeRig() {
  const bones = {};
  const mk = (name, pos, parent) => {
    const b = new THREE.Bone();
    b.name = name;
    b.position.copy(pos);
    if (parent) parent.add(b); 
    bones[name] = b;
    return b;
  };
  const root = new THREE.Object3D();
  const spine2 = mk('Spine2', new THREE.Vector3(0, 1.2957, 0.0112));
  root.add(spine2);
  const neck = mk('Neck', new THREE.Vector3(0, 0.1908, 0), spine2);
  const head = mk('Head', new THREE.Vector3(0, 0.1289, 0), neck);
  mk('LeftEye', new THREE.Vector3(0.0358, 0.0792, 0.0721), head);
  mk('RightEye', new THREE.Vector3(-0.0352, 0.0781, 0.0735), head);
  mk('LeftShoulder', new THREE.Vector3(0.0542, 0.1707, 0.0037), spine2);
  mk('RightShoulder', new THREE.Vector3(-0.0542, 0.1707, 0.0037), spine2);
  root.updateMatrixWorld(true);
  return { root, bones };
}

function makeModel() {
  const { root, bones } = makeRig();
  const meshes = Object.entries(report.meshes)
    .filter(([, i]) => i.count > 0)
    .map(([name, info]) => {
      const dict = {};
      info.names.forEach((n, i) => { dict[n] = i; });
      return { name, morphTargetDictionary: dict, morphTargetInfluences: new Array(info.names.length).fill(0) };
    });
  return {
    root, bones,
    morphs: new MorphWriter(meshes, { strict: true }),
    cfg: { model: { neckBone: 'Neck', headBone: 'Head', chestBone: 'Spine2',
                    eyeBones: ['LeftEye', 'RightEye'],
                    shoulderBones: ['LeftShoulder', 'RightShoulder'] } },
    meshes,
  };
}

/** Прогнать N кадров по 1/60 с. */
function run(beh, model, seconds, onFrame) {
  const dt = 1 / 60;
  const n = Math.round(seconds / dt);
  for (let i = 0; i < n; i++) {
    model.morphs.begin();
    beh.update(dt, i * dt);
    model.morphs.commit();
    if (onFrame) onFrame(i * dt, i);
  }
}

describe('моргание', () => {
  test('времена асимметричны: закрытие быстрее открытия', () => {
    const model = makeModel();
    const beh = new Microbehavior(model, cfg);
    beh.setEnabled('gaze', false); beh.setEnabled('head', false); beh.setEnabled('breath', false);

    // Ловим одно моргание целиком и меряем фазы по кадрам.
    const phases = [];
    beh.blink.next = 0.05;
    run(beh, model, 3, () => phases.push(beh.blink.phase));

    const count = (p) => phases.filter((x) => x === p).length;
    const closeMs = count('close') / 60 * 1000;
    const openMs = count('open') / 60 * 1000;
    assert.ok(openMs > closeMs * 1.3,
      `открытие ${openMs.toFixed(0)} мс должно быть заметно дольше закрытия ${closeMs.toFixed(0)} мс`);
  });

  test('веко проходит весь путь до полного закрытия и обратно', () => {
    const model = makeModel();
    const beh = new Microbehavior(model, cfg);
    beh.setEnabled('gaze', false); beh.setEnabled('head', false); beh.setEnabled('breath', false);
    beh.blink.next = 0.05;
    let max = 0, endValue = 1;
    run(beh, model, 3, () => { max = Math.max(max, beh.blink.value); endValue = beh.blink.value; });
    assert.equal(max, 1, 'глаз должен закрыться полностью');
    assert.equal(endValue, 0, 'и открыться обратно');
  });

  test('интервалы логнормальные: длинный хвост, а не равномерный поток', () => {
    const rnd = new Lognormal(cfg.blink.seed, cfg.blink.medianSec, cfg.blink.sigma);
    const a = Array.from({ length: 20000 }, () => rnd.sample()).sort((x, y) => x - y);
    const mean = a.reduce((p, c) => p + c, 0) / a.length;
    const median = a[a.length >> 1];
    assert.ok(mean > 3.4 && mean < 4.6, `среднее ${mean.toFixed(2)} с, ожидалось около 4`);
    assert.ok(mean > median, 'среднее правее медианы — признак правого хвоста');
    assert.ok(a.at(-1) > median * 3, `хвост слишком короткий: max ${a.at(-1).toFixed(1)} с`);
  });

  test('частота близка к человеческой, а не к тику', () => {
    // Первая версия притягивала моргание к саккадам добавлением, а не сдвигом,
    // и давала 43 моргания в минуту при медианном промежутке 1.1 с. В разговоре
    // человек моргает примерно 15-25 раз в минуту.
    const model = makeModel();
    const beh = new Microbehavior(model, cfg);
    const times = [];
    let prev = 'idle';
    const MIN = 5;
    run(beh, model, MIN * 60, (t) => {
      if (beh.blink.phase === 'close' && prev !== 'close') times.push(t);
      prev = beh.blink.phase;
    });
    // Двойное моргание — одно поведенческое событие, а не два: второе смыкание
    // идёт через doubleGapMs и рефрактерным промежутком не ограничено.
    const groupGap = (cfg.blink.doubleGapMs + cfg.blink.openMs + cfg.blink.closeMs) / 1000 + 0.1;
    const groups = times.filter((t, i) => i === 0 || t - times[i - 1] > groupGap);
    const perMin = groups.length / MIN;
    assert.ok(perMin >= 12 && perMin <= 26, `${perMin.toFixed(1)} морганий в минуту`);
    assert.ok(times.length > groups.length, 'двойных морганий не встретилось за 5 минут');

    const gaps = groups.slice(1).map((v, i) => v - groups[i]);
    const sorted = [...gaps].sort((a, b) => a - b);
    assert.ok(sorted[0] >= cfg.blink.refractorySec - 0.05,
      `промежуток ${sorted[0].toFixed(2)} с короче рефрактерного`);
    assert.ok(sorted.at(-1) > sorted[sorted.length >> 1] * 2,
      'нет длинного хвоста: поток слишком равномерный');
  });

  test('микроопускание бровей идёт вместе с морганием', () => {
    const model = makeModel();
    const beh = new Microbehavior(model, cfg);
    beh.setEnabled('gaze', false); beh.setEnabled('head', false); beh.setEnabled('breath', false);
    beh.blink.next = 0.05;
    let sawBrow = false;
    run(beh, model, 3, () => {
      if (beh.blink.value > 0.5) sawBrow = model.morphs.get('browDownLeft') > 0;
    });
    assert.ok(sawBrow, 'бровь должна опускаться на моргании, иначе веко дёргается в одиночку');
  });
});

describe('саккады', () => {
  test('взгляд прыгает, а не плывёт: доля кадров в броске мала', () => {
    const model = makeModel();
    const beh = new Microbehavior(model, cfg);
    beh.setEnabled('blink', false); beh.setEnabled('head', false); beh.setEnabled('breath', false);
    let sacc = 0, total = 0;
    run(beh, model, 60, () => { total++; if (beh.gaze.phase === 'saccade') sacc++; });
    const share = sacc / total;
    // 30-80 мс броска на 200-600 мс фиксации — это заведомо меньше четверти.
    assert.ok(share < 0.25, `в броске ${(share * 100).toFixed(0)}% кадров — взгляд плывёт, а не прыгает`);
    assert.ok(share > 0.02, `бросков почти нет (${(share * 100).toFixed(1)}%) — взгляд застыл`);
  });

  test('длительность броска лежит в 30-80 мс', () => {
    const model = makeModel();
    const beh = new Microbehavior(model, cfg);
    beh.setEnabled('blink', false); beh.setEnabled('head', false); beh.setEnabled('breath', false);
    const durations = [];
    let prev = 'fixate';
    run(beh, model, 120, () => {
      if (beh.gaze.phase === 'saccade' && prev === 'fixate') durations.push(beh.gaze.duration * 1000);
      prev = beh.gaze.phase;
    });
    assert.ok(durations.length > 50, `слишком мало саккад: ${durations.length}`);
    const min = Math.min(...durations), max = Math.max(...durations);
    assert.ok(min >= cfg.gaze.saccade.minMs - 1e-6, `бросок короче 30 мс: ${min.toFixed(1)}`);
    assert.ok(max <= cfg.gaze.saccade.maxMs + 1e-6, `бросок длиннее 80 мс: ${max.toFixed(1)}`);
  });

  test('взгляд не уходит в случайное блуждание и держится у собеседника', () => {
    const model = makeModel();
    const beh = new Microbehavior(model, cfg);
    beh.setEnabled('blink', false); beh.setEnabled('head', false); beh.setEnabled('breath', false);
    let maxAbs = 0;
    const samples = [];
    run(beh, model, 180, () => {
      maxAbs = Math.max(maxAbs, Math.abs(beh.gaze.yaw));
      samples.push(Math.hypot(beh.gaze.yaw, beh.gaze.pitch));
    });
    const mean = samples.reduce((p, c) => p + c, 0) / samples.length;
    assert.ok(maxAbs < cfg.gaze.limitDeg.yaw,
      `взгляд упёрся в предел (${maxAbs.toFixed(1)}°) — это случайное блуждание`);
    assert.ok(mean < 6, `взгляд в среднем в ${mean.toFixed(1)}° от собеседника, должен держаться ближе`);
  });

  test('фиксация почти неподвижна: микродрожь, а не дрейф', () => {
    const model = makeModel();
    const beh = new Microbehavior(model, cfg);
    beh.setEnabled('blink', false); beh.setEnabled('head', false); beh.setEnabled('breath', false);
    let maxStep = 0, prevYaw = null;
    run(beh, model, 30, () => {
      if (beh.gaze.phase === 'fixate') {
        if (prevYaw !== null) maxStep = Math.max(maxStep, Math.abs(beh.gaze.yaw - prevYaw));
        prevYaw = beh.gaze.yaw;
      } else prevYaw = null;
    });
    assert.ok(maxStep < 0.05, `во время фиксации взгляд шевелится на ${maxStep.toFixed(3)}°/кадр`);
  });
});

describe('компенсация движения головы (вестибулоокулярный рефлекс)', () => {
  test('при повороте головы точка взгляда остаётся на месте', () => {
    const model = makeModel();
    const beh = new Microbehavior(model, cfg);
    beh.setEnabled('blink', false); beh.setEnabled('head', false); beh.setEnabled('breath', false);
    const anchor = new THREE.Vector3(0, 1.694, 1.2);
    beh.setAnchor(anchor);
    // Заморозить саккады: проверяем только влияние головы.
    beh.gaze.duration = 1e9; beh.gaze.phase = 'fixate';
    beh.gaze.yaw = 0; beh.gaze.pitch = 0;
    beh.cfg = { ...cfg, gaze: { ...cfg.gaze, fixation: { ...cfg.gaze.fixation, tremorDeg: 0 } } };

    const hitPoint = () => {
      model.root.updateMatrixWorld(true);
      const eye = model.bones.LeftEye;
      const p = eye.getWorldPosition(new THREE.Vector3());
      const dir = new THREE.Vector3(0, 0, 1).applyQuaternion(eye.getWorldQuaternion(new THREE.Quaternion()));
      // Куда смотрит глаз на плоскости z = anchor.z
      const t = (anchor.z - p.z) / dir.z;
      return new THREE.Vector3().copy(p).addScaledVector(dir, t);
    };

    run(beh, model, 0.2);
    const before = hitPoint();

    // Резко повернуть шею на 12° — как если бы это сделал шум Перлина.
    model.bones.Neck.quaternion.setFromEuler(new THREE.Euler(0.06, 0.21, 0));
    run(beh, model, 0.2);
    const after = hitPoint();

    const drift = before.distanceTo(after);
    assert.ok(drift < 0.01,
      `точка взгляда уехала на ${(drift * 100).toFixed(1)} см при повороте головы — ` +
      `глаза не компенсируют движение головы`);
  });
});

describe('дыхание', () => {
  test('вдох быстрее выдоха', () => {
    const model = makeModel();
    const beh = new Microbehavior(model, cfg);
    const rising = [], falling = [];
    let prev = beh._breathCurve(0);
    for (let i = 1; i <= 1000; i++) {
      const v = beh._breathCurve(i / 1000);
      (v > prev ? rising : falling).push(i);
      prev = v;
    }
    assert.ok(rising.length < falling.length,
      `фаза роста ${rising.length} против спада ${falling.length} — вдох должен быть короче выдоха`);
    assert.ok(Math.abs(rising.length / 1000 - cfg.breath.inhaleFraction) < 0.05);
  });

  test('период гуляет: фиксированной периодичности нет', () => {
    const model = makeModel();
    const beh = new Microbehavior(model, cfg);
    const periods = new Set();
    for (let i = 0; i < 50; i++) { beh._newBreathPeriod(); periods.add(beh.breath.period.toFixed(3)); }
    assert.ok(periods.size > 40, `периодов всего ${periods.size} — дыхание слишком регулярное`);
  });
});

describe('каналы независимы', () => {
  // Порог здесь не «поменьше — значит лучше». У шума период 256 ячеек, поэтому
  // независимых отсчётов принципиально немного, и выборочная корреляция гуляет
  // сама по себе. Замерено на 400 парах заведомо независимых сидов: медиана
  // |r| = 0.048, p95 = 0.142, максимум 0.186, СКО 0.071. Значимой считается
  // корреляция выше трёх СКО, то есть 0.21; всё, что ниже, — шум оценки, и
  // более строгий порог отбраковывал бы нормальные сиды.
  const CORR_LIMIT = 0.21;

  const corr = (a, b) => {
    let cov = 0, sa = 0, sb = 0;
    for (let i = 0; i < 5120; i++) {
      const x = a.fbm(i * 0.05), y = b.fbm(i * 0.05);
      cov += x * y; sa += x * x; sb += y * y;
    }
    return cov / Math.sqrt(sa * sb);
  };

  test('все реально используемые потоки шума независимы', () => {
    const model = makeModel();
    const beh = new Microbehavior(model, cfg);
    const streams = Object.entries({
      headYaw: beh.headNoise.yaw, headPitch: beh.headNoise.pitch, headRoll: beh.headNoise.roll,
      tremorYaw: beh.tremorNoise.yaw, tremorPitch: beh.tremorNoise.pitch,
    });
    const bad = [];
    for (let i = 0; i < streams.length; i++) {
      for (let j = i + 1; j < streams.length; j++) {
        const r = corr(streams[i][1], streams[j][1]);
        if (Math.abs(r) >= CORR_LIMIT) bad.push(`${streams[i][0]}~${streams[j][0]}=${r.toFixed(3)}`);
      }
    }
    assert.deepEqual(bad, [], 'коррелирующие пары каналов — их периодичности совпадут');
  });

  test('период шума заведомо длиннее демонстрации', () => {
    // Таблица перестановок из 256 ячеек, аргумент = время * hz. Повтор рисунка
    // наступает через 256/hz секунд; на демо это должно быть недостижимо.
    const periodSec = 256 / cfg.head.hz;
    assert.ok(periodSec > 20 * 60,
      `рисунок движений головы повторится через ${(periodSec / 60).toFixed(0)} мин`);
  });

  test('заморозка канала выключает только его', () => {
    const model = makeModel();
    const beh = new Microbehavior(model, cfg);
    beh.setEnabled('blink', false);
    beh.blink.next = 0.05;
    let blinked = false, headMoved = false;
    const q0 = model.bones.Neck.quaternion.clone();
    run(beh, model, 3, () => {
      if (model.morphs.get('eyeBlinkLeft') > 0) blinked = true;
      if (!model.bones.Neck.quaternion.equals(q0)) headMoved = true;
    });
    assert.equal(blinked, false, 'замороженное моргание не должно писать в веки');
    assert.ok(headMoved, 'остальные каналы должны продолжать работать');
  });
});

describe('бюджет кадра', () => {
  test('update() не аллоцирует', (t) => {
    if (typeof global.gc !== 'function') { t.skip('нужен --expose-gc'); return; }
    const model = makeModel();
    const beh = new Microbehavior(model, cfg);
    run(beh, model, 5);                       // прогрев и JIT

    global.gc();
    const before = process.memoryUsage().heapUsed;
    run(beh, model, 120);                     // две минуты анимации
    global.gc();
    const grew = process.memoryUsage().heapUsed - before;
    assert.ok(grew < 128 * 1024, `куча выросла на ${grew} байт за 2 минуты анимации`);
  });
});

test('negative expression pitch lowers the chin', () => {
  const model = makeModel();
  const local = structuredClone(cfg);
  local.head.amplitudeDeg = {yaw:0,pitch:0,roll:0};
  local.head.gazeFollow.enabled = false;
  const behavior = new Microbehavior(model, local);
  behavior.setEnabled('gaze', false); behavior.setEnabled('breath', false);
  behavior.setHeadPose(0, -3, 0);
  run(behavior, model, 0.1);
  const forward = new THREE.Vector3(0,0,1);
  model.root.updateMatrixWorld(true);
  forward.applyQuaternion(model.bones.Head.getWorldQuaternion(new THREE.Quaternion()));
  assert.ok(forward.y < -0.04, `negative pitch must look down, got y=${forward.y}`);
});
