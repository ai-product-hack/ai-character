// Тесты слоя артикуляции.
//
// Проверяется то, что названо в приёмке: буферизация трека целиком, отмена по
// generation_id без единого кадра от старой генерации, асимметричные времена,
// опережение, ограничение амплитуды на быстрой речи, и что время берётся
// только у часов плеера.

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import { SOURCE, VisemeLayer } from '../src/viseme.js';
import { MorphWriter } from '../src/morphs.js';
import { ManualClock } from '../src/clock.js';
import { LAYERS } from '../src/zones.js';

const here = dirname(fileURLToPath(import.meta.url));
const cfg = JSON.parse(readFileSync(resolve(here, '../visemes.json'), 'utf8'));
const report = JSON.parse(readFileSync(resolve(here, '../../bench/results/r5_glb_inspect.json'), 'utf8'))[0];

function makeWriter() {
  const meshes = Object.entries(report.meshes)
    .filter(([, i]) => i.count > 0)
    .map(([name, info]) => {
      const dict = {};
      info.names.forEach((n, i) => { dict[n] = i; });
      return { name, morphTargetDictionary: dict, morphTargetInfluences: new Array(info.names.length).fill(0) };
    });
  return { writer: new MorphWriter(meshes, { strict: true }), meshes };
}

function setup() {
  const { writer, meshes } = makeWriter();
  // Конфиг клонируется на каждый тест: слой держит матрицу по ссылке (это нужно,
  // чтобы слайдеры правили её на лету), поэтому без клона мутация из одного
  // теста протекает в остальные.
  const layer = new VisemeLayer(writer, structuredClone(cfg));
  const clock = new ManualClock(0.024);        // outputLatency 24 мс, как в S2
  layer.attachClock(clock);
  return { layer, writer, clock, meshes };
}

/** Прогнать кадры по 1/60 с, продвигая часы. */
function run(layer, writer, clock, seconds, onFrame) {
  const dt = 1 / 60;
  for (let i = 0; i < Math.round(seconds / dt); i++) {
    clock.advance(dt);
    writer.begin();
    layer.update(dt);
    writer.commit();
    if (onFrame) onFrame(clock.nowMs(), i);
  }
}

const TRACK = [
  { pts_ms: 0, viseme: 'SIL' },
  { pts_ms: 200, viseme: 'AA' },
  { pts_ms: 500, viseme: 'MBP' },
  { pts_ms: 800, viseme: 'OU' },
  { pts_ms: 1200, viseme: 'SIL' },
];

describe('матрица и модель согласованы', () => {
  test('все морфы матрицы есть в модели', () => {
    const { layer } = setup();
    assert.deepEqual(layer.validate(), []);
  });

  test('матрица не выходит за пределы зоны viseme', () => {
    const { layer, writer } = setup();
    writer.begin();
    for (const [name, weights] of Object.entries(cfg.matrix)) {
      for (const morph of Object.keys(weights)) {
        if (morph.startsWith('_')) continue;
        assert.equal(writer.write(LAYERS.VISEME, morph, 0.1), true,
          `висема ${name}: морф ${morph} не принадлежит зоне viseme`);
      }
    }
  });

  test('битый вес в матрице не рвёт геометрию', () => {
    const { layer, writer, clock, meshes } = setup();
    const byName = Object.fromEntries(meshes.map((m) => [m.name, m]));
    // Так выглядит опечатка в visemes.json или сорвавшийся слайдер.
    layer.matrix.AA = { viseme_aa: undefined, jawOpen: '0.7', mouthFunnel: NaN };
    clock.anchor(0);
    layer.playGeneration('g', [{ pts_ms: 0, viseme: 'AA' }, { pts_ms: 2000, viseme: 'AA' }]);
    run(layer, writer, clock, 0.5);
    for (const mesh of meshes) {
      for (const v of mesh.morphTargetInfluences) {
        assert.ok(Number.isFinite(v), `в ${mesh.name} завёлся ${v}`);
      }
    }
  });

  test('validate() называет битые веса поимённо', () => {
    const { layer } = setup();
    layer.matrix.AA = { viseme_aa: undefined, jawOpen: 0.7 };
    layer.validate();
    assert.deepEqual(layer.badWeights, ['AA.viseme_aa = undefined']);
  });

  test('в матрице ровно 13 висем', () => {
    assert.equal(Object.keys(cfg.matrix).length, 13);
  });
});

describe('время берётся у часов плеера', () => {
  test('без часов слой молчит', () => {
    const { layer, writer } = setup();
    layer.attachClock(null);
    layer.playGeneration('g1', TRACK);
    writer.begin(); layer.update(1 / 60); writer.commit();
    assert.equal(writer.get('jawOpen'), 0);
  });

  test('без якоря слой молчит: генерация ещё не началась', () => {
    const { layer, writer, clock } = setup();
    layer.playGeneration('g1', TRACK);
    assert.equal(clock.nowMs(), null);
    writer.begin(); layer.update(1 / 60); writer.commit();
    assert.equal(writer.get('jawOpen'), 0);
  });

  test('outputLatency вычитается: рот не опережает звук на 24 мс', () => {
    const { clock } = setup();
    clock.anchor(0);
    clock.advance(1.0);
    assert.equal(Math.round(clock.nowMs()), 976, '1000 мс минус 24 мс задержки вывода');
  });
});

describe('артикуляция по треку', () => {
  test('рот открывается на AA и закрывается на MBP', () => {
    const { layer, writer, clock } = setup();
    clock.anchor(0);
    layer.playGeneration('g1', TRACK);

    let jawOnAA = 0, jawOnMBP = 1, closeOnMBP = 0;
    run(layer, writer, clock, 1.4, (nowMs) => {
      if (nowMs > 300 && nowMs < 450) jawOnAA = Math.max(jawOnAA, writer.get('jawOpen'));
      if (nowMs > 600 && nowMs < 750) {
        jawOnMBP = Math.min(jawOnMBP, writer.get('jawOpen'));
        closeOnMBP = Math.max(closeOnMBP, writer.get('mouthClose'));
      }
    });
    // Пороги считаются от самой матрицы: её подкручивают слайдерами, и
    // абсолютные числа в тестах устаревали бы после каждой правки.
    const jawFull = cfg.matrix.AA.jawOpen;
    const closeFull = cfg.matrix.MBP.mouthClose;
    assert.ok(jawOnAA > jawFull * 0.7,
      `на AA челюсть открылась на ${jawOnAA.toFixed(2)} из ${jawFull}`);
    assert.ok(jawOnMBP < jawFull * 0.2, `на MBP челюсть не закрылась: ${jawOnMBP.toFixed(2)}`);
    assert.ok(closeOnMBP > closeFull * 0.7,
      `губы не сомкнулись: ${closeOnMBP.toFixed(2)} из ${closeFull}`);
  });

  test('веса доходят до зубов и языка, а не только до головы', () => {
    const { layer, writer, clock, meshes } = setup();
    const byName = Object.fromEntries(meshes.map((m) => [m.name, m]));
    const infl = (mesh, morph) => mesh.morphTargetInfluences[mesh.morphTargetDictionary[morph]];
    clock.anchor(0);
    layer.playGeneration('g1', TRACK);
    let teeth = 0, tongue = 0;
    run(layer, writer, clock, 0.6, () => {
      teeth = Math.max(teeth, infl(byName.Teeth_Mesh, 'jawOpen'));
      tongue = Math.max(tongue, infl(byName.Tongue_Mesh, 'viseme_aa'));
    });
    assert.ok(teeth > cfg.matrix.AA.jawOpen * 0.7, 'зубы должны опускаться вместе с челюстью');
    assert.ok(tongue > cfg.matrix.AA.viseme_aa * 0.7, 'язык должен двигаться вместе с висемой');
  });
});

describe('три вещи, отличающие живую артикуляцию от механической', () => {
  test('атака быстрее спада', () => {
    const { layer, writer, clock } = setup();
    clock.anchor(0);
    layer.playGeneration('g1', [
      { pts_ms: 0, viseme: 'AA' },
      { pts_ms: 400, viseme: 'SIL' },
      { pts_ms: 2000, viseme: 'SIL' },
    ]);
    const dt = 1 / 60;
    let riseFrames = 0, fallFrames = 0, peaked = false, peak = 0;
    for (let i = 0; i < 120; i++) {
      clock.advance(dt);
      writer.begin(); layer.update(dt); writer.commit();
      const v = writer.get('jawOpen');
      if (!peaked) {
        if (v > cfg.matrix.AA.jawOpen * 0.6) { peaked = true; peak = v; } else riseFrames++;
      } else if (v > cfg.matrix.AA.jawOpen * 0.1) fallFrames++;
    }
    assert.ok(fallFrames > riseFrames,
      `подъём ${riseFrames} кадров, спад ${fallFrames} — спад должен быть длиннее`);
  });

  test('на быстрой речи амплитуда режется вдвое', () => {
    const { layer, writer, clock } = setup();
    clock.anchor(0);
    // Интервал 40 мс — меньше порога fastSpeechMs = 80.
    const fast = [];
    for (let i = 0; i < 20; i++) fast.push({ pts_ms: i * 40, viseme: i % 2 ? 'AA' : 'MBP' });
    fast.push({ pts_ms: 2000, viseme: 'SIL' });
    layer.playGeneration('g1', fast);
    let maxJaw = 0;
    // Мерить только внутри быстрой части. Последняя висема перед длинной
    // паузой стоит долго, интервал до следующей большой — она законно
    // раскрывается полностью, и хвост испортил бы замер.
    run(layer, writer, clock, 0.8, (nowMs) => {
      if (nowMs < 700) maxJaw = Math.max(maxJaw, writer.get('jawOpen'));
    });
    const full = cfg.matrix.AA.jawOpen;
    assert.ok(maxJaw < full * 0.75,
      `на быстрой речи челюсть дошла до ${maxJaw.toFixed(2)} при полном ${full} — не режется`);
  });

  test('подряд идущие одинаковые висемы не считаются быстрой речью', () => {
    // Иначе ограничитель срабатывает на любом треке с шагом мельче порога,
    // и рот не открывается никогда. Найдено на живой странице: при шаге 40 мс
    // челюсть доходила до 0.108 при 0.28 в матрице.
    const { layer, writer, clock } = setup();
    clock.anchor(0);
    // Шаг 40 мс — меньше порога 80, но висема одна и та же: рот не движется.
    const track = [];
    for (let i = 0; i < 12; i++) track.push({ pts_ms: i * 40, viseme: 'AA' });
    track.push({ pts_ms: 3000, viseme: 'SIL' });
    layer.playGeneration('g', track);
    let maxJaw = 0;
    run(layer, writer, clock, 0.45, () => { maxJaw = Math.max(maxJaw, writer.get('jawOpen')); });
    assert.ok(maxJaw > cfg.matrix.AA.jawOpen * 0.8,
      `челюсть дошла до ${maxJaw.toFixed(3)} из ${cfg.matrix.AA.jawOpen} — ` +
      `ограничитель сработал там, где смены висемы нет`);
  });

  test('быстрая СМЕНА висем всё ещё режется', () => {
    const { layer, writer, clock } = setup();
    clock.anchor(0);
    const track = [];
    for (let i = 0; i < 20; i++) track.push({ pts_ms: i * 40, viseme: i % 2 ? 'AA' : 'MBP' });
    track.push({ pts_ms: 3000, viseme: 'SIL' });
    layer.playGeneration('g', track);
    let maxJaw = 0;
    run(layer, writer, clock, 0.7, (nowMs) => {
      if (nowMs < 700) maxJaw = Math.max(maxJaw, writer.get('jawOpen'));
    });
    assert.ok(maxJaw < cfg.matrix.AA.jawOpen * 0.75,
      `на быстрой смене челюсть дошла до ${maxJaw.toFixed(3)} — не режется`);
  });

  test('опережение: висема начинается раньше своего таймкода', () => {
    const { layer, writer, clock } = setup();
    clock.anchor(0);
    layer.playGeneration('g1', [
      { pts_ms: 0, viseme: 'SIL' },
      { pts_ms: 500, viseme: 'AA' },
      { pts_ms: 2000, viseme: 'SIL' },
    ]);
    let firstMoveMs = null;
    run(layer, writer, clock, 1.0, (nowMs) => {
      if (firstMoveMs === null && writer.get('jawOpen') > 0.01) firstMoveMs = nowMs;
    });
    assert.ok(firstMoveMs !== null, 'рот вообще не открылся');
    assert.ok(firstMoveMs < 500,
      `рот пошёл на ${firstMoveMs.toFixed(0)} мс, а должен раньше таймкода 500`);
    assert.ok(firstMoveMs > 500 - cfg.timing.leadMs - 40,
      `опережение ${(500 - firstMoveMs).toFixed(0)} мс — больше заявленного ${cfg.timing.leadMs}`);
  });
});

describe('дрейф и предпрокрутка — разные вещи', () => {
  test('до начала трека дрейф нулевой, а не равен запасу планирования', () => {
    const { layer, writer, clock } = setup();
    // Генерация запланирована на 120 мс вперёд, как на реальном старте.
    clock.anchor(0.12);
    layer.playGeneration('g', [{ pts_ms: 0, viseme: 'SIL' }, { pts_ms: 2000, viseme: 'AA' }]);
    let maxDrift = 0, sawPreroll = false;
    run(layer, writer, clock, 0.1, () => {
      maxDrift = Math.max(maxDrift, Math.abs(layer.stats.drift_ms));
      if (layer.stats.preroll_ms > 0) sawPreroll = true;
    });
    assert.equal(maxDrift, 0, 'предпрокрутка не должна показываться как дрейф');
    assert.ok(sawPreroll, 'но сама предпрокрутка должна быть видна отдельным числом');
  });

  test('внутри трека дрейф остаётся нулевым', () => {
    const { layer, writer, clock } = setup();
    clock.anchor(0);
    layer.playGeneration('g', TRACK);
    let maxDrift = 0;
    run(layer, writer, clock, 1.1, () => {
      maxDrift = Math.max(maxDrift, Math.abs(layer.stats.drift_ms));
    });
    assert.ok(maxDrift < 50, `дрейф ${maxDrift.toFixed(1)} мс превышает порог приёмки`);
  });
});

describe('отмена по generation_id', () => {
  test('после cancel ни один кадр не несёт старую артикуляцию', () => {
    const { layer, writer, clock } = setup();
    clock.anchor(0);
    layer.playGeneration('gen-1', TRACK);
    run(layer, writer, clock, 0.4);
    assert.ok(writer.get('jawOpen') > cfg.matrix.AA.jawOpen * 0.6,
      'до отмены рот должен быть открыт');

    const before = { jaw: writer.get('jawOpen'), aa: writer.get('viseme_aa') };
    layer.cancel('gen-1');
    assert.equal(layer.isPlaying, false);
    assert.equal(layer.genId, null);
    assert.equal(layer.track, null, 'трек должен быть отброшен, а не доигран');

    // Мгновенно обнулить веса нельзя — это щелчок. Требование другое: после
    // отмены артикуляция может только затухать, ни одна висема старой
    // генерации не должна нарасти ни в одном кадре.
    let prevJaw = before.jaw, prevAa = before.aa, grew = null;
    run(layer, writer, clock, 1.0, (nowMs) => {
      const jaw = writer.get('jawOpen'), aa = writer.get('viseme_aa');
      if (grew === null && (jaw > prevJaw + 1e-6 || aa > prevAa + 1e-6)) grew = nowMs;
      prevJaw = jaw; prevAa = aa;
    });
    assert.equal(grew, null, `артикуляция нарастала после отмены на ${grew} мс`);
    assert.equal(writer.get('jawOpen'), 0, 'рот должен закрыться полностью');
  });

  test('перебивание закрывает рот быстро, а не за обычный спад', () => {
    const { layer, writer, clock } = setup();
    clock.anchor(0);
    layer.playGeneration('gen-1', TRACK);
    run(layer, writer, clock, 0.4);

    // Состояние interrupted просит быстрое схлопывание.
    layer.cancel('gen-1', 45);
    let closedAt = null;
    const t0 = clock.nowMs();
    run(layer, writer, clock, 0.6, (nowMs) => {
      if (closedAt === null && writer.get('jawOpen') < cfg.matrix.AA.jawOpen * 0.05) closedAt = nowMs - t0;
    });
    assert.ok(closedAt !== null && closedAt < 250,
      `рот закрывался ${closedAt === null ? '>600' : closedAt.toFixed(0)} мс — для перебивания долго`);
  });

  test('после схлопывания времена возвращаются к обычным', () => {
    const { layer, writer, clock } = setup();
    clock.anchor(0);
    layer.playGeneration('gen-1', TRACK);
    run(layer, writer, clock, 0.4);
    layer.cancel('gen-1', 45);
    run(layer, writer, clock, 0.5);
    assert.equal(layer.releaseMs, undefined, 'ускоренный спад не должен залипать');
  });

  test('cancel чужого id не трогает текущую генерацию', () => {
    const { layer, writer, clock } = setup();
    clock.anchor(0);
    layer.playGeneration('gen-2', TRACK);
    assert.equal(layer.cancel('gen-1'), false, 'чужой id не должен отменять');
    assert.equal(layer.isPlaying, true);
    run(layer, writer, clock, 0.4);
    assert.ok(writer.get('jawOpen') > cfg.matrix.AA.jawOpen * 0.6);
  });

  test('новая генерация вытесняет старую немедленно', () => {
    const { layer, writer, clock } = setup();
    clock.anchor(0);
    layer.playGeneration('gen-1', TRACK);
    run(layer, writer, clock, 0.4);
    layer.playGeneration('gen-2', [{ pts_ms: 0, viseme: 'SIL' }, { pts_ms: 3000, viseme: 'SIL' }]);
    assert.equal(layer.genId, 'gen-2');
    assert.equal(layer.track.length, 2);
  });
});

describe('буферизация и подвисания', () => {
  test('трек буферизуется целиком и сортируется', () => {
    const { layer } = setup();
    layer.playGeneration('g', [
      { pts_ms: 500, viseme: 'MBP' },
      { pts_ms: 0, viseme: 'SIL' },
      { pts_ms: 200, viseme: 'AA' },
    ]);
    assert.deepEqual(layer.track.map((t) => t.pts_ms), [0, 200, 500]);
  });

  test('выход за конец ПОТОКА считается подвисанием', () => {
    const { layer, writer, clock } = setup();
    clock.anchor(0);
    // complete: false — кадры ещё ждём, значит это настоящее подвисание.
    layer.playGeneration('g', [{ pts_ms: 0, viseme: 'AA' }, { pts_ms: 100, viseme: 'AA' }],
      { complete: false });
    run(layer, writer, clock, 0.5);
    assert.ok(layer.stats.underruns > 0,
      'рендер вышел за пределы кадров — это должно считаться, как в S2');
  });

  test('выход за конец ПОЛНОГО трека подвисанием не считается', () => {
    // Иначе счётчик тонет в шуме: на dev-странице законченная фраза давала
    // 179 «подвисаний», хотя рендер ни разу не остался без кадров.
    const { layer, writer, clock } = setup();
    clock.anchor(0);
    layer.playGeneration('g', [{ pts_ms: 0, viseme: 'AA' }, { pts_ms: 100, viseme: 'SIL' }]);
    run(layer, writer, clock, 1.0);
    assert.equal(layer.stats.underruns, 0);
    assert.equal(layer.finished, true, 'генерация должна отметиться законченной');
    assert.equal(writer.get('jawOpen'), 0, 'рот должен вернуться в покой');
  });

  test('досланные кадры продлевают ту же генерацию', () => {
    const { layer, writer, clock } = setup();
    clock.anchor(0);
    layer.playGeneration('g', [{ pts_ms: 0, viseme: 'SIL' }], { complete: false });
    assert.equal(layer.appendVisemes('g', [{ pts_ms: 200, viseme: 'AA' }]), true);
    assert.equal(layer.track.length, 2);
    run(layer, writer, clock, 0.45);
    assert.ok(writer.get('jawOpen') > cfg.matrix.AA.jawOpen * 0.5,
      'досланная висема должна отыграться');
  });

  test('досылка в чужую генерацию отбрасывается', () => {
    const { layer } = setup();
    layer.playGeneration('gen-2', [{ pts_ms: 0, viseme: 'SIL' }], { complete: false });
    assert.equal(layer.appendVisemes('gen-1', [{ pts_ms: 100, viseme: 'AA' }]), false);
    assert.equal(layer.track.length, 1);
  });
});

describe('лестница отступления', () => {
  test('источник рта подменяется на AnalyserNode', () => {
    const { layer, writer } = setup();
    // Заглушка анализатора: постоянная амплитуда.
    layer.attachAnalyser({
      fftSize: 256,
      getFloatTimeDomainData(buf) { buf.fill(0.3); },
    });
    layer.setSource(SOURCE.ANALYSER);
    const dt = 1 / 60;
    for (let i = 0; i < 30; i++) { writer.begin(); layer.update(dt); writer.commit(); }
    assert.ok(writer.get('jawOpen') > 0.5,
      'уровень 3 должен открывать рот по огибающей без всякого трека');
    assert.equal(layer.debug().source, 'analyser');
  });

  test('переключение источника не требует ни трека, ни часов', () => {
    const { layer } = setup();
    layer.attachClock(null);
    layer.setSource(SOURCE.ANALYSER);
    layer.attachAnalyser({ fftSize: 64, getFloatTimeDomainData(b) { b.fill(0); } });
    assert.doesNotThrow(() => layer.update(1 / 60));
  });
});

describe('бюджет кадра', () => {
  test('update() не аллоцирует', (t) => {
    if (typeof global.gc !== 'function') { t.skip('нужен --expose-gc'); return; }
    const { layer, writer, clock } = setup();
    clock.anchor(0);
    const long = [];
    for (let i = 0; i < 500; i++) long.push({ pts_ms: i * 60, viseme: ['AA', 'MBP', 'OU', 'SS'][i % 4] });
    layer.playGeneration('g', long);
    run(layer, writer, clock, 5);                 // прогрев

    global.gc();
    const before = process.memoryUsage().heapUsed;
    run(layer, writer, clock, 25);
    global.gc();
    const grew = process.memoryUsage().heapUsed - before;
    assert.ok(grew < 128 * 1024, `куча выросла на ${grew} байт за 25 с артикуляции`);
  });
});
