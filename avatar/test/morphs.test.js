// Тесты writer'а морфов.
//
// Фикстура строится из реального отчёта об инвентаризации модели
// (bench/results/r5_glb_inspect.json), а не из выдуманных имён: проверяется
// раскладка по тем мешам и индексам, которые действительно лежат в data/model.glb.
// Если модель поменяют — тест это заметит, а не тихо продолжит проверять миф.

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import { MorphWriter } from '../src/morphs.js';
import { LAYERS, OWNER, ZONE_MORPHS, UNUSED_MORPHS } from '../src/zones.js';

const here = dirname(fileURLToPath(import.meta.url));
const REPORT = resolve(here, '../../bench/results/r5_glb_inspect.json');

/** Меши-заглушки формы THREE.Mesh, собранные по реальному отчёту. */
function fixtureMeshes() {
  const report = JSON.parse(readFileSync(REPORT, 'utf8'))[0];
  return Object.entries(report.meshes)
    .filter(([, info]) => info.count > 0)
    .map(([name, info]) => {
      const morphTargetDictionary = {};
      info.names.forEach((n, i) => { morphTargetDictionary[n] = i; });
      return {
        name,
        morphTargetDictionary,
        morphTargetInfluences: new Array(info.names.length).fill(0),
      };
    });
}

const byName = (meshes) => Object.fromEntries(meshes.map((m) => [m.name, m]));
const influence = (mesh, morph) => mesh.morphTargetInfluences[mesh.morphTargetDictionary[morph]];

// Веса живут в Float32Array — так их и держит three.js, и так дешевле по памяти
// на каждый кадр. Сравнение с допуском, а не побитовое: 0.7 в float32 это
// 0.699999988, и требовать точного равенства значило бы тестировать IEEE 754.
const near = (actual, expected, msg) =>
  assert.ok(Math.abs(actual - expected) < 1e-6,
    `${msg || ''}: получено ${actual}, ожидалось ${expected}`);

describe('раскладка веса по всем мешам, где морф объявлен', () => {
  // Главный тест. Именно этот баг — «губы шевелятся, зубы стоят» — был пойман
  // на инвентаризации модели, и он должен быть закрыт проверкой, а не вниманием.
  test('jawOpen доходит до Head_Mesh, Teeth_Mesh и Tongue_Mesh', () => {
    const meshes = fixtureMeshes();
    const w = new MorphWriter(meshes, { strict: true });
    const m = byName(meshes);

    w.begin();
    w.set(LAYERS.VISEME, 'jawOpen', 0.7);
    w.commit();

    near(influence(m.Head_Mesh, 'jawOpen'), 0.7, 'челюсть на голове');
    near(influence(m.Teeth_Mesh, 'jawOpen'), 0.7, 'зубы должны опуститься вместе с губами');
    near(influence(m.Tongue_Mesh, 'jawOpen'), 0.7, 'язык тоже');
    assert.deepEqual(
      [...w.meshesOf.get('jawOpen')].sort(),
      ['Head_Mesh', 'Teeth_Mesh', 'Tongue_Mesh']);
  });

  test('вес кладётся по имени, а не по индексу: индексы у мешей разные', () => {
    const meshes = fixtureMeshes();
    const m = byName(meshes);
    // Ради этого весь класс и существует: 49 против 16 против 16.
    assert.equal(m.Head_Mesh.morphTargetDictionary.jawOpen, 49);
    assert.equal(m.Teeth_Mesh.morphTargetDictionary.jawOpen, 16);
    assert.notEqual(
      m.Head_Mesh.morphTargetDictionary.jawOpen,
      m.Teeth_Mesh.morphTargetDictionary.jawOpen);

    const w = new MorphWriter(meshes, { strict: true });
    w.begin();
    w.set(LAYERS.VISEME, 'jawOpen', 1);
    w.commit();
    // В зубах должен быть тронут ровно индекс 16 и ничего больше.
    assert.equal(m.Teeth_Mesh.morphTargetInfluences[16], 1);
    assert.equal(m.Teeth_Mesh.morphTargetInfluences[49], undefined);
    assert.equal(m.Teeth_Mesh.morphTargetInfluences.filter((v) => v !== 0).length, 1);
  });

  test('висема доходит до головы, зубов и языка', () => {
    const meshes = fixtureMeshes();
    const w = new MorphWriter(meshes, { strict: true });
    const m = byName(meshes);
    w.begin();
    w.set(LAYERS.VISEME, 'viseme_aa', 0.9);
    w.commit();
    for (const name of ['Head_Mesh', 'Teeth_Mesh', 'Tongue_Mesh']) {
      near(influence(m[name], 'viseme_aa'), 0.9, name);
    }
  });

  test('бровь доходит до головы, ресниц и EyeAO — ресницы не отстают от бровей', () => {
    const meshes = fixtureMeshes();
    const w = new MorphWriter(meshes, { strict: true });
    const m = byName(meshes);
    w.begin();
    w.set(LAYERS.EMOTION, 'browInnerUp', 0.5);
    w.commit();
    for (const name of ['Head_Mesh', 'EyeAO_Mesh', 'Eyelash_Mesh']) {
      near(influence(m[name], 'browInnerUp'), 0.5, name);
    }
  });

  test('моргание доходит до трёх мешей', () => {
    const meshes = fixtureMeshes();
    const w = new MorphWriter(meshes, { strict: true });
    const m = byName(meshes);
    w.begin();
    w.set(LAYERS.IDLE, 'eyeBlinkLeft', 1);
    w.commit();
    for (const name of ['Head_Mesh', 'EyeAO_Mesh', 'Eyelash_Mesh']) {
      assert.equal(influence(m[name], 'eyeBlinkLeft'), 1, name);
    }
  });
});

describe('владение зонами', () => {
  test('чужой слой отбрасывается, значение владельца выживает', () => {
    const meshes = fixtureMeshes();
    const w = new MorphWriter(meshes);           // не strict: считаем, а не бросаем
    w.begin();
    w.set(LAYERS.VISEME, 'jawOpen', 0.8);
    const accepted = w.set(LAYERS.EMOTION, 'jawOpen', 0.1);   // эмоция лезет в рот
    w.commit();

    assert.equal(accepted, false);
    near(w.get('jawOpen'), 0.8, 'побеждает владелец, а не последняя запись');
    assert.equal(w.stats.trespassWrites, 1);
    assert.match(w.stats.lastTrespass, /emotion -> jawOpen/);
  });

  test('порядок записей не влияет: владелец побеждает и когда пишет первым, и когда вторым', () => {
    const meshes = fixtureMeshes();
    const w = new MorphWriter(meshes);
    w.begin();
    w.set(LAYERS.EMOTION, 'jawOpen', 0.1);       // чужой первым
    w.set(LAYERS.VISEME, 'jawOpen', 0.8);
    w.commit();
    near(w.get('jawOpen'), 0.8);
  });

  test('в strict нарушение владения — исключение', () => {
    const w = new MorphWriter(fixtureMeshes(), { strict: true });
    w.begin();
    assert.throws(() => w.set(LAYERS.IDLE, 'mouthFunnel', 1), /владеет «viseme»/);
  });

  test('агрегаты Avaturn не принадлежат никому и не пишутся', () => {
    const w = new MorphWriter(fixtureMeshes());
    w.begin();
    for (const layer of Object.values(LAYERS)) {
      assert.equal(w.set(layer, 'mouthOpen', 1), false, `${layer} не должен писать в mouthOpen`);
    }
    w.commit();
    assert.equal(w.get('mouthOpen'), 0);
  });
});

describe('клампинг и обнуление кадра', () => {
  test('сумма вкладов одного слоя клампится в [0, 1]', () => {
    const meshes = fixtureMeshes();
    const w = new MorphWriter(meshes, { strict: true });
    const m = byName(meshes);
    w.begin();
    w.add(LAYERS.VISEME, 'jawOpen', 0.7);   // висема
    w.add(LAYERS.VISEME, 'jawOpen', 0.6);   // дожим
    w.commit();
    assert.equal(w.get('jawOpen'), 1);
    assert.equal(influence(m.Teeth_Mesh, 'jawOpen'), 1, 'кламп доезжает до всех мешей');
  });

  test('отрицательный вес клампится в 0', () => {
    const w = new MorphWriter(fixtureMeshes(), { strict: true });
    w.begin();
    w.set(LAYERS.VISEME, 'jawOpen', -0.5);
    w.commit();
    assert.equal(w.get('jawOpen'), 0);
  });

  test('begin() обнуляет кадр целиком, включая меши', () => {
    const meshes = fixtureMeshes();
    const w = new MorphWriter(meshes, { strict: true });
    const m = byName(meshes);
    w.begin();
    w.set(LAYERS.VISEME, 'jawOpen', 1);
    w.commit();
    w.begin();
    w.commit();                              // кадр без единой записи
    assert.equal(w.get('jawOpen'), 0);
    assert.equal(influence(m.Teeth_Mesh, 'jawOpen'), 0, 'рот не должен залипнуть открытым');
  });
});

describe('модель и таблица зон не разошлись', () => {
  const w = new MorphWriter(fixtureMeshes());

  test('в модели нет морфов, не учтённых таблицей зон', () => {
    assert.deepEqual(w.unclaimed, [], 'морфы модели вне зон и вне UNUSED_MORPHS');
  });

  test('таблица зон не ссылается на морфы, которых в модели нет', () => {
    assert.deepEqual(w.missing, []);
  });

  test('счёт сходится: 67 владельцев + 5 агрегатов = 72 морфа модели', () => {
    const d = w.describe();
    assert.equal(d.morphs, 72);
    assert.equal(d.meshes, 6);
    assert.equal(OWNER.size + UNUSED_MORPHS.length, 72);
    assert.equal(d.byLayer.viseme, ZONE_MORPHS[LAYERS.VISEME].length);
    assert.equal(d.unowned, UNUSED_MORPHS.length);
  });

  test('запись в несуществующий морф считается, а не молча теряется', () => {
    const w2 = new MorphWriter(fixtureMeshes());
    w2.begin();
    assert.equal(w2.set(LAYERS.VISEME, 'viseme_ы', 1), false);
    assert.equal(w2.stats.unknownWrites, 1);
    assert.equal(w2.stats.lastUnknown, 'viseme_ы');
  });
});

describe('горячий путь', () => {
  test('slotOf даёт стабильный слот, запись по слоту равна записи по имени', () => {
    const meshes = fixtureMeshes();
    const w = new MorphWriter(meshes, { strict: true });
    const m = byName(meshes);
    const slot = w.slotOf('viseme_O');
    assert.ok(slot >= 0);
    assert.equal(w.slotOf('нет такого'), -1);
    w.begin();
    w.setSlot(LAYERS.VISEME, slot, 0.42);
    w.commit();
    near(influence(m.Tongue_Mesh, 'viseme_O'), 0.42);
  });

  test('commit() не аллоцирует: кадры не растят кучу', (t) => {
    // Без принудительной сборки мусора этот замер ловит работу тест-раннера,
    // а не writer'а, и падает примерно в четверти прогонов. Запускается через
    // `node --expose-gc` (см. package.json); без флага теста просто нет —
    // лучше пропуск, чем красный тест, который ничего не доказывает.
    if (typeof global.gc !== 'function') {
      t.skip('нужен --expose-gc');
      return;
    }
    const w = new MorphWriter(fixtureMeshes(), { strict: true });
    const slot = w.slotOf('jawOpen');
    const batch = (n) => {
      for (let i = 0; i < n; i++) {
        w.begin();
        w.setSlot(LAYERS.VISEME, slot, (i % 100) / 100);
        w.commit();
      }
    };
    batch(2000);                                   // прогрев и JIT

    global.gc();
    const before = process.memoryUsage().heapUsed;
    batch(20000);
    global.gc();
    const grew = process.memoryUsage().heapUsed - before;

    // Запись в предвыделенные типизированные массивы не аллоцирует ничего;
    // порог в 64 КБ оставлен на служебный шум, а не на вклад writer'а.
    assert.ok(grew < 64 * 1024, `куча выросла на ${grew} байт за 20000 кадров`);
  });
});
