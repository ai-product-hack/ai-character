// Тесты слоя «символ -> висема».
//
// Проверяются те расхождения письма и произношения, из-за которых артикуляция
// читается как «иностранец читает по бумажке». Слова взяты из области, где
// модуль будет работать: реплики интервьюера из bench/r4-lipsync/align_offline.py.

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

import { DEVOICING, VISEMES, textToVisemes, timedToTrack, wordToVisemes } from '../src/g2p.js';

/** Только последовательность висем, без долей и блендов. */
const seq = (word) => wordToVisemes(word).map((v) => v.viseme);
const seqText = (t) => textToVisemes(t).map((v) => v.viseme);

describe('йотированные дают две висемы на один символ', () => {
  test('я = IH + AA', () => assert.deepEqual(seq('я'), ['IH', 'AA']));
  test('ю = IH + OU', () => assert.deepEqual(seq('ю'), ['IH', 'OU']));
  test('е = IH + EE', () => assert.deepEqual(seq('е'), ['IH', 'EE']));
  test('ё = IH + OH', () => assert.deepEqual(seq('ё'), ['IH', 'OH']));

  test('«моя» — три символа, четыре висемы', () => {
    assert.deepEqual(seq('моя'), ['MBP', 'OH', 'IH', 'AA']);
  });

  test('скольжение короче гласной', () => {
    const [glide, vowel] = wordToVisemes('я');
    assert.ok(glide.share < vowel.share,
      `скольжение ${glide.share} должно быть короче гласной ${vowel.share}`);
  });
});

describe('ь и ъ не дают висемы и не дают паузу', () => {
  test('мягкий знак съедается', () => {
    // Если бы ь давал SIL, здесь была бы пауза посреди слова.
    assert.deepEqual(seq('семь'), ['SS', 'IH', 'EE', 'MBP']);
    assert.ok(!seq('семь').includes('SIL'), 'паузы посреди слова быть не должно');
  });

  test('твёрдый знак съедается', () => {
    assert.ok(!seq('объём').includes('SIL'));
    // о-б-ъ-ё-м: ъ пропадает, ё даёт две висемы
    assert.deepEqual(seq('объём'), ['OH', 'MBP', 'IH', 'OH', 'MBP']);
  });

  test('мягкий знак смягчает предыдущий согласный, а не заменяет его', () => {
    const v = wordToVisemes('пять');
    const last = v.at(-1);
    assert.equal(last.viseme, 'TH', 'т остаётся т');
    assert.ok(last.blend && last.blend.IH > 0, 'но со сдвигом в сторону IH');
  });
});

describe('аканье', () => {
  test('безударное о получает подмешанное AA', () => {
    // «хорошо»: ударение без словаря неизвестно, поэтому все о редуцируются
    // частично — ошибка ограничена в обе стороны.
    const os = wordToVisemes('хорошо').filter((v) => v.viseme === 'OH');
    assert.equal(os.length, 3);
    for (const o of os) {
      assert.ok(o.blend && o.blend.AA > 0, 'безударное о должно тянуться к а');
    }
  });

  test('в односложном слове о ударно и не редуцируется', () => {
    const v = wordToVisemes('он');
    const o = v.find((x) => x.viseme === 'OH');
    assert.ok(!o.blend, 'единственная гласная — ударная, редукции быть не должно');
  });

  test('ё всегда ударно и никогда не редуцируется', () => {
    const v = wordToVisemes('ёлка');
    const o = v.find((x) => x.viseme === 'OH');
    assert.ok(!o.blend, 'ё ударно по определению');
  });
});

describe('удвоенные согласные', () => {
  test('«класс» даёт одну SS, а не две', () => {
    assert.deepEqual(seq('класс'), ['KG', 'L', 'AA', 'SS']);
  });

  test('«группа» даёт одну MBP на удвоенную п', () => {
    assert.deepEqual(seq('группа'), ['KG', 'KG', 'OU', 'MBP', 'AA']);
  });

  test('«ванна» даёт одну TH', () => {
    assert.deepEqual(seq('ванна'), ['FV', 'AA', 'TH', 'AA']);
  });

  test('соседние РАЗНЫЕ буквы одной висемы не схлопываются', () => {
    // «гр» — это г и р, обе KG, но это разные звуки в разных слотах времени.
    // Схлопнуть их значило бы потерять таймкод; рот и так удержит форму.
    const v = wordToVisemes('гром');
    assert.deepEqual(v.map((x) => x.viseme), ['KG', 'KG', 'OH', 'MBP']);
    assert.notEqual(v[0].src, v[1].src, 'у них разные исходные символы');
  });

  test('удвоенная гласная не схлопывается', () => {
    // «зоопарк»: две о подряд — это два разных слога, не удвоение согласной
    const v = seq('зоопарк');
    assert.equal(v.filter((x) => x === 'OH').length, 2);
  });
});

describe('оглушение на конце слова', () => {
  // Правило в модуле НЕ применяется, и тест стережёт основание этого решения:
  // в наборе из 13 висем звонкий и глухой каждой пары сидят в одном классе,
  // поэтому оглушение визуально ничего не меняет. Если раскладку разведут,
  // тест упадёт и напомнит включить правило.
  test('пары звонкий/глухой дают одну и ту же висему — правило было бы пустым', () => {
    const same = [];
    for (const [voiced, voiceless] of Object.entries(DEVOICING)) {
      const a = seq(voiced), b = seq(voiceless);
      if (a[0] !== b[0]) same.push(`${voiced}->${a[0]} но ${voiceless}->${b[0]}`);
    }
    assert.deepEqual(same, [],
      'раскладка развела пару по разным висемам — пора включить оглушение в g2p.js');
  });

  test('«зуб» и «зуп» дают одинаковый трек', () => {
    assert.deepEqual(seq('зуб'), seq('зуп'));
  });
});

describe('фраза целиком', () => {
  test('пробел даёт одну паузу, а не по одной на символ', () => {
    const v = seqText('да  нет');
    assert.equal(v.filter((x) => x === 'SIL').length, 1);
  });

  test('знаки препинания дают паузу', () => {
    assert.ok(seqText('да, нет').includes('SIL'));
  });

  test('реплика интервьюера раскладывается без пауз внутри слов', () => {
    const parsed = textToVisemes('Понятно. А что именно делали лично вы?');
    const sils = parsed.filter((v) => v.viseme === 'SIL');
    // Паузы только между словами и в конце: 6 промежутков + финальный знак.
    assert.ok(sils.length >= 6 && sils.length <= 8, `пауз ${sils.length}`);
    for (const v of parsed) {
      assert.ok(VISEMES.includes(v.viseme), `неизвестная висема ${v.viseme}`);
    }
  });

  test('все висемы из известного набора', () => {
    const long = 'Здравствуйте! Расскажите, пожалуйста, о вашем последнем проекте.';
    for (const v of textToVisemes(long)) {
      assert.ok(VISEMES.includes(v.viseme), `неизвестная висема ${v.viseme}`);
    }
  });
});

describe('раскладка таймкодов', () => {
  /** Посимвольные таймкоды с шагом 40 мс, как отдаёт GigaAM. */
  const timed = (text, step = 40) =>
    [...text].map((ch, i) => ({ ch, ms: i * step }));

  test('таймкоды монотонны', () => {
    const track = timedToTrack(timed('привет всем'));
    for (let i = 1; i < track.length; i++) {
      assert.ok(track[i].pts_ms >= track[i - 1].pts_ms,
        `таймкод пошёл назад на ${i}: ${track[i - 1].pts_ms} -> ${track[i].pts_ms}`);
    }
  });

  test('две висемы одного символа делят его слот, а не встают на один таймкод', () => {
    const track = timedToTrack(timed('яма'));
    assert.equal(track[0].viseme, 'IH');
    assert.equal(track[1].viseme, 'AA');
    assert.equal(track[0].pts_ms, 0);
    assert.ok(track[1].pts_ms > 0 && track[1].pts_ms < 40,
      `вторая висема на ${track[1].pts_ms} мс, должна быть внутри слота 0-40`);
  });

  test('съеденный мягкий знак не оставляет дыры в таймлайне', () => {
    const track = timedToTrack(timed('пять'));
    // 4 символа -> 4 висемы (ь съеден), последняя начинается не позже слота т
    assert.equal(track.length, 4);
    assert.equal(track.at(-1).viseme, 'TH');
    assert.ok(track.at(-1).pts_ms <= 120);
  });

  test('трек не длиннее исходных таймкодов', () => {
    const t = timed('здравствуйте');
    const track = timedToTrack(t);
    assert.ok(track.at(-1).pts_ms <= t.at(-1).ms + 40);
  });
});
