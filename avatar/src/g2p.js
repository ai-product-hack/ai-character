// Русская орфография -> висемы.
//
// Это НЕ матрица висем. Матрица отвечает на вопрос «как выглядит висема AA»,
// а этот модуль — на вопрос «какие висемы и когда» для написанного текста.
// Разделение принципиальное: матрица подкручивается на глаз слайдерами, а
// правила ниже проверяются юнит-тестами на словах.
//
// Вход — посимвольные таймкоды от GigaAM (шаг 40 мс). Русское письмо заметно
// расходится с произношением, и без этого слоя артикуляция читается как
// «иностранец читает по бумажке»:
//
//   - йотированные дают ДВЕ висемы на один символ: я = IH+AA, ю = IH+OU,
//     е = IH+EE, ё = IH+OH;
//   - ь и ъ висемы не дают вообще, но таймкод у них есть — если не съесть,
//     получится пауза посреди слова;
//   - аканье: безударное о артикулируется ближе к а;
//   - удвоенные согласные дают одну висему, а не две.
//
// Про оглушение на конце слова см. комментарий у DEVOICING ниже: в выбранном
// наборе из 13 висем это правило оказалось пустым, и вместо мёртвого кода
// здесь стоит тест, который стережёт само основание.

/** 13 висем из R4. SIL — пауза, всё в ноль. */
export const VISEMES = Object.freeze([
  'SIL', 'AA', 'EE', 'IH', 'OH', 'OU', 'MBP', 'FV', 'L', 'WQ', 'SS', 'TH', 'KG',
]);

/** Базовая раскладка графем. Совпадает с таблицей R4 за вычетом её ошибок. */
const BASE = Object.freeze({
  а: 'AA', о: 'OH', у: 'OU', ы: 'IH', и: 'IH', э: 'EE', й: 'IH',
  м: 'MBP', б: 'MBP', п: 'MBP',
  ф: 'FV', в: 'FV',
  л: 'L',
  ш: 'WQ', ж: 'WQ', щ: 'WQ', ч: 'WQ',
  с: 'SS', з: 'SS', ц: 'SS',
  т: 'TH', д: 'TH', н: 'TH',
  к: 'KG', г: 'KG', х: 'KG', р: 'KG',
});

/**
 * Йотированные: две висемы на один символ. Первая короткая — это скольжение
 * [й], вторая несёт основную гласную.
 */
const YOTATED = Object.freeze({
  я: ['IH', 'AA'],
  ю: ['IH', 'OU'],
  е: ['IH', 'EE'],
  ё: ['IH', 'OH'],
});

/** Доля слота, уходящая на скольжение [й]. Остальное — гласной. */
const GLIDE_SHARE = 0.35;

/** Знаки без собственной висемы. Мягкий знак вдобавок смягчает предыдущий согласный. */
const SOFT_SIGN = 'ь';
const HARD_SIGN = 'ъ';

/** Согласные, которые мягкий знак сдвигает в сторону IH. */
const SOFTENABLE = new Set(['TH', 'SS', 'L', 'MBP', 'FV', 'KG']);

/**
 * Оглушение звонких на конце слова. Оставлено таблицей, но НЕ применяется:
 * в наборе из 13 висем каждая пара звонкий/глухой уже сидит в одном классе
 * (б и п — MBP, в и ф — FV, з и с — SS, д и т — TH, г и к — KG, ж и ш — WQ),
 * поэтому «зуб -> зуп» не меняет ни одной висемы. Правило было бы мёртвым
 * кодом. Таблица нужна тесту, который проверяет само это основание: если
 * раскладку BASE когда-нибудь разведут по разным висемам, тест упадёт и
 * напомнит, что правило пора включать.
 */
export const DEVOICING = Object.freeze({ б: 'п', в: 'ф', г: 'к', д: 'т', ж: 'ш', з: 'с' });

const VOWELS = new Set(['а', 'о', 'у', 'ы', 'и', 'э', 'я', 'ю', 'е', 'ё']);
const isLetter = (c) => BASE[c] !== undefined || YOTATED[c] !== undefined ||
                        c === SOFT_SIGN || c === HARD_SIGN;

/**
 * Аканье. Ударения без словаря не взять, поэтому вместо бинарного выбора
 * «о или а» безударное о даётся СМЕСЬЮ OH и AA. Ошибка тогда ограничена в обе
 * стороны: перепутать ударение — значит промахнуться на половину округления
 * губ, а не поставить круглый рот вместо плоского.
 *
 * Ударение приближается двумя надёжными признаками: ё всегда ударно, и в
 * односложном слове ударение однозначно. Во всех остальных случаях о
 * редуцируется частично. Словарь ударений это заметно улучшит — место для
 * него отмечено.
 */
function stressedVowelIndex(word) {
  const yo = word.indexOf('ё');
  if (yo >= 0) return yo;
  const vowels = [...word].map((c, i) => (VOWELS.has(c) ? i : -1)).filter((i) => i >= 0);
  if (vowels.length === 1) return vowels[0];
  return -1;                       // не знаем; см. комментарий выше
}

/**
 * Разложить слово в висемы.
 * @param {string} word одно слово, уже в нижнем регистре
 * @param {object} cfg параметры из visemes.json (раздел g2p)
 * @returns {Array<{viseme:string, share:number, src:number, blend?:object, note?:string}>}
 *   share — доля временного слота исходного символа
 *   src — индекс символа в слове, чтобы разложить таймкоды
 */
export function wordToVisemes(word, cfg = {}) {
  const unstressedO = cfg.unstressedOBlend ?? 0.6;   // сколько AA подмешать в безударное о
  const softenBlend = cfg.softenBlend ?? 0.3;        // сколько IH подмешать в мягкий согласный
  const out = [];
  const stressAt = stressedVowelIndex(word);

  for (let i = 0; i < word.length; i++) {
    const c = word[i];
    const next = word[i + 1];

    if (c === HARD_SIGN) continue;                   // разделительный, висемы нет
    if (c === SOFT_SIGN) continue;                   // обработан при предыдущем символе

    if (YOTATED[c]) {
      const [glide, vowel] = YOTATED[c];
      out.push({ viseme: glide, share: GLIDE_SHARE, src: i, note: 'йот' });
      // ё всегда ударно, поэтому его OH не редуцируется.
      out.push({ viseme: vowel, share: 1 - GLIDE_SHARE, src: i });
      continue;
    }

    const base = BASE[c];
    if (base === undefined) continue;                // цифры, латиница и прочее

    // Удвоенная согласная — одна висема, а не две.
    if (next === c && !VOWELS.has(c)) continue;

    const entry = { viseme: base, share: 1, src: i };

    if (c === 'о' && i !== stressAt) {
      entry.blend = { AA: unstressedO };
      entry.note = 'аканье';
    }
    if (next === SOFT_SIGN && SOFTENABLE.has(base)) {
      entry.blend = { ...(entry.blend || {}), IH: softenBlend };
      entry.note = entry.note ? `${entry.note}+мягкость` : 'мягкость';
    }
    out.push(entry);
  }
  return out;
}

/**
 * Разложить фразу. Пробелы и знаки препинания дают SIL.
 * @returns {Array<{viseme, share, src, word?, blend?, note?}>} src — индекс в исходной строке
 */
export function textToVisemes(text, cfg = {}) {
  const lower = text.toLowerCase().replace(/ё/g, 'ё');
  const out = [];
  let i = 0;
  while (i < lower.length) {
    if (!isLetter(lower[i])) {
      // Один SIL на группу пробелов и знаков подряд, а не по одному на символ.
      const start = i;
      while (i < lower.length && !isLetter(lower[i])) i++;
      out.push({ viseme: 'SIL', share: 1, src: start, span: i - start });
      continue;
    }
    const start = i;
    while (i < lower.length && isLetter(lower[i])) i++;
    const word = lower.slice(start, i);
    for (const v of wordToVisemes(word, cfg)) {
      out.push({ ...v, src: start + v.src, word });
    }
  }
  return out;
}

/**
 * Собрать трек висем из посимвольных таймкодов.
 *
 * @param {Array<{ch:string, ms:number}>} timed символы с таймкодами, по возрастанию
 * @param {object} cfg
 * @returns {Array<{pts_ms:number, viseme:string, weight:number}>}
 */
export function timedToTrack(timed, cfg = {}) {
  const text = timed.map((t) => t.ch).join('');
  const parsed = textToVisemes(text, cfg);

  // Длительность слота символа: до следующего таймкода, для последнего — по
  // среднему шагу. Шаг таймкодов GigaAM — 40 мс.
  const step = cfg.timecodeStepMs ?? 40;
  const slotAt = (idx) => {
    const t = timed[idx];
    if (!t) return { start: 0, dur: step };
    const nextT = timed[idx + 1] ? timed[idx + 1].ms : t.ms + step;
    return { start: t.ms, dur: Math.max(1, nextT - t.ms) };
  };

  const track = [];
  // Несколько висем на один символ делят его слот по share.
  let k = 0;
  while (k < parsed.length) {
    const src = parsed[k].src;
    const group = [];
    while (k < parsed.length && parsed[k].src === src) group.push(parsed[k++]);
    const { start, dur } = slotAt(src);
    const total = group.reduce((s, g) => s + g.share, 0) || 1;
    let acc = 0;
    for (const g of group) {
      track.push({
        pts_ms: Math.round(start + (acc / total) * dur),
        viseme: g.viseme,
        weight: 1,
        blend: g.blend,
      });
      acc += g.share;
    }
  }
  return track;
}
