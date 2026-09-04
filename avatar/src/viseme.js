// Слой артикуляции: трек висем -> веса морфов рта.
//
// Владеет зоной viseme (15 висем Oculus, челюсть, губы, язык) и ничем больше.
// Источник данных для рта подменяем: уровень 2 — трек висем с таймкодами,
// уровень 3 — AnalyserNode и один морф раскрытия. Переключатель заложен
// в архитектуру сразу, как требует лестница отступления.
//
// Время берётся ТОЛЬКО у часов плеера (см. clock.js). Своих таймеров нет.

import { LAYERS } from './zones.js';

export const SOURCE = Object.freeze({ VISEMES: 'visemes', ANALYSER: 'analyser' });

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

/**
 * Целевые веса морфов для висемы, с учётом подмесей от слоя g2p
 * (аканье даёт OH+AA, мягкость даёт согласный+IH).
 */
function resolveWeights(matrix, viseme, blend, out) {
  for (const k in out) delete out[k];
  const base = matrix[viseme];
  // Нечисловые веса отбрасываются здесь же: строка или undefined в матрице
  // дала бы NaN, а NaN в morphTargetInfluences рвёт геометрию.
  if (base) {
    for (const [morph, w] of Object.entries(base)) {
      if (morph.startsWith('_') || !Number.isFinite(w)) continue;
      out[morph] = w;
    }
  }
  if (blend) {
    for (const [other, amount] of Object.entries(blend)) {
      const mix = matrix[other];
      if (!mix || !Number.isFinite(amount)) continue;
      for (const [morph, w] of Object.entries(mix)) {
        if (morph.startsWith('_') || !Number.isFinite(w)) continue;
        out[morph] = (out[morph] || 0) * (1 - amount) + w * amount;
      }
    }
  }
  return out;
}

export class VisemeLayer {
  /**
   * @param {import('./morphs.js').MorphWriter} morphs
   * @param {object} cfg содержимое visemes.json
   */
  constructor(morphs, cfg) {
    this.morphs = morphs;
    this.setConfig(cfg);

    this.source = SOURCE.VISEMES;
    this.clock = null;
    this.analyser = null;
    this._analyserBuf = null;

    /** Текущая генерация. Всё с чужим generation_id отбрасывается безусловно. */
    this.genId = null;
    this.track = null;
    this.trackIndex = 0;
    this.releaseMs = undefined;
    this.complete = true;
    this.finished = false;

    // Текущие и целевые веса. Ключи — имена морфов, значения — веса.
    this.current = new Map();
    this._target = {};
    this._targetPrev = {};

    this.stats = { drift_ms: 0, preroll_ms: 0, underruns: 0, frames: 0,
                   lastViseme: 'SIL', ahead_ms: 0 };
  }

  setConfig(cfg) {
    this.cfg = cfg;
    this.matrix = cfg.matrix;
    this.timing = cfg.timing;
    // Слоты резолвятся лениво: матрицу правят слайдерами, набор морфов меняется.
    this._slots = new Map();
  }

  /** Проверить, что все морфы матрицы есть в модели. Зовётся один раз на загрузке. */
  validate() {
    const missing = new Set();
    const bad = [];
    for (const [viseme, weights] of Object.entries(this.matrix)) {
      for (const [morph, w] of Object.entries(weights)) {
        if (morph.startsWith('_')) continue;
        if (this.morphs.slotOf(morph) < 0) missing.add(morph);
        if (!Number.isFinite(w)) bad.push(`${viseme}.${morph} = ${w}`);
      }
    }
    this.badWeights = bad;
    return [...missing];
  }

  attachClock(clock) { this.clock = clock; }

  /** Уровень 3 лестницы отступления: огибающая вместо таймкодов. */
  attachAnalyser(analyser) {
    this.analyser = analyser;
    this._analyserBuf = analyser ? new Float32Array(analyser.fftSize) : null;
  }

  setSource(source) { this.source = source; }

  /**
   * Принять трек висем на генерацию. Трек буферизуется ЦЕЛИКОМ до начала
   * воспроизведения: в S2 зафиксировано ~3% подвисаний — это моменты, когда
   * рендер вышел за пределы имеющихся кадров и держал последний.
   */
  playGeneration(genId, visemeTrack, opts = {}) {
    this.genId = genId;
    this.releaseMs = undefined;
    this.track = (visemeTrack || []).slice().sort((a, b) => a.pts_ms - b.pts_ms);
    this.trackIndex = 0;
    this.stats.underruns = 0;
    // Завершена ли генерация. Различие существенное: выход за конец ПОЛНОГО
    // трека — это просто конец реплики, а выход за конец ПОТОКА, куда кадры
    // ещё придут, — то самое подвисание из S2, когда рендер держит последний
    // кадр. Считать их одним счётчиком значит утопить настоящую проблему в шуме:
    // на dev-странице это дало 179 «подвисаний» на фразе, где их не было ни одного.
    this.complete = opts.complete !== false;
    this.finished = false;
  }

  /** Дослать кадры в текущую генерацию. Так это будет работать в потоке. */
  appendVisemes(genId, moreVisemes, opts = {}) {
    if (genId !== this.genId) return false;      // чужая генерация — молча мимо
    for (const v of moreVisemes) this.track.push(v);
    this.track.sort((a, b) => a.pts_ms - b.pts_ms);
    if (opts.complete) this.complete = true;
    return true;
  }

  /**
   * Отбросить генерацию. Безусловно и немедленно: никакого «доиграть последний
   * кадр». Рот идёт в покой спадом, которым владеет автомат состояний.
   */
  cancel(genId, releaseMs) {
    if (genId !== undefined && genId !== this.genId) return false;
    this.genId = null;
    this.track = null;
    this.trackIndex = 0;
    // Мгновенно обнулить веса нельзя — рот телепортировался бы в покой, и это
    // видно как щелчок. Но и обычный спад в 80 мс для перебивания медленный:
    // состояние interrupted просит закрыть рот быстро. Поэтому вызывающий
    // может задать своё время схлопывания; оно действует до конца спада.
    this.releaseMs = releaseMs;
    return true;
  }

  get isPlaying() { return this.track !== null && this.genId !== null; }

  /** 0..1 — насколько сейчас активно артикулирует рот, для остального лица. */
  get activity() {
    let value = 0;
    for (const weight of this.current.values()) value = Math.max(value, weight);
    return clamp01(value);
  }

  /**
   * Найти пару висем, между которыми сейчас находится время, двоичным поиском.
   * Возвращает индекс левой висемы или -1, если время левее трека.
   */
  _indexAt(ms) {
    const t = this.track;
    if (!t.length || ms < t[0].pts_ms) return -1;
    let lo = 0, hi = t.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (t[mid].pts_ms <= ms) lo = mid; else hi = mid;
    }
    return t[hi].pts_ms <= ms ? hi : lo;
  }

  /**
   * Один кадр. Вызывается между begin() и commit() writer'а морфов.
   * @param {number} dt секунды с прошлого кадра
   */
  update(dt) {
    this.stats.frames++;
    const T = this.timing;

    if (this.source === SOURCE.ANALYSER) {
      this._updateFromAnalyser(dt);
      return;
    }

    let target = this._target;
    for (const k in target) delete target[k];

    if (this.isPlaying && this.clock) {
      const nowMs = this.clock.nowMs();
      if (nowMs !== null) {
        // Опережение: артикуляция следующего звука начинается ДО самого звука.
        // Сдвиг вперёд выглядит естественнее точного совпадения.
        const lookAt = nowMs + T.leadMs;
        const i = this._indexAt(lookAt);
        if (i < 0) {
          // Время ещё не дошло до начала трека: генерацию запланировали с
          // запасом вперёд. Это предпрокрутка, а не дрейф — рендерить пока
          // нечего, и рассинхрону взяться неоткуда. Считать её дрейфом значит
          // показывать на оверлее 50+ мс там, где расхождения нет ни одного,
          // а по этому числу проверяют приёмку.
          this.stats.preroll_ms = this.track[0].pts_ms - lookAt;
          this.stats.drift_ms = 0;
        } else {
          this.stats.preroll_ms = 0;
          const a = this.track[i];
          const b = this.track[i + 1];
          this.stats.lastViseme = a.viseme;
          this.stats.drift_ms = 0;
          this.trackIndex = i;

          // Интервал считается до следующей СМЕНЫ висемы: подряд идущие
          // одинаковые висемы рот не двигают, и мерить по ним значит объявлять
          // быстрой речью любой трек с шагом мельче порога.
          let j = i + 1;
          while (j < this.track.length && this.track[j].viseme === a.viseme) j++;
          const nextChange = this.track[j];

          if (!b) {
            if (this.complete) {
              // Полный трек кончился — реплика отговорена, рот идёт в покой.
              this.finished = true;
              this.stats.lastViseme = 'SIL';
            } else {
              // Кадры ещё ждём, а рендер уже за их пределами: держим последнюю
              // висему и считаем подвисание, ровно как в S2.
              this.stats.underruns++;
              resolveWeights(this.matrix, a.viseme, a.blend, target);
            }
          } else {
            const gap = (nextChange ? nextChange.pts_ms : b.pts_ms) - a.pts_ms;
            // Ограничение амплитуды на быстрой речи: иначе челюсть стучит.
            const scale = gap < T.fastSpeechMs ? T.fastSpeechScale : 1;
            resolveWeights(this.matrix, a.viseme, a.blend, target);
            if (scale !== 1) for (const k in target) target[k] *= scale;
          }
        }
        this.stats.ahead_ms = T.leadMs;
      }
    }

    this._approach(target, dt);
    this._write();
  }

  /**
   * Уровень 3: огибающая с AnalyserNode в один морф раскрытия рта.
   * Отставание 5 мс при корреляции 0.94 — измерено в S2.
   */
  _updateFromAnalyser(dt) {
    const target = this._target;
    for (const k in target) delete target[k];
    if (this.analyser) {
      this.analyser.getFloatTimeDomainData(this._analyserBuf);
      let sum = 0;
      for (let i = 0; i < this._analyserBuf.length; i++) {
        const v = this._analyserBuf[i];
        sum += v * v;
      }
      const rms = Math.sqrt(sum / this._analyserBuf.length);
      const A = this.cfg.analyser || {};
      // Порог тишины: без него шум и хвосты реверберации держат челюсть
      // приоткрытой в паузах, и рот не закрывается ни в одном кадре.
      const gated = Math.max(0, rms - (A.gate ?? 0));
      target.jawOpen = clamp01(gated * (A.gain ?? 6)) * (A.maxOpen ?? 1);
      this.stats.lastViseme = 'RMS';
    }
    this._approach(target, dt);
    this._write();
  }

  /**
   * Приблизить текущие веса к целевым с АСИММЕТРИЧНЫМИ временами: атака 40 мс,
   * спад 80 мс. Одинаковый lerp читается как марионетка — это самая заметная
   * из трёх вещей, отличающих живую артикуляцию от механической.
   */
  _approach(target, dt) {
    const T = this.timing;
    const decayMs = this.releaseMs ?? T.decayMs;
    const kAttack = T.attackMs > 0 ? 1 - Math.exp(-dt * 1000 / T.attackMs) : 1;
    const kDecay = decayMs > 0 ? 1 - Math.exp(-dt * 1000 / decayMs) : 1;

    // Морфы, которых нет в цели, спадают к нулю.
    for (const [morph, value] of this.current) {
      if (!(morph in target) && value > 0) {
        const v = value + (0 - value) * kDecay;
        if (v < 1e-4) this.current.delete(morph); else this.current.set(morph, v);
      }
    }
    for (const morph in target) {
      const want = target[morph];
      const have = this.current.get(morph) || 0;
      const k = want > have ? kAttack : kDecay;
      this.current.set(morph, have + (want - have) * k);
    }
    // Ускоренное схлопывание живёт ровно до покоя, дальше времена обычные.
    if (this.releaseMs !== undefined && this.current.size === 0) this.releaseMs = undefined;
  }

  /** Разложить текущие веса в writer. Слой владеет только зоной viseme. */
  _write() {
    for (const [morph, value] of this.current) {
      if (value <= 0) continue;
      let slot = this._slots.get(morph);
      if (slot === undefined) {
        slot = this.morphs.slotOf(morph);
        this._slots.set(morph, slot);
      }
      if (slot >= 0) this.morphs.writeSlot(LAYERS.VISEME, slot, value);
    }
  }

  /** Состояние для оверлея. */
  debug() {
    return {
      source: this.source,
      genId: this.genId,
      playing: this.isPlaying,
      viseme: this.stats.lastViseme,
      trackLength: this.track ? this.track.length : 0,
      complete: this.complete,
      finished: this.finished,
      trackIndex: this.trackIndex,
      drift_ms: this.stats.drift_ms,
      preroll_ms: this.stats.preroll_ms,
      underruns: this.stats.underruns,
      activeMorphs: this.current.size,
    };
  }
}
