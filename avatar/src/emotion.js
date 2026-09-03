// Слой эмоции: брови, веки, уголки рта.
//
// Владеет зоной emotion и общими морфами `eyeBlink*` и `browDown*`. Рта не
// касается — им распоряжаются висемы. Уголки рта (`mouthSmile*`,
// `mouthFrown*`, `mouthDimple*`) в зоне эмоции законно: это ДРУГИЕ морфы, чем
// те, что двигают висемы (челюсть, смыкание, вытягивание), и таблица зон это
// разделение стережёт — попытка слоя висем тронуть улыбку будет отброшена.
//
// В слой пишут двое: сама эмоция и автомат состояний. Их вклады СКЛАДЫВАЮТСЯ и
// клампятся, поэтому «скептичен и при этом перебит» даёт сумму обеих поз, а не
// одну из них. Отдельного writer'а у автомата состояний нет намеренно: два
// писателя в одну зону — это ровно тот конфликт, который таблица зон и
// разводит, поэтому автомат отдаёт позу сюда.

import { LAYERS } from './zones.js';

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

export class EmotionLayer {
  /**
   * @param {import('./morphs.js').MorphWriter} morphs
   * @param {object} cfg содержимое expression.config.json
   */
  constructor(morphs, cfg) {
    this.morphs = morphs;
    this.setConfig(cfg);

    this.emotion = { name: 'neutral', intensity: 0 };
    /** Поза от автомата состояний. Складывается с эмоцией. */
    this.statePose = {};

    // Текущие и целевые веса. Ключи — имена морфов.
    this.current = new Map();
    this._target = new Map();
    this._slots = new Map();

    this.missing = [];
  }

  setConfig(cfg) {
    this.cfg = cfg;
    this.emotions = cfg.emotions;
    this.transitionMs = cfg.transitionMs ?? 200;
  }

  /** Проверить, что все морфы поз есть в модели и лежат в разрешённых зонах. */
  validate() {
    const missing = new Set();
    const check = (pose) => {
      for (const morph of Object.keys(pose || {})) {
        if (morph.startsWith('_')) continue;
        if (this.morphs.slotOf(morph) < 0) missing.add(morph);
      }
    };
    for (const [name, e] of Object.entries(this.emotions)) {
      if (name.startsWith('_') || !e) continue;
      check(e.pose);
    }
    for (const [name, s] of Object.entries(this.cfg.states)) {
      if (name.startsWith('_') || !s) continue;
      check(s.pose);
      if (s.impatience) check(s.impatience.pose);
    }
    this.missing = [...missing];
    return this.missing;
  }

  setEmotion(name, intensity = 1) {
    const e = this.emotions[name];
    if (!e) throw new Error(`EmotionLayer: неизвестная эмоция «${name}»`);
    this.emotion = { name, intensity: clamp01(intensity) };
  }

  /** Поза от автомата состояний. Складывается с позой эмоции. */
  setStatePose(pose) {
    this.statePose = pose || {};
  }

  /** Множитель интервала моргания от эмоции — читает автомат состояний. */
  get blinkScale() {
    const e = this.emotions[this.emotion.name];
    if (!e) return 1;
    return 1 + ((e.blinkScale ?? 1) - 1) * this.emotion.intensity;
  }

  /** Множитель скорости артикуляции. Эмоция влияет на неё, но не на форму рта. */
  get articulationRate() {
    const e = this.emotions[this.emotion.name];
    if (!e) return 1;
    return 1 + ((e.articulationRate ?? 1) - 1) * this.emotion.intensity;
  }

  /** Наклон головы от эмоции, в градусах. Применяет слой микроповедения. */
  get headPitchDeg() {
    const e = this.emotions[this.emotion.name];
    if (!e) return 0;
    return (e.headPitchDeg ?? 0) * this.emotion.intensity;
  }

  /**
   * Один кадр. Вызывается между begin() и commit() writer'а морфов.
   * @param {number} dt секунды
   * @param {number} [transitionMs] время перехода; автомат состояний просит
   *        более быстрый вход в `interrupted`
   */
  update(dt, transitionMs) {
    const target = this._target;
    target.clear();

    const e = this.emotions[this.emotion.name];
    if (e && e.pose) {
      for (const [morph, w] of Object.entries(e.pose)) {
        if (morph.startsWith('_') || !Number.isFinite(w)) continue;
        target.set(morph, w * this.emotion.intensity);
      }
    }
    for (const [morph, w] of Object.entries(this.statePose)) {
      if (morph.startsWith('_') || !Number.isFinite(w)) continue;
      target.set(morph, (target.get(morph) || 0) + w);
    }

    // Один экспоненциальный подход на всё: у эмоции нет причин иметь
    // асимметричные времена, в отличие от артикуляции.
    const ms = transitionMs ?? this.transitionMs;
    const k = ms > 0 ? 1 - Math.exp(-dt * 1000 / ms) : 1;

    for (const [morph, value] of this.current) {
      if (!target.has(morph)) {
        const v = value + (0 - value) * k;
        if (v < 1e-4) this.current.delete(morph); else this.current.set(morph, v);
      }
    }
    for (const [morph, want] of target) {
      const have = this.current.get(morph) || 0;
      this.current.set(morph, have + (want - have) * k);
    }

    this._write();
  }

  _write() {
    for (const [morph, value] of this.current) {
      if (value <= 0) continue;
      let slot = this._slots.get(morph);
      if (slot === undefined) {
        slot = this.morphs.slotOf(morph);
        this._slots.set(morph, slot);
      }
      if (slot >= 0) this.morphs.writeSlot(LAYERS.EMOTION, slot, clamp01(value));
    }
  }

  debug() {
    return {
      emotion: this.emotion.name,
      intensity: this.emotion.intensity,
      activeMorphs: this.current.size,
      blinkScale: +this.blinkScale.toFixed(2),
      articulationRate: +this.articulationRate.toFixed(2),
    };
  }
}
