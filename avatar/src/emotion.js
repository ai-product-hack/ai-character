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
import { compileFacialClip, sampleFacialClip } from './facial-clip.js';

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
    this.speechActivity = 0;

    // Facial mocap clips are optional. Tests and emergency rollback keep using
    // the static poses until loadClips() has registered a valid clip.
    this.clips = new Map();
    this.clipErrors = [];
    this.clipTimeMs = 0;
    this.clipMotionEnabled = true;
    this._emotionFrame = new Map();
    // Цель кадра до кроссфейда. Живёт полем, а не локальной переменной: кадр
    // считается 60 раз в секунду, и бюджет «update() не аллоцирует» проверяется
    // тестом.
    this._wantFrame = new Map();
    this._crossfadeFrom = new Map();
    this._crossfadeMs = Infinity;
    this._phaseMs = 0;

    this.missing = [];
  }

  setConfig(cfg) {
    this.cfg = cfg;
    this.emotions = cfg.emotions;
    this.transitionMs = cfg.transitionMs ?? 200;
  }

  async loadClips(fetcher = fetch) {
    const C = this.cfg.clips || {};
    this.clips.clear();
    this.clipErrors.length = 0;
    if (!C.enabled) return;
    await Promise.all(Object.entries(this.emotions).map(async ([name, emotion]) => {
      if (name.startsWith('_') || !emotion?.clip) return;
      try {
        const path = `${C.basePath || '/avatar/clips'}/${emotion.clip}`;
        const response = await fetcher(path);
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        this.registerClip(name, await response.json());
      } catch (error) {
        this.clipErrors.push(`${name}: ${error.message}`);
      }
    }));
    this._pickPhase(this.emotion.name);
  }

  registerClip(name, raw) {
    const clip = compileFacialClip(raw, this.morphs);
    this.clips.set(name, clip);
    return clip;
  }

  /** Dev freeze: contribution stays visible, only the recorded time stops. */
  setClipMotionEnabled(on) { this.clipMotionEnabled = !!on; }

  _pickPhase(name) {
    const clip = this.clips.get(name);
    if (!clip) { this._phaseMs = 0; return; }
    if (this.cfg.clips?.randomizePhase === false) {
      let hash = 2166136261;
      for (const ch of name) hash = Math.imul(hash ^ ch.charCodeAt(0), 16777619);
      this._phaseMs = (hash >>> 0) % clip.durationMs;
    } else {
      this._phaseMs = Math.random() * clip.durationMs;
    }
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
    const changed = name !== this.emotion.name;
    if (changed) {
      this._crossfadeFrom.clear();
      for (const [morph, value] of this._emotionFrame) this._crossfadeFrom.set(morph, value);
      this._crossfadeMs = 0;
      this._pickPhase(name);
    }
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
  update(dt, transitionMs, speechActivity = 0) {
    const target = this._target;
    target.clear();

    const e = this.emotions[this.emotion.name];
    this._applyEmotion(target, e, dt);
    for (const [morph, w] of Object.entries(this.statePose)) {
      if (morph.startsWith('_') || !Number.isFinite(w)) continue;
      target.set(morph, (target.get(morph) || 0) + w);
    }

    // Речь двигает не только челюсть: мягко подключаем щёки, нос и брови к
    // активности артикуляции. Отдельное сглаживание не даёт им дёргаться на
    // каждом 40-миллисекундном таймкоде.
    const speech = this.cfg.speechMotion || {};
    const wantSpeech = clamp01(speechActivity);
    const speechMs = wantSpeech > this.speechActivity
      ? (speech.attackMs ?? 100) : (speech.decayMs ?? 180);
    const speechK = speechMs > 0 ? 1 - Math.exp(-dt * 1000 / speechMs) : 1;
    this.speechActivity += (wantSpeech - this.speechActivity) * speechK;
    for (const [morph, w] of Object.entries(speech.pose || {})) {
      if (morph.startsWith('_') || !Number.isFinite(w)) continue;
      target.set(morph, (target.get(morph) || 0) + w * this.speechActivity);
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

  _applyEmotion(target, emotion, dt) {
    const C = this.cfg.clips || {};
    const clip = C.enabled ? this.clips.get(this.emotion.name) : null;
    if (!clip) {
      this._emotionFrame.clear();
      if (emotion?.pose) {
        for (const [morph, w] of Object.entries(emotion.pose)) {
          if (morph.startsWith('_') || !Number.isFinite(w)) continue;
          const value = w * this.emotion.intensity;
          this._emotionFrame.set(morph, value);
          target.set(morph, value);
        }
      }
      return;
    }

    if (this.clipMotionEnabled) this.clipTimeMs += dt * 1000;
    const sample = sampleFacialClip(clip, this.clipTimeMs + this._phaseMs, C.seamMs ?? 400);
    const amplitude = (emotion.clipAmplitude ?? C.amplitude ?? 0.4) * this.emotion.intensity;
    const blend = C.blend !== false;
    const motion = (C.motion ?? C.amplitude ?? 0.4) * this.emotion.intensity;
    const duration = C.crossfadeMs ?? 400;
    this._crossfadeMs += dt * 1000;
    const x = duration > 0 ? Math.min(1, this._crossfadeMs / duration) : 1;
    const mix = x * x * (3 - 2 * x);

    // Цель кадра считается ЦЕЛИКОМ до кроссфейда, иначе поза, которой нет
    // среди каналов клипа, встала бы в кадр мимо смешивания и переключалась
    // рывком.
    //
    // Поза задаёт ФОРМУ, запись добавляет ДВИЖЕНИЕ вокруг неё.
    //
    // Раньше клип позу заменял целиком: при наличии записи `emotion.pose` не
    // применялась вовсе. Из-за этого вся продуманная статика для пяти
    // записанных эмоций не работала, а работала запись — со своей формой,
    // которая местами противоположна задуманной (в снятом `pressing` брови
    // идут ВВЕРХ, и давление читалось как лёгкое удивление). Вдобавок половину
    // записи выбрасывает таблица зон, так что до лица доезжала пятая часть
    // снятого.
    //
    // Вычитая среднее канала, берём от записи ровно то, чем она ценна, —
    // живое движение, — и не тащим её абсолютную форму. `clips.blend: false`
    // возвращает прежнее поведение без правки кода.
    const want = this._wantFrame;
    want.clear();
    if (blend) {
      for (const [morph, w] of Object.entries(emotion.pose || {})) {
        if (morph.startsWith('_') || !Number.isFinite(w)) continue;
        want.set(morph, w * this.emotion.intensity);
      }
    }
    // Вниз движение ограничено долей позы, вверх — свободно. Без этого запись
    // с большим размахом (у снятого `warming` улыбка гуляет почти на всю
    // шкалу) в нижней точке обнуляла заданную форму: улыбка на теплеющем лице
    // периодически пропадала совсем. Морфы, которых в позе нет, начинаются с
    // нуля и ограничения не получают — им двигаться неоткуда и некуда падать.
    const floor = C.motionFloor ?? 0.5;
    for (let i = 0; i < clip.channels.length; i++) {
      const morph = clip.channels[i].name;
      if (!blend) { want.set(morph, sample[i] * amplitude); continue; }
      const base = want.get(morph) || 0;
      const moved = base + (sample[i] - clip.means[i]) * motion;
      want.set(morph, Math.max(base * floor, Math.max(0, moved)));
    }

    // Кроссфейд: гасим то, что было только в прошлой эмоции, и подводим
    // остальное к цели.
    this._emotionFrame.clear();
    for (const [morph, from] of this._crossfadeFrom) {
      if (want.has(morph)) continue;
      const value = from * (1 - mix);
      if (value > 1e-5) this._emotionFrame.set(morph, value);
    }
    for (const [morph, value] of want) {
      const from = this._crossfadeFrom.get(morph) || 0;
      this._emotionFrame.set(morph, from + (value - from) * mix);
    }
    for (const [morph, value] of this._emotionFrame) target.set(morph, value);
    if (mix >= 1 && this._crossfadeFrom.size) this._crossfadeFrom.clear();
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
      speechActivity: +this.speechActivity.toFixed(2),
      clip: this.clips.has(this.emotion.name) ? this.emotion.name : null,
      clipMotion: this.clipMotionEnabled,
      clipErrors: this.clipErrors.length,
    };
  }
}
