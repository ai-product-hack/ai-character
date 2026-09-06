// Публичный API модуля аватара.
//
//   avatar.load(url)
//   avatar.setState(state)                    listening | thinking | speaking | interrupted
//   avatar.setEmotion(emotion, intensity)     neutral | skeptical | pressing | warming |
//                                             impressed | angry | anxious
//   avatar.noteActivity()                     человек печатает — не торопить
//   avatar.playGeneration(genId, visemeTrack)
//   avatar.cancel(genId)
//   avatar.attachClock(audioClock)
//
// Порядок в кадре здесь и нигде больше: обнулить вклады -> слои пишут в своих
// зонах -> разложить по мешам -> нарисовать. Слои между собой не общаются,
// конфликты за морфы разрешает таблица зон.

import * as THREE from 'three';
import { Look } from './look.js';
import { loadAvatarModel } from './model.js';
import { Microbehavior } from './behavior.js';
import { BodyIdle } from './body-idle.js';
import { VisemeLayer, SOURCE } from './viseme.js';
import { EmotionLayer } from './emotion.js';
import { StateMachine } from './states.js';
import { AudioClock } from './clock.js';

export { STATES } from './states.js';
// Первые пять лежат на оси «холодно → тепло» и выводятся из накопленных
// оценок; angry и anxious на ней не лежат — их назначает персона сценария или
// разметка [emo:] в реплике. Для слоя эмоции разницы нет, она вся в том, кто
// эмоцию включает.
export const EMOTIONS = Object.freeze([
  'neutral', 'skeptical', 'pressing', 'warming', 'impressed', 'angry', 'anxious',
]);

export class Avatar {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{look: object, behavior: object, visemes: object, expression: object}} configs
   */
  constructor(canvas, configs) {
    this.canvas = canvas;
    this.configs = configs;
    this.look = new Look(canvas, configs.look);
    this.model = null;
    this.behavior = null;
    this.bodyIdle = null;
    this.visemes = null;
    this.emotionLayer = null;
    this.states = null;
    this.clock = null;

    this._lastFrameMs = 0;
    this._speechBeatIn = 0;
    this._pendingAccents = [];
    this._running = false;
    this.stats = { fps: 0, frameMs: 0, cpuMs: 0 };
    this._fps = { frames: 0, acc: 0 };
  }

  /** Загрузить модель и собрать слои. Путь берётся из конфига, не из кода. */
  async load(url) {
    const cfg = url ? { ...this.configs.look, model: { ...this.configs.look.model, url } }
                    : this.configs.look;
    this.model = await loadAvatarModel(cfg);
    this.look.scene.add(this.model.root);
    this.look.frameOn(this.model.frameTarget());

    this.behavior = new Microbehavior(this.model, this.configs.behavior);
    this.behavior.setAnchor(this.look.camera.position);
    this.bodyIdle = new BodyIdle(this.model, this.configs.behavior);
    await this.bodyIdle.loadClips();

    this.visemes = new VisemeLayer(this.model.morphs, this.configs.visemes);
    const missing = this.visemes.validate();
    if (missing.length) {
      console.warn('avatar: морфов матрицы нет в модели:', missing);
    }

    this.emotionLayer = new EmotionLayer(this.model.morphs, this.configs.expression);
    await this.emotionLayer.loadClips();
    const missingEmo = this.emotionLayer.validate();
    if (missingEmo.length) {
      console.warn('avatar: морфов поз нет в модели:', missingEmo);
    }
    this.states = new StateMachine(this.behavior, this.emotionLayer, this.configs.expression);
    if (this.clock) this.visemes.attachClock(this.clock);
    return this;
  }

  /**
   * Источник времени. Только часы аудиографа — никаких setInterval и никаких
   * собственных таймеров.
   */
  attachClock(audioClock) {
    this.clock = audioClock;
    if (this.visemes) this.visemes.attachClock(audioClock);
    return this;
  }

  /** Уровень 3 лестницы отступления. */
  useAnalyser(analyser) {
    if (!this.visemes) return this;
    this.visemes.attachAnalyser(analyser);
    this.visemes.setSource(analyser ? SOURCE.ANALYSER : SOURCE.VISEMES);
    return this;
  }

  setState(state) {
    if (!this.states) throw new Error('avatar: модель ещё не загружена');
    this.states.set(state);
    return this;
  }

  /**
   * Человек печатает. Гасит нарастающее нетерпение, не меняя состояния:
   * персонаж ждёт спокойно, пока видит работу, и начинает торопить, только
   * когда экран замер.
   */
  noteActivity() {
    if (this.states) this.states.noteActivity();
    return this;
  }

  /** Queue a gesture until articulation is actually playing. */
  queueSpeechAccent(kind = 'speech', ptsMs = 0) {
    this._pendingAccents.push({ kind, ptsMs });
    this._pendingAccents.sort((a, b) => a.ptsMs - b.ptsMs);
    return this;
  }

  /** Осмысленный жест на начале заполнителя или содержательной реплики. */
  speechAccent(kind = 'speech') {
    if (!this.behavior) return this;
    const cfg = this.configs.expression.speechMotion || {};
    const accent = this.configs.expression.emotions[this.emotion.name]?.accent;
    if (kind === 'backchannel') {
      this.behavior.nod(cfg.backchannelNodDeg ?? 2, cfg.nodDurationMs ?? 520);
    } else if (accent) {
      const strength = this.emotion.name === 'neutral' ? 1 : this.emotion.intensity;
      this.behavior.gesture({ ...accent, pitchDeg: accent.pitchDeg * strength,
        yawDeg: accent.yawDeg * strength, rollDeg: accent.rollDeg * strength });
    } else {
      this.behavior.nod(cfg.nodDeg ?? 1.2, cfg.nodDurationMs ?? 520);
    }
    this._speechBeatIn = cfg.beatEverySec ?? 2.8;
    this.behavior.notifyEvent();
    return this;
  }

  setEmotion(emotion, intensity = 1) {
    if (!this.emotionLayer) throw new Error('avatar: модель ещё не загружена');
    this.emotionLayer.setEmotion(emotion, intensity);
    // Эмоция меняет частоту моргания и наклон головы — пересобрать модуляцию.
    this.states.apply();
    return this;
  }

  get state() { return this.states ? this.states.state : 'listening'; }
  get emotion() {
    return this.emotionLayer ? this.emotionLayer.emotion : { name: 'neutral', intensity: 0 };
  }

  playGeneration(genId, visemeTrack) {
    if (!this.visemes) throw new Error('avatar: модель ещё не загружена');
    this.visemes.playGeneration(genId, visemeTrack);
    return this;
  }

  /**
   * Отбросить генерацию. Всё с чужим generation_id умирает безусловно;
   * рот схлопывается быстро, потому что зритель должен увидеть реакцию,
   * а не тишину.
   */
  cancel(genId) {
    if (!this.visemes) return false;
    const releaseMs = this.configs.visemes.timing.interruptReleaseMs ?? 45;
    const ok = this.visemes.cancel(genId, releaseMs);
    if (ok) { this._pendingAccents.length = 0; this._speechBeatIn = 0; }
    if (ok && this.behavior) this.behavior.notifyEvent();
    // Отмена и реакция — одно событие: жюри должно увидеть не тишину, а лицо.
    if (ok && this.states) this.states.set('interrupted');
    return ok;
  }

  setSize(w, h) {
    this.look.setSize(w, h);
    if (this.model) this.look.frameOn(this.model.frameTarget());
    if (this.behavior) this.behavior.setAnchor(this.look.camera.position);
  }

  /**
   * Один кадр. Порядок фиксирован: слои вносят вклады в writer, writer
   * раскладывает их по всем мешам, дальше рендер.
   */
  frame(nowMs) {
    const dt = this._lastFrameMs ? Math.min((nowMs - this._lastFrameMs) / 1000, 0.1) : 1 / 60;
    this._lastFrameMs = nowMs;

    const t0 = performance.now();
    if (this.model) {
      const morphs = this.model.morphs;
      // Порядок фиксирован: автомат состояний задаёт позу и модуляцию, слои
      // пишут в своих зонах, writer раскладывает по мешам.
      const fastMs = this.states ? this.states.update(dt) : null;
      morphs.begin();
      if (this.visemes) this.visemes.update(dt * this.emotionLayer.articulationRate);
      if (this.emotionLayer) {
        const speechActivity = this.visemes && this.visemes.isPlaying
          ? this.visemes.activity : 0;
        this.emotionLayer.update(dt, fastMs, speechActivity);
        this._speechBeatIn -= dt;
        const audioMs = this.clock?.nowMs();
        let accent = null;
        while (audioMs != null && this._pendingAccents.length &&
               this._pendingAccents[0].ptsMs <= audioMs) {
          accent = this._pendingAccents.shift();
        }
        if (this.state === 'speaking' && audioMs != null && audioMs >= 0 && speechActivity > 0.18 &&
            (accent || this._speechBeatIn <= 0)) {
          this.speechAccent(accent?.kind || 'speech');
        } else if (accent && this.state === 'speaking') {
          // A clause can start with silence: retain its cue until the first phoneme.
          this._pendingAccents.unshift(accent);
        }
      }
      if (this.bodyIdle) this.bodyIdle.update(dt, this.state, this.emotion);
      if (this.behavior) this.behavior.update(dt, nowMs / 1000);
      morphs.commit();
    }
    this.stats.cpuMs = performance.now() - t0;

    this.look.render(nowMs / 1000);

    this._fps.frames++;
    this._fps.acc += dt;
    if (this._fps.acc >= 0.5) {
      this.stats.fps = this._fps.frames / this._fps.acc;
      this._fps.frames = 0; this._fps.acc = 0;
    }
    this.stats.frameMs = dt * 1000;
  }

  /** Сводка для оверлея. */
  debug() {
    return {
      state: this.state,
      emotion: this.emotion,
      fps: this.stats.fps,
      frameMs: this.stats.frameMs,
      cpuMs: this.stats.cpuMs,
      outputLatencyMs: this.clock ? this.clock.outputLatency * 1000 : null,
      audioMs: this.clock ? this.clock.nowMs() : null,
      viseme: this.visemes ? this.visemes.debug() : null,
      behavior: this.behavior ? this.behavior.debug() : null,
      bodyIdle: this.bodyIdle ? this.bodyIdle.debug() : null,
      states: this.states ? this.states.debug() : null,
      emotionLayer: this.emotionLayer ? this.emotionLayer.debug() : null,
      postEnabled: this.look.postEnabled,
    };
  }
}

/** Собрать аватар, прочитав все три конфига. */
export async function createAvatar(canvas, urls) {
  const [look, behavior, visemes, expression] = await Promise.all([
    fetch(urls.look).then((r) => r.json()),
    fetch(urls.behavior).then((r) => r.json()),
    fetch(urls.visemes).then((r) => r.json()),
    fetch(urls.expression).then((r) => r.json()),
  ]);
  return new Avatar(canvas, { look, behavior, visemes, expression });
}

export { AudioClock, THREE };
