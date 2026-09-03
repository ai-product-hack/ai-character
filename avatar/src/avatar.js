// Публичный API модуля аватара.
//
//   avatar.load(url)
//   avatar.setState(state)                    listening | thinking | speaking | interrupted
//   avatar.setEmotion(emotion, intensity)     skeptical | pressing | warming | impressed | neutral
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
import { VisemeLayer, SOURCE } from './viseme.js';
import { AudioClock } from './clock.js';

export const STATES = Object.freeze(['listening', 'thinking', 'speaking', 'interrupted']);
export const EMOTIONS = Object.freeze(['neutral', 'skeptical', 'pressing', 'warming', 'impressed']);

export class Avatar {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{look: object, behavior: object, visemes: object}} configs
   */
  constructor(canvas, configs) {
    this.canvas = canvas;
    this.configs = configs;
    this.look = new Look(canvas, configs.look);
    this.model = null;
    this.behavior = null;
    this.visemes = null;
    this.clock = null;

    this.state = 'listening';
    this.emotion = { name: 'neutral', intensity: 0 };

    this._lastFrameMs = 0;
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

    this.visemes = new VisemeLayer(this.model.morphs, this.configs.visemes);
    const missing = this.visemes.validate();
    if (missing.length) {
      console.warn('avatar: морфов матрицы нет в модели:', missing);
    }
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
    if (!STATES.includes(state)) throw new Error(`avatar: неизвестное состояние «${state}»`);
    this.state = state;
    // Автомат состояний — шаг 6. Пока состояние только запоминается и видно
    // в оверлее; поведение по нему появится там же.
    return this;
  }

  setEmotion(emotion, intensity = 1) {
    if (!EMOTIONS.includes(emotion)) throw new Error(`avatar: неизвестная эмоция «${emotion}»`);
    this.emotion = { name: emotion, intensity: Math.max(0, Math.min(1, intensity)) };
    // Слой эмоции — шаг 7.
    return this;
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
    if (ok && this.behavior) this.behavior.notifyEvent();
    return ok;
  }

  setSize(w, h) {
    this.look.setSize(w, h);
    if (this.model) this.look.frameOn(this.model.frameTarget());
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
      morphs.begin();
      if (this.visemes) this.visemes.update(dt);
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
      postEnabled: this.look.postEnabled,
    };
  }
}

/** Собрать аватар, прочитав все три конфига. */
export async function createAvatar(canvas, urls) {
  const [look, behavior, visemes] = await Promise.all([
    fetch(urls.look).then((r) => r.json()),
    fetch(urls.behavior).then((r) => r.json()),
    fetch(urls.visemes).then((r) => r.json()),
  ]);
  return new Avatar(canvas, { look, behavior, visemes });
}

export { AudioClock, THREE };
