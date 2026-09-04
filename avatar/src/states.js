// Автомат состояний: listening, thinking, speaking, interrupted.
//
// Состояние НЕ пишет в морфы напрямую. Оно отдаёт позу слою эмоции (там позы
// складываются и клампятся) и смещает слой микроповедения через setGazeBias,
// setBlinkScale, setHeadPose и snapGaze. Причина не в чистоте: два писателя в
// одну зону — это ровно тот конфликт, который таблица зон разводит, и заводить
// четвёртого владельца бровей значило бы вернуть его обратно.
//
// Единственный источник времени — тот же, что у всего модуля: секунды кадра.
// Никаких setTimeout, включая задержку выхода из `interrupted`.

export const STATES = Object.freeze(['listening', 'thinking', 'speaking', 'interrupted']);

export class StateMachine {
  /**
   * @param {import('./behavior.js').Microbehavior} behavior
   * @param {import('./emotion.js').EmotionLayer} emotion
   * @param {object} cfg содержимое expression.config.json
   */
  constructor(behavior, emotion, cfg) {
    this.behavior = behavior;
    this.emotion = emotion;
    this.setConfig(cfg);

    this.state = 'listening';
    this.sinceEnter = 0;
    // Часы нетерпения идут отдельно от часов состояния: человек, который
    // печатает ответ, не «молчит», и подгонять его нечестно. Набор текста
    // сбрасывает именно этот счётчик, не трогая ни кивки, ни само состояние.
    this.idleSince = 0;
    this.impatience = 0;          // 0..1, растёт на паузе БЕЗ активности
    this._nodIn = this._nextNodDelay();
    this._pose = {};              // переиспользуемый объект, без аллокаций в кадре
    this._transitionMs = null;    // разовое ускорение перехода
    this.apply();
  }

  setConfig(cfg) {
    this.cfg = cfg;
    this.states = cfg.states;
  }

  get config() { return this.states[this.state] || {}; }

  /**
   * Сменить состояние. Повторная установка того же состояния ничего не
   * сбрасывает: `setState('speaking')` на каждую реплику не должен обнулять
   * накопленное нетерпение или таймер кивка.
   */
  set(state) {
    if (!STATES.includes(state)) throw new Error(`StateMachine: неизвестное состояние «${state}»`);
    if (state === this.state) return this;
    this.state = state;
    this.sinceEnter = 0;
    this.idleSince = 0;
    if (state !== 'listening') this.impatience = 0;

    const c = this.config;
    // Вход в interrupted быстрее обычного: реакция должна быть резкой.
    this._transitionMs = c.enterMs ?? null;
    if (c.snapGaze) this.behavior.snapGaze();
    this.apply();
    return this;
  }

  _nextNodDelay() {
    const n = this.config.nod;
    if (!n) return Infinity;
    const [lo, hi] = n.everySec;
    return lo + Math.random() * (hi - lo);
  }

  /** Разложить текущее состояние в позу и модуляцию. */
  apply() {
    const c = this.config;
    const pose = this._pose;
    for (const k in pose) delete pose[k];

    for (const [morph, w] of Object.entries(c.pose || {})) {
      if (morph.startsWith('_')) continue;
      pose[morph] = w;
    }

    let biasYaw = (c.gazeBias || [0, 0])[0];
    let biasPitch = (c.gazeBias || [0, 0])[1];
    let blink = c.blinkScale ?? 1;

    // Нетерпение подмешивается пропорционально, а не включается порогом:
    // скачок был бы виден как подёргивание.
    const imp = c.impatience;
    if (imp && this.impatience > 0) {
      const k = this.impatience;
      for (const [morph, w] of Object.entries(imp.pose || {})) {
        if (morph.startsWith('_')) continue;
        pose[morph] = (pose[morph] || 0) + w * k;
      }
      biasYaw += (imp.gazeBias[0] - biasYaw) * k;
      biasPitch += (imp.gazeBias[1] - biasPitch) * k;
      blink += ((imp.blinkScale ?? 1) - blink) * k;
    }

    this.emotion.setStatePose(pose);
    this.behavior.setGazeBias(biasYaw, biasPitch);
    this.behavior.setGazeStyle(c.gazeStyle);
    // Множители моргания от состояния и от эмоции перемножаются: «думает и при
    // этом скептичен» должно давать оба эффекта, а не последний назначенный.
    this.behavior.setBlinkScale(blink * this.emotion.blinkScale);
    this.behavior.setHeadPose(c.headTiltDeg ?? 0, this.emotion.headPitchDeg);
  }

  /**
   * Один кадр. Вызывается ДО слоя эмоции, потому что задаёт ему позу.
   * @returns {number|null} время перехода для слоя эмоции, если оно не обычное
   */
  /**
   * Человек проявил активность — печатает. Нетерпение откатывается к нулю
   * плавно, обычным переходом позы: резкий сброс с 0.8 на 0 читался бы как
   * дёрганье.
   */
  noteActivity() {
    this.idleSince = 0;
    return this;
  }

  update(dt) {
    this.sinceEnter += dt;
    this.idleSince += dt;
    const c = this.config;

    // Нетерпение: на паузе дольше afterSec взгляд уходит в сторону, бровь
    // поднимается. Нарастает плавно за rampSec.
    if (c.impatience) {
      const over = this.idleSince - c.impatience.afterSec;
      const want = over <= 0 ? 0 : Math.min(1, over / c.impatience.rampSec);
      if (want !== this.impatience) {
        this.impatience = want;
        this.apply();
      }
    }

    // Микрокивок.
    if (c.nod) {
      this._nodIn -= dt;
      if (this._nodIn <= 0) {
        this.behavior.nod(c.nod.amplitudeDeg, c.nod.durationMs);
        this._nodIn = this._nextNodDelay();
      }
    }

    // Выход из interrupted по часам кадра, а не по setTimeout.
    if (c.holdMs && this.sinceEnter * 1000 >= c.holdMs) this.set('listening');

    // Ускоренный вход действует, пока поза не сойдётся, а не один кадр.
    // Первая версия сбрасывала его сразу после первого возврата, и брови
    // выходили на 80% за 273 мс вместо обещанных 90 — то есть «резкая реакция»
    // была резкой только в конфиге.
    const ms = this._transitionMs;
    if (ms !== null && this.sinceEnter * 1000 >= ms * 4) this._transitionMs = null;
    return ms;
  }

  debug() {
    return {
      state: this.state,
      sinceEnter: +this.sinceEnter.toFixed(2),
      impatience: +this.impatience.toFixed(2),
      nodIn: this._nodIn === Infinity ? null : +this._nodIn.toFixed(1),
    };
  }
}
