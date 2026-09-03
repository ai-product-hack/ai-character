// Часы. Единственный допустимый источник времени — AudioContext.currentTime.
//
// Требование жёсткое и не декоративное: никаких setInterval и никаких
// собственных таймеров. Рендер живёт по часам аудиографа, а ухо слышит позже,
// поэтому из времени плеера вычитается outputLatency. В S2 измерено 24 мс:
// без вычитания рот опережает звук ровно на эту величину.
//
// Класс намеренно тонкий и без зависимости от three.js: его подменяют в тестах
// и на dev-странице, где звука может не быть вовсе.

export class AudioClock {
  /**
   * @param {AudioContext} ctx
   * @param {{compensateOutputLatency?: boolean}} [opts]
   */
  constructor(ctx, opts = {}) {
    this.ctx = ctx;
    this.compensate = opts.compensateOutputLatency !== false;
    this.t0 = null;              // якорь генерации в секундах графа
  }

  /** Задержка вывода устройства в секундах. У некоторых движков её нет. */
  get outputLatency() {
    return this.compensate ? (this.ctx.outputLatency || 0) : 0;
  }

  /**
   * Поставить якорь на начало генерации. Один якорь на всю генерацию: каждый
   * последующий кусок планируется относительно него, поэтому опоздавший кусок
   * оставляет дыру, а не сдвигает таймлайн и не рассинхронизирует лицо.
   */
  anchor(atSec) {
    this.t0 = atSec !== undefined ? atSec : this.ctx.currentTime;
    return this.t0;
  }

  reset() { this.t0 = null; }

  /** Сколько миллисекунд генерации сейчас слышит ухо. null, если якоря нет. */
  nowMs() {
    if (this.t0 === null) return null;
    return (this.ctx.currentTime - this.outputLatency - this.t0) * 1000;
  }
}

/**
 * Часы для dev-страницы и тестов: то же поведение, но время задаётся вручную.
 * Нужны, чтобы отлаживать лицо без конвейера и без звука.
 */
export class ManualClock {
  constructor(outputLatencySec = 0) {
    this.currentTime = 0;
    this.outputLatency = outputLatencySec;
    this.t0 = null;
  }

  anchor(atSec) { this.t0 = atSec !== undefined ? atSec : this.currentTime; return this.t0; }
  reset() { this.t0 = null; }
  advance(dtSec) { this.currentTime += dtSec; }
  nowMs() {
    if (this.t0 === null) return null;
    return (this.currentTime - this.outputLatency - this.t0) * 1000;
  }
}
