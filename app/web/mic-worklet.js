// Захват микрофона: копит кадры и отдаёт их пачками по ~100 мс.
//
// Работает в аудиопотоке, поэтому здесь нельзя ничего тяжёлого и нельзя
// таймеров: кадры приходят ровно тогда, когда их отдаёт железо, и это
// единственные честные часы у входа.
class MicCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const ms = (options && options.processorOptions
                && options.processorOptions.chunkMs) || 100;
    // sampleRate — глобальная константа воркета, равна частоте контекста.
    this.size = Math.round(sampleRate * ms / 1000);
    this.buf = new Float32Array(this.size);
    this.at = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.buf[this.at++] = ch[i];
      if (this.at === this.size) {
        // Копия обязательна: буфер переиспользуется, а сообщение уедет позже.
        this.port.postMessage(this.buf.slice());
        this.at = 0;
      }
    }
    return true;
  }
}

registerProcessor('mic-capture', MicCapture);
