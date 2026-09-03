// Aggregates mic samples into fixed 20 ms RMS frames. No setInterval anywhere:
// the audio thread is the clock.
class MeterProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frameLen = Math.round(sampleRate * 0.02); // 20 ms
    this.acc = 0;
    this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.acc += ch[i] * ch[i];
      if (++this.n >= this.frameLen) {
        this.port.postMessage({ t: currentTime, rms: Math.sqrt(this.acc / this.n) });
        this.acc = 0;
        this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor('meter', MeterProcessor);
