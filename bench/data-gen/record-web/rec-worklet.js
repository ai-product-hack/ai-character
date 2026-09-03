// Накапливает сырые отсчёты микрофона. Отдаёт их одним куском по команде stop:
// для 35 коротких фраз проще и точнее, чем стримить.
class Rec extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunks = [];
    this.on = false;
    this.port.onmessage = (e) => {
      if (e.data.cmd === 'start') { this.chunks = []; this.on = true; }
      else if (e.data.cmd === 'stop') {
        this.on = false;
        let n = 0;
        for (const c of this.chunks) n += c.length;
        const out = new Float32Array(n);
        let o = 0;
        for (const c of this.chunks) { out.set(c, o); o += c.length; }
        this.chunks = [];
        this.port.postMessage({ ev: 'clip', pcm: out }, [out.buffer]);
      }
    };
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (this.on && ch && ch.length) {
      this.chunks.push(new Float32Array(ch));
      let s = 0;
      for (let i = 0; i < ch.length; i++) s += ch[i] * ch[i];
      this.port.postMessage({ ev: 'level', rms: Math.sqrt(s / ch.length) });
    }
    return true;
  }
}
registerProcessor('rec', Rec);
