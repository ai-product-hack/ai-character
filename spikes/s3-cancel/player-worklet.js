// Ring-buffered PCM player. It is the only place that knows when sound actually
// stopped, so it — not the main thread — reports the stop timestamp.
class Player extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(sampleRate * 60);
    this.r = 0; this.w = 0;
    this.fading = false; this.fadeLeft = 0; this.fadeLen = 0;
    this.lastAudible = 0;
    this.draining = false;
    this.tickN = 0;
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.cmd === 'push') {
        const s = m.pcm;
        for (let i = 0; i < s.length; i++) {
          this.buf[this.w % this.buf.length] = s[i];
          this.w++;
        }
      } else if (m.cmd === 'flush') {
        // Hard stop: drop everything still queued.
        this.r = this.w; this.fading = false;
        this.port.postMessage({ ev: 'stopped', t: currentTime, mode: 'flush' });
      } else if (m.cmd === 'fade') {
        this.fadeLen = Math.max(1, Math.round(sampleRate * (m.ms / 1000)));
        this.fadeLeft = this.fadeLen; this.fading = true;
      } else if (m.cmd === 'drain') {
        // Naive barge-in: stop feeding, let the queue play out. The worklet
        // reports the exact moment it runs dry — no polling from the main
        // thread, so the number is not blurred by timer granularity.
        this.draining = true;
      } else if (m.cmd === 'query') {
        this.port.postMessage({ ev: 'depth', frames: this.w - this.r, t: currentTime });
      }
    };
  }

  process(_, outputs) {
    const out = outputs[0][0];
    for (let i = 0; i < out.length; i++) {
      let v = 0;
      if (this.r < this.w) {
        v = this.buf[this.r % this.buf.length];
        this.r++;
        if (this.fading) {
          v *= this.fadeLeft / this.fadeLen;
          if (--this.fadeLeft <= 0) {
            this.r = this.w;                 // fade done -> drop the rest
            this.fading = false;
            this.port.postMessage({ ev: 'stopped', t: currentTime, mode: 'fade' });
          }
        }
      }
      out[i] = v;
      if (v !== 0) this.lastAudible = currentTime + i / sampleRate;
      if (this.draining && this.r >= this.w) {
        this.draining = false;
        this.port.postMessage({ ev: 'stopped', t: currentTime + i / sampleRate,
                                mode: 'feed-stop' });
      }
    }
    // Throttled: a message per render quantum would be ~375/s for nothing.
    if ((this.tickN++ & 15) === 0) {
      this.port.postMessage({ ev: 'tick', last: this.lastAudible,
                              depth: this.w - this.r, t: currentTime });
    }
    return true;
  }
}
registerProcessor('player', Player);
