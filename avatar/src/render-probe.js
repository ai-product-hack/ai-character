// Стенд реального времени считает не «длительность кадра» из requestAnimationFrame
// (она снизу ограничена VSYNC), а число вертикальных интервалов на одну
// фактически выполненную отрисовку. 1 — кадр не пропущен, 2 — пропущен один.

function percentile(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * q) - 1)];
}

export class RenderProbe {
  constructor() { this.reset(); }

  reset() {
    this.startedAt = null;
    this.lastAt = null;
    this.deltas = [];
    this.spans = [];
    this.cpu = [];
    this.drawCalls = [];
    this.renderCount = 0;
    this.focusedCount = 0;
    this.visibleCount = 0;
    this.postCount = 0;
    this.vsyncMs = 1000 / 60;
  }

  sample(nowMs, { cpuMs = 0, drawCalls = 0, focused = true, visible = true,
                  postEnabled = true } = {}) {
    if (this.startedAt === null) this.startedAt = nowMs;
    if (this.lastAt !== null) {
      const delta = nowMs - this.lastAt;
      if (delta > 3 && delta < 100 && this.deltas.length < 180) {
        this.deltas.push(delta);
        // Медиана коротких интервалов оценивает физический период дисплея:
        // нижний дециль слишком чувствителен к джиттеру таймера и занижает
        // 16.67 до 15.x мс. Длинные интервалы от пропусков сюда не попадают.
        if (this.deltas.length % 30 === 0) {
          const short = this.deltas.filter((value) => value < 25);
          if (short.length >= 15) this.vsyncMs = percentile(short, 0.5);
        }
      }
      this.spans.push(Math.max(1, Math.round(delta / this.vsyncMs)));
    }
    this.lastAt = nowMs;
    this.renderCount++;
    this.cpu.push(cpuMs);
    this.drawCalls.push(drawCalls);
    if (focused) this.focusedCount++;
    if (visible) this.visibleCount++;
    if (postEnabled) this.postCount++;
  }

  result() {
    const elapsedMs = this.lastAt === null || this.startedAt === null
      ? 0 : this.lastAt - this.startedAt;
    const intervals = this.spans.reduce((sum, n) => sum + n, 0);
    const measuredRenders = this.spans.length;
    const missed = Math.max(0, intervals - measuredRenders);
    return {
      durationSec: elapsedMs / 1000,
      renders: this.renderCount,
      vsyncMs: this.vsyncMs,
      vsyncIntervals: intervals,
      missedVsyncs: missed,
      missedPercent: intervals ? missed / intervals * 100 : 0,
      spanP99: percentile(this.spans, 0.99),
      spanMax: this.spans.length ? Math.max(...this.spans) : 0,
      stalledRenders: this.spans.filter((span) => span > 1).length,
      fps: elapsedMs > 0 ? (this.renderCount - 1) * 1000 / elapsedMs : 0,
      cpuP50Ms: percentile(this.cpu, 0.5),
      cpuP99Ms: percentile(this.cpu, 0.99),
      drawCallsP99: percentile(this.drawCalls, 0.99),
      focusedPercent: this.renderCount ? this.focusedCount / this.renderCount * 100 : 0,
      visiblePercent: this.renderCount ? this.visibleCount / this.renderCount * 100 : 0,
      postEnabledPercent: this.renderCount ? this.postCount / this.renderCount * 100 : 0,
    };
  }
}
