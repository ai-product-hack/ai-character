import { LAYERS, RULES } from './zones.js';

const clamp01 = (v) => Math.max(0, Math.min(1, Number(v) || 0));
const smooth = (t) => t * t * (3 - 2 * t);

/**
 * Validate and flatten an authoring clip for the render hot path. Channels
 * outside the emotion zone are discarded here, before a frame can write them.
 */
export function compileFacialClip(raw, morphs) {
  if (!raw || raw.version !== 1 || raw.kind !== 'face-mocap' ||
      !Array.isArray(raw.channels) || !Array.isArray(raw.frames) || raw.frames.length < 2) {
    throw new Error('FacialClip: неверный формат');
  }
  const kept = [];
  const filtered = [];
  raw.channels.forEach((name, sourceIndex) => {
    const rule = RULES.get(name);
    const slot = morphs.slotOf(name);
    if (slot >= 0 && rule?.layers.has(LAYERS.EMOTION)) {
      kept.push({ name, slot, sourceIndex });
    } else {
      filtered.push(name);
    }
  });
  const times = new Float32Array(raw.frames.length);
  const values = new Float32Array(raw.frames.length * kept.length);
  let previous = -1;
  raw.frames.forEach((frame, frameIndex) => {
    if (!Number.isFinite(frame.tMs) || frame.tMs < previous ||
        !Array.isArray(frame.weights) || frame.weights.length !== raw.channels.length) {
      throw new Error(`FacialClip: повреждён кадр ${frameIndex}`);
    }
    previous = frame.tMs;
    times[frameIndex] = frame.tMs;
    kept.forEach((channel, channelIndex) => {
      values[frameIndex * kept.length + channelIndex] = clamp01(frame.weights[channel.sourceIndex]);
    });
  });
  const durationMs = Number(raw.durationMs) || times.at(-1);
  if (!(durationMs > 0)) throw new Error('FacialClip: нулевая длительность');
  // Среднее по каждому каналу. Нужно, чтобы использовать запись как ДВИЖЕНИЕ
  // вокруг заданной позы, а не как саму позу: вычитая среднее, мы оставляем от
  // записи только то, чем она живая, и не тащим её абсолютную форму. Форма у
  // записи бывает попросту чужой — в снятом `pressing` брови идут вверх, тогда
  // как давление это брови вниз.
  const means = new Float32Array(kept.length);
  for (let frameIndex = 0; frameIndex < times.length; frameIndex++) {
    const base = frameIndex * kept.length;
    for (let c = 0; c < kept.length; c++) means[c] += values[base + c];
  }
  for (let c = 0; c < kept.length; c++) means[c] /= times.length || 1;
  return {
    name: raw.name,
    durationMs,
    times,
    values,
    means,
    channels: kept,
    filtered,
    sample: new Float32Array(kept.length),
  };
}

function framePair(clip, t) {
  const times = clip.times;
  let lo = 0, hi = times.length - 1;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= t) lo = mid; else hi = mid;
  }
  const span = times[hi] - times[lo];
  return [lo, hi, span > 0 ? (t - times[lo]) / span : 1];
}

function sampleLinear(clip, t, out) {
  const n = clip.channels.length;
  if (t <= clip.times[0]) {
    for (let c = 0; c < n; c++) out[c] = clip.values[c];
    return;
  }
  const last = clip.times.length - 1;
  if (t >= clip.times[last]) {
    const base = last * n;
    for (let c = 0; c < n; c++) out[c] = clip.values[base + c];
    return;
  }
  const [a, b, k] = framePair(clip, t);
  const baseA = a * n, baseB = b * n;
  for (let c = 0; c < n; c++) {
    const va = clip.values[baseA + c];
    out[c] = va + (clip.values[baseB + c] - va) * k;
  }
}

/** Sample a loop and make its tail converge exactly to frame zero. */
export function sampleFacialClip(clip, timeMs, seamMs = 400) {
  const duration = clip.durationMs;
  const t = ((timeMs % duration) + duration) % duration;
  sampleLinear(clip, t, clip.sample);
  const seam = Math.min(Math.max(0, seamMs), duration * 0.45);
  if (seam > 0 && t > duration - seam) {
    const k = smooth((t - (duration - seam)) / seam);
    for (let c = 0; c < clip.channels.length; c++) {
      clip.sample[c] += (clip.values[c] - clip.sample[c]) * k;
    }
  }
  return clip.sample;
}
