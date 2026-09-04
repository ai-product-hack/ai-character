const clamp01 = (v) => Math.max(0, Math.min(1, Number(v) || 0));

/** Canonical, compact representation used by the recorder and runtime. */
export function makeClip(name, frames, channels, meta = {}) {
  if (!Array.isArray(frames) || frames.length < 2) {
    throw new Error('Клип должен содержать минимум два кадра');
  }
  const cleanChannels = [...channels];
  const cleanFrames = frames.map((frame) => ({
    tMs: Math.max(0, Math.round(frame.tMs)),
    weights: cleanChannels.map((_, i) => clamp01(frame.weights?.[i])),
    ...(Array.isArray(frame.head) ? { head: frame.head.slice(0, 3).map(Number) } : {}),
  })).sort((a, b) => a.tMs - b.tMs);
  const origin = cleanFrames[0].tMs;
  for (const frame of cleanFrames) frame.tMs -= origin;
  return {
    version: 1,
    kind: 'face-mocap',
    name,
    channels: cleanChannels,
    durationMs: cleanFrames.at(-1).tMs,
    frames: cleanFrames,
    source: {
      tool: 'MediaPipe Face Landmarker',
      ...meta,
    },
  };
}

export function trimClip(clip, fromMs, toMs) {
  const start = Math.max(0, Math.min(fromMs, clip.durationMs));
  const end = Math.max(start + 1, Math.min(toMs, clip.durationMs));
  const frames = clip.frames
    .filter((frame) => frame.tMs >= start && frame.tMs <= end)
    .map((frame) => ({ ...frame, tMs: frame.tMs - start }));
  if (!frames.length || frames[0].tMs > 0) frames.unshift(sampleClip(clip, start, false));
  frames[0] = { ...frames[0], tMs: 0 };
  if (frames.at(-1).tMs < end - start) {
    frames.push({ ...sampleClip(clip, end, false), tMs: end - start });
  }
  return makeClip(clip.name, frames, clip.channels, clip.source);
}

/** Linear sampling. Looping cross-fades the tail into the beginning. */
export function sampleClip(clip, timeMs, loop = true, seamMs = 350) {
  const duration = Math.max(1, clip.durationMs);
  let t = loop ? ((timeMs % duration) + duration) % duration
               : Math.max(0, Math.min(timeMs, duration));
  const sampleLinear = (at) => {
    let hi = clip.frames.findIndex((frame) => frame.tMs >= at);
    if (hi <= 0) return { ...clip.frames[0], weights: [...clip.frames[0].weights] };
    if (hi < 0) return { ...clip.frames.at(-1), weights: [...clip.frames.at(-1).weights] };
    const a = clip.frames[hi - 1], b = clip.frames[hi];
    const k = b.tMs === a.tMs ? 1 : (at - a.tMs) / (b.tMs - a.tMs);
    return {
      tMs: at,
      weights: a.weights.map((v, i) => v + (b.weights[i] - v) * k),
      ...(a.head && b.head ? { head: a.head.map((v, i) => v + (b.head[i] - v) * k) } : {}),
    };
  };
  const out = sampleLinear(t);
  const seam = Math.min(Math.max(0, seamMs), duration * 0.45);
  if (loop && seam > 0 && t > duration - seam) {
    const k = (t - (duration - seam)) / seam;
    // The end must converge exactly to frame zero. Blending against a moving
    // segment from the beginning leaves a discontinuity at modulo wrap.
    const beginning = sampleLinear(0);
    out.weights = out.weights.map((v, i) => v + (beginning.weights[i] - v) * k);
    if (out.head && beginning.head) {
      out.head = out.head.map((v, i) => v + (beginning.head[i] - v) * k);
    }
  }
  return out;
}

export function validateClip(value) {
  if (!value || value.version !== 1 || value.kind !== 'face-mocap') return 'неизвестный формат';
  if (!Array.isArray(value.channels) || !value.channels.length) return 'нет каналов';
  if (!Array.isArray(value.frames) || value.frames.length < 2) return 'слишком мало кадров';
  if (!Number.isFinite(value.durationMs) || value.durationMs <= 0) return 'неверная длительность';
  if (value.frames.some((f) => !Number.isFinite(f.tMs) || f.weights?.length !== value.channels.length)) {
    return 'кадр не совпадает со списком каналов';
  }
  return null;
}
