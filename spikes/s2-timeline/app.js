// S2: audio and face on one clock. The audio clock is the master; the render
// loop asks it what time it is on every frame. No setInterval anywhere.
const $ = (id) => document.getElementById(id);
const log = (m) => { $('log').textContent += m + '\n'; $('log').scrollTop = 1e9; };

let ctx, t0 = null, face = [], playing = false, analyser = null, gainNode = null;
let drift = [], rafGaps = [], underruns = 0, lastRaf = 0, nextAudioAt = 0;
// (audioMs, jawOpen) sampled from whichever source drives the mouth. In
// analyser mode this is cross-correlated against the ground-truth track to
// measure how far the analyser lags the audio it is reading.
let trace = [];

function lerp(a, b, k) {
  const o = {};
  for (const key in a) o[key] = a[key] + (b[key] - a[key]) * k;
  return o;
}

// Binary search for the bracketing pair, then interpolate. Holding the last
// frame instead would quantise the mouth to 33 ms steps.
function poseAt(ms) {
  if (!face.length) return null;
  let lo = 0, hi = face.length - 1;
  if (ms <= face[0].pts_ms) return { pose: face[0].v, drift: face[0].pts_ms - ms, exact: false };
  if (ms >= face[hi].pts_ms) return { pose: face[hi].v, drift: face[hi].pts_ms - ms, exact: false };
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (face[mid].pts_ms <= ms) lo = mid; else hi = mid;
  }
  const a = face[lo], b = face[hi];
  const k = (ms - a.pts_ms) / (b.pts_ms - a.pts_ms);
  return { pose: lerp(a.v, b.v, k), drift: 0, exact: true, gap: b.pts_ms - a.pts_ms };
}

function render(now) {
  requestAnimationFrame(render);
  if (lastRaf) rafGaps.push(now - lastRaf);
  lastRaf = now;
  if (!playing || t0 === null) return;

  const outLat = $('comp').checked ? (ctx.outputLatency || 0) : 0;
  // What the ear is hearing right now, not what the buffer has swallowed.
  const audioMs = (ctx.currentTime - outLat - t0) * 1000;
  const r = poseAt(audioMs);
  if (!r) return;
  if (!r.exact) underruns++;
  drift.push(r.drift);

  let jaw, funnel;
  if ($('src').value === 'analyser' && analyser) {
    // Level 3 of R4: amplitude straight off the graph, no timing metadata at
    // all. Cheapest possible lipsync — the question is only how late it is.
    const buf = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    jaw = Math.min(1, Math.sqrt(sum / buf.length) * 6.0);
    funnel = 0;
  } else {
    jaw = r.pose.jawOpen; funnel = r.pose.mouthFunnel;
  }
  trace.push([audioMs, jaw]);
  $('mouth').style.height = (8 + jaw * 70).toFixed(1) + 'px';
  $('mouth').style.width = (110 - funnel * 45).toFixed(1) + 'px';
  $('hud').textContent =
    `аудио: ${audioMs.toFixed(0)} мс | jawOpen ${jaw.toFixed(3)} | ` +
    `дрейф кадра ${r.drift.toFixed(1)} мс | rAF ${(rafGaps.at(-1) || 0).toFixed(1)} мс | ` +
    `подвисаний ${underruns} | outputLatency ${((ctx.outputLatency || 0) * 1000).toFixed(1)} мс`;
}

async function start() {
  ctx = new AudioContext();
  await ctx.resume();
  drift = []; rafGaps = []; underruns = 0; face = []; trace = [];
  gainNode = ctx.createGain();
  gainNode.connect(ctx.destination);
  analyser = ctx.createAnalyser();
  analyser.fftSize = +$('fft').value;
  analyser.connect(ctx.destination);
  const resp = await fetch(`/stream?speed=${$('speed').value}`);
  const sr = +resp.headers.get('X-Sample-Rate');
  const reader = resp.body.getReader();
  let acc = new Uint8Array(0);
  playing = true;
  log(`старт | sampleRate=${ctx.sampleRate} | outputLatency=${((ctx.outputLatency||0)*1e3).toFixed(1)} мс`);

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const n = new Uint8Array(acc.length + value.length);
    n.set(acc); n.set(value, acc.length); acc = n;
    while (acc.length >= 8) {
      const dv = new DataView(acc.buffer, acc.byteOffset, acc.byteLength);
      const jl = dv.getUint32(0, true), pl = dv.getUint32(4, true);
      if (acc.length < 8 + jl + pl) break;
      const hdr = JSON.parse(new TextDecoder().decode(acc.subarray(8, 8 + jl)));
      const payload = acc.subarray(8 + jl, 8 + jl + pl);
      acc = acc.subarray(8 + jl + pl);

      if (hdr.kind === 'face') { face.push(hdr); continue; }

      const i16 = new Int16Array(payload.slice().buffer);
      const buf = ctx.createBuffer(1, i16.length, sr);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < i16.length; i++) ch[i] = i16[i] / 32768;
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(analyser);
      if (t0 === null) {
        // One anchor for the whole generation: every later chunk is scheduled
        // against it, so a chunk arriving late leaves a gap instead of shifting
        // the timeline and desynchronising the face.
        t0 = ctx.currentTime + 0.12;
        nextAudioAt = t0;
      }
      const at = t0 + hdr.pts_ms / 1000;
      src.start(Math.max(at, ctx.currentTime));
      nextAudioAt = at + buf.duration;
    }
  }
  // The stream ends long before playback does — the server runs ahead of real
  // time. Keep measuring until the last scheduled chunk has actually played,
  // otherwise the sample covers only the first second.
  while (ctx.currentTime < nextAudioAt) {
    await new Promise((r) => setTimeout(r, 100));
  }
  playing = false;
  report();
}

function pct(a, p) { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length * p)] || 0; }

// Lag of the analyser-driven mouth against the ground-truth blendshape track:
// the shift that maximises correlation. Objective, no eyeballing.
function measureLag() {
  if (trace.length < 30 || !face.length) return null;
  const ref = (ms) => { const r = poseAt(ms); return r ? r.pose.jawOpen : 0; };
  let best = null;
  for (let shift = -200; shift <= 200; shift += 5) {
    let sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0, n = 0;
    for (const [ms, jaw] of trace) {
      const v = ref(ms + shift);
      sx += jaw; sy += v; sxy += jaw * v; sxx += jaw * jaw; syy += v * v; n++;
    }
    const num = n * sxy - sx * sy;
    const den = Math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
    const c = den > 0 ? num / den : 0;
    if (!best || c > best.c) best = { c, shift };
  }
  return best;
}

function report() {
  const abs = drift.map(Math.abs);
  const lag = measureLag();
  const rec = {
    source: $('src').value, fft: +$('fft').value,
    analyser_lag_ms: lag ? -lag.shift : null,
    analyser_corr: lag ? +lag.c.toFixed(3) : null,
    compensate: $('comp').checked, speed: +$('speed').value,
    frames: drift.length,
    drift_p50_ms: +pct(abs, 0.5).toFixed(1),
    drift_p95_ms: +pct(abs, 0.95).toFixed(1),
    drift_max_ms: +Math.max(...abs, 0).toFixed(1),
    raf_p50_ms: +pct(rafGaps, 0.5).toFixed(1),
    raf_p95_ms: +pct(rafGaps, 0.95).toFixed(1),
    underruns, output_latency_ms: +((ctx.outputLatency || 0) * 1000).toFixed(1),
    sample_rate: ctx.sampleRate, ua: navigator.userAgent,
  };
  fetch('/report', { method: 'POST', body: JSON.stringify(rec) });
  if (rec.analyser_lag_ms !== null) {
    log(`источник ${rec.source}: отставание рта от аудио ${rec.analyser_lag_ms} мс ` +
        `(корреляция ${rec.analyser_corr})`);
  }
  log(`ИТОГ compensate=${rec.compensate}: дрейф p50 ${rec.drift_p50_ms} мс, ` +
      `p95 ${rec.drift_p95_ms} мс, макс ${rec.drift_max_ms} мс | ` +
      `rAF p50 ${rec.raf_p50_ms} мс | подвисаний ${rec.underruns} | ` +
      `outputLatency ${rec.output_latency_ms} мс`);
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${rec.source}</td><td>${rec.analyser_lag_ms ?? '—'}</td>
    <td>${rec.compensate ? 'да' : 'нет'}</td><td>${rec.speed}×</td>
    <td>${rec.frames}</td><td><b>${rec.drift_p50_ms}</b></td><td>${rec.drift_p95_ms}</td>
    <td>${rec.raf_p50_ms}</td><td>${rec.output_latency_ms}</td><td>${rec.underruns}</td>`;
  $('rows').appendChild(tr);
}

$('go').onclick = () => { t0 = null; start().catch((e) => log('ERR ' + e)); };
requestAnimationFrame(render);
