// S3: cancel by generation_id and find where the stop latency actually accrues.
const $ = (id) => document.getElementById(id);
const log = (m) => { $('log').textContent += m + '\n'; $('log').scrollTop = 1e9; };

let ctx, node, gen = 0, reader = null;
let stats = { pushed: 0, dropped: 0, lastAudible: 0, depth: 0 };
let cancelAt = null, pending = null;

async function init() {
  ctx = new AudioContext();
  await ctx.audioWorklet.addModule('player-worklet.js');
  node = new AudioWorkletNode(ctx, 'player');
  node.connect(ctx.destination);
  node.port.onmessage = (e) => {
    const m = e.data;
    if (m.ev === 'tick') { stats.lastAudible = m.last; stats.depth = m.depth; }
    if (m.ev === 'stopped' && pending) finish(m.t, m.mode);
  };
  log(`AudioContext ${ctx.sampleRate} Hz | baseLatency=${(ctx.baseLatency*1e3).toFixed(1)} мс ` +
      `outputLatency=${((ctx.outputLatency||0)*1e3).toFixed(1)} мс`);
}

async function play() {
  gen++;
  const myGen = gen;
  stats = { pushed: 0, dropped: 0, lastAudible: 0, depth: 0 };
  log(`\n=== старт gen=${myGen}, скорость сервера ${$('speed').value}× реального времени ===`);
  const resp = await fetch(`/stream?gen=${myGen}&speed=${$('speed').value}`);
  const sr = +resp.headers.get('X-Sample-Rate');
  reader = resp.body.getReader();
  let acc = new Uint8Array(0);
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const n = new Uint8Array(acc.length + value.length);
    n.set(acc); n.set(value, acc.length); acc = n;
    // Frames: [u32 jsonLen][u32 pcmLen][json][pcm]
    while (acc.length >= 8) {
      const dv = new DataView(acc.buffer, acc.byteOffset, acc.byteLength);
      const jl = dv.getUint32(0, true), pl = dv.getUint32(4, true);
      if (acc.length < 8 + jl + pl) break;
      const hdr = JSON.parse(new TextDecoder().decode(acc.subarray(8, 8 + jl)));
      const pcmBytes = acc.subarray(8 + jl, 8 + jl + pl);
      acc = acc.subarray(8 + jl + pl);
      // The client's own guard: anything from an older generation dies here,
      // no matter what the server did.
      if (hdr.gen !== gen) { stats.dropped++; continue; }
      const i16 = new Int16Array(pcmBytes.slice().buffer);
      const f32 = new Float32Array(i16.length);
      for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
      node.port.postMessage({ cmd: 'push', pcm: f32 }, [f32.buffer]);
      stats.pushed += i16.length;
    }
    $('depth').textContent =
      `в буфере ${(stats.depth / ctx.sampleRate * 1000).toFixed(0)} мс | ` +
      `получено ${(stats.pushed / sr * 1000).toFixed(0)} мс | отброшено кадров ${stats.dropped}`;
  }
  log(`gen=${myGen}: поток закончился`);
}

function finish(tStopped, mode) {
  const p = pending; pending = null;
  const outLat = (ctx.outputLatency || 0) * 1000;
  const worklet = (tStopped - p.tCancel) * 1000;
  const rec = {
    mode, speed: +$('speed').value,
    buffered_ms_at_cancel: Math.round(p.depth / ctx.sampleRate * 1000),
    stop_latency_ms: Math.round(worklet),
    stop_latency_incl_output_ms: Math.round(worklet + outLat),
    output_latency_ms: Math.round(outLat),
    last_sample_abs: p.lastAbs,
    sample_rate: ctx.sampleRate, ua: navigator.userAgent,
  };
  fetch('/report', { method: 'POST', body: JSON.stringify(rec) });
  log(`режим ${mode}: остановка через ${rec.stop_latency_ms} мс ` +
      `(+${rec.output_latency_ms} мс на выход устройства = ${rec.stop_latency_incl_output_ms} мс), ` +
      `в буфере на момент отмены было ${rec.buffered_ms_at_cancel} мс`);
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${mode}</td><td>${rec.speed}×</td><td>${rec.buffered_ms_at_cancel}</td>
    <td><b>${rec.stop_latency_ms}</b></td><td>${rec.stop_latency_incl_output_ms}</td>`;
  $('rows').appendChild(tr);
}

function barge(mode) {
  if (!ctx) return;
  const tCancel = ctx.currentTime;
  const depth = stats.depth;
  gen++;                                   // everything older is now invalid
  fetch('/cancel', { method: 'POST', body: JSON.stringify({ gen: gen - 1 }) });
  pending = { tCancel, depth, lastAbs: null };
  if (mode === 'feed-stop') {
    // The naive version: stop feeding and let the buffer drain. The worklet
    // reports the exact moment it runs dry.
    node.port.postMessage({ cmd: 'drain' });
  } else if (mode === 'flush') {
    node.port.postMessage({ cmd: 'flush' });
  } else {
    node.port.postMessage({ cmd: 'fade', ms: +$('fade').value });
  }
}

$('init').onclick = async () => {
  await init(); $('init').disabled = true;
  ['play', 'b1', 'b2', 'b3'].forEach((i) => ($(i).disabled = false));
};
$('play').onclick = () => play().catch((e) => log('ERR ' + e));
$('b1').onclick = () => barge('feed-stop');
$('b2').onclick = () => barge('flush');
$('b3').onclick = () => barge('fade');
