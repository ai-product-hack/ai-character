// R6: does the browser AEC see audio we play ourselves, and does a naive
// energy VAD fire on the agent's own voice? Measured, not assumed.

const $ = (id) => document.getElementById(id);
const log = (msg) => { $('log').textContent += msg + '\n'; $('log').scrollTop = 1e9; };

const CFG = {
  floorMs: 2000,        // phase A: noise floor
  frameMs: 20,
  vadFactor: 3.0,       // threshold = floor_rms * factor  (~9.5 dB over floor)
  vadMinSpeechFrames: 5,   // 100 ms
  vadHangoverFrames: 15,   // 300 ms
};

let ctx, micStream, micNode, meter, frames = [], collecting = false;
let agentBuf = null, agentBlobUrl = null;

const dbfs = (r) => 20 * Math.log10(Math.max(r, 1e-9));
const mean = (a) => a.reduce((x, y) => x + y, 0) / (a.length || 1);

async function loadAgent() {
  const resp = await fetch('assets/agent_ru.wav');
  const ab = await resp.arrayBuffer();
  agentBlobUrl = URL.createObjectURL(new Blob([ab.slice(0)], { type: 'audio/wav' }));
  agentBuf = await ctx.decodeAudioData(ab.slice(0));
  log(`agent_ru.wav: ${agentBuf.duration.toFixed(2)} s @ ${agentBuf.sampleRate} Hz`);
}

async function openMic(echoCancellation) {
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation,
      // Isolate AEC: everything else off, so a change in the number is AEC.
      noiseSuppression: false,
      autoGainControl: false,
    },
  });
  const track = micStream.getAudioTracks()[0];
  const settings = track.getSettings();
  log(`mic: ${track.label} | settings ${JSON.stringify({
    echoCancellation: settings.echoCancellation,
    noiseSuppression: settings.noiseSuppression,
    autoGainControl: settings.autoGainControl,
  })}`);
  if (micNode) micNode.disconnect();
  micNode = ctx.createMediaStreamSource(micStream);
  micNode.connect(meter);
  return settings;
}

function collect(ms) {
  frames = [];
  collecting = true;
  return new Promise((res) => setTimeout(() => { collecting = false; res(frames.slice()); }, ms));
}

// Energy VAD with hysteresis — deliberately the naive design, because that is
// what a hackathon build reaches for first. Counts speech ONSETS.
function vad(rmsFrames, floorRms) {
  const thr = floorRms * CFG.vadFactor;
  let run = 0, hang = 0, inSpeech = false, events = 0, hot = 0;
  for (const r of rmsFrames) {
    if (r > thr) {
      run++; hang = CFG.vadHangoverFrames;
      if (!inSpeech && run >= CFG.vadMinSpeechFrames) { inSpeech = true; events++; }
    } else {
      run = 0;
      if (inSpeech && --hang <= 0) inSpeech = false;
    }
    if (inSpeech) hot++;
  }
  return { events, hotFrames: hot, thresholdDbfs: +dbfs(thr).toFixed(1) };
}

function playAgent(path, gain) {
  // Returns {stop, done}. Two output paths, same signal, same nominal level.
  if (path === 'webaudio') {
    const src = ctx.createBufferSource();
    const g = ctx.createGain();
    g.gain.value = gain;
    src.buffer = agentBuf;
    src.connect(g).connect(ctx.destination);
    src.start();
    return { stop: () => { try { src.stop(); } catch (e) {} },
             done: new Promise((r) => (src.onended = r)) };
  }
  const el = new Audio(agentBlobUrl);
  el.volume = gain;
  el.play();
  return { stop: () => { el.pause(); el.src = ''; },
           done: new Promise((r) => (el.onended = r)) };
}

async function runCell() {
  const path = $('path').value;
  const ec = $('ec').value === 'true';
  const device = $('device').value;
  const gain = parseFloat($('gain').value);
  const cell = `${device}/ec=${ec}/${path}`;
  log(`\n=== ${cell} ===`);

  const settings = await openMic(ec);
  if (settings.echoCancellation !== ec) {
    log(`!! browser refused echoCancellation=${ec}, actual=${settings.echoCancellation}`);
  }

  $('phase').textContent = 'A: тишина, не говорите';
  await new Promise((r) => setTimeout(r, 300));
  const floorFrames = await collect(CFG.floorMs);
  const floorRms = mean(floorFrames.map((f) => f.rms));
  log(`floor: ${dbfs(floorRms).toFixed(1)} dBFS over ${floorFrames.length} frames`);

  $('phase').textContent = 'B: играет агент — МОЛЧИТЕ';
  const player = playAgent(path, gain);
  const echoP = collect(Math.round(agentBuf.duration * 1000));
  await player.done;
  const echoFrames = await echoP;
  const echoRms = mean(echoFrames.map((f) => f.rms));
  const p95 = echoFrames.map((f) => f.rms).sort((a, b) => a - b)[Math.floor(echoFrames.length * 0.95)] || 0;

  const v = vad(echoFrames.map((f) => f.rms), floorRms);
  const leakDb = +(dbfs(echoRms) - dbfs(floorRms)).toFixed(1);

  const rec = {
    cell, device, echo_cancellation_requested: ec,
    echo_cancellation_actual: settings.echoCancellation,
    output_path: path, output_gain: gain,
    ua: navigator.userAgent,
    sample_rate: ctx.sampleRate,
    floor_dbfs: +dbfs(floorRms).toFixed(1),
    echo_mean_dbfs: +dbfs(echoRms).toFixed(1),
    echo_p95_dbfs: +dbfs(p95).toFixed(1),
    leak_db: leakDb,
    vad_events: v.events,
    vad_false_frames: v.hotFrames,
    vad_total_frames: echoFrames.length,
    vad_hot_ratio: +(v.hotFrames / (echoFrames.length || 1)).toFixed(3),
    vad_threshold_dbfs: v.thresholdDbfs,
    raw_floor_dbfs: floorFrames.map((f) => +dbfs(f.rms).toFixed(1)),
    raw_echo_dbfs: echoFrames.map((f) => +dbfs(f.rms).toFixed(1)),
  };
  await fetch('/result', { method: 'POST', body: JSON.stringify(rec) });

  $('phase').textContent = 'готово';
  log(`leak над шумом: ${leakDb} dB | VAD ложных срабатываний: ${v.events} ` +
      `(${(rec.vad_hot_ratio * 100).toFixed(0)}% времени речи агента)`);
  log(leakDb < 6
    ? '=> AEC ГАСИТ наш выход: голосовое перебивание на этой конфигурации возможно'
    : '=> AEC НЕ гасит: VAD будет ловить агента, нужен мьют или push-to-talk');
  renderRow(rec);
}

function renderRow(r) {
  const tr = document.createElement('tr');
  const verdict = r.leak_db < 6 ? '✅ AEC гасит' : (r.leak_db < 15 ? '⚠️ частично' : '❌ течёт');
  tr.innerHTML = `<td>${r.device}</td><td>${r.echo_cancellation_actual}</td>
    <td>${r.output_path}</td><td>${r.floor_dbfs}</td><td>${r.echo_mean_dbfs}</td>
    <td><b>${r.leak_db}</b></td><td>${r.vad_events}</td>
    <td>${(r.vad_hot_ratio * 100).toFixed(0)}%</td><td>${verdict}</td>`;
  $('rows').appendChild(tr);
}

// Phase C: real barge-in. Human speaks over the agent; we measure whether the
// VAD separates the human from the leaked agent voice at all.
async function runBargeIn() {
  const ec = $('ec').value === 'true';
  const path = $('path').value;
  log(`\n=== barge-in: ${$('device').value}/ec=${ec}/${path} ===`);
  await openMic(ec);
  $('phase').textContent = 'A: тишина';
  const floorRms = mean((await collect(CFG.floorMs)).map((f) => f.rms));

  const player = playAgent(path, parseFloat($('gain').value));
  const t0 = ctx.currentTime;
  $('phase').textContent = 'B: молчите 3 с…';
  setTimeout(() => { $('phase').textContent = '>>> ГОВОРИТЕ: «стоп, подождите» <<<'; }, 3000);
  const framesP = collect(Math.min(8000, agentBuf.duration * 1000));
  const got = await framesP;
  player.stop();

  const thr = floorRms * CFG.vadFactor;
  const pre = got.filter((f) => f.t - t0 < 3.0).map((f) => f.rms);
  const during = got.filter((f) => f.t - t0 >= 3.2).map((f) => f.rms);
  const rec = {
    cell: `bargein/${$('device').value}/ec=${ec}/${path}`,
    kind: 'barge_in', device: $('device').value,
    echo_cancellation_requested: ec, output_path: path,
    floor_dbfs: +dbfs(floorRms).toFixed(1),
    agent_only_dbfs: +dbfs(mean(pre)).toFixed(1),
    agent_plus_human_dbfs: +dbfs(mean(during)).toFixed(1),
    separation_db: +(dbfs(mean(during)) - dbfs(mean(pre))).toFixed(1),
    vad_threshold_dbfs: +dbfs(thr).toFixed(1),
    raw_dbfs: got.map((f) => +dbfs(f.rms).toFixed(1)),
  };
  await fetch('/result', { method: 'POST', body: JSON.stringify(rec) });
  log(`агент один: ${rec.agent_only_dbfs} dBFS | агент+человек: ${rec.agent_plus_human_dbfs} dBFS`);
  log(`запас на различение человека: ${rec.separation_db} dB ` +
      (rec.separation_db > 6 ? '=> человека видно поверх утечки' : '=> человека НЕ отличить от утечки'));
  $('phase').textContent = 'готово';
}

$('start').onclick = async () => {
  if (!ctx) {
    ctx = new AudioContext();
    await ctx.audioWorklet.addModule('meter-worklet.js');
    meter = new AudioWorkletNode(ctx, 'meter');
    meter.port.onmessage = (e) => { if (collecting) frames.push(e.data); };
    await loadAgent();
    log(`AudioContext: ${ctx.sampleRate} Hz, baseLatency=${(ctx.baseLatency * 1000).toFixed(1)} ms, ` +
        `outputLatency=${((ctx.outputLatency || 0) * 1000).toFixed(1)} ms`);
  }
  await ctx.resume();
  $('start').disabled = true; $('run').disabled = false; $('barge').disabled = false;
};
$('run').onclick = () => runCell().catch((e) => log('ERR ' + e));
$('barge').onclick = () => runBargeIn().catch((e) => log('ERR ' + e));
