// Диктофон для живых записей. Разрешение на микрофон запрашивает браузер,
// поэтому доступ терминала к микрофону не нужен.
const $ = (id) => document.getElementById(id);
let ctx, node, stream, list = [], idx = 0, recording = false, lastRms = 0;

async function init() {
  stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  ctx = new AudioContext();
  await ctx.audioWorklet.addModule('rec-worklet.js');
  node = new AudioWorkletNode(ctx, 'rec');
  ctx.createMediaStreamSource(stream).connect(node);
  node.port.onmessage = (e) => {
    if (e.data.ev === 'level') {
      lastRms = e.data.rms;
      $('meter').style.width = Math.min(100, e.data.rms * 400) + '%';
    } else if (e.data.ev === 'clip') {
      upload(e.data.pcm);
    }
  };
  $('mic').textContent = `микрофон: ${stream.getAudioTracks()[0].label} @ ${ctx.sampleRate} Гц`;
  $('init').disabled = true;
  $('rec').disabled = false;
  render();
}

async function load() {
  list = await (await fetch('/phrases')).json();
  idx = list.findIndex((p) => !p.recorded);
  if (idx < 0) idx = 0;
  render();
}

function render() {
  const p = list[idx];
  if (!p) return;
  const doneN = list.filter((x) => x.recorded).length;
  $('progress').textContent = `${doneN} / ${list.length} записано`;
  $('bar').style.width = (100 * doneN / list.length) + '%';
  $('set').textContent = p.set;
  $('id').textContent = p.id;
  $('text').textContent = p.text;
  $('state').textContent = p.recorded ? '✅ уже записано' : '';
  $('rec').textContent = recording ? '■ Стоп (пробел)' : '● Записать (пробел)';
}

function toggle() {
  if (!ctx) return;
  if (!recording) {
    recording = true;
    node.port.postMessage({ cmd: 'start' });
    $('state').textContent = '● идёт запись…';
  } else {
    recording = false;
    node.port.postMessage({ cmd: 'stop' });
    $('state').textContent = 'сохраняю…';
  }
  render();
}

async function upload(pcm) {
  const p = list[idx];
  const r = await fetch('/clip', {
    method: 'POST',
    headers: { 'X-Meta': JSON.stringify({ set: p.set, id: p.id, sampleRate: ctx.sampleRate }) },
    body: pcm.buffer,
  });
  const info = await r.json();
  if (info.error) { $('state').textContent = 'ошибка: ' + info.error; return; }
  p.recorded = true;
  const quiet = info.rms < 0.001;
  // Для фраз с заминками главное не громкость, а наличие реальной паузы:
  // прочитанная бегло фраза для R2 бесполезна.
  const needGap = p.set === 'r2_hesitations';
  const gap = info.max_gap_ms;
  const noGap = needGap && gap !== null && gap !== undefined && gap < 400;
  if (quiet) {
    $('state').textContent = `⚠️ тишина (${info.duration_s} с, RMS ${info.rms}) — перезапишите`;
  } else if (noGap) {
    $('state').textContent =
      `⚠️ пауза всего ${gap} мс — прочитано слишком бегло. Нужна заминка ≥ 400 мс ` +
      `на многоточии: тяните «эээ», думайте вслух. Перезапишите.`;
  } else {
    $('state').textContent = `✅ ${info.duration_s} с, RMS ${info.rms}` +
      (needGap ? `, пауза ${gap} мс` : '');
  }
  if (!quiet && !noGap) next();
  render();
}

function next() { if (idx < list.length - 1) { idx++; render(); } }
function prev() { if (idx > 0) { idx--; render(); } }

$('init').onclick = () => init().catch((e) => ($('mic').textContent = 'ошибка: ' + e));
$('rec').onclick = toggle;
$('next').onclick = next;
$('prev').onclick = prev;
$('manifest').onclick = async () => {
  const r = await (await fetch('/manifest')).json();
  $('state').textContent = `манифест записан: ${r.written} клипов`;
};
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { e.preventDefault(); toggle(); }
  if (e.code === 'ArrowRight') next();
  if (e.code === 'ArrowLeft') prev();
});
load();
