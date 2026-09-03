// Dev-страница модуля аватара.
//
// Работает БЕЗ бэкенда: трек висем синтезируется на месте из текста тем же
// слоем g2p, что будет разбирать таймкоды GigaAM, а часы — настоящий
// AudioContext, потому что другого источника времени в модуле нет.

import { Avatar, EMOTIONS, STATES, createAvatar } from '../src/avatar.js';
import { AudioClock } from '../src/clock.js';
import { CHANNELS } from '../src/behavior.js';
import { SOURCE } from '../src/viseme.js';
import { timedToTrack } from '../src/g2p.js';
import { LAYERS, ZONE_MORPHS } from '../src/zones.js';

const $ = (id) => document.getElementById(id);
const fail = (e) => {
  $('err').style.display = 'block';
  $('err').textContent += (e && e.stack ? e.stack : e) + '\n';
};
window.addEventListener('error', (e) => fail(e.error || e.message));
window.addEventListener('unhandledrejection', (e) => fail(e.reason));

const URLS = {
  look: '/avatar/look.config.json',
  behavior: '/avatar/behavior.config.json',
  visemes: '/avatar/visemes.json',
};

let avatar, clock, audioCtx;
let genCounter = 0;
let samples = [];              // записи из dev/samples/index.json
let decoded = new Map();       // id -> AudioBuffer, декодируем один раз
let voice = null;              // { source, gain } текущей реплики
let analyser = null;           // постоянный узел: уровень 3 и проверка звука
let trackSource = 'record';    // record | synth
let heldViseme = null;
let matrix = null;            // правится слайдерами, экспортируется целиком
let pristineMatrix = null;

async function boot() {
  avatar = await createAvatar($('c'), URLS);
  avatar.setSize(window.innerWidth, window.innerHeight);
  await avatar.load();
  avatar.setSize(window.innerWidth, window.innerHeight);

  matrix = avatar.configs.visemes.matrix;
  pristineMatrix = structuredClone(matrix);

  await loadSamples();
  buildStates();
  buildEmotions();
  buildChannels();
  buildSources();
  buildVisemePanel();
  buildTrackSource();
  wireRender();

  window.__dev = { avatar, get clock() { return clock; }, URLS };
  requestAnimationFrame(loop);
}

/**
 * Часы. AudioContext создаётся по первому жесту пользователя — иначе браузер
 * держит его в suspended, и currentTime стоит.
 */
async function ensureClock() {
  if (clock) return clock;
  audioCtx = new AudioContext();
  await audioCtx.resume();
  clock = new AudioClock(audioCtx);
  avatar.attachClock(clock);

  // Анализатор стоит в тракте голоса постоянно. Он нужен уровню 3 лестницы
  // отступления (огибающая -> один морф раскрытия рта); без него переключатель
  // «analyser» на панели давал бы нулевую огибающую и выглядел бы сломанным.
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  analyser.connect(audioCtx.destination);
  avatar.visemes.attachAnalyser(analyser);
  return clock;
}

/** Пиковая амплитуда в тракте голоса — проверка, что звук действительно идёт. */
function voicePeak() {
  if (!analyser) return 0;
  const buf = new Float32Array(analyser.fftSize);
  analyser.getFloatTimeDomainData(buf);
  let peak = 0;
  for (const v of buf) { const a = Math.abs(v); if (a > peak) peak = a; }
  return peak;
}

function loop(now) {
  requestAnimationFrame(loop);
  avatar.frame(now);
  drawHud();
}

// ------------------------------------------------------------------ оверлей

function drawHud() {
  const d = avatar.debug();
  const v = d.viseme || {};
  const b = d.behavior || {};
  const drift = v.drift_ms || 0;
  const driftClass = Math.abs(drift) > 50 ? ' class="warn"' : '';
  $('hud').innerHTML =
    `FPS            <b>${d.fps.toFixed(0)}</b>  (кадр ${d.frameMs.toFixed(1)} мс, ` +
      `CPU ${d.cpuMs.toFixed(2)} мс)\n` +
    `состояние      <b>${d.state}</b>\n` +
    `эмоция         ${d.emotion.name} ${d.emotion.intensity.toFixed(2)}\n` +
    `generation_id  ${v.genId === null || v.genId === undefined ? '—' : v.genId}\n` +
    `outputLatency  ${d.outputLatencyMs === null ? '— (нет часов)' : d.outputLatencyMs.toFixed(1) + ' мс'}\n` +
    `время аудио    ${d.audioMs === null ? '—' : d.audioMs.toFixed(0) + ' мс'}\n` +
    `дрейф         <span${driftClass}>${drift.toFixed(1)} мс</span>  (порог 50)` +
      ((v.preroll_ms || 0) > 0 ? `   предпрокрутка ${v.preroll_ms.toFixed(0)} мс` : '') + `\n` +
    `висема         ${v.viseme || '—'}  ${v.trackIndex ?? 0}/${v.trackLength ?? 0}` +
      `  подвисаний ${v.underruns ?? 0}\n` +
    `источник рта   ${v.source || '—'}\n` +
    `звук           ${analyser ? (voicePeak() > 0.001 ? 'идёт, пик ' + voicePeak().toFixed(3) : 'тишина') : '—'}\n` +
    `пост           ${d.postEnabled ? 'вкл' : 'ВЫКЛ'}\n` +
    `\n` +
    `взгляд         ${b.gazePhase || '—'}  ` +
      `${(b.gazeYaw ?? 0).toFixed(1)}° / ${(b.gazePitch ?? 0).toFixed(1)}°\n` +
    `моргание       ${b.blinkPhase || '—'}  ${(b.blinkValue ?? 0).toFixed(2)}\n` +
    `дыхание        ${((b.breathPhase ?? 0) * 100).toFixed(0)}%`;
}

// ------------------------------------------------------- состояния и эмоции

function toggleGroup(container, items, initial, onPick) {
  const buttons = new Map();
  for (const name of items) {
    const btn = document.createElement('button');
    btn.textContent = name;
    btn.onclick = () => {
      for (const [n, b] of buttons) b.classList.toggle('on', n === name);
      onPick(name);
    };
    buttons.set(name, btn);
    container.appendChild(btn);
  }
  buttons.get(initial)?.classList.add('on');
  return buttons;
}

function buildStates() {
  toggleGroup($('states'), STATES, avatar.state, (s) => avatar.setState(s));
}

function buildEmotions() {
  toggleGroup($('emotions'), EMOTIONS, avatar.emotion.name,
    (e) => avatar.setEmotion(e, +$('emoInt').value));
  $('emoInt').oninput = () => {
    $('emoIntV').textContent = (+$('emoInt').value).toFixed(2);
    avatar.setEmotion(avatar.emotion.name, +$('emoInt').value);
  };
}

function buildChannels() {
  for (const ch of CHANNELS) {
    const btn = document.createElement('button');
    btn.textContent = ch;
    btn.classList.add('on');
    btn.onclick = () => {
      const on = !btn.classList.contains('on');
      btn.classList.toggle('on', on);
      btn.classList.toggle('off', !on);
      avatar.behavior.setEnabled(ch, on);
    };
    $('channels').appendChild(btn);
  }
  const ev = document.createElement('button');
  ev.textContent = 'событие (моргнуть)';
  ev.onclick = () => avatar.behavior.notifyEvent();
  $('channels').appendChild(ev);
}

function buildSources() {
  toggleGroup($('sources'), [SOURCE.VISEMES, SOURCE.ANALYSER], SOURCE.VISEMES, (s) => {
    if (s === SOURCE.ANALYSER && !analyser) {
      fail(new Error('уровень 3 требует AnalyserNode: нажмите «проиграть», ' +
                     'чтобы поднять аудиограф'));
    }
    avatar.visemes.setSource(s);
  });
}

// ------------------------------------------------------- звук и настоящие тайминги

/**
 * Список записей. Звук синтезирован Silero, таймкоды получены выравниванием
 * ЭТОГО ЖЕ звука распознавателем GigaAM (bench/r4-lipsync/chartimes.py),
 * поэтому губы и звук проверяются друг против друга, а не против синтетики.
 */
async function loadSamples() {
  try {
    samples = await (await fetch('./samples/index.json')).json();
  } catch (e) {
    samples = [];
    fail(new Error('нет записей: запустите bench/r4-lipsync/chartimes.py — ' +
                   'страница будет работать только на синтетическом треке'));
    return;
  }
  const sel = $('sample');
  for (const s of samples) {
    const o = document.createElement('option');
    o.value = String(s.id);
    o.textContent = `${s.text.slice(0, 34)}…  (${s.audio_s} с)`;
    sel.appendChild(o);
  }
}

function buildTrackSource() {
  const has = samples.length > 0;
  toggleGroup($('trackSrc'), ['record', 'synth'], has ? 'record' : 'synth', (v) => {
    trackSource = v;
    $('recordBox').style.display = v === 'record' ? '' : 'none';
    $('synthBox').style.display = v === 'synth' ? '' : 'none';
  });
  if (!has) {
    trackSource = 'synth';
    $('recordBox').style.display = 'none';
    $('synthBox').style.display = '';
  }
  $('compLat').onchange = () => {
    if (clock) clock.compensate = $('compLat').checked;
  };
}

/** Декодировать WAV один раз и запомнить. */
async function bufferFor(sample) {
  if (decoded.has(sample.id)) return decoded.get(sample.id);
  // Путь абсолютный от корня репозитория; в именах файлов есть пробелы и скобки.
  const bytes = await (await fetch(encodeURI(sample.audio))).arrayBuffer();
  const buf = await audioCtx.decodeAudioData(bytes);
  decoded.set(sample.id, buf);
  return buf;
}

/**
 * Остановить голос. Мгновенный обрыв даёт щелчок, поэтому fade 25 мс — та же
 * величина, что измерена в S3 для отмены аудио.
 */
function stopVoice(fadeMs = 25) {
  if (!voice) return;
  const { source, gain } = voice;
  voice = null;
  const t = audioCtx.currentTime;
  gain.gain.cancelScheduledValues(t);
  gain.gain.setValueAtTime(gain.gain.value, t);
  gain.gain.linearRampToValueAtTime(0, t + fadeMs / 1000);
  try { source.stop(t + fadeMs / 1000 + 0.005); } catch (_) { /* уже остановлен */ }
}

// -------------------------------------------------- фейковый трек и перебой

/**
 * Синтетические посимвольные таймкоды.
 *
 * Длительность символа и шаг квантования — РАЗНЫЕ величины, и путать их дорого:
 * шаг таймкодов GigaAM 40 мс, но это точность, а не темп. При 40 мс на символ
 * получается 25 символов в секунду, вдвое быстрее живой речи, ограничитель
 * быстрой речи срабатывает на каждой паре, и рот не открывается вообще.
 * Символ длится ~75 мс, а таймкод квантуется по сетке 40 мс — как в бою.
 */
function synthTimecodes(text, rate) {
  const g = avatar.configs.visemes.g2p;
  const charMs = (g.charMs ?? 75) / rate;
  const grid = g.timecodeStepMs ?? 40;
  return [...text].map((ch, i) => ({ ch, ms: Math.round((i * charMs) / grid) * grid }));
}

async function playPhrase() {
  await ensureClock();
  stopVoice(10);
  const genId = `gen-${++genCounter}`;
  const g = avatar.configs.visemes.g2p;

  if (trackSource === 'record' && samples.length) {
    const sample = samples[+$('sample').value] || samples[0];
    const buf = await bufferFor(sample);

    // Один якорь на всю генерацию: звук и лицо планируются от него, поэтому
    // опоздавший кусок оставил бы дыру, а не сдвинул таймлайн.
    const startAt = audioCtx.currentTime + 0.12;
    const gain = audioCtx.createGain();
    gain.connect(analyser);            // -> analyser -> destination
    const source = audioCtx.createBufferSource();
    source.buffer = buf;
    source.connect(gain);
    source.start(startAt);
    voice = { source, gain };
    source.onended = () => { if (voice && voice.source === source) voice = null; };

    clock.anchor(startAt);
    // Таймкоды выровнены по этому же звуку, раскладку в висемы делает наш g2p.
    avatar.playGeneration(genId, timedToTrack(sample.chars, g));
  } else {
    const track = timedToTrack(synthTimecodes($('phrase').value, +$('rate').value), g);
    clock.anchor(audioCtx.currentTime + 0.08);
    avatar.playGeneration(genId, track);
  }

  avatar.setState('speaking');
  syncStateButtons();
}

async function barge() {
  await ensureClock();
  // Перебивание: новая генерация обесценивает старую, старый id не должен
  // проявиться ни в одном кадре.
  const dying = avatar.visemes.genId;
  avatar.cancel(dying);
  stopVoice();                       // flush + fade 25 мс, как в S3
  avatar.setState('interrupted');
  syncStateButtons();
  setTimeoutOnClock(0.6, () => { avatar.setState('listening'); syncStateButtons(); });
}

/**
 * Задержка по часам аудиографа, а не setTimeout: своих таймеров в модуле нет,
 * и на dev-странице тоже не заводим.
 */
function setTimeoutOnClock(sec, fn) {
  const target = audioCtx.currentTime + sec;
  const tick = () => {
    if (audioCtx.currentTime >= target) fn();
    else requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function syncStateButtons() {
  for (const btn of $('states').children) {
    btn.classList.toggle('on', btn.textContent === avatar.state);
  }
}

/**
 * Проверка синхронности рта и звука.
 *
 * Меряется НЕ корреляция огибающей с раскрытием рта: это разные по природе
 * сигналы (на согласном звук громкий, а рот закрыт), и на висемном липсинке
 * такая корреляция даёт 0.07–0.40 просто по построению. В S2 она была 0.94
 * потому, что там оба сигнала выводились из одной огибающей.
 *
 * Меряется совпадение по времени: звук выше порога против «рот работает».
 * Важен не столько процент совпадения, сколько СДВИГ, на котором совпадение
 * максимально: если он около нуля, систематического рассинхрона нет.
 */
async function measureSync() {
  await ensureClock();
  if (!samples.length) { $('syncOut').textContent = 'нет записей'; return; }
  stopVoice(5);
  $('syncOut').textContent = 'меряю…';
  const rows = [];
  for (const s of samples) {
    const buf = await bufferFor(s);
    const ch = buf.getChannelData(0), sr = buf.sampleRate, win = Math.round(sr * 0.01);
    const env = [];
    for (let i = 0; i + win <= ch.length; i += win) {
      let q = 0;
      for (let k = 0; k < win; k++) { const v = ch[i + k]; q += v * v; }
      env.push(Math.sqrt(q / win));
    }
    const emax = Math.max(...env) || 1;
    const speech = env.map((v) => (v / emax > 0.08 ? 1 : 0));

    const track = timedToTrack(s.chars, avatar.configs.visemes.g2p);
    const startAt = audioCtx.currentTime + 0.15;
    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    const gain = audioCtx.createGain();
    gain.gain.value = 0.0001;               // измеряем, а не слушаем
    src.connect(gain); gain.connect(analyser);
    src.start(startAt);
    clock.anchor(startAt);
    avatar.playGeneration('sync-' + s.id, track);

    const mouth = [];
    await new Promise((res) => {
      const tick = () => {
        const ms = clock.nowMs();
        if (ms !== null && ms >= 0) mouth.push([ms, mouthActivity()]);
        if (ms === null || ms < s.audio_s * 1000) requestAnimationFrame(tick); else res();
      };
      requestAnimationFrame(tick);
    });
    try { src.stop(); } catch (_) { /* уже кончился */ }

    const amax = Math.max(...mouth.map((v) => v[1])) || 1;
    const active = mouth.map(([ms, v]) => [ms, v / amax > 0.12 ? 1 : 0]);
    const agreeAt = (lag) => {
      let ok = 0, n = 0;
      for (const [ms, a] of active) {
        const i = Math.round((ms + lag) / 10);
        if (i < 0 || i >= speech.length) continue;
        if (a === speech[i]) ok++;
        n++;
      }
      return n ? ok / n : 0;
    };
    let best = { lag: 0, agree: -1 };
    for (let lag = -200; lag <= 200; lag += 10) {
      const a = agreeAt(lag);
      if (a > best.agree) best = { lag, agree: a };
    }
    rows.push({ id: s.id, at0: agreeAt(0), bestLag: best.lag, best: best.agree });
  }
  const lags = rows.map((r) => r.bestLag).sort((a, b) => a - b);
  const median = lags[lags.length >> 1];
  $('syncOut').innerHTML =
    rows.map((r) => `#${r.id}: совпадение ${(r.at0 * 100).toFixed(0)}%, ` +
                    `лучший сдвиг ${r.bestLag > 0 ? '+' : ''}${r.bestLag} мс`).join('<br>') +
    `<br><b>медианный сдвиг ${median > 0 ? '+' : ''}${median} мс</b> — ` +
    (Math.abs(median) <= 40 ? 'систематического рассинхрона нет' : 'есть систематический сдвиг');
  avatar.setState('listening'); syncStateButtons();
}

/** Суммарная активность зоны рта — для проверки синхронности. */
function mouthActivity() {
  const m = avatar.model.morphs;
  let sum = 0;
  for (const morph of ['jawOpen', 'mouthClose', 'viseme_aa', 'viseme_E', 'viseme_I',
                       'viseme_O', 'viseme_U', 'viseme_SS', 'viseme_PP', 'viseme_CH',
                       'viseme_DD', 'viseme_kk', 'viseme_nn', 'viseme_FF']) {
    sum += m.get(morph) || 0;
  }
  return sum;
}

$('measure').onclick = () => measureSync().catch(fail);
$('play').onclick = () => playPhrase().catch(fail);
$('barge').onclick = () => barge().catch(fail);

// ------------------------------------------------------ панель матрицы висем

function buildVisemePanel() {
  const sel = $('visemeSel');
  for (const name of Object.keys(matrix)) {
    const o = document.createElement('option');
    o.value = o.textContent = name;
    sel.appendChild(o);
  }
  sel.value = 'AA';
  sel.onchange = () => renderSliders();

  const add = $('addMorph');
  add.innerHTML = '<option value="">—</option>';
  for (const morph of ZONE_MORPHS[LAYERS.VISEME]) {
    const o = document.createElement('option');
    o.value = o.textContent = morph;
    add.appendChild(o);
  }
  add.onchange = () => {
    if (!add.value) return;
    matrix[sel.value][add.value] = 0;
    add.value = '';
    renderSliders();
  };

  $('hold').onclick = () => { heldViseme = sel.value; applyHold(); };
  $('release').onclick = () => { heldViseme = null; };
  $('export').onclick = exportMatrix;
  $('resetMatrix').onclick = () => {
    Object.assign(matrix, structuredClone(pristineMatrix));
    for (const k of Object.keys(matrix)) {
      if (!(k in pristineMatrix)) delete matrix[k];
    }
    avatar.visemes.setConfig(avatar.configs.visemes);
    renderSliders();
  };
  renderSliders();
}

/**
 * Удержание висемы: подменяем трек на бесконечный, состоящий из одной висемы.
 * Так слайдеры видно сразу на лице, а не через проигрывание фразы.
 */
function applyHold() {
  if (!heldViseme) return;
  ensureClock().then(() => {
    clock.anchor(audioCtx.currentTime);
    avatar.playGeneration('hold', [
      { pts_ms: 0, viseme: heldViseme },
      { pts_ms: 600000, viseme: heldViseme },
    ]);
  });
}

function renderSliders() {
  const name = $('visemeSel').value;
  const row = matrix[name];
  const box = $('sliders');
  box.innerHTML = '';
  const morphs = Object.keys(row).filter((k) => !k.startsWith('_'));
  if (!morphs.length) {
    box.innerHTML = '<div class="hint">Пусто. SIL — это все морфы в нуле.</div>';
    return;
  }
  for (const morph of morphs) {
    const wrap = document.createElement('div');
    wrap.className = 'row';
    const label = document.createElement('label');
    label.textContent = morph;
    label.title = morph;
    const range = document.createElement('input');
    range.type = 'range'; range.min = '0'; range.max = '1'; range.step = '0.01';
    range.value = String(row[morph]);
    const val = document.createElement('span');
    val.className = 'v';
    val.textContent = (+row[morph]).toFixed(2);
    range.oninput = () => {
      row[morph] = +range.value;
      val.textContent = (+range.value).toFixed(2);
      // Матрица читается слоем по ссылке, поэтому правка видна сразу.
      avatar.visemes.setConfig(avatar.configs.visemes);
      if (heldViseme === name) applyHold();
    };
    wrap.append(label, range, val);
    box.appendChild(wrap);
  }
}

function exportMatrix() {
  const out = JSON.stringify(avatar.configs.visemes, null, 2);
  const blob = new Blob([out], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'visemes.json';
  a.click();
  URL.revokeObjectURL(a.href);
}

// ------------------------------------------------------------------- рендер

function wireRender() {
  const post = $('post');
  post.classList.toggle('on', avatar.look.postEnabled);
  post.onclick = () => {
    avatar.look.setPostEnabled(!avatar.look.postEnabled);
    post.classList.toggle('on', avatar.look.postEnabled);
  };
  $('wire').onclick = () => {
    for (const m of avatar.model.meshes) {
      const mats = Array.isArray(m.material) ? m.material : [m.material];
      mats.forEach((x) => { x.wireframe = !x.wireframe; });
    }
  };
  $('reload').onclick = async () => {
    const fresh = await Promise.all(Object.values(URLS).map((u) =>
      fetch(u + '?t=' + Date.now()).then((r) => r.json())));
    avatar.configs.look = fresh[0];
    avatar.configs.behavior = fresh[1];
    avatar.configs.visemes = fresh[2];
    matrix = avatar.configs.visemes.matrix;
    pristineMatrix = structuredClone(matrix);
    avatar.look.cfg = fresh[0];
    avatar.visemes.setConfig(fresh[2]);
    avatar.look.frameOn(avatar.model.frameTarget());
    renderSliders();
  };
}

window.addEventListener('resize', () => avatar && avatar.setSize(window.innerWidth, window.innerHeight));

boot().catch(fail);
