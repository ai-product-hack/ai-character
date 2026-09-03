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

  buildStates();
  buildEmotions();
  buildChannels();
  buildSources();
  buildVisemePanel();
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
  return clock;
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
    if (s === SOURCE.ANALYSER && !avatar.visemes.analyser) {
      fail(new Error('уровень 3 требует AnalyserNode — на dev-странице звука нет, ' +
                     'источник переключён, но огибающая будет нулевой'));
    }
    avatar.visemes.setSource(s);
  });
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
  const text = $('phrase').value;
  const rate = +$('rate').value;
  const track = timedToTrack(synthTimecodes(text, rate), avatar.configs.visemes.g2p);
  const genId = `gen-${++genCounter}`;
  clock.anchor(audioCtx.currentTime + 0.08);   // небольшой запас, как на реальном старте
  avatar.setState('speaking');
  syncStateButtons();
  avatar.playGeneration(genId, track);
}

async function barge() {
  await ensureClock();
  // Перебивание: новая генерация обесценивает старую, старый id не должен
  // проявиться ни в одном кадре.
  const dying = avatar.visemes.genId;
  avatar.cancel(dying);
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
