import { createAvatar, EMOTIONS } from '../src/avatar.js';
import { AudioClock } from '../src/clock.js';
import { timedToTrack } from '../src/g2p.js';

const $ = id => document.getElementById(id);
const descriptions = {
  neutral: ['Спокойствие', 'Расслабленное лицо, внимание к собеседнику.'],
  skeptical: ['Сомнение', 'Асимметричные брови, прищур, небольшой поворот головы в речи.'],
  pressing: ['Требовательность', 'Сведённые брови, собранный взгляд, короткие речевые акценты.'],
  warming: ['Теплота', 'Улыбка в уголках губ и щеках, мягкий наклон головы.'],
  impressed: ['Впечатление', 'Приподнятые брови, открытое внимание, лёгкая улыбка.'],
  angry: ['Гнев', 'Опущенные брови и уголки губ, напряжённый нос, сдержанное тело.'],
  anxious: ['Тревога', 'Внутренние края бровей вверх, напряжённые уголки рта.'],
};
const stateNames = {listening:'Слушает', thinking:'Думает', speaking:'Говорит'};
let avatar, tour = null, audit = null, ctx, clock, source, generation = 0, last = 0, hudAt = 0;
const buttons = new Map(), stateButtons = new Map();
const fail = error => { $('error').textContent = error.message || String(error); };
window.addEventListener('error', event => fail(event.error || event.message));
window.addEventListener('unhandledrejection', event => fail(event.reason));

function choose(name) {
  avatar.setEmotion(name, +$('intensity').value);
  $('name').textContent = descriptions[name][0];
  $('meaning').textContent = descriptions[name][1];
  for (const [key, button] of buttons) button.setAttribute('aria-pressed', key === name);
}
function endTour() { tour = null; $('tour').textContent = 'Сравнить 7 эмоций'; }
function stopSpeech() {
  generation++;
  if (source) { source.onended = null; source.stop(); source = null; }
  if (avatar.visemes.genId) avatar.cancel(avatar.visemes.genId);
  // An empty track releases the previous one even if the source has finished.
  avatar.playGeneration(`rehearsal-${generation}`, []);
  if (clock) clock.reset();
}
function setState(state) {
  avatar.setState(state);
  for (const [key, button] of stateButtons) button.setAttribute('aria-pressed', key === state);
}
async function speak() {
  endTour(); audit = null; stopSpeech();
  const token = generation;
  $('speak').disabled = true;
  try {
    ctx ||= new AudioContext();
    await ctx.resume();
    clock ||= new AudioClock(ctx);
    avatar.attachClock(clock);
    const samples = await (await fetch('/avatar/dev/samples/index.json')).json();
    const sample = samples[0];
    const response = await fetch(encodeURI(sample.audio));
    if (!response.ok) throw new Error('Запись недоступна. Выражения можно проверить без речи.');
    const buffer = await ctx.decodeAudioData(await response.arrayBuffer());
    if (token !== generation) return;
    source = ctx.createBufferSource(); source.buffer = buffer; source.connect(ctx.destination);
    clock.anchor(ctx.currentTime + 0.12);
    avatar.playGeneration(`rehearsal-${generation}`, timedToTrack(sample.chars, avatar.configs.visemes.g2p));
    setState('speaking'); avatar.queueSpeechAccent();
    $('meaning').textContent = sample.text;
    source.onended = () => { if (token === generation) { source = null; setState('listening'); } };
    source.start(clock.t0);
  } catch (error) { fail(error); } finally { $('speak').disabled = false; }
}
async function boot() {
  avatar = await createAvatar($('c'), {
    look:'/avatar/look.config.json', behavior:'/avatar/behavior.config.json',
    visemes:'/avatar/visemes.json', expression:'/avatar/expression.config.json',
  });
  const resize = () => { const r = $('stage').getBoundingClientRect(); avatar.setSize(r.width, r.height); };
  resize(); await avatar.load(); resize();
  new ResizeObserver(resize).observe($('stage'));
  for (const name of EMOTIONS) {
    const button = document.createElement('button'); button.textContent = descriptions[name][0];
    button.onclick = () => { endTour(); choose(name); };
    buttons.set(name, button); $('emotions').append(button);
  }
  for (const [name, label] of Object.entries(stateNames)) {
    const button = document.createElement('button'); button.textContent = label;
    button.onclick = () => { endTour(); audit = null; stopSpeech(); setState(name); };
    stateButtons.set(name, button); $('states').append(button);
  }
  $('intensity').oninput = () => { $('strength').textContent = (+$('intensity').value).toFixed(2); choose(avatar.emotion.name); };
  $('mocap').onchange = () => { avatar.emotionLayer.cfg.clips.enabled = $('mocap').checked; };
  $('body').onchange = () => avatar.bodyIdle.setMotionEnabled($('body').checked);
  $('activity').onclick = () => avatar.noteActivity();
  $('tour').onclick = () => {
    if (tour) { endTour(); return; }
    audit = null; stopSpeech(); setState('listening'); choose(EMOTIONS[0]);
    tour = {index:0, elapsed:0}; $('tour').textContent = 'Остановить сравнение';
  };
  $('wait').onclick = () => {
    endTour(); stopSpeech(); setState('thinking'); setState('listening'); choose('neutral');
    audit = {elapsed:0, contact:0, away:0, longest:0};
  };
  $('speak').onclick = speak;
  for (const id of ['tour','speak','activity','wait']) $(id).disabled = false;
  choose('neutral'); setState('listening');
  requestAnimationFrame(frame);
}
function frame(now) {
  requestAnimationFrame(frame);
  const dt = last ? Math.min((now-last)/1000, 0.1) : 1/60; last = now;
  if (tour) {
    tour.elapsed += dt;
    if (tour.elapsed >= 4) {
      tour.elapsed = 0; tour.index++;
      if (tour.index >= EMOTIONS.length) endTour(); else choose(EMOTIONS[tour.index]);
    }
  }
  avatar.frame(now);
  if (audit && audit.elapsed < 40) {
    audit.elapsed += dt;
    const g = avatar.behavior.gaze;
    if (Math.hypot(g.yaw,g.pitch) <= 1.5) { audit.contact += dt; audit.away = 0; }
    else { audit.away += dt; audit.longest = Math.max(audit.longest,audit.away); }
  }
  if (now > hudAt) {
    hudAt = now + 250;
    const d = avatar.debug();
    $('status').textContent = `${stateNames[d.state] || d.state} · ${Math.round(d.fps)} FPS\nВзгляд ${d.behavior.gazeYaw.toFixed(1)}° / ${d.behavior.gazePitch.toFixed(1)}°\nКлипы лица: ${avatar.emotionLayer.clips.size}, ошибок: ${d.emotionLayer.clipErrors}`;
    if (audit) $('status').textContent += `\nОжидание: ${Math.min(40,audit.elapsed).toFixed(0)} / 40 с\nКонтакт ±1.5°: ${Math.round(100*audit.contact/audit.elapsed)}%\nМакс. отвод: ${audit.longest.toFixed(2)} с`;
    for (const [key, button] of stateButtons) button.setAttribute('aria-pressed', key === avatar.state);
  }
}
boot().catch(fail);
