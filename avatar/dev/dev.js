// Dev-страница. Шаг 2: смотрим на кадр, свет, пост и глаза.
// Фейковый источник данных для рта появится на шаге 4 — модуль должен
// разрабатываться без бэкенда.

import * as THREE from 'three';
import { Look } from '../src/look.js';
import { loadAvatarModel } from '../src/model.js';
import { CHANNELS, Microbehavior } from '../src/behavior.js';

const $ = (id) => document.getElementById(id);
const fail = (e) => { $('err').style.display = 'block'; $('err').textContent += (e.stack || e) + '\n'; };
window.addEventListener('error', (e) => fail(e.error || e.message));
window.addEventListener('unhandledrejection', (e) => fail(e.reason));

const CONFIG_URL = '/avatar/look.config.json';
const BEHAVIOR_URL = '/avatar/behavior.config.json';
let look, model, cfg, behCfg, beh;
let frames = 0, fpsAcc = 0, fps = 0, lastT = performance.now(), cpuAcc = 0;

async function boot() {
  [cfg, behCfg] = await Promise.all([
    (await fetch(CONFIG_URL)).json(),
    (await fetch(BEHAVIOR_URL)).json(),
  ]);
  look = new Look($('c'), cfg);
  resize();

  model = await loadAvatarModel(cfg, (p) => {
    if (p.lengthComputable) $('hud').textContent = `модель ${(p.loaded / p.total * 100) | 0}%`;
  });
  look.scene.add(model.root);
  look.frameOn(model.frameTarget());

  const d = model.morphs.describe();
  console.log('морфы:', d);
  if (d.unclaimed.length) console.warn('морфы вне зон:', d.unclaimed);

  beh = new Microbehavior(model, behCfg);
  beh.setAnchor(look.camera.position);      // смотрит на собеседника = в камеру
  buildChannelToggles();

  // Ссылки для отладки из консоли: свет и поза подбираются на глаз, и делать
  // это перезагрузкой страницы на каждое число — потерянный день.
  window.__dev = { THREE, look, model, cfg, get beh() { return beh; } };

  requestAnimationFrame(loop);
}

function resize() {
  if (!look) return;                      // resize может прийти раньше boot()
  look.setSize(window.innerWidth, window.innerHeight);
  if (model) look.frameOn(model.frameTarget());
}
window.addEventListener('resize', resize);

function loop(now) {
  requestAnimationFrame(loop);
  const dt = (now - lastT) / 1000; lastT = now;
  fpsAcc += dt; frames++;
  if (fpsAcc >= 0.5) { fps = frames / fpsAcc; frames = 0; fpsAcc = 0; }

  // Порядок в кадре: обнулить вклады -> слои пишут -> разложить по мешам.
  const cpu0 = performance.now();
  if (beh) {
    model.morphs.begin();
    beh.update(Math.min(dt, 0.1), now / 1000);
    model.morphs.commit();
  }
  const cpuMs = performance.now() - cpu0;
  cpuAcc = cpuAcc * 0.9 + cpuMs * 0.1;

  look.render(now / 1000);

  const info = look.renderer.info;
  const b = beh ? beh.debug() : null;
  $('hud').innerHTML =
    `FPS            <b>${fps.toFixed(0)}</b>  (кадр ${(dt * 1000).toFixed(1)} мс)\n` +
    `микроповедение ${cpuAcc.toFixed(3)} мс CPU\n` +
    `треугольников  ${info.render.triangles.toLocaleString('ru')}\n` +
    `вызовов        ${info.render.calls}\n` +
    `fov            ${look.camera.fov.toFixed(1)}° (${cfg.camera.focalLengthMm} мм)\n` +
    `pixelRatio     ${look.renderer.getPixelRatio()}\n` +
    `пост           ${look.postEnabled ? 'вкл' : 'ВЫКЛ'}\n` +
    (b ? `\nвзгляд         ${b.gazePhase}  ${b.gazeYaw.toFixed(1)}° / ${b.gazePitch.toFixed(1)}°\n` +
         `моргание       ${b.blinkPhase}  ${b.blinkValue.toFixed(2)}  (через ${b.nextBlinkSec.toFixed(1)} с)\n` +
         `дыхание        ${(b.breathPhase * 100).toFixed(0)}%  период ${b.breathPeriod.toFixed(2)} с\n` +
         `голова за взгл ${b.headFollowYaw.toFixed(2)}°` : '');
}

// Заморозка каналов по одному — единственный способ понять, какой из слоёв
// делает лицо странным.
function buildChannelToggles() {
  const box = document.createElement('div');
  box.style.cssText = 'margin-top:8px;border-top:1px solid #232a38;padding-top:6px';
  box.innerHTML = '<div style="opacity:.7;margin-bottom:4px">каналы микроповедения</div>';
  for (const ch of CHANNELS) {
    const b = document.createElement('button');
    b.textContent = ch;
    b.dataset.on = '1';
    b.onclick = () => {
      const on = b.dataset.on !== '1';
      b.dataset.on = on ? '1' : '0';
      b.style.opacity = on ? '1' : '0.4';
      b.style.textDecoration = on ? 'none' : 'line-through';
      beh.setEnabled(ch, on);
    };
    box.appendChild(b);
  }
  const ev = document.createElement('button');
  ev.textContent = 'событие (моргнуть)';
  ev.onclick = () => beh.notifyEvent();
  box.appendChild(ev);
  $('panel').appendChild(box);
}

$('wire').onclick = () => {
  model.meshes.forEach((m) => {
    const mats = Array.isArray(m.material) ? m.material : [m.material];
    mats.forEach((x) => { x.wireframe = !x.wireframe; });
  });
};
$('eyesOff').onclick = () => {
  const v = !model.eyes.catchlights[0]?.visible;
  model.eyes.setVisible(v);
};
$('post').onclick = () => look.setPostEnabled(!look.postEnabled);
$('reload').onclick = async () => {
  const fresh = await (await fetch(CONFIG_URL + '?t=' + Date.now())).json();
  look.cfg = cfg = fresh;
  look.frameOn(model.frameTarget());
  const g = cfg.post.grade;
  look.grade.uniforms.grain.value = g.grain;
  look.grade.uniforms.ca.value = g.chromaticAberration;
  look.grade.uniforms.vignette.value = g.vignette;
  look.grade.uniforms.vignetteSoftness.value = g.vignetteSoftness;
  if (look.bloom) {
    look.bloom.strength = cfg.post.bloom.strength;
    look.bloom.radius = cfg.post.bloom.radius;
    look.bloom.threshold = cfg.post.bloom.threshold;
  }
  look.renderer.toneMappingExposure = cfg.renderer.exposure;
};

boot().catch(fail);
