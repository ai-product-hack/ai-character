// Dev-страница. Шаг 2: смотрим на кадр, свет, пост и глаза.
// Фейковый источник данных для рта появится на шаге 4 — модуль должен
// разрабатываться без бэкенда.

import * as THREE from 'three';
import { Look } from '../src/look.js';
import { loadAvatarModel } from '../src/model.js';

const $ = (id) => document.getElementById(id);
const fail = (e) => { $('err').style.display = 'block'; $('err').textContent += (e.stack || e) + '\n'; };
window.addEventListener('error', (e) => fail(e.error || e.message));
window.addEventListener('unhandledrejection', (e) => fail(e.reason));

const CONFIG_URL = '/avatar/look.config.json';
let look, model, cfg;
let frames = 0, fpsAcc = 0, fps = 0, lastT = performance.now();

async function boot() {
  cfg = await (await fetch(CONFIG_URL)).json();
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

  // Ссылки для отладки из консоли: свет и поза подбираются на глаз, и делать
  // это перезагрузкой страницы на каждое число — потерянный день.
  window.__dev = { THREE, look, model, cfg };

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

  look.render(now / 1000);

  const info = look.renderer.info;
  $('hud').innerHTML =
    `FPS            <b>${fps.toFixed(0)}</b>\n` +
    `треугольников  ${info.render.triangles.toLocaleString('ru')}\n` +
    `вызовов        ${info.render.calls}\n` +
    `fov            ${look.camera.fov.toFixed(1)}° (${cfg.camera.focalLengthMm} мм)\n` +
    `дистанция      ${look.camera.position.distanceTo(look.target).toFixed(3)} м\n` +
    `фокус DOF      ${(look.focusDistance || 0).toFixed(3)} м\n` +
    `pixelRatio     ${look.renderer.getPixelRatio()}`;
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
$('post').onclick = () => {
  for (const p of look.composer.passes) {
    if (p === look.composer.passes[0]) continue;      // RenderPass не трогаем
    if (p.constructor.name === 'OutputPass') continue; // без него всё уйдёт в линейное
    p.enabled = !p.enabled;
  }
};
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
