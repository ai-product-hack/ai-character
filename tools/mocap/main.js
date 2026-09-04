import * as THREE from 'three';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { Look } from '../../avatar/src/look.js';
import { loadAvatarModel } from '../../avatar/src/model.js';
import { LAYERS, RULES } from '../../avatar/src/zones.js';
import { makeClip, sampleClip, trimClip, validateClip } from './clip.js';

const $ = (id) => document.getElementById(id);
const status = (text) => { $('status').textContent = text; };
const fail = (error) => {
  $('error').style.display = 'block';
  $('error').textContent = error?.stack || String(error);
  status('ошибка');
};

let faceLandmarker, stream, look, model;
let latest = null;
let rawFrames = [];
let clip = null;
let recording = false;
let recordStart = 0;
let previewStart = null;
let lastVideoTime = -1;
let neutralHead = null;
const slots = new Map();

async function boot() {
  const [lookCfg] = await Promise.all([
    fetch('/avatar/look.config.json').then((r) => r.json()),
  ]);
  look = new Look($('avatar'), lookCfg);
  model = await loadAvatarModel(lookCfg);
  look.scene.add(model.root);
  resize();
  look.frameOn(model.frameTarget());

  const vision = await FilesetResolver.forVisionTasks(
    '/avatar/node_modules/@mediapipe/tasks-vision/wasm');
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: '/tools/mocap/models/face_landmarker.task', delegate: 'CPU' },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  status('готово — включите камеру');
  requestAnimationFrame(frame);
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false,
  });
  $('camera').srcObject = stream;
  await $('camera').play();
  $('cameraBtn').textContent = 'камера включена';
  $('cameraBtn').disabled = true;
  $('recordBtn').disabled = false;
  status('лицо ищется…');
}

function headEuler(result) {
  const data = result.facialTransformationMatrixes?.[0]?.data;
  if (!data || data.length !== 16) return null;
  const rotation = new THREE.Matrix4().fromArray(data).extractRotation(new THREE.Matrix4().fromArray(data));
  const e = new THREE.Euler().setFromRotationMatrix(rotation, 'YXZ');
  const current = [e.x * 180 / Math.PI, e.y * 180 / Math.PI, e.z * 180 / Math.PI];
  if (!neutralHead) neutralHead = current;
  return current.map((v, i) => v - neutralHead[i]);
}

function readResult(result, now) {
  const categories = result.faceBlendshapes?.[0]?.categories;
  if (!categories?.length) { latest = null; status('лицо не найдено'); return; }
  const weights = Object.fromEntries(categories.map((c) => [c.categoryName, c.score]));
  latest = { weights, head: headEuler(result) };
  status(recording ? `запись ${(now - recordStart).toFixed(0)} мс` : 'лицо найдено');
  drawCoefficients(categories);
  if (recording) {
    rawFrames.push({ tMs: now - recordStart, weights, head: latest.head });
  }
}

function drawCoefficients(categories) {
  const box = $('coeffs'); box.textContent = '';
  for (const c of [...categories].sort((a, b) => b.score - a.score)) {
    const n = document.createElement('span'), v = document.createElement('span');
    n.textContent = c.categoryName; v.textContent = c.score.toFixed(3);
    box.append(n, v);
  }
}

function previewWeights(weights, head) {
  const morphs = model.morphs;
  morphs.begin();
  for (const [name, value] of Object.entries(weights)) {
    const rule = RULES.get(name);
    if (!rule?.layers.has(LAYERS.EMOTION)) continue;
    let slot = slots.get(name);
    if (slot === undefined) { slot = morphs.slotOf(name); slots.set(name, slot); }
    if (slot >= 0) morphs.writeSlot(LAYERS.EMOTION, slot, value);
  }
  morphs.commit();
  if (head && model.bones.Head) {
    const rest = model.bones.Head.userData.mocapRest || model.bones.Head.quaternion.clone();
    model.bones.Head.userData.mocapRest = rest;
    const e = new THREE.Euler(head[0] * Math.PI / 180, head[1] * Math.PI / 180, head[2] * Math.PI / 180);
    model.bones.Head.quaternion.copy(rest).multiply(new THREE.Quaternion().setFromEuler(e));
  }
}

function frame(now) {
  requestAnimationFrame(frame);
  const video = $('camera');
  if (faceLandmarker && stream && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    readResult(faceLandmarker.detectForVideo(video, now), now);
  }
  if (previewStart !== null && clip) {
    const trimmed = currentTrim();
    const sample = sampleClip(trimmed, now - previewStart, true);
    previewWeights(Object.fromEntries(trimmed.channels.map((name, i) => [name, sample.weights[i]])), sample.head);
  } else if (latest) {
    previewWeights(latest.weights, latest.head);
  }
  look.render(now / 1000);
}

function toggleRecord() {
  if (!recording) {
    recording = true; rawFrames = []; recordStart = performance.now(); neutralHead = null;
    $('recordBtn').textContent = 'остановить'; $('recordBtn').classList.add('recording');
    previewStart = null;
    return;
  }
  recording = false;
  $('recordBtn').textContent = 'записать'; $('recordBtn').classList.remove('recording');
  const channels = [...new Set(rawFrames.flatMap((frame) => Object.keys(frame.weights)))].sort();
  const frames = rawFrames.map((frame) => ({
    tMs: frame.tMs,
    weights: channels.map((name) => frame.weights[name] || 0),
    head: frame.head,
  }));
  clip = makeClip($('clipName').value, frames, channels, {
    capturedAt: new Date().toISOString(), rawAmplitude: 1,
  });
  setupTimeline();
  previewStart = performance.now();
}

function setupTimeline() {
  $('timeline').hidden = false;
  for (const id of ['trimStart', 'trimEnd']) { $(id).max = String(clip.durationMs); }
  $('trimStart').value = '0'; $('trimEnd').value = String(clip.durationMs);
  updateTrimLabels();
}

function updateTrimLabels() {
  $('trimStartV').textContent = `${(+$('trimStart').value / 1000).toFixed(2)} c`;
  $('trimEndV').textContent = `${(+$('trimEnd').value / 1000).toFixed(2)} c`;
}

function currentTrim() {
  return trimClip(clip, +$('trimStart').value, +$('trimEnd').value);
}

async function saveClip() {
  const trimmed = currentTrim();
  trimmed.name = $('clipName').value;
  const response = await fetch('./api/save-clip', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: trimmed.name, clip: trimmed }),
  });
  if (!response.ok) throw new Error(`Сохранение не удалось: ${response.status}`);
  const result = await response.json();
  status(`сохранено: ${result.path}`);
}

$('cameraBtn').onclick = () => startCamera().catch(fail);
$('recordBtn').onclick = () => { try { toggleRecord(); } catch (e) { fail(e); } };
$('previewBtn').onclick = () => { previewStart = performance.now(); };
$('saveBtn').onclick = () => saveClip().catch(fail);
$('trimStart').oninput = () => {
  if (+$('trimStart').value >= +$('trimEnd').value) $('trimStart').value = String(+$('trimEnd').value - 10);
  updateTrimLabels(); previewStart = performance.now();
};
$('trimEnd').oninput = () => {
  if (+$('trimEnd').value <= +$('trimStart').value) $('trimEnd').value = String(+$('trimStart').value + 10);
  updateTrimLabels(); previewStart = performance.now();
};
$('loadClip').onchange = async () => {
  try {
    const value = JSON.parse(await $('loadClip').files[0].text());
    const error = validateClip(value); if (error) throw new Error(error);
    clip = value; $('clipName').value = value.name; setupTimeline(); previewStart = performance.now();
  } catch (e) { fail(e); }
};
function resize() { if (look) look.setSize($('avatar').clientWidth, $('avatar').clientHeight); }
window.addEventListener('resize', resize);
window.addEventListener('beforeunload', () => stream?.getTracks().forEach((track) => track.stop()));
boot().catch(fail);
