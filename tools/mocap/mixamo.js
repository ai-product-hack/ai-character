import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { animationToBodyClip } from './mixamo-clip.js';

const $ = (id) => document.getElementById(id);
let clip = null;

$('fbx').onchange = async () => {
  clip = null; $('save').disabled = true; $('error').textContent = '';
  const file = $('fbx').files[0];
  if (!file) return;
  try {
    const object = new FBXLoader().parse(await file.arrayBuffer(), '');
    if (!object.animations?.length) throw new Error('FBX не содержит анимацию');
    clip = animationToBodyClip(object.animations[0], $('name').value, file.name);
    $('summary').textContent = `${clip.durationMs} мс · ${Object.keys(clip.tracks).join(', ')}`;
    $('save').disabled = false;
  } catch (error) {
    $('error').textContent = error.stack || String(error);
  }
};

$('name').onchange = () => { if (clip) clip.name = $('name').value; };

$('save').onclick = async () => {
  try {
    clip.name = $('name').value;
    const response = await fetch('/tools/mocap/api/save-body-clip', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: $('name').value, clip }),
    });
    if (!response.ok) throw new Error(await response.text());
    const result = await response.json();
    $('summary').textContent = `сохранено: ${result.path}`;
  } catch (error) {
    $('error').textContent = error.stack || String(error);
  }
};

