// Deterministic rehearsal clips used until the team records a human performer.
// They deliberately identify themselves as bootstrap material in source.tool.
//
//   node tools/mocap/seed-clips.mjs                 только недостающие клипы
//   node tools/mocap/seed-clips.mjs angry anxious   именованные, поверх всего
//   node tools/mocap/seed-clips.mjs --all           весь набор заново
//
// По умолчанию скрипт больше НЕ перезаписывает существующие клипы. Пять из них
// сняты с живого исполнителя через MediaPipe, и один запуск этого скрипта
// стирал захват обратно в синтетику — молча и без следа в выводе.
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '../..');
const cfg = JSON.parse(readFileSync(resolve(root, 'avatar/expression.config.json')));
const destination = resolve(root, 'avatar/clips');
mkdirSync(destination, { recursive: true });

const extra = {
  neutral: { browInnerUp: 0.05, cheekSquintLeft: 0.025, cheekSquintRight: 0.02 },
  skeptical: { mouthDimpleRight: 0.08, browOuterUpLeft: 0.04 },
  pressing: { cheekSquintLeft: 0.08, cheekSquintRight: 0.09 },
  warming: { noseSneerLeft: 0.025, browDownRight: 0.03 },
  impressed: { cheekSquintLeft: 0.055, cheekSquintRight: 0.045 },
  angry: { cheekSquintLeft: 0.1, cheekSquintRight: 0.085, browOuterUpRight: 0.02 },
  anxious: { cheekSquintLeft: 0.03, mouthDimpleRight: 0.06, noseSneerLeft: 0.02 },
};
const frames = 61;
const durationMs = 4000;

const args = process.argv.slice(2);
const all = args.includes('--all');
const named = args.filter((a) => !a.startsWith('--'));
const written = [];
const kept = [];

for (const [name, emotion] of Object.entries(cfg.emotions)) {
  if (name.startsWith('_')) continue;
  const target = resolve(destination, `${name}.json`);
  const wanted = all || (named.length ? named.includes(name) : !existsSync(target));
  if (!wanted) { kept.push(name); continue; }
  const pose = { ...(emotion.pose || {}), ...(extra[name] || {}) };
  const channels = Object.keys(pose).filter((key) => !key.startsWith('_')).sort();
  const rows = Array.from({ length: frames }, (_, i) => {
    const phase = i / (frames - 1);
    const t = phase * Math.PI * 2;
    const weights = channels.map((channel, ci) => {
      const base = Math.min(1, pose[channel] / cfg.clips.amplitude);
      // Two slow, incommensurate waves plus per-channel phase create subtle
      // asymmetry while keeping the first and final frame identical.
      const motion = 1 + 0.055 * Math.sin(t + ci * 0.71) + 0.025 * Math.sin(t * 2 + ci * 1.13);
      return Math.max(0, Math.min(1, base * motion));
    });
    return { tMs: Math.round(phase * durationMs), weights };
  });
  const clip = {
    version: 1, kind: 'face-mocap', name, channels, durationMs, frames: rows,
    source: {
      tool: 'bootstrap motion derived from static pose',
      capturedAt: null,
      replaceWithHumanCapture: true,
    },
  };
  writeFileSync(target, JSON.stringify(clip, null, 2) + '\n');
  written.push(name);
}

console.log(`записано: ${written.join(', ') || '—'}`);
if (kept.length) {
  console.log(`не тронуто (клип уже есть): ${kept.join(', ')}`);
  console.log('перезаписать: --all или перечислить имена');
}
