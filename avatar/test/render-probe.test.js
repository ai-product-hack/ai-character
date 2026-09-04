import test from 'node:test';
import assert from 'node:assert/strict';
import { RenderProbe } from '../src/render-probe.js';

test('render probe counts clean VSYNC intervals instead of reporting raw RAF p99', () => {
  const p = new RenderProbe();
  for (let i = 0; i < 600; i++) {
    p.sample(i * 1000 / 60, { cpuMs: i === 400 ? 0.9 : 0.2, drawCalls: 52 });
  }
  const r = p.result();
  assert.equal(r.spanP99, 1);
  assert.equal(r.missedVsyncs, 0);
  assert.ok(Math.abs(r.fps - 60) < 0.01);
  assert.equal(r.drawCallsP99, 52);
});

test('render probe records missed presentation slots and invalid focus', () => {
  const p = new RenderProbe();
  let now = 0;
  for (let i = 0; i < 200; i++) {
    now += (i % 40 === 0) ? 1000 / 30 : 1000 / 60;
    p.sample(now, { cpuMs: 0.3, focused: i < 100, visible: i < 150, postEnabled: true });
  }
  const r = p.result();
  assert.ok(r.missedVsyncs >= 4);
  assert.equal(r.spanP99, 2);
  assert.equal(r.focusedPercent, 50);
  assert.equal(r.visiblePercent, 75);
});
