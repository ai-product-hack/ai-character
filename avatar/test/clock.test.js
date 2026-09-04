import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

import { scheduleAudioClause, shiftTimeline } from '../src/clock.js';

describe('опоздавшие аудиоклаузы', () => {
  test('клауза в будущем сохраняет серверный PTS', () => {
    const s = scheduleAudioClause(10, 10.1, 200, 40);
    assert.ok(Math.abs(s.atSec - 10.2) < 1e-9);
    assert.ok(s.slipMs < 1e-6);
  });

  test('опоздавшая клауза получает динамический сдвиг', () => {
    const s = scheduleAudioClause(10, 10.5, 200, 40);
    assert.ok(Math.abs(s.atSec - 10.54) < 1e-9);
    assert.ok(Math.abs(s.slipMs - 340) < 1e-6);
    assert.deepEqual(shiftTimeline([{pts_ms: 200, viseme: 'AA'}], s.slipMs),
      [{pts_ms: 540, viseme: 'AA'}]);
  });

  test('нулевой сдвиг не копирует трек', () => {
    const track = [{pts_ms: 0, viseme: 'SIL'}];
    assert.equal(shiftTimeline(track, 0), track);
  });
});
