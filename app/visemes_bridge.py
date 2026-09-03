"""Мост к слою g2p из модуля аватара.

Раскладка символов в висемы живёт в `avatar/src/g2p.js` и покрыта тестами там.
Дублировать её на Python значило бы завести вторую правду: правила русской
орфографии тонкие, и две реализации разойдутся на первом же исключении.

Поэтому вызываем ту же самую, через node. Стоимость — запуск процесса, поэтому
node держится поднятым и общается построчно.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
AVATAR = ROOT / "avatar"

_BRIDGE_JS = r"""
import { readFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { timedToTrack } from '%(g2p)s';

const cfg = JSON.parse(readFileSync('%(cfg)s', 'utf8'));
const rl = createInterface({ input: process.stdin });
rl.on('line', (line) => {
  if (!line.trim()) return;
  try {
    const chars = JSON.parse(line);
    process.stdout.write(JSON.stringify(timedToTrack(chars, cfg.g2p)) + '\n');
  } catch (e) {
    process.stdout.write(JSON.stringify({ error: String(e) }) + '\n');
  }
});
"""


class VisemeBridge:
    """Долгоживущий node-процесс с тем же g2p, что и в браузере."""

    def __init__(self):
        script = ROOT / "app" / "_g2p_bridge.mjs"
        script.write_text(_BRIDGE_JS % {
            "g2p": (AVATAR / "src" / "g2p.js").as_posix(),
            "cfg": (AVATAR / "visemes.json").as_posix(),
        }, encoding="utf-8")
        self.proc = subprocess.Popen(
            ["node", str(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, encoding="utf-8")
        self.lock = threading.Lock()

    def __call__(self, chars: list[dict]) -> list[dict]:
        if not chars:
            return []
        with self.lock:
            self.proc.stdin.write(json.dumps(chars, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        data = json.loads(line)
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"g2p: {data['error']}")
        return data

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=3)
        except Exception:                                  # noqa: BLE001
            self.proc.kill()
