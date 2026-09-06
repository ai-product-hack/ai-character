"""Приём звука: сколько его держит распознавание на лету.

Проверяется одна вещь — успевает ли сервер принимать куски микрофона, пока
разбирает уже сказанное. Клиент шлёт куски по 100 мс строго по одному
(очередь, один запрос в полёте) и держит потолок в 20 кусков: если сервер
отвечает медленнее, чем куски приходят, очередь растёт и звук теряется.

Живая жалоба, ради которой это написано: «спустя несколько реплик распознавание
тормозит где-то на четвёртом-пятом слове, и почти всю остальную реплику он не
заносит — отправляет недоделанную».

    bench/r1-stt/.venv/bin/python bench/r9-voice/intake.py
"""
import json
import pathlib
import sys
import threading
import time
import types

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.server import GigaAMAligner, Session          # noqa: E402
from app.voice import VoiceInput                       # noqa: E402

CHUNK_MS = 100          # столько шлёт браузер за раз
CAP = 20                # потолок очереди на клиенте
LENGTHS = (5, 12, 25)   # короткая, средняя и длинная реплики, секунд


def make_session(aligner, sync: bool):
    s = Session.__new__(Session)
    s.models = {"aligner": aligner, "voice_cfg": {"partial_every_ms": 500}}
    s.voice = VoiceInput(endpointer=types.SimpleNamespace(
        speech_ms=0, push=lambda pcm: False, reset=lambda: None))
    s.voice.enabled = True
    s.partial_text = ""
    s._partial_sent = ""
    s._partial_at = 0.0
    s._partial_cost = 0.0
    s._turn = 0
    s._partial_thread = None
    s._partial_lock = threading.Lock()
    s.marks = []
    s.spec = types.SimpleNamespace(on_typing=lambda t: None)
    if sync:
        # Как было до починки: разбор прямо в обработчике запроса.
        s._kick_partial = lambda: _sync_partial(s)
    return s


def _sync_partial(s):
    every = s.models["voice_cfg"]["partial_every_ms"]
    now = time.perf_counter() * 1000
    if now - s._partial_at < every:
        return
    s._partial_at = now
    audio = s.voice.snapshot()
    if audio is None or len(audio) < 8000:
        return
    s._partial_lock.acquire()
    s._run_partial(audio, s._turn)          # синхронно, в потоке запроса
    s._partial_at = time.perf_counter() * 1000


def run(aligner, sync: bool, seconds: int):
    """Гоняем реплику в реальном времени и смотрим на очередь клиента."""
    s = make_session(aligner, sync)
    rng = np.random.default_rng(0)
    n = int(seconds * 1000 / CHUNK_MS)
    chunk = int(16000 * CHUNK_MS / 1000)

    queue, dropped, waits, peak = [], 0, [], 0
    next_at = time.perf_counter()
    produced = 0
    while produced < n or queue:
        now = time.perf_counter()
        # Микрофон кладёт кусок в очередь каждые 100 мс, независимо от сервера.
        while produced < n and now >= next_at:
            if len(queue) > CAP:
                queue.pop(0)
                dropped += 1
            queue.append(rng.standard_normal(chunk).astype(np.float32) * 0.05)
            produced += 1
            peak = max(peak, len(queue))
            next_at += CHUNK_MS / 1000
        if not queue:
            time.sleep(0.001)
            continue
        pcm = queue.pop(0)
        t0 = time.perf_counter()
        s.push_audio(pcm)
        waits.append((time.perf_counter() - t0) * 1000)

    if s._partial_thread is not None:
        s._partial_thread.join(timeout=10)
    waits.sort()
    return {
        "кусков": n,
        "потеряно": dropped,
        "потеряно_%": round(100 * dropped / n, 1),
        "очередь_пик": peak,
        "приём_медиана_мс": round(waits[len(waits) // 2], 1),
        "приём_максимум_мс": round(waits[-1], 1),
        "приём_p95_мс": round(waits[int(len(waits) * 0.95)], 1),
        "расшифровок": s.partial_text.count(" ") + 1 if s.partial_text else 0,
    }


def main():
    al = GigaAMAligner()
    al(np.zeros(16000, np.float32), 16000)          # прогрев

    out = {}
    print(f"{'реплика':>9} {'путь':>12} {'приём p95':>11} {'максимум':>10} "
          f"{'очередь':>9} {'потеряно':>9}")
    for sec in LENGTHS:
        for sync in (True, False):
            r = run(al, sync=sync, seconds=sec)
            out[f"{sec}с_{'синхронно' if sync else 'фоном'}"] = r
            print(f"{sec:>8}с {'синхронно' if sync else 'фоном':>12} "
                  f"{r['приём_p95_мс']:>9} мс {r['приём_максимум_мс']:>8} мс "
                  f"{r['очередь_пик']:>9} {r['потеряно']:>9}")
    dst = ROOT / "bench" / "results" / "r9_intake.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
