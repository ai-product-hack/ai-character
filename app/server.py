#!/usr/bin/env python3
"""Сервер вертикального среза: два экрана и весь путь между ними.

    bench/r1-stt/.venv/bin/python app/server.py
    http://localhost:8town/app/web/methodist.html

Каркас взят из спайков S2 и S3: тот же кадровый протокол
`[u32 jsonLen][u32 pcmLen][json][pcm]`, тот же принцип «клиент безусловно
дропает всё с чужим generation_id». Переписывать не стал — оно померено.

Красоты здесь нет намеренно: срез ищет интеграционные сюрпризы, а не вёрстку.
"""
from __future__ import annotations

import json
import http.server
import pathlib
import socketserver
import struct
import sys
import threading
import time
import urllib.parse

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import llm as llm_mod, report as report_mod        # noqa: E402
from app.actions import parse_reply                         # noqa: E402
from app.agent import SYSTEM, apply as apply_action, build_prompt   # noqa: E402
from app.dialogue import DialogueState                      # noqa: E402
from app.evaluator import BackgroundEvaluator               # noqa: E402
from app.generation import GenerationRegistry               # noqa: E402
from app.media import GigaAMAligner, SileroTTS, stream_deepseek     # noqa: E402
from app.pipeline import ReplyPipeline, subtitle_cues       # noqa: E402
from app.scenario import Criterion, load_all                # noqa: E402
from app.visemes_bridge import VisemeBridge                 # noqa: E402

PORT = 8010


class Session:
    """Один тренировочный диалог. Живёт в процессе сервера."""

    def __init__(self, scenario, criteria_text: str, models):
        self.scenario = scenario
        if criteria_text.strip():
            self.scenario.criteria = parse_criteria(criteria_text) or scenario.criteria
        self.state = DialogueState(scenario)
        self.registry = GenerationRegistry()
        self.models = models
        self.stream = stream_deepseek(models["llm"])
        self.pipe = ReplyPipeline(self.stream, models["tts"], models["aligner"],
                                  models["bridge"], self.registry)
        self.evaluator = BackgroundEvaluator(models["llm"])
        self.frames: list[tuple[dict, bytes]] = []
        self.frame_cv = threading.Condition()
        self.marks: list[dict] = []
        self.lock = threading.Lock()

    # ------------------------------------------------------------- генерация

    def speak(self, user_text: str | None) -> None:
        """Сгенерировать реплику агента и разложить её в кадры."""
        with self.lock:
            if user_text is not None:
                self.state.add_user(user_text)
                self._emit({"kind": "user", "text": user_text})
            gen = self.registry.start()
            prompt = build_prompt(self.state, user_text)

        t0 = time.perf_counter()
        marks = {"t_request": t0, "t_first_audio": None}

        def on_result(r):
            if marks["t_first_audio"] is None:
                marks["t_first_audio"] = (time.perf_counter() - t0) * 1000
            pcm16 = (r.pcm * 32767).astype("<i2").tobytes()
            self._emit({
                "kind": "audio", "generation_id": r.generation_id,
                "clause": r.index, "start_ms": r.start_ms,
                "sample_rate": r.sample_rate, "audio_ms": r.audio_ms,
            }, pcm16)
            self._emit({
                "kind": "visemes", "generation_id": r.generation_id,
                "clause": r.index, "track": r.visemes,
            })
            self._emit({
                "kind": "subtitles", "generation_id": r.generation_id,
                "clause": r.index, "cues": subtitle_cues(r),
            })

        results = self.pipe.run(SYSTEM, prompt, gen, on_result=on_result)
        if not gen.check():
            return                              # перебили — ничего не дописываем

        reply = parse_reply(self.pipe.last_raw)
        with self.lock:
            happened = apply_action(self.state, reply, gen.id)
        # Оценка уходит в фон и диалог не задерживает.
        self.evaluator.submit(self.state)

        self.marks.append({
            "generation_id": gen.id,
            "t_first_token_ms": round(self.pipe.last_stats.get("t_first_token") or 0),
            "t_first_audio_ms": round(marks["t_first_audio"] or 0),
            "clauses": len(results),
            "action": happened["action"],
            "forced": happened.get("forced", False),
        })
        self._emit({
            "kind": "state", "generation_id": gen.id,
            "stage": self.state.stage_id,
            "stage_index": self.state.stage_index,
            "stages_total": len(self.scenario.stages),
            "finished": self.state.finished,
            "action": happened["action"],
            "t_first_audio_ms": round(marks["t_first_audio"] or 0),
        })
        if self.state.finished:
            self._emit({"kind": "finished"})

    def cancel(self) -> str | None:
        """Перебивание: гасим ВСЮ цепочку одним движением."""
        gid = self.registry.cancel()
        self.stream.cancel()
        if gid:
            with self.lock:
                self.state.drop_generation(gid)
            self._emit({"kind": "cancel", "generation_id": gid})
        return gid

    # ----------------------------------------------------------------- кадры

    def _emit(self, header: dict, payload: bytes = b"") -> None:
        with self.frame_cv:
            self.frames.append((header, payload))
            self.frame_cv.notify_all()

    def frames_from(self, index: int, timeout: float = 25):
        """Дождаться кадров начиная с index. Блокирует, пока их нет."""
        with self.frame_cv:
            if index >= len(self.frames):
                self.frame_cv.wait(timeout=timeout)
            return self.frames[index:]

    # ---------------------------------------------------------------- отчёт

    def report(self) -> dict:
        self.evaluator.drain(timeout=20)
        rep = report_mod.build(self.state, self.evaluator.log)
        d = rep.to_dict()
        d["marks"] = self.marks
        d["evaluator"] = {"calls": self.evaluator.log.calls,
                          "errors": self.evaluator.log.errors,
                          "total_ms": round(self.evaluator.log.total_ms)}
        return d


def parse_criteria(text: str) -> list[Criterion]:
    """Критерии от методиста: по строке на критерий, `ключ: название`.

    Формат нарочно примитивный: поле свободного ввода, а не форма.
    """
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, title = line.partition(":")
        key = key.strip().replace(" ", "_").lower()
        if not key:
            continue
        out.append(Criterion(key=key, title=(title.strip() or key), scale="1-5",
                             anchor_1="не проявлено", anchor_5="проявлено полностью"))
    return out


class App:
    def __init__(self):
        print("поднимаю модели…")
        t0 = time.perf_counter()
        self.models = {
            "tts": SileroTTS(voice="eugene"),
            "aligner": GigaAMAligner(),
            "bridge": VisemeBridge(),
            "llm": llm_mod.DeepSeekLLM(),
        }
        self.scenarios = load_all(ROOT / "data" / "scenarios")
        self.session: Session | None = None
        print(f"готово за {time.perf_counter() - t0:.1f} с, "
              f"сценариев {len(self.scenarios)}")

    def start(self, scenario_id: str, criteria_text: str) -> Session:
        sc = next((s for s in self.scenarios if s.id == scenario_id), self.scenarios[0])
        # Свежая копия сценария: критерии методиста не должны протечь в
        # следующую сессию.
        from app.scenario import Scenario
        sc = Scenario.from_dict(sc.to_dict())
        self.session = Session(sc, criteria_text, self.models)
        threading.Thread(target=self.session.speak, args=(None,), daemon=True).start()
        return self.session


APP: App | None = None


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, fmt, *args):
        if "404" in (fmt % args):
            super().log_message(fmt, *args)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # ------------------------------------------------------------------ GET

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/scenarios":
            return self._json([{"id": s.id, "title": s.title, "type": s.type,
                                "stages": len(s.stages),
                                "criteria": [{"key": c.key, "title": c.title}
                                             for c in s.criteria]}
                               for s in APP.scenarios])
        if u.path == "/api/report":
            if not APP.session:
                return self._json({"error": "сессия не начата"}, 400)
            return self._json(APP.session.report())
        if u.path == "/api/stream":
            return self._stream(int(urllib.parse.parse_qs(u.query).get("from", ["0"])[0]))
        return super().do_GET()

    def _stream(self, start: int):
        if not APP.session:
            return self._json({"error": "сессия не начата"}, 400)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        idx = start
        try:
            while True:
                batch = APP.session.frames_from(idx)
                if not batch:
                    # Держим соединение: браузер переподключаться не должен.
                    self.wfile.write(struct.pack("<II", 2, 0) + b"{}")
                    self.wfile.flush()
                    continue
                for header, payload in batch:
                    hdr = json.dumps({**header, "seq": idx}, ensure_ascii=False).encode()
                    self.wfile.write(struct.pack("<II", len(hdr), len(payload))
                                     + hdr + payload)
                    idx += 1
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ----------------------------------------------------------------- POST

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        data = json.loads(body or b"{}")

        if u.path == "/api/start":
            APP.start(data.get("scenario"), data.get("criteria", ""))
            return self._json({"ok": True})
        if u.path == "/api/message":
            if not APP.session:
                return self._json({"error": "сессия не начата"}, 400)
            # Enter во время речи агента = перебивание. Одно движение гасит всё.
            gid = APP.session.cancel()
            threading.Thread(target=APP.session.speak,
                             args=(data.get("text", ""),), daemon=True).start()
            return self._json({"ok": True, "cancelled": gid})
        if u.path == "/api/cancel":
            return self._json({"ok": True, "cancelled": APP.session.cancel()})
        return self._json({"error": "нет такого метода"}, 404)

    def _json(self, obj, code=200):
        blob = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)


class Threaded(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    global APP
    APP = App()
    with Threaded(("", PORT), Handler) as httpd:
        print(f"методист:  http://localhost:{PORT}/app/web/methodist.html")
        print(f"сотрудник: http://localhost:{PORT}/app/web/trainee.html")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
