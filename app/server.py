#!/usr/bin/env python3
"""Сервер вертикального среза: два экрана и весь путь между ними.

    bench/r1-stt/.venv/bin/python app/server.py
    http://localhost:8010/app/web/methodist.html

Каркас взят из спайков S2 и S3: тот же кадровый протокол
`[u32 jsonLen][u32 pcmLen][json][pcm]`, тот же принцип «клиент безусловно
дропает всё с чужим generation_id». Переписывать не стал — оно померено.

Красоты здесь нет намеренно: срез ищет интеграционные сюрпризы, а не вёрстку.
"""
from __future__ import annotations

import argparse
import json
import http.server
import queue
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
from app.actions import parse_reply, repair_action          # noqa: E402
from app.agent import SYSTEM, apply as apply_action, build_prompt   # noqa: E402
from app.backchannel import Backchannel                     # noqa: E402
from app.dialogue import DialogueState                      # noqa: E402
from app.emotion_tags import SYSTEM_HINT as EMO_HINT        # noqa: E402
from app.evaluator import BackgroundEvaluator               # noqa: E402
from app.generation import GenerationRegistry               # noqa: E402
from app.media import (GigaAMAligner, PROVIDERS, SwitchableTTS,   # noqa: E402
                       stream_deepseek, tts_config)
from app.pipeline import ReplyPipeline, subtitle_cues       # noqa: E402
from app.scenario import Criterion, load_all                # noqa: E402
from app.speculation import Speculator                      # noqa: E402
from app.typing_signal import TypingTracker                 # noqa: E402
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
        # Поток создаётся НА КАЖДУЮ реплику. CancellableStream одноразовый:
        # после отмены он закрыт навсегда, и переиспользование давало вторую
        # реплику с нулём клауз и нулевым TTFT — модель просто не звалась.
        self.stream = None
        self.pipe = ReplyPipeline(None, models["tts"], models["aligner"],
                                  models["bridge"], self.registry)
        # Генерации, которые доиграли до конца. Их реплики пользователь
        # услышал, и стирать их из истории при следующей отмене нельзя.
        self.completed: set[str] = set()
        # Текст клауз, реально ушедших клиенту, по генерациям. При перебивании
        # это единственный источник правды о том, что человек услышал:
        # `apply_action` до истории уже не доберётся.
        self.spoken: dict[str, list[str]] = {}
        # Ход пользователя, породивший генерацию. Нужен, чтобы перебитый обмен
        # не списывал бюджет этапа.
        self.gen_turn: dict[str, object] = {}
        self.evaluator = BackgroundEvaluator(models["llm"])
        self.backchannel = models.get("backchannel")
        self.repair_llm = models.get("repair_llm")
        self.bc_cfg = models.get("bc_cfg", {})
        # Спекуляция на недопечатанном: запрос уходит, пока человек ещё печатает.
        self.spec = Speculator(
            make_stream=lambda: stream_deepseek(models["llm"]),
            build_prompt=lambda text: build_prompt(self.state, text),
            system=SYSTEM, cfg=models.get("spec_cfg", {}))
        # Динамика набора: время до первого нажатия, паузы, стирания. Часы
        # отсчитываются от конца реплики агента — их ставит сервер, а не клиент:
        # у клиента нет момента, когда персонаж договорил.
        self.typing = TypingTracker()
        self.t_agent_done_ms: float | None = None
        self.frames: list[tuple[dict, bytes]] = []
        self.frame_cv = threading.Condition()
        self.marks: list[dict] = []
        self.lock = threading.Lock()

    # ------------------------------------------------------------- генерация

    def speak(self, user_text: str | None) -> None:
        """Сгенерировать реплику агента и разложить её в кадры.

        Кадры отдаёт НЕ конвейер, а этот метод: между готовностью клаузы и её
        отправкой стоит решение про заполнитель, и принять его может только
        тот, кто видит обе стороны.
        """
        # Нуль отсчёта — вход в метод, то есть фактически нажатие Enter.
        # Ставится ДО разбора и построения промпта: иначе замер начинался бы
        # уже после части работы и льстил бы себе.
        t0 = time.perf_counter()
        with self.lock:
            if user_text is not None:
                # Нуль отсчёта — момент, когда агент договорил. Он мог
                # сдвинуться уже после первых нажатий, поэтому подставляется
                # здесь, а не при наблюдении.
                sig = self.typing.summarise(final_length=len(user_text),
                                            origin_ms=self.t_agent_done_ms)
                self.typing.reset()
                user_turn = self.state.add_user(user_text, typing=sig.to_dict())
                self._emit({"kind": "user", "text": user_text,
                            "typing": sig.to_dict()})
            else:
                user_turn = None
            gen = self.registry.start()
            if user_turn is not None:
                self.gen_turn[gen.id] = user_turn
            prompt = build_prompt(self.state, user_text)
            # Если спекулятивный запрос попал — берём его: токены уже летят,
            # а то и накопились. Промах стоит потраченных токенов, не задержки.
            flight, cover = (self.spec.take(user_text) if user_text is not None
                             else (None, 0.0))
            if flight is not None:
                self.stream = flight.stream
                self.pipe.llm_stream = _ReplayStream(flight)
                spec_hit = True
                # Сколько запрос летел ДО нажатия Enter. Это и есть вся польза
                # спекуляции: попадание с форой в 200 мс экономит 200 мс, а не
                # TTFT целиком, сколько бы «попаданий из попаданий» ни было в
                # сводке.
                em_head = (t0 - flight.t_start) * 1000
                em_tokens = len(flight.tokens)
            else:
                self.stream = stream_deepseek(self.models["llm"])
                self.pipe.llm_stream = self.stream
                spec_hit = False
                em_head = None
                em_tokens = None

        pending: queue.Queue = queue.Queue()
        DONE = object()
        em = {"offset_ms": None, "clause_base": 0, "used_bc": False,
              "t_first_audio": None, "t_first_speech": None, "bc_text": None,
              "t_bc_emit": None}

        emitter = threading.Thread(target=self._emit_loop,
                                   args=(gen, t0, pending, DONE, em), daemon=True)
        emitter.start()

        results = self.pipe.run(SYSTEM + "\n\n" + EMO_HINT, prompt, gen,
                                on_result=pending.put)
        pending.put(DONE)
        emitter.join(timeout=10)

        if not gen.check():
            return                              # перебили — ничего не дописываем

        reply = parse_reply(self.pipe.last_raw)
        # Реплика без управляющей строки: спрашиваем отдельно, что это было.
        # Реплика уже звучит, так что на первый звук это не влияет.
        t_repair = None
        fell_back = reply.action.fell_back      # починка перезапишет действие
        if fell_back and self.repair_llm is not None:
            t_r0 = time.perf_counter()
            st = self.state.stage
            fixed = repair_action(self.repair_llm, reply.speakable,
                                  st.goal if st else "", self.state.is_last_stage)
            t_repair = (time.perf_counter() - t_r0) * 1000
            if fixed is not None:
                reply.action = fixed
        with self.lock:
            happened = apply_action(self.state, reply, gen.id)
            self.completed.add(gen.id)
        self._emit({"kind": "agent", "generation_id": gen.id,
                    "text": reply.speakable, "stage": self.state.stage_id})
        self.evaluator.submit(self.state)

        ttft = round(self.pipe.last_stats.get("t_first_token") or 0)
        if user_text is not None:
            self.spec.note_ttft(ttft, spec_hit)
        self.marks.append({
            "generation_id": gen.id,
            "spec_hit": spec_hit,
            "spec_cover": round(cover, 3) if user_text is not None else None,
            # Фора: сколько запрос летел до Enter, и сколько токенов успело
            # накопиться к этому моменту. Без этих двух чисел «попаданий 55 из
            # 55» ничего не говорит о сэкономленном времени.
            "spec_head_ms": round(em_head) if em_head is not None else None,
            "spec_tokens_ready": em_tokens,
            "t_first_token_ms": ttft,
            # Первый звук — то, что слышит человек. Первая содержательная
            # реплика — другая величина, и знать надо обе.
            "t_first_audio_ms": round(em["t_first_audio"] or 0),
            "t_first_speech_ms": round(em["t_first_speech"] or 0),
            "backchannel": em["bc_text"],
            "clauses": len(results),
            "action": happened["action"],
            "forced": happened.get("forced", False),
            # Без этих двух полей доля реплик без управляющей строки не
            # измеряется — а именно её мы и чиним.
            "fell_back": fell_back,
            "repaired": reply.action.repaired,
            "t_repair_ms": round(t_repair) if t_repair is not None else None,
        })
        self._emit({
            "kind": "state", "generation_id": gen.id,
            "stage": self.state.stage_id,
            "stage_index": self.state.stage_index,
            "stages_total": len(self.scenario.stages),
            "finished": self.state.finished,
            "action": happened["action"],
            # Реплика уже целиком отправлена, но браузер ещё может её доигрывать.
            # Он применит listening по окончании последнего AudioBufferSource.
            "face": "listening",
            "after_audio": True,
            "t_first_audio_ms": round(em["t_first_audio"] or 0),
            "t_first_speech_ms": round(em["t_first_speech"] or 0),
            # Оверлею: то, что видно только серверу. Без этих полей на показе
            # нечем показать, что спекуляция и починка вообще работают.
            "spec_hit": spec_hit,
            "spec_cover": round(cover, 3) if user_text is not None else None,
            "repaired": reply.action.repaired,
            "t_first_token_ms": ttft,
        })
        # Нуль отсчёта набора: с этого момента ход человека. Наблюдения НЕ
        # сбрасываются — набранное, пока агент ещё говорил, тоже сигнал.
        self.t_agent_done_ms = time.perf_counter() * 1000
        if self.state.finished:
            self._emit({"kind": "finished"})

    def note_typing(self, text: str) -> None:
        """Наблюдение за набором. Часы абсолютные, нуль ставится при подведении.

        Если агент ещё не говорил, нулём становится первое наблюдение: паузы и
        стирания от этого не зависят, а «до первого нажатия» на первом ходу и
        нечему мерить.
        """
        now = time.perf_counter() * 1000
        if self.t_agent_done_ms is None:
            self.t_agent_done_ms = now
        self.typing.observe(now, len(text))

    # ------------------------------------------------------------- отправка

    def _emit_loop(self, gen, t0, pending, DONE, em) -> None:
        """Отдаёт клаузы клиенту, вставляя заполнитель, если ответ задерживается."""
        after_s = (self.bc_cfg.get("after_ms", 600)) / 1000

        first = None
        try:
            first = pending.get(timeout=after_s)
        except queue.Empty:
            pass

        # Заполнитель нужен только когда ответа ещё нет. Если модель ответила
        # быстро, он звучит навязчиво.
        if first is None and self.backchannel and self.backchannel.ready \
                and self.bc_cfg.get("enabled", True) and gen.check():
            f = self.backchannel.pick()
            if f is not None:
                self._emit_filler(gen, f)
                em["used_bc"] = True
                em["bc_text"] = f.text
                em["offset_ms"] = f.audio_ms
                em["clause_base"] = 1
                em["t_first_audio"] = (time.perf_counter() - t0) * 1000
                em["t_bc_emit"] = time.perf_counter()
                # Заполнитель отзвучал — лицо уходит думать, пока идёт модель.
                self._emit({"kind": "state", "generation_id": gen.id,
                            "face": "thinking", "after_ms": f.audio_ms})

        while True:
            r = first if first is not None else pending.get()
            first = None
            if r is DONE:
                break
            if not gen.check():
                continue                    # перебили: дочитываем очередь молча
            self._emit_clause(gen, r, em, t0)

    def _emit_filler(self, gen, f) -> None:
        """Заполнитель — клауза 0 той же генерации.

        Не отдельный поток и не отдельный таймлайн: тот же `generation_id`, тот
        же PTS, та же отмена. Аватар получает его через `playGeneration`, и
        основная реплика потом досылается в тот же трек — поэтому рот между
        ними не захлопывается, если пауза короткая.
        """
        pcm16 = (f.pcm * 32767).astype("<i2").tobytes()
        self._emit({"kind": "audio", "generation_id": gen.id, "clause": 0,
                    "start_ms": 0.0, "sample_rate": f.sample_rate,
                    "audio_ms": f.audio_ms, "backchannel": True}, pcm16)
        self._emit({"kind": "visemes", "generation_id": gen.id, "clause": 0,
                    "track": f.visemes, "backchannel": True})
        self._emit({"kind": "subtitles", "generation_id": gen.id, "clause": 0,
                    "cues": [{"pts_ms": 0, "text": f.text, "clause": 0,
                              "generation_id": gen.id}], "backchannel": True})

    def _emit_clause(self, gen, r, em, t0) -> None:
        if em["offset_ms"] is None:
            em["offset_ms"] = 0.0
        elif em["clause_base"] and em.get("_placed") is None:
            # Первая содержательная клауза после заполнителя. Смещение НЕЛЬЗЯ
            # брать равным длине заполнителя: клауза может быть готова гораздо
            # позже, и тогда звук уедет от таймлайна висем — плеер поставит его
            # «не раньше сейчас», а мимика останется на своих 700 мс.
            # Берём фактическое время готовности плюс запас на сеть и декод.
            elapsed = (time.perf_counter() - em["t_bc_emit"]) * 1000
            lead = self.bc_cfg.get("lead_ms", 150)
            em["offset_ms"] = max(em["offset_ms"], elapsed + lead)
        em["_placed"] = True
        self.spoken.setdefault(r.generation_id, []).append(r.text)

        off = em["offset_ms"]
        idx = r.index + em["clause_base"]
        if em["t_first_audio"] is None:
            em["t_first_audio"] = (time.perf_counter() - t0) * 1000
        if em["t_first_speech"] is None:
            em["t_first_speech"] = (time.perf_counter() - t0) * 1000

        # После заполнителя лицо успело перейти в thinking. Первая настоящая
        # клауза обязана вернуть speaking ДО аудио: раньше такого события не
        # было, и весь содержательный ответ аватар произносил, продолжая
        # смотреть вверх-влево как при размышлении.
        if em["used_bc"] and not em.get("_content_speaking"):
            em["_content_speaking"] = True
            self._emit({"kind": "state", "generation_id": r.generation_id,
                        "face": "speaking"})

        pcm16 = (r.pcm * 32767).astype("<i2").tobytes()
        self._emit({"kind": "audio", "generation_id": r.generation_id,
                    "clause": idx, "start_ms": r.start_ms + off,
                    "sample_rate": r.sample_rate, "audio_ms": r.audio_ms}, pcm16)
        self._emit({"kind": "visemes", "generation_id": r.generation_id,
                    "clause": idx,
                    "track": [{**v, "pts_ms": v["pts_ms"] + off} for v in r.visemes]})
        self._emit({"kind": "subtitles", "generation_id": r.generation_id,
                    "clause": idx,
                    "cues": [{**c, "pts_ms": c["pts_ms"] + off, "clause": idx}
                             for c in subtitle_cues(r)]})
        if r.emotions:
            self._emit({"kind": "emotions", "generation_id": r.generation_id,
                        "clause": idx,
                        "marks": [{**e, "pts_ms": e["pts_ms"] + off}
                                  for e in r.emotions]})

    def cancel(self) -> str | None:
        """Перебивание: гасим ВСЮ цепочку одним движением."""
        gid = self.registry.cancel()
        if self.stream is not None:
            self.stream.cancel()
        # Спекуляцию здесь НЕ гасим. Она относится к сообщению пользователя, а
        # не к реплике агента: перебивание как раз и есть отправка сообщения,
        # ради которого запрос улетел вперёд. Первая версия звала drop() отсюда
        # и убивала ровно тот запрос, который собиралась использовать —
        # попаданий было ноль. Устаревший запрос гасится сам: при перезапуске
        # по росту буфера и при промахе по префиксу.
        if gid:
            with self.lock:
                if gid not in self.completed:
                    # Недоговорённое из истории убираем, но НЕ целиком:
                    # прозвучавшую часть пользователь слышал, и агент обязан
                    # её помнить, иначе он переспросит то, на что уже получил
                    # ответ. Возвращаем ровно озвученные клаузы.
                    self.state.drop_generation(gid)
                    said = " ".join(self.spoken.get(gid, [])).strip()
                    if said:
                        self.state.add_interrupted_agent(said, gid)
                    # Обмен не состоялся — агент не договорил свой вопрос.
                    # Бюджет этапа за это списывать нельзя: иначе перебивание,
                    # то есть нормальная живость разговора, приближало бы
                    # сценарий к принудительному концу.
                    turn = self.gen_turn.get(gid)
                    if turn is not None:
                        turn.counted = False
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
        d["speculation"] = self.spec.stats.summary()
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


CONFIG = ROOT / "app" / "config.json"


def load_config() -> dict:
    """Конфиг приложения. Переключение провайдера синтеза — одна строка здесь."""
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise SystemExit(f"{CONFIG}: {e}")


class _ReplayStream:
    """Обёртка вокруг летящего запроса под интерфейс потока конвейера.

    Конвейер зовёт `stream(system, prompt)` и получает генератор. Здесь аргументы
    игнорируются: запрос уже ушёл со своим промптом, и переспрашивать модель
    заново значило бы выбросить весь выигрыш.
    """

    def __init__(self, flight):
        self.flight = flight

    def cancel(self):
        self.flight.cancel()

    def __call__(self, system, prompt):
        return self.flight.replay()


class App:
    def __init__(self):
        print("поднимаю модели…")
        t0 = time.perf_counter()
        self.models = {
            "tts": SwitchableTTS(tts_config()),
            "aligner": GigaAMAligner(),
            "bridge": VisemeBridge(),
            "llm": llm_mod.DeepSeekLLM(),
            # Отдельный дешёвый клиент для второго разбора: ответ — одно слово,
            # и держать под него бюджет основной реплики незачем.
            "repair_llm": llm_mod.DeepSeekLLM(max_tokens=8, temperature=0),
        }
        # Заполнители готовятся здесь и лежат в памяти: по Enter не считается
        # ничего, иначе смысл теряется.
        bc = Backchannel().warm(self.models["tts"],
                                lambda pcm, sr: self.models["aligner"](pcm, sr),
                                self.models["bridge"],
                                cache_key=self.models["tts"].provider)
        self.models["backchannel"] = bc
        # after_ms — сколько ждать ответа, прежде чем ставить заполнитель.
        # Порог короткий намеренно: полсекунды тишины и есть то, что заполнитель
        # убирает. Случай «ответ пришёл быстро» — это попадание спекуляции с
        # уже готовым результатом, а не ожидание в шестьсот миллисекунд.
        cfg = load_config()
        self.models["bc_cfg"] = {k: v for k, v in cfg.get("backchannel", {}).items()
                                 if not k.startswith("_")} or \
            {"enabled": True, "after_ms": 120, "lead_ms": 150}
        self.models["spec_cfg"] = {k: v for k, v in cfg.get("speculation", {}).items()
                                   if not k.startswith("_")}
        print(f"  заполнители: {bc.describe()}")
        self.scenarios = load_all(ROOT / "data" / "scenarios")
        self.session: Session | None = None
        print(f"  синтез: {self.models['tts'].describe()}")
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


    def switch_tts(self, provider: str) -> dict:
        """Сменить синтезатор на ходу и пересобрать заполнители.

        Заполнители переозвучиваются обязательно: они звучат непосредственно
        перед репликой, и если голос сменится между ними, на стыке будет слышно
        двух разных людей. Текущая генерация гасится по той же причине — иначе
        голос сменился бы посреди реплики.
        """
        if provider not in PROVIDERS:
            raise ValueError(f"нет провайдера «{provider}»")
        tts = self.models["tts"]
        t0 = time.perf_counter()
        changed = tts.switch(provider)          # бросит, если движок не собрался
        if changed:
            if self.session:
                self.session.cancel()
            self.models["backchannel"].warm(
                tts, lambda pcm, sr: self.models["aligner"](pcm, sr),
                self.models["bridge"], cache_key=provider)
        return {"changed": changed, "took_ms": round((time.perf_counter() - t0) * 1000),
                "tts": tts.describe(),
                "backchannel": self.models["backchannel"].describe()}


APP: App | None = None


class Handler(http.server.SimpleHTTPRequestHandler):
    # HTTP/1.1 с явным chunked. На 1.0 браузер копил ответ и отдавал его
    # порциями по своему усмотрению: первые кадры доезжали (их было много
    # килобайт разом), а всё, что приходило после паузы, застревало в буфере.
    # Сырой сокет при этом получал кадры мгновенно — то есть сервер работал, и
    # искать поломку в нём было бы напрасно. Спайкам S2 и S3 хватало 1.0,
    # потому что там поток шёл непрерывно и буфер не успевал застояться.
    protocol_version = "HTTP/1.1"

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
        if u.path == "/api/tts":
            tts = APP.models["tts"]
            return self._json({"providers": list(PROVIDERS),
                               "provider_options": tts.provider_options(),
                               "tts": tts.describe()})
        if u.path == "/api/health":
            # Чем синтезируем прямо сейчас. На показе это единственный способ
            # заметить, что сетевой голос отвалился и говорит запасной.
            tts = APP.models["tts"]
            return self._json({"tts": tts.describe(),
                               "scenarios": len(APP.scenarios),
                               "session": bool(APP.session)})
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
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        idx = start
        try:
            while True:
                batch = APP.session.frames_from(idx)
                if not batch:
                    # Держим соединение: браузер переподключаться не должен.
                    self._chunk(struct.pack("<II", 2, 0) + b"{}")
                    continue
                blob = b""
                for header, payload in batch:
                    hdr = json.dumps({**header, "seq": idx}, ensure_ascii=False).encode()
                    blob += struct.pack("<II", len(hdr), len(payload)) + hdr + payload
                    idx += 1
                self._chunk(blob)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _chunk(self, data: bytes) -> None:
        """Один кусок chunked-потока. Браузер отдаёт его читателю сразу."""
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

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
        if u.path == "/api/typing":
            if not APP.session:
                return self._json({"ok": False})
            sess = APP.session
            sess.note_typing(data.get("text", ""))
            launched = sess.spec.on_typing(data.get("text", ""))
            return self._json({"ok": True, "launched": launched})
        if u.path == "/api/tts":
            try:
                return self._json(APP.switch_tts(data.get("provider", "")))
            except SystemExit as e:            # нет ключа, нет такого голоса
                return self._json({"error": str(e)}, 400)
            except Exception as e:             # noqa: BLE001 — сеть, API, что угодно
                return self._json({"error": f"{type(e).__name__}: {e}"}, 400)
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
    # Порт задаётся снаружи, чтобы репетиция могла поднять свой сервер, не
    # выбивая тот, что уже открыт в браузере.
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    port = ap.parse_args().port
    APP = App()
    with Threaded(("", port), Handler) as httpd:
        print(f"методист:  http://localhost:{port}/app/web/methodist.html")
        print(f"сотрудник: http://localhost:{port}/app/web/trainee.html")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
