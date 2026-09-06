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
import os
import http.server
import queue
import pathlib
import socketserver
import struct
import sys
import threading
import time
import urllib.parse
import uuid

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import generator as gen_mod, llm as llm_mod, report as report_mod  # noqa: E402
from app.actions import parse_reply, repair_action          # noqa: E402
from app.agent import SYSTEM, apply as apply_action, build_prompt   # noqa: E402
from app.backchannel import Backchannel                     # noqa: E402
from app.dialogue import DialogueState                      # noqa: E402
from app.emotion_drive import mood_from                     # noqa: E402
from app.emotion_tags import EMOTIONS, SYSTEM_HINT as EMO_HINT   # noqa: E402
from app.panels import (SYSTEM_HINT as PANEL_HINT,          # noqa: E402
                        material_payload)
from app.evaluator import BackgroundEvaluator               # noqa: E402
from app.generation import GenerationRegistry               # noqa: E402
from app.media import (GigaAMAligner, PROVIDERS, SwitchableTTS,   # noqa: E402
                       tts_config)
from app.pipeline import ReplyPipeline, subtitle_cues       # noqa: E402
from app.scenario import Criterion, Scenario, load_all      # noqa: E402
from app.templates import TYPES                             # noqa: E402
from app.speculation import Speculator                      # noqa: E402
from app.typing_signal import TypingTracker                 # noqa: E402
from app.visemes_bridge import VisemeBridge                 # noqa: E402
from app.voice import Utterance, VoiceInput, transcribe     # noqa: E402

PORT = 8010


class Session:
    """Один тренировочный диалог. Живёт в процессе сервера."""

    def __init__(self, scenario, criteria_text: str, models, session_id: str = ""):
        self.scenario = scenario
        # Идентификатор сессии. По нему сотрудник открывает свой экран, а
        # методист забирает отчёт: словарь в памяти процесса, никакой базы.
        self.id = session_id or uuid.uuid4().hex[:8]
        # Отчёт пишется на диск ровно один раз — на `finish`. Опрашивают его
        # каждые две секунды, и без этого флага история прогонов состояла бы из
        # сотни копий одного разговора.
        self.saved = ""
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
        # Клаузы, ушедшие клиенту, с их местом на таймлайне. При перебивании
        # отсюда берётся то, что человек успел УСЛЫШАТЬ, — по позиции его
        # часов аудио: `apply_action` до истории уже не доберётся.
        self.spoken: dict[str, list[dict]] = {}
        # Ход пользователя, породивший генерацию. Нужен, чтобы перебитый обмен
        # не списывал бюджет этапа.
        self.gen_turn: dict[str, object] = {}
        self.evaluator = BackgroundEvaluator(
            models["llm"], summary_llm=models.get("summary_llm"))
        self.backchannel = models.get("backchannel")
        self.repair_llm = models.get("repair_llm")
        self.bc_cfg = models.get("bc_cfg", {})
        # Спекуляция на недопечатанном: запрос уходит, пока человек ещё печатает.
        self.spec = Speculator(
            make_stream=lambda: llm_mod.make_stream(models["llm"]),
            build_prompt=lambda text: build_prompt(self.state, text),
            system=SYSTEM, cfg=models.get("spec_cfg", {}))
        # Динамика набора: время до первого нажатия, паузы, стирания. Часы
        # отсчитываются от конца реплики агента — их ставит сервер, а не клиент:
        # у клиента нет момента, когда персонаж договорил.
        self.typing = TypingTracker()
        self.t_agent_done_ms: float | None = None
        # Голосовой ввод под тумблером. Текстовый путь остаётся основным и
        # ничего о голосе не знает: сюда приходит уже готовая реплика.
        self.voice = VoiceInput(models.get("endpointer"))
        self.voice_stats: list[dict] = []
        # Распознавание на лету: пока человек говорит, разбираем уже сказанное.
        # Даёт две вещи сразу — слова в интерфейсе (человек видит, что его
        # слышат) и текст для спекуляции (запрос уходит до конца реплики).
        self.partial_text = ""
        self._partial_at = 0.0
        self._partial_lock = threading.Lock()
        self.frames: list[tuple[dict, bytes]] = []
        self.frame_cv = threading.Condition()
        self.marks: list[dict] = []
        self.lock = threading.Lock()

    # ------------------------------------------------------------- генерация

    def speak(self, user_text: str | None, t_start: float | None = None) -> None:
        """Сгенерировать реплику агента и разложить её в кадры.

        Кадры отдаёт НЕ конвейер, а этот метод: между готовностью клаузы и её
        отправкой стоит решение про заполнитель, и принять его может только
        тот, кто видит обе стороны.
        """
        # Нуль отсчёта — вход в метод, то есть фактически нажатие Enter.
        # Ставится ДО разбора и построения промпта: иначе замер начинался бы
        # уже после части работы и льстил бы себе.
        #
        # При голосовом вводе нуль приходит снаружи — это момент, когда
        # эндпоинтер решил, что человек договорил. Заполнитель обязан
        # стартовать оттуда, а не от конца распознавания: иначе к паузе
        # добавилось бы ещё и время расшифровки.
        t0 = t_start if t_start is not None else time.perf_counter()
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
                self.stream = llm_mod.make_stream(self.models["llm"])
                self.pipe.llm_stream = self.stream
                spec_hit = False
                em_head = None
                em_tokens = None

        pending: queue.Queue = queue.Queue()
        DONE = object()
        em = {"offset_ms": None, "clause_base": 0, "used_bc": False,
              "t_first_audio": None, "t_first_speech": None, "bc_text": None,
              "t_bc_emit": None, "panels": 0}

        emitter = threading.Thread(
            target=self._emit_loop,
            # Открывающая реплика — не ответ на вопрос, и заполнителю там
            # нечего заполнять.
            args=(gen, t0, pending, DONE, em, user_text is not None), daemon=True)
        emitter.start()

        results = self.pipe.run(
            "\n\n".join((SYSTEM, EMO_HINT, PANEL_HINT)), prompt, gen,
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
        # Переход на новый этап — законный повод показать его материал. Модель
        # вызывает панель сама, но делает это редко: на 91 реплике прогона ни
        # разу, даже с явным поводом в подсказке. Как и с эмоциями, разметка
        # модели главнее, а механика продукта заполняет молчание.
        #
        # Раньше здесь показывалась карта накопленных оценок — и это была
        # ошибка, замеченная на живом прогоне: «не вижу смысла показывать
        # статистику, как человек отвечает». Своя оценка посреди разговора —
        # подсказка на экзамене, а не материал. Теперь показывается то, О ЧЁМ
        # спрашивают на новом этапе, и только если материал у него есть:
        # пустая панель хуже отсутствующей.
        material = self._panel_data("material")
        if happened.get("advanced") and material and not em.get("panels") \
                and gen.check():
            self._emit({"kind": "panels", "generation_id": gen.id,
                        "clause": len(results), "from_stage": True,
                        "marks": [{"pts_ms": self.pipe.last_timeline.total_ms
                                   + (em["offset_ms"] or 0),
                                   "panel": "material", "arg": "",
                                   "data": material}]})

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
        # Агент договорил — микрофон снова слушает.
        self.voice.muted_by_agent = False
        self.voice.start()
        if self.state.finished:
            self._emit({"kind": "finished"})

    def push_audio(self, pcm) -> dict:
        """Кусок с микрофона. Если ход закончился — запускаем ответ.

        Распознавание идёт здесь, но нулём отсчёта для ответа служит момент
        решения эндпоинтера: заполнитель обязан стартовать оттуда, иначе к
        паузе добавится ещё и расшифровка.
        """
        endpointer = self.voice.endpointer
        speech_before = endpointer.speech_ms if endpointer is not None else 0
        utt = self.voice.push(pcm)
        if utt is None:
            speech_after = endpointer.speech_ms if endpointer is not None else 0
            # Reuse VAD evidence for listening gestures; silence and the muted
            # microphone during agent speech must not look like user activity.
            out = {"ok": True, "endpoint": False,
                   "user_speaking": speech_after > speech_before}
            partial = self._maybe_partial()
            if partial is not None:
                out["partial"] = partial
            return out
        return self._on_utterance(utt)

    def _maybe_partial(self) -> str | None:
        """Разобрать сказанное на лету, если пора. Возвращает текст или None.

        Дважды одновременно не запускается: пока идёт разбор, следующий кусок
        просто проходит мимо. Опоздать здесь не страшно — это подсказка
        человеку и корм для спекуляции, а не источник правды. Правду даёт
        разбор целой реплики на конце хода.
        """
        # Без распознавателя разбирать нечем: голос в такой сборке не
        # включается вовсе, и приём звука не должен об этом спотыкаться.
        if self.models.get("aligner") is None or self.voice.endpointer is None:
            return None
        every = self.models.get("voice_cfg", {}).get("partial_every_ms", 500)
        now = time.perf_counter() * 1000
        if now - self._partial_at < every:
            return None
        if not self._partial_lock.acquire(blocking=False):
            return None
        try:
            self._partial_at = now
            audio = self.voice.snapshot()
            if audio is None or len(audio) < 8000:      # меньше полусекунды
                return None
            text = transcribe(self.models["aligner"],
                              Utterance(pcm=audio, sample_rate=16000,
                                        speech_ms=0.0, t_endpoint=0.0))
            if not text or text == self.partial_text:
                return text or ""
            self.partial_text = text
            # Спекуляция на недосказанном — тот же механизм, что на
            # недопечатанном. Растущий текст, порог по длине, перезапуск по
            # приросту, проверка префикса на конце хода: всё уже написано,
            # голосу оно просто не досталось, потому что висело на /api/typing.
            self.spec.on_typing(text)
            return text
        except Exception as e:                          # noqa: BLE001
            # Разбор на лету не имеет права ломать приём звука.
            self.marks.append({"partial_error": f"{type(e).__name__}: {e}"})
            return None
        finally:
            self._partial_lock.release()

    def end_utterance(self) -> dict:
        """Push-to-talk отпущена: ход кончился по воле человека."""
        utt = self.voice.flush()
        if utt is None:
            return {"ok": True, "endpoint": False}
        return self._on_utterance(utt, forced=True)

    def _on_utterance(self, utt, forced: bool = False) -> dict:
        t0 = time.perf_counter()
        text = transcribe(self.models["aligner"], utt)
        asr_ms = (time.perf_counter() - t0) * 1000
        self.voice_stats.append({
            "duration_ms": round(utt.duration_ms),
            "speech_ms": round(utt.speech_ms),
            "asr_ms": round(asr_ms),
            "chars": len(text),
            "forced": forced,
        })
        if not text:
            # Распозналась пустота — это не реплика. Молчим, микрофон дальше
            # слушает: иначе агент отвечал бы на шум.
            self.voice.start()
            return {"ok": True, "endpoint": True, "text": "", "asr_ms": round(asr_ms)}
        # Пока агент говорит, микрофон закрыт: на колонках VAD услышал бы его
        # самого и перебил бы его же репликой.
        self.voice.muted_by_agent = True
        self.partial_text = ""
        self.cancel()
        threading.Thread(target=self.speak, args=(text, utt.t_endpoint),
                         daemon=True).start()
        return {"ok": True, "endpoint": True, "text": text, "asr_ms": round(asr_ms)}

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

    def _emit_loop(self, gen, t0, pending, DONE, em, answering: bool = True) -> None:
        """Отдаёт клаузы клиенту, вставляя заполнитель, если ответ задерживается.

        `answering` — отвечаем ли мы на реплику человека. У открывающей реплики
        это False: заполнитель там звучит абсурдно («Понятно… Здравствуйте!»),
        потому что понимать ещё нечего — никто ничего не сказал.
        """
        after_s = (self.bc_cfg.get("after_ms", 600)) / 1000

        first = None
        try:
            first = pending.get(timeout=after_s)
        except queue.Empty:
            pass

        # Заполнитель нужен только когда ответа ещё нет. Если модель ответила
        # быстро, он звучит навязчиво.
        use_bc = (first is None and answering and self.backchannel
                  and self.backchannel.ready
                  and self.bc_cfg.get("enabled", True) and gen.check())
        if first is None and not use_bc and gen.check():
            # Ответа нет, а заполнителя не будет: так бывает на открывающей
            # реплике. Молчащее лицо в позе слушателя читается как «сломалось»,
            # поэтому хотя бы уводим его думать — это ничего не стоит и
            # закрывает паузу тем единственным, что у нас есть мгновенно.
            self._emit({"kind": "state", "generation_id": gen.id,
                        "face": "thinking"})
        if use_bc:
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

        off = em["offset_ms"]
        idx = r.index + em["clause_base"]
        # Клауза запоминается ВМЕСТЕ с её местом на таймлайне клиента — тем
        # самым `start_ms`, что уходит в кадре звука. Раньше здесь копился
        # голый текст, и на перебивании в историю попадало всё ОТПРАВЛЕННОЕ.
        # А конвейер работает с опережением намеренно: синтез бежит впереди
        # воспроизведения. Значит две-три следующие клаузы уже отправлены и
        # засчитывались как сказанные, хотя человек их не слышал — агент потом
        # продолжал так, будто произнёс весь абзац.
        self.spoken.setdefault(r.generation_id, []).append(
            {"start_ms": r.start_ms + off, "audio_ms": r.audio_ms, "text": r.text})
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
        elif r.index == 0:
            # Модель ничего не разметила — берём настроение из накопленных
            # оценок. Разметка от модели главнее: если она сказала
            # «impressed», это точнее среднего балла. Но молчит она часто, а
            # ровное лицо весь показ — хуже, чем эмоция от механики продукта.
            mood = mood_from(self.evaluator.log, self.scenario.criteria,
                             baseline=self.scenario.persona.start_emotion)
            self._emit({"kind": "emotions", "generation_id": r.generation_id,
                        "clause": idx, "from_scores": True,
                        "marks": [{"pts_ms": r.start_ms + off, "emotion": mood.emotion,
                                   "intensity": mood.intensity}]})
        if r.panels:
            em["panels"] = em.get("panels", 0) + len(r.panels)
            # Данные кладём сразу: панель рисуется из уже накопленного отчёта,
            # и второго запроса ради неё быть не должно.
            self._emit({"kind": "panels", "generation_id": r.generation_id,
                        "clause": idx,
                        "marks": [{**c, "pts_ms": c["pts_ms"] + off,
                                   "data": self._panel_data(c["panel"])}
                                  for c in r.panels]})

    def _panel_data(self, panel: str) -> dict:
        """Содержимое панели из состояния, которое уже есть.

        Карта навыков — это оценки фоновой сессии, они считаются по ходу
        разговора независимо от панели. Ничего не генерируется.
        """
        if panel == "material":
            # Ничего не считаем и не запрашиваем: материал лежит в сценарии.
            return material_payload(self.state.stage)
        if panel == "scenario":
            return {"stages": [{"id": st.id, "goal": st.goal}
                               for st in self.scenario.stages],
                    "current": self.state.stage_index}
        return {}

    def cancel(self, heard_ms: float | None = None) -> str | None:
        """Перебивание: гасим ВСЮ цепочку одним движением.

        `heard_ms` — позиция воспроизведения у клиента в момент нажатия, по
        часам аудио. Это единственный источник правды о том, что человек
        услышал: сервер знает лишь то, что ОТПРАВИЛ, а отправляет он с
        опережением. Без него (скриптовые прогоны, старый клиент) остаётся
        прежнее поведение — считаем услышанным всё отправленное.
        """
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
                clauses = self.spoken.get(gid, [])
                # «Дописана» и «дослушана» — разные вещи, и это оказалось
                # главным в жалобе «перебил на половине, а в контекст ушёл весь
                # абзац». Конвейер отправляет клаузы с опережением и помечает
                # генерацию завершённой, когда отправил ПОСЛЕДНЮЮ, — а звук в
                # этот момент играет вторую. Проверка `gid not in completed`
                # поэтому почти никогда не срабатывала на живом перебивании, и
                # реплика оставалась в истории целиком.
                cut_short = was_cut_short(clauses, heard_ms)
                if gid not in self.completed or cut_short:
                    # Недоговорённое из истории убираем, но НЕ целиком:
                    # прозвучавшую часть пользователь слышал, и агент обязан
                    # её помнить, иначе он переспросит то, на что уже получил
                    # ответ. Возвращаем ровно озвученные клаузы.
                    self.state.drop_generation(gid)
                    said = heard_text(clauses, heard_ms)
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
        # Потолок диалога проверяется и здесь, а не только после удавшейся
        # реплики. Проверка жила в `apply_action`, то есть срабатывала лишь
        # когда генерация доходила до конца, — и разговор, у которого
        # перебиты последние ходы, проезжал мимо неё: замерено на прогоне с
        # 13 перебиваниями, 23 хода при потолке 23 и `finished=False`. Это
        # ровно тот случай, ради которого потолок и заведён: собеседник,
        # перебивающий каждую реплику, иначе не даёт сценарию закончиться.
        self._finish_if_out_of_budget()
        return gid

    def _finish_if_out_of_budget(self) -> bool:
        """Закрыть диалог, если ходы кончились. Идемпотентно."""
        with self.lock:
            if self.state.finished or not self.state.dialogue_budget_spent:
                return False
            self.state.finish("бюджет диалога исчерпан")
        self._emit({"kind": "finished", "reason": self.state.finish_reason})
        return True

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
        # Выводы стоят одного запроса и потому считаются только на `finish`:
        # пока диалог идёт, отчёт опрашивают каждые две секунды.
        texts = (self.evaluator.conclusion(self.state)
                 if self.state.finished else {})
        rep = report_mod.build(self.state, self.evaluator.log,
                               session_id=self.id,
                               conclusion=texts.get("trainee", ""),
                               conclusion_methodist=texts.get("methodist", ""))
        d = rep.to_dict()
        d["scenario"] = self.scenario.to_dict()
        d["marks"] = self.marks
        d["speculation"] = self.spec.stats.summary()
        d["evaluator"] = {"calls": self.evaluator.log.calls,
                          "errors": self.evaluator.log.errors,
                          "total_ms": round(self.evaluator.log.total_ms)}
        if self.state.finished and not self.saved:
            self.saved = report_mod.save(d)
        d["saved_to"] = self.saved
        return d


def _heard_position(clauses: list, heard_ms) -> float | None:
    """Позицию воспроизведения — в число, если ей можно верить.

    None означает «доверять нечему, считай услышанным всё»: так ведут себя
    скриптовые прогоны без звука, старый клиент и сломанные часы.
    """
    if not clauses or not isinstance(clauses[0], dict) or heard_ms is None:
        return None
    try:
        heard = float(heard_ms)
    except (TypeError, ValueError):
        return None
    if heard != heard or heard in (float("inf"), float("-inf")):   # noqa: PLR0124
        return None
    return heard


def was_cut_short(clauses: list, heard_ms) -> bool:
    """Оборвали ли реплику на полуслове.

    Не то же самое, что «генерация не завершилась»: конвейер помечает её
    завершённой, когда ОТПРАВИЛ последнюю клаузу, а звук в этот момент играет
    вторую. Судить надо по часам человека, а не по состоянию отправки.
    """
    heard = _heard_position(clauses, heard_ms)
    if heard is None:
        return False
    end = max(c["start_ms"] + c["audio_ms"] for c in clauses)
    return heard < end


def heard_text(clauses: list, heard_ms: float | None) -> str:
    """Что из отправленного человек успел услышать.

    Клауза засчитывается, если она успела НАЧАТЬСЯ к моменту перебивания.
    Начавшаяся прозвучала хотя бы частично, и агент вправе о ней помнить;
    не начавшаяся не звучала вовсе, и держать её в истории — значит заставлять
    агента продолжать так, будто он договорил.

    Дробить клаузу по середине не пытаемся: у нас нет отображения времени в
    символы внутри неё, а клауза и так короткая — это одна фраза, а не абзац.
    """
    if not clauses:
        return ""
    everything = " ".join(c if isinstance(c, str) else c["text"]
                          for c in clauses).strip()
    # Старый формат, прогоны без звука, сломанные часы: всё отправленное.
    # Безопасная сторона здесь именно эта: обрезать по недостоверным часам
    # значило бы стирать то, что человек на самом деле слышал.
    heard = _heard_position(clauses, heard_ms)
    if heard is None or not was_cut_short(clauses, heard):
        return everything
    return " ".join(c["text"] for c in clauses if c["start_ms"] <= heard).strip()


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
        cfg = load_config()
        # Кто ведёт диалог — строка в конфиге. Три клиента с разными бюджетами
        # токенов: реплика, второй разбор действия (ответ в одно слово) и общий
        # вывод отчёта (два текста, на 300 токенах обрывался на середине JSON).
        dialogue = llm_mod.build_dialogue(cfg.get("dialogue", {}))
        self.dialogue_provider = dialogue["provider"]
        self.models = {
            "tts": SwitchableTTS(tts_config()),
            "aligner": GigaAMAligner(),
            "bridge": VisemeBridge(),
            "llm": dialogue["llm"],
            "repair_llm": dialogue["repair_llm"],
            "summary_llm": dialogue["summary_llm"],
        }
        # Заполнители готовятся здесь и лежат в памяти: по Enter не считается
        # ничего, иначе смысл теряется.
        bc = Backchannel().warm(self.models["tts"],
                                lambda pcm, sr: self.models["aligner"](pcm, sr),
                                self.models["bridge"],
                                cache_key=self.models["tts"].provider)
        self.models["backchannel"] = bc
        # Эндпоинтер не обязателен: без него голосовой ввод просто не включится,
        # а текстовый путь — основной и его судят — не должен от этого страдать.
        voice_cfg = {k: v for k, v in cfg.get("voice", {}).items()
                     if not k.startswith("_")}
        self.models["voice_cfg"] = voice_cfg
        try:
            from app.voice import SileroEndpointer
            self.models["endpointer"] = SileroEndpointer(
                silence_ms=voice_cfg.get("silence_ms", 1000))
            print(f"  голосовой ввод: порог тишины "
                  f"{self.models['endpointer'].silence_ms} мс")
        except Exception as e:                                # noqa: BLE001
            self.models["endpointer"] = None
            print(f"  голосовой ввод недоступен ({type(e).__name__}: {e}); "
                  f"текстовый путь работает как обычно")
        # after_ms — сколько ждать ответа, прежде чем ставить заполнитель.
        # Порог короткий намеренно: полсекунды тишины и есть то, что заполнитель
        # убирает. Случай «ответ пришёл быстро» — это попадание спекуляции с
        # уже готовым результатом, а не ожидание в шестьсот миллисекунд.
        self.models["bc_cfg"] = {k: v for k, v in cfg.get("backchannel", {}).items()
                                 if not k.startswith("_")} or \
            {"enabled": True, "after_ms": 120, "lead_ms": 150}
        self.models["spec_cfg"] = {k: v for k, v in cfg.get("speculation", {}).items()
                                   if not k.startswith("_")}
        print(f"  заполнители: {bc.describe()}")
        self.scenarios = load_all(ROOT / "data" / "scenarios")
        # Текущая сессия остаётся: экран сотрудника, репетиция и стенды
        # обращаются к ней без идентификатора, и ломать это ради ссылки
        # незачем. Реестр рядом — для второго экрана и для истории.
        self.session: Session | None = None
        self.sessions: dict[str, Session] = {}
        # Утверждённые артефакты: замороженный JSON по идентификатору сессии.
        # Словарь в памяти процесса, никакой базы — как и договаривались.
        self.approved: dict[str, dict] = {}
        # Генерации сценариев из текста. Ключ Anthropic отдельный от DeepSeek:
        # диалог и генерация — разные задачи с разными требованиями.
        self.jobs = gen_mod.Jobs(self._make_generator)
        self.gen_cfg = {k: v for k, v in load_config().get("generator", {}).items()
                        if not k.startswith("_")}
        self._generator = None
        self._generator_lock = threading.Lock()
        print(f"  синтез: {self.models['tts'].describe()}")
        # Кто ведёт диалог — на показе это единственный способ заметить, что
        # переключение в конфиге не подхватилось.
        print(f"  диалог: {dialogue['provider']} / {self.models['llm'].model}")
        print(f"готово за {time.perf_counter() - t0:.1f} с, "
              f"сценариев {len(self.scenarios)}")

    def _make_generator(self):
        """Генератор сценариев. Клиент один на процесс, поднимается лениво.

        Лениво — потому что ключа Anthropic может не быть вовсе, а сервер
        обязан подниматься и без него: селектор готовых сценариев работает
        всегда, и это его смысл.
        """
        with self._generator_lock:
            if self._generator is None:
                self._generator = gen_mod.AnthropicGenerator(**self.gen_cfg)
            return self._generator

    def resolve(self, spec) -> Scenario:
        """Сценарий из того, что прислал экран методиста.

        Три источника, и все три нужны: идентификатор из селектора готовых
        (страховка на показе), утверждённый артефакт по ссылке (сотрудник
        открывает свой экран) и сценарий целиком (методист нажал «начать»
        сразу после approve).
        """
        if isinstance(spec, dict):
            return Scenario.from_dict(spec)
        if isinstance(spec, str) and spec in self.approved:
            return Scenario.from_dict(self.approved[spec]["scenario"])
        sc = next((s for s in self.scenarios if s.id == spec), None)
        return (sc or self.scenarios[0]).copy()

    def approve(self, raw: dict) -> dict:
        """Заморозить артефакт и выдать ссылку.

        Approve превращает артефакт в неизменяемый JSON. Правки после него на
        идущий диалог не влияют: сессия работает с копией, снятой здесь, —
        иначе поехала бы метрика завершения, ведь этапы можно и удалить.
        """
        # Ключи и id дописываются здесь: критерий, добавленный методистом
        # руками, приходит без ключа — придумывать идентификаторы его никто
        # не просил.
        sc = Scenario.from_dict(gen_mod.fill_identifiers(raw))
        problems = sc.validate()
        if problems:
            raise ValueError("; ".join(problems))
        sid = uuid.uuid4().hex[:8]
        frozen = sc.to_dict()
        self.approved[sid] = {"scenario": frozen, "at": time.time(),
                              "title": sc.title}
        return {"session": sid, "scenario": frozen,
                "trainee_url": f"/app/web/trainee.html?session={sid}",
                "report_url": f"/app/web/methodist.html?session={sid}"}

    def start(self, spec, criteria_text: str = "", session_id: str = "") -> Session:
        # Свежая копия сценария: критерии методиста не должны протечь в
        # следующую сессию.
        sc = self.resolve(spec).copy()
        sid = session_id or (spec if isinstance(spec, str) and spec in self.approved
                             else uuid.uuid4().hex[:8])
        self.session = Session(sc, criteria_text, self.models, session_id=sid)
        self.sessions[sid] = self.session
        threading.Thread(target=self.session.speak, args=(None,), daemon=True).start()
        return self.session

    def session_for(self, sid: str | None) -> Session | None:
        """Сессия по идентификатору; без него — текущая."""
        return self.sessions.get(sid) if sid else self.session


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

    def _sid(self, u, data: dict | None = None) -> str | None:
        """Идентификатор сессии из запроса. Нет — значит текущая.

        Экран сотрудника и репетиция обращаются без него; ссылка методиста
        приносит его в query. Обе формы обязаны работать.
        """
        if data and data.get("session"):
            return str(data["session"])
        q = urllib.parse.parse_qs(u.query).get("session")
        return q[0] if q else None

    def _need_session(self, u, data=None):
        sess = APP.session_for(self._sid(u, data))
        if sess is None:
            self._json({"error": "сессия не начата"}, 400)
        return sess

    def _need_live_session(self, u, data=None):
        """Сессия, которая ещё принимает реплики.

        Завершение обязано быть настоящим конечным состоянием. Раньше
        `finished` только ставился и уходил кадром — и на этом всё: сессия
        принимала реплики дальше, микрофон жил, тренажёр по факту не
        заканчивался. Отчёт при этом уже показан, то есть человек продолжает
        разговор, итог которого подведён.
        """
        sess = self._need_session(u, data)
        if sess is not None and sess.state.finished:
            self._json({"error": "диалог завершён", "finished": True,
                        "reason": sess.state.finish_reason}, 409)
            return None
        return sess

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/scenarios":
            return self._json({
                "scenarios": [{"id": s.id, "title": s.title, "type": s.type,
                               "stages": len(s.stages), "source": s.source,
                               "persona": s.persona.to_dict(),
                               "criteria": [{"key": c.key, "title": c.title}
                                            for c in s.criteria]}
                              for s in APP.scenarios],
                "types": TYPES,
                # Палитра эмоций отдаётся сервером, а не дублируется в
                # странице. Дубль уже стоил ошибки: страница знала пять
                # состояний, генератор вернул шестое, и селектор молча
                # подставил neutral поверх выбора генератора.
                "emotions": list(EMOTIONS),
            })
        if u.path == "/api/scenario":
            sid = urllib.parse.parse_qs(u.query).get("id", [""])[0]
            sc = next((x for x in APP.scenarios if x.id == sid), None)
            if sc is None:
                return self._json({"error": "нет такого сценария"}, 404)
            return self._json({"scenario": sc.to_dict()})
        if u.path == "/api/generate":
            job = APP.jobs.get(urllib.parse.parse_qs(u.query).get("id", [""])[0])
            if job is None:
                return self._json({"error": "нет такой генерации"}, 404)
            return self._json(job.to_dict())
        if u.path == "/api/artifact":
            # Утверждённый артефакт по ссылке: это открывает экран сотрудника.
            sid = urllib.parse.parse_qs(u.query).get("session", [""])[0]
            art = APP.approved.get(sid)
            if art is None:
                return self._json({"error": "нет такой сессии"}, 404)
            return self._json({"session": sid, **art,
                               "started": sid in APP.sessions})
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
                               "dialogue": {"provider": APP.dialogue_provider,
                                            "model": APP.models["llm"].model},
                               "scenarios": len(APP.scenarios),
                               "voice": APP.models.get("endpointer") is not None,
                               "generator": bool(os.environ.get("ANTHROPIC_API_KEY")),
                               "sessions": len(APP.sessions),
                               "session": bool(APP.session)})
        if u.path == "/api/report":
            sess = self._need_session(u)
            return None if sess is None else self._json(sess.report())
        if u.path == "/api/stream":
            q = urllib.parse.parse_qs(u.query)
            return self._stream(int(q.get("from", ["0"])[0]), self._sid(u))
        return super().do_GET()

    def _stream(self, start: int, sid: str | None = None):
        session = APP.session_for(sid)
        if session is None:
            return self._json({"error": "сессия не начата"}, 400)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        idx = start
        try:
            while True:
                batch = session.frames_from(idx)
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

        # Звук приходит сырым PCM16, а не JSON: разбирать его как текст нельзя,
        # поэтому ветка стоит до общего разбора тела.
        if u.path == "/api/audio":
            sess = self._need_live_session(u)
            if sess is None:
                return None
            pcm = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768
            return self._json(sess.push_audio(pcm))

        data = json.loads(body or b"{}")

        # ------------------------------------------------------- артефакт
        if u.path == "/api/generate":
            text = (data.get("text") or "").strip()
            if len(text) < 40:
                return self._json({"error": "текст слишком короткий: из двух "
                                            "строк сценарий не выводится"}, 400)
            job = APP.jobs.start(text, data.get("kind", ""))
            return self._json(job.to_dict())
        if u.path == "/api/refine":
            # Чат-слой: обёртка над артефактом, а не отдельный продукт.
            # Синхронный запрос — правка занимает столько же, сколько
            # генерация, и ждать её методист согласен, глядя на артефакт.
            msg = (data.get("message") or "").strip()
            if not msg:
                return self._json({"error": "пустая просьба"}, 400)
            try:
                out = gen_mod.refine(APP._make_generator(), data.get("scenario") or {},
                                     msg, data.get("protect") or [])
            except Exception as e:                          # noqa: BLE001
                return self._json({"error": f"{type(e).__name__}: {e}"}, 400)
            summary = "готово"
            if out["lost"]:
                summary = ("артефакт перегенерирован; состав этапов или критериев "
                           "изменился, поэтому часть ваших правок вернуть на место "
                           "не удалось — проверьте: " + ", ".join(out["lost"]))
            return self._json({**out, "summary": summary})
        if u.path == "/api/approve":
            try:
                return self._json(APP.approve(data.get("scenario") or {}))
            except (ValueError, KeyError, TypeError) as e:
                return self._json({"error": f"артефакт не проходит проверку: {e}"}, 400)

        if u.path == "/api/start":
            # Сценарий приходит либо целиком (утверждённый артефакт), либо
            # идентификатором из селектора готовых, либо идентификатором
            # утверждённой сессии.
            spec = data.get("artifact") or data.get("session") or data.get("scenario")
            sess = APP.start(spec, data.get("criteria", ""))
            return self._json({"ok": True, "session": sess.id,
                               "scenario": sess.scenario.to_dict()})
        if u.path == "/api/message":
            sess = self._need_live_session(u, data)
            if sess is None:
                return None
            # Enter во время речи агента = перебивание. Одно движение гасит всё.
            # `heard_ms` — сколько человек успел услышать по своим часам аудио.
            gid = sess.cancel(data.get("heard_ms"))
            threading.Thread(target=sess.speak,
                             args=(data.get("text", ""),), daemon=True).start()
            return self._json({"ok": True, "cancelled": gid})
        if u.path == "/api/typing":
            sess = APP.session_for(self._sid(u, data))
            if sess is None or sess.state.finished:
                # Спекуляция в законченной сессии — потраченные токены на
                # реплику, которую никто не примет.
                return self._json({"ok": False})
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
        if u.path == "/api/voice":
            # Выключить микрофон в законченной сессии можно всегда: это уборка,
            # а не реплика. Включить — уже нет.
            sess = (self._need_session(u, data) if data.get("on") is False
                    else self._need_live_session(u, data))
            if sess is None:
                return None
            v = sess.voice
            if v.endpointer is None:
                return self._json({"error": "эндпоинтер не поднялся: "
                                            "нет пакета silero-vad"}, 400)
            if "on" in data:
                v.enabled = bool(data["on"])
                v.start()
            if data.get("ptt") == "up":
                return self._json(sess.end_utterance())
            return self._json({"ok": True, "on": v.enabled,
                               "silence_ms": v.endpointer.silence_ms,
                               "stats": v.stats})
        if u.path == "/api/cancel":
            sess = self._need_session(u, data)
            return None if sess is None else self._json(
                {"ok": True, "cancelled": sess.cancel(data.get("heard_ms"))})
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
