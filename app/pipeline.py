"""Конвейер реплики: LLM -> клаузы -> TTS -> выравнивание -> висемы и субтитры.

Каждая клауза проходит путь целиком и уходит клиенту, не дожидаясь остальных:
первый звук должен прозвучать, пока модель ещё договаривает.

Что здесь важно и почему:

* PTS считается от начала генерации, смещение берётся из ФАКТИЧЕСКОЙ длины
  аудио. Ожидаемая длительность из длины текста не выводится, и любая оценка
  копила бы дрейф от клаузы к клаузе.
* Принадлежность генерации проверяется перед КАЖДЫМ шагом, а не только на
  входе: клауза, начатая до перебивания, доедет до выравнивания уже после него.
* Синтез и выравнивание идут в отдельном потоке: они блокирующие, а поток
  токенов от модели ждать не должен.
"""
from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass, field

from .actions import strip_control
from .clauses import Clause, ClauseSplitter
from .generation import Generation, GenerationRegistry, PTSTimeline


@dataclass
class ClauseResult:
    """Готовая клауза: звук, висемы и субтитр на общем таймлайне."""
    generation_id: str
    index: int
    text: str
    first: bool
    start_ms: float
    audio_ms: float
    pcm: object                     # np.ndarray float32
    sample_rate: int
    visemes: list[dict]
    chars: list[dict]
    timings: dict = field(default_factory=dict)


class ReplyPipeline:
    """Один проход по реплике агента.

    `llm_stream(system, prompt)` — генератор токенов.
    `tts(text)` -> (pcm, sample_rate)
    `align(pcm, sample_rate)` -> список {ch, ms}
    `to_visemes(chars)` -> список {pts_ms, viseme, ...}
    """

    def __init__(self, llm_stream, tts, align, to_visemes,
                 registry: GenerationRegistry, splitter_kw: dict | None = None):
        self.llm_stream = llm_stream
        self.tts = tts
        self.align = align
        self.to_visemes = to_visemes
        self.registry = registry
        self.splitter_kw = splitter_kw or {}
        # Сырой ответ последней реплики: по нему разбирается действие агента.
        self.last_raw = ""
        self.last_stats = {}
        self.last_timeline = None

    def run(self, system: str, prompt: str, gen: Generation,
            on_result=None, on_token=None) -> list[ClauseResult]:
        """Отработать реплику целиком. `on_result` зовётся на каждой готовой
        клаузе — это и есть путь к плееру."""
        splitter = ClauseSplitter(**self.splitter_kw)
        timeline = PTSTimeline()
        results: list[ClauseResult] = []

        # Очередь клауз в рабочий поток: синтез блокирующий, а токены ждать
        # его не должны.
        work: queue.Queue = queue.Queue()
        done = threading.Event()
        t_start = time.perf_counter()
        stats = {"t_first_token": None, "t_first_audio": None, "tokens": 0}
        # Сырой ответ модели копится отдельно от речи. Разделение обязательное:
        # из клауз управляющий блок вырезан (иначе он звучал бы вслух), и
        # разбирать действие по ним значит не находить его никогда. Стоило
        # 104 хода подряд с действием stay — сценарии доходили до конца
        # исключительно принудительными переходами по бюджету этапа.
        raw_parts: list[str] = []

        def worker():
            while True:
                item = work.get()
                if item is None:
                    break
                clause: Clause = item
                if not gen.check():
                    # Генерация отменена, пока клауза ждала очереди.
                    work.task_done()
                    continue
                try:
                    r = self._process(clause, gen, timeline, t_start, stats)
                except Exception as e:                    # noqa: BLE001
                    r = None
                    stats.setdefault("errors", []).append(f"{type(e).__name__}: {e}")
                if r is not None and gen.check():
                    results.append(r)
                    if on_result:
                        on_result(r)
                work.task_done()
            done.set()

        th = threading.Thread(target=worker, daemon=True)
        th.start()

        try:
            for token in self.llm_stream(system, prompt):
                if not gen.check():
                    break
                if stats["t_first_token"] is None:
                    stats["t_first_token"] = (time.perf_counter() - t_start) * 1000
                stats["tokens"] += 1
                raw_parts.append(token)
                if on_token:
                    on_token(token)
                for c in splitter.push(token):
                    work.put(c)
            if gen.check():
                for c in splitter.flush():
                    work.put(c)
        finally:
            # Гасим и сам запрос к модели, а не только доставку кадров: иначе
            # поток стоит в блокирующем чтении до первого токена.
            if not gen.check() and hasattr(self.llm_stream, "cancel"):
                self.llm_stream.cancel()
            work.put(None)
            done.wait(timeout=30)

        gen.clauses_done = len(results)
        gen.audio_ms = timeline.total_ms
        self.last_stats = stats
        self.last_timeline = timeline
        self.last_raw = "".join(raw_parts)
        return results

    def _process(self, clause: Clause, gen: Generation, timeline: PTSTimeline,
                 t_start: float, stats: dict) -> ClauseResult | None:
        # Управляющий JSON не должен попасть в синтез. В нестримовом пути он
        # срезался из целого ответа, но здесь клаузы уходят в TTS по мере
        # готовности, и хвостовой блок приезжает приклеенным к последней —
        # агент буквально произносил бы «фигурная скобка action next stage».
        text = strip_control(clause.text)
        if not text:
            stats["control_only_clauses"] = stats.get("control_only_clauses", 0) + 1
            return None
        if text != clause.text:
            stats["clauses_with_control"] = stats.get("clauses_with_control", 0) + 1
        clause = Clause(clause.index, text, clause.first)

        t0 = time.perf_counter()
        pcm, sr = self.tts(clause.text)
        t_tts = (time.perf_counter() - t0) * 1000
        if not gen.check():
            return None                     # отменили, пока синтезировали

        t1 = time.perf_counter()
        chars = self.align(pcm, sr)
        t_align = (time.perf_counter() - t1) * 1000
        if not gen.check():
            return None                     # отменили, пока выравнивали

        visemes = self.to_visemes(chars)
        audio_ms = len(pcm) / sr * 1000
        start_ms, shifted = timeline.add_clause(audio_ms, visemes, clause.text)

        if stats["t_first_audio"] is None:
            stats["t_first_audio"] = (time.perf_counter() - t_start) * 1000

        return ClauseResult(
            generation_id=gen.id, index=clause.index, text=clause.text,
            first=clause.first, start_ms=start_ms, audio_ms=audio_ms,
            pcm=pcm, sample_rate=sr, visemes=shifted, chars=chars,
            timings={"tts_ms": round(t_tts), "align_ms": round(t_align)},
        )


def subtitle_cues(result: ClauseResult, chars: list[dict] | None = None) -> list[dict]:
    """Субтитры из тех же таймкодов, по которым едут висемы.

    Отдельного выравнивания не требуется, поэтому пункт почти бесплатный:
    расшифровка уже посимвольно привязана ко времени, остаётся разложить
    исходный текст клаузы по словам и раздать им таймкоды.
    """
    src = chars if chars is not None else result.chars
    if not src:
        return [{"pts_ms": result.start_ms, "text": result.text,
                 "clause": result.index, "generation_id": result.generation_id}]

    # Слова расшифровки с их временем начала.
    spoken, cur, start = [], "", None
    for c in src:
        if c["ch"] == " ":
            if cur:
                spoken.append((start, cur))
                cur, start = "", None
            continue
        if start is None:
            start = c["ms"]
        cur += c["ch"]
    if cur:
        spoken.append((start, cur))

    # Слова оригинала: их и показываем — с пунктуацией и заглавными,
    # которых в расшифровке нет.
    original = re.findall(r"\S+", result.text)
    cues = []
    for i, word in enumerate(original):
        ms = spoken[i][0] if i < len(spoken) else (
            spoken[-1][0] if spoken else 0)
        cues.append({
            "pts_ms": result.start_ms + ms,
            "text": word,
            "clause": result.index,
            "generation_id": result.generation_id,
        })
    return cues
