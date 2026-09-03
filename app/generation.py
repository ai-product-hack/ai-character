"""Сквозной generation_id.

Раньше он жил в трёх подсистемах независимо: спайк конвейера, плеер и аватар
каждый вёл свой счётчик. Здесь источник один на сессию, и отмена гасит ВСЮ
цепочку: запрос к модели, синтез уже улетевшей вперёд клаузы, выравнивание,
плеер, аватар, субтитры.

Проверка принадлежности дешёвая и делается на каждом шаге конвейера, а не
только на входе: клауза, которую начали синтезировать до отмены, доедет до
выравнивания уже после неё.
"""
from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field


@dataclass
class Generation:
    """Одна реплика агента от запроса к модели до последнего кадра висем."""
    id: str
    session: "GenerationRegistry"
    cancelled: bool = False
    t0: float | None = None          # якорь аудио, секунды часов плеера
    clauses_done: int = 0
    audio_ms: float = 0.0            # фактическая длительность уже синтезированного

    @property
    def alive(self) -> bool:
        return not self.cancelled and self.session.current_id == self.id

    def check(self) -> bool:
        """Жива ли генерация. Зовётся перед каждым шагом конвейера."""
        return self.alive


class GenerationRegistry:
    """Единственный источник generation_id на сессию."""

    def __init__(self, prefix: str = "gen"):
        self._counter = itertools.count(1)
        self._prefix = prefix
        self._lock = threading.Lock()
        self.current_id: str | None = None
        self._generations: dict[str, Generation] = {}
        self.cancelled_ids: list[str] = []

    def start(self) -> Generation:
        """Начать новую генерацию. Прежняя тем самым становится чужой."""
        with self._lock:
            prev = self.current_id
            gid = f"{self._prefix}-{next(self._counter)}"
            self.current_id = gid
            g = Generation(gid, self)
            self._generations[gid] = g
            if prev and prev in self._generations:
                self._generations[prev].cancelled = True
                self.cancelled_ids.append(prev)
            return g

    def cancel(self, gid: str | None = None) -> str | None:
        """Погасить генерацию. Без аргумента — текущую."""
        with self._lock:
            target = gid or self.current_id
            if not target or target not in self._generations:
                return None
            self._generations[target].cancelled = True
            self.cancelled_ids.append(target)
            if self.current_id == target:
                self.current_id = None
            return target

    def get(self, gid: str) -> Generation | None:
        return self._generations.get(gid)

    def alive(self, gid: str | None) -> bool:
        """Годится ли этот id прямо сейчас. Всё остальное отбрасывается."""
        return bool(gid) and gid == self.current_id and \
            not self._generations[gid].cancelled


class PTSTimeline:
    """Склейка PTS через клаузы.

    Главный источник накапливающегося дрейфа: каждая клауза синтезируется и
    выравнивается отдельно, а `pts_ms` должен считаться ОТ НАЧАЛА ГЕНЕРАЦИИ.

    Смещение берётся из ФАКТИЧЕСКОЙ длины уже синтезированного аудио, а не из
    ожидаемой: длительность синтеза не выводится из длины текста, и любая
    оценка разъедется тем быстрее, чем длиннее реплика.
    """

    def __init__(self):
        self.offset_ms = 0.0
        self.clauses: list[dict] = []

    def add_clause(self, audio_ms: float, visemes: list[dict],
                   text: str = "") -> tuple[float, list[dict]]:
        """Положить клаузу на общий таймлайн.

        Возвращает (смещение клаузы, сдвинутый трек висем).
        """
        start = self.offset_ms
        shifted = [{**v, "pts_ms": v["pts_ms"] + start} for v in visemes]
        self.clauses.append({
            "index": len(self.clauses), "text": text,
            "start_ms": start, "audio_ms": audio_ms,
            "end_ms": start + audio_ms, "visemes": len(shifted),
        })
        self.offset_ms += audio_ms
        return start, shifted

    @property
    def total_ms(self) -> float:
        return self.offset_ms

    def gap_before(self, index: int) -> float:
        """Пауза перед клаузой: расстояние от конца предыдущей до её начала.

        Между клаузами рот не должен захлопываться в ноль, если пауза короткая,
        поэтому конвейеру нужно знать её величину.
        """
        if index <= 0 or index >= len(self.clauses):
            return 0.0
        return self.clauses[index]["start_ms"] - self.clauses[index - 1]["end_ms"]
