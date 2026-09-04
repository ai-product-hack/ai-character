"""Динамика набора как сигнал.

Текстовый ввод даёт то, чего голос не давал: видно, как человек думает. Время
до первого нажатия, паузы внутри ответа, стирания, число правок — всё это
собирается бесплатно и говорит об уверенности больше, чем сам текст.

Два применения:

* в отчёт — сигнал уверенности по каждому ответу;
* в поведение — состояние `listening` умеет нарастающее нетерпение, но пока
  считает его по таймеру. Человек печатает — персонаж ждёт спокойно; человек
  замер — начинает проявлять нетерпение.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TypingSample:
    """Одно наблюдение за набором, как его прислал клиент."""
    at_ms: float                   # от появления реплики агента
    length: int                    # длина текста в этот момент


@dataclass
class TypingSignal:
    """Сводка по одному ответу пользователя."""
    first_key_ms: float | None = None
    total_ms: float = 0.0
    final_length: int = 0
    max_length: int = 0
    deletions: int = 0             # сколько раз текст становился короче
    deleted_chars: int = 0
    pauses: list[float] = field(default_factory=list)
    samples: int = 0

    @property
    def longest_pause_ms(self) -> float:
        return max(self.pauses) if self.pauses else 0.0

    @property
    def chars_per_sec(self) -> float:
        if self.total_ms <= 0:
            return 0.0
        return round(self.final_length / (self.total_ms / 1000), 2)

    @property
    def rewrite_ratio(self) -> float:
        """Доля стёртого от написанного. Чем выше, тем больше человек правил."""
        return round(self.deleted_chars / self.max_length, 3) if self.max_length else 0.0

    @property
    def confidence(self) -> str:
        """Грубая метка уверенности.

        Намеренно грубая: это сигнал для методиста, а не оценка. Три исхода
        читаются с одного взгляда, пять уже требуют легенды.
        """
        if self.samples < 2:
            return "нет данных"
        slow_start = (self.first_key_ms or 0) > 4000
        long_pause = self.longest_pause_ms > 3000
        heavy_edit = self.rewrite_ratio > 0.25
        score = sum((slow_start, long_pause, heavy_edit))
        return ("уверенно", "с заминками", "неуверенно")[min(score, 2)]

    def to_dict(self) -> dict:
        return {
            "first_key_ms": round(self.first_key_ms) if self.first_key_ms else None,
            "total_ms": round(self.total_ms),
            "final_length": self.final_length,
            "deletions": self.deletions,
            "deleted_chars": self.deleted_chars,
            "longest_pause_ms": round(self.longest_pause_ms),
            "chars_per_sec": self.chars_per_sec,
            "rewrite_ratio": self.rewrite_ratio,
            "confidence": self.confidence,
        }


class TypingTracker:
    """Копит наблюдения по текущему ответу и подводит итог по отправке."""

    def __init__(self, pause_threshold_ms: float = 1200):
        self.pause_threshold_ms = pause_threshold_ms
        self.samples: list[TypingSample] = []
        self._last_at: float | None = None
        self._last_len = 0

    def reset(self) -> None:
        self.samples.clear()
        self._last_at = None
        self._last_len = 0

    def observe(self, at_ms: float, length: int) -> None:
        self.samples.append(TypingSample(at_ms, length))
        self._last_at = at_ms
        self._last_len = length

    @property
    def idle_ms(self) -> float:
        """Сколько прошло с последнего наблюдения. Нужно нетерпению."""
        return 0.0 if self._last_at is None else self._last_at

    def summarise(self, final_length: int | None = None,
                  total_ms: float | None = None) -> TypingSignal:
        s = TypingSignal()
        s.samples = len(self.samples)
        if not self.samples:
            s.final_length = final_length or 0
            return s
        s.first_key_ms = self.samples[0].at_ms
        s.final_length = final_length if final_length is not None else self.samples[-1].length
        s.max_length = max(x.length for x in self.samples)
        s.total_ms = total_ms if total_ms is not None else \
            (self.samples[-1].at_ms - self.samples[0].at_ms)
        prev = self.samples[0]
        for cur in self.samples[1:]:
            gap = cur.at_ms - prev.at_ms
            if gap >= self.pause_threshold_ms:
                s.pauses.append(gap)
            if cur.length < prev.length:
                s.deletions += 1
                s.deleted_chars += prev.length - cur.length
            prev = cur
        return s
