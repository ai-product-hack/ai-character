"""Бэкчэннел: короткая реплика-заполнитель, уходящая в плеер сразу по Enter.

Смысл не в вежливости, а в бюджете. Первый звук упирается в TTFT провайдера
(2.25 с из 2.57), и своими силами конвейер это не исправит: синтез клаузы стоит
36 мс, выравнивание 64. Заполнитель убирает ожидание из восприятия целиком —
человек слышит ответ через десятки миллисекунд, а содержательная реплика
приходит следом.

Побочный, но не менее важный эффект: TTFB синтезатора перестаёт быть статьёй
бюджета первого звука. Без этого выбор голоса прикован к офлайновым 9 мс Silero
и сетевой синтез рассматривать нельзя.

Всё готовится на старте сервиса: PCM, посимвольные таймкоды и трек висем лежат
в памяти. По Enter не считается ничего.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field


@dataclass
class Filler:
    """Готовый заполнитель: звук и мимика к нему."""
    text: str
    pcm: object                    # np.ndarray float32
    sample_rate: int
    audio_ms: float
    visemes: list[dict] = field(default_factory=list)
    chars: list[dict] = field(default_factory=list)


class Backchannel:
    """Набор заполнителей, синтезированных заранее.

    `pick()` не повторяет предыдущий: одно и то же «угу» дважды подряд слышно
    как заедание.
    """

    DEFAULT = ["Так…", "Угу.", "Понятно.", "Хорошо.", "Ага.", "Так, ясно."]

    def __init__(self, texts: list[str] | None = None, seed: int | None = None):
        self.texts = list(texts or self.DEFAULT)
        self.fillers: list[Filler] = []
        self._last = -1
        self._rnd = random.Random(seed)
        self.warmup_ms = 0.0

    def warm(self, tts, align, to_visemes) -> "Backchannel":
        """Синтезировать и разметить всё заранее. Зовётся один раз на старте."""
        t0 = time.perf_counter()
        for text in self.texts:
            pcm, sr = tts(text)
            chars = align(pcm, sr)
            self.fillers.append(Filler(
                text=text, pcm=pcm, sample_rate=sr,
                audio_ms=len(pcm) / sr * 1000,
                visemes=to_visemes(chars), chars=chars,
            ))
        self.warmup_ms = (time.perf_counter() - t0) * 1000
        return self

    @property
    def ready(self) -> bool:
        return bool(self.fillers)

    def pick(self) -> Filler | None:
        if not self.fillers:
            return None
        if len(self.fillers) == 1:
            return self.fillers[0]
        i = self._last
        while i == self._last:
            i = self._rnd.randrange(len(self.fillers))
        self._last = i
        return self.fillers[i]

    def describe(self) -> dict:
        return {
            "count": len(self.fillers),
            "warmup_ms": round(self.warmup_ms),
            "items": [{"text": f.text, "audio_ms": round(f.audio_ms),
                       "visemes": len(f.visemes)} for f in self.fillers],
        }
