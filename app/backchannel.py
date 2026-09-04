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

from .pronounce import normalize


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
        self._cache: dict[str, list[Filler]] = {}
        self._last = -1
        self._rnd = random.Random(seed)
        self.warmup_ms = 0.0
        self.cache_hit = False

    def warm(self, tts, align, to_visemes,
             cache_key: str | None = None) -> "Backchannel":
        """Синтезировать и разметить всё заранее.

        Зовётся на старте и ещё раз при смене голоса: заполнитель обязан
        звучать тем же голосом, что и реплика за ним, иначе на стыке слышно
        двух разных людей. Поэтому список сначала чистится — повторный вызов
        заменяет заполнители, а не добавляет вторые.
        """
        t0 = time.perf_counter()
        if cache_key is not None and cache_key in self._cache:
            self.fillers = list(self._cache[cache_key])
            self._last = -1
            self.warmup_ms = 0.0
            self.cache_hit = True
            return self

        self.fillers.clear()
        self._last = -1
        self.cache_hit = False
        spoken = [normalize(text) for text in self.texts]
        synthesize_many = getattr(tts, "synthesize_many", None)
        audio = (synthesize_many(spoken) if synthesize_many
                 else [tts(text) for text in spoken])
        for text, (pcm, sr) in zip(self.texts, audio):
            # По умолчанию заполнители чисто русские, но методист может задать
            # свои — правило «в синтез идёт нормализованное» действует и здесь.
            chars = align(pcm, sr)
            audio_ms = len(pcm) / sr * 1000
            visemes = list(to_visemes(chars))
            # Основная реплика может приехать и дополнить тот же трек ещё до
            # конца заполнителя. Без явной тишины её первая висема становится
            # ``b`` после последней висемы «угу», и рендер держит последнюю
            # форму рта до начала ответа. SIL в фактическом конце аудио делает
            # паузу настоящей независимо от того, когда приехал хвост трека.
            if not visemes or visemes[-1].get("viseme") != "SIL":
                visemes.append({"pts_ms": audio_ms, "viseme": "SIL"})
            self.fillers.append(Filler(
                text=text, pcm=pcm, sample_rate=sr,
                audio_ms=audio_ms, visemes=visemes, chars=chars,
            ))
        if cache_key is not None:
            self._cache[cache_key] = list(self.fillers)
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
            "cache_hit": self.cache_hit,
            "items": [{"text": f.text, "audio_ms": round(f.audio_ms),
                       "visemes": len(f.visemes)} for f in self.fillers],
        }
