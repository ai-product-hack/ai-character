"""Спекулятивная генерация на недопечатанном сообщении.

Прямой перенос механики из S1: там запрос к модели уходил по растущему
транскрипту речи и перезапускался, когда буфер вырастал на 40%. Здесь ровно то
же, только буфер растёт от нажатий клавиш.

Условия лучше, чем в R-фазе: печатают 5–10 секунд, а не говорят 2, значит
перезапусков успевает пройти больше и покрытие финального текста выше.

Смысл: TTFT провайдера — 2.25 с из 2.57 с первого звука. Если запрос ушёл, пока
человек ещё печатал, к нажатию Enter ответ уже в пути.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class SpecStats:
    launches: int = 0
    relaunches: int = 0
    hits: int = 0
    misses: int = 0
    last_cover: float = 0.0
    ttft_hit_ms: list[float] = field(default_factory=list)
    ttft_miss_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        med = lambda a: round(sorted(a)[len(a) // 2]) if a else None   # noqa: E731
        total = self.hits + self.misses
        return {
            "launches": self.launches, "relaunches": self.relaunches,
            "hits": self.hits, "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else None,
            "ttft_hit_ms": med(self.ttft_hit_ms),
            "ttft_miss_ms": med(self.ttft_miss_ms),
        }


class InFlight:
    """Запрос, уже летящий к модели. Токены копятся, пока их никто не читает."""

    def __init__(self, stream, system: str, prompt: str, user_text: str):
        self.stream = stream
        self.user_text = user_text
        self.tokens: list[str] = []
        self.done = threading.Event()
        self.error: Exception | None = None
        self.t_start = time.perf_counter()
        self.t_first_token: float | None = None
        self._cv = threading.Condition()
        self._cancelled = False
        self.thread = threading.Thread(
            target=self._run, args=(system, prompt), daemon=True)
        self.thread.start()

    def _run(self, system: str, prompt: str) -> None:
        try:
            for tok in self.stream(system, prompt):
                if self._cancelled:
                    break
                with self._cv:
                    if self.t_first_token is None:
                        self.t_first_token = (time.perf_counter() - self.t_start) * 1000
                    self.tokens.append(tok)
                    self._cv.notify_all()
        except Exception as e:                                  # noqa: BLE001
            self.error = e
        finally:
            self.done.set()
            with self._cv:
                self._cv.notify_all()

    def cancel(self) -> None:
        self._cancelled = True
        try:
            self.stream.cancel()
        except AttributeError:
            pass
        with self._cv:
            self._cv.notify_all()

    def replay(self):
        """Генератор: сначала накопленное, потом всё новое до конца потока.

        Именно им подменяется обычный поток в конвейере, поэтому конвейер о
        спекуляции ничего не знает.
        """
        i = 0
        while True:
            with self._cv:
                while i >= len(self.tokens) and not self.done.is_set():
                    self._cv.wait(timeout=0.2)
                if i >= len(self.tokens):
                    if self.done.is_set():
                        return
                    continue
                tok = self.tokens[i]
                i += 1
            yield tok


class Speculator:
    """Один спекулятивный запрос в полёте на сессию.

    В полёте держится только самый свежий: старые отменяются. Один ранний
    запуск бесполезен — к моменту отправки его промпт покрывает лишь начало
    сообщения, и результат приходится выбрасывать.
    """

    def __init__(self, make_stream, build_prompt, system: str, cfg: dict | None = None):
        self.make_stream = make_stream
        self.build_prompt = build_prompt
        self.system = system
        cfg = cfg or {}
        self.min_chars = cfg.get("min_chars", 15)
        self.idle_ms = cfg.get("idle_ms", 400)
        self.growth = cfg.get("relaunch_growth", 0.4)
        self.reuse_cover = cfg.get("reuse_cover", 0.6)
        self.enabled = cfg.get("enabled", True)

        self.flight: InFlight | None = None
        self.stats = SpecStats()
        self._last_typed_at = 0.0
        self._pending_text = ""
        self._lock = threading.Lock()

    # ------------------------------------------------------------- печатание

    def on_typing(self, text: str) -> bool:
        """Пользователь напечатал. Возвращает True, если запустили запрос."""
        if not self.enabled:
            return False
        text = (text or "").strip()
        now = time.perf_counter()
        with self._lock:
            self._pending_text = text
            self._last_typed_at = now
            if len(text) < self.min_chars:
                return False
            cur = self.flight
            if cur is not None and cur.user_text == text:
                return False
            # Перезапуск только когда буфер заметно вырос: иначе каждый символ
            # порождал бы новый запрос.
            if cur is not None and len(text) < len(cur.user_text) * (1 + self.growth):
                return False
            if cur is not None:
                cur.cancel()
                self.stats.relaunches += 1
            self.flight = InFlight(self.make_stream(), self.system,
                                   self.build_prompt(text), text)
            self.stats.launches += 1
            return True

    def idle_enough(self) -> bool:
        """Пауза в наборе — сигнал, что мысль дописана."""
        return (time.perf_counter() - self._last_typed_at) * 1000 >= self.idle_ms

    # -------------------------------------------------------------- отправка

    def take(self, final_text: str) -> tuple[InFlight | None, float]:
        """Забрать результат под финальный текст.

        Годится только если ранний промпт — префикс финального и покрыл его
        заметную часть. Иначе платим потраченными токенами, а не задержкой.
        """
        final = (final_text or "").strip()
        with self._lock:
            cur, self.flight = self.flight, None
            if cur is None:
                self.stats.misses += 1
                return None, 0.0
            cover = len(cur.user_text) / max(1, len(final))
            self.stats.last_cover = cover
            if final.startswith(cur.user_text) and cover >= self.reuse_cover:
                self.stats.hits += 1
                return cur, cover
            cur.cancel()
            self.stats.misses += 1
            return None, cover

    def note_ttft(self, ms: float, hit: bool) -> None:
        """TTFT считается ОТ НАЖАТИЯ ENTER, а не от запуска спекулятивного
        запроса. Внутреннее время полёта тут ни при чём: при попадании токены
        уже накоплены, и первый из них приходит мгновенно — а нас интересует,
        сколько ждал человек."""
        (self.stats.ttft_hit_ms if hit else self.stats.ttft_miss_ms).append(ms)

    def drop(self) -> None:
        """Погасить всё, что в полёте. Зовётся при перебивании."""
        with self._lock:
            if self.flight is not None:
                self.flight.cancel()
                self.flight = None
