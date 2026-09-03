"""Нарезка потока модели на клаузы.

Первая клауза короткая — до первой границы предложения или ~50 символов, что
раньше. Это условие быстрого первого звука: синтез стоит 13 мс медианы, а вот
ждать конца всей реплики значит ждать генерацию целиком.

Дальше клаузы могут быть длиннее: слушатель уже слышит речь, и запас времени
появился.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SENTENCE_END = ".!?…"
CLAUSE_END = ",;:—–"


@dataclass
class Clause:
    index: int
    text: str
    first: bool


class ClauseSplitter:
    """Скармливаешь токены — отдаёт готовые клаузы.

    Состояние живёт в объекте, потому что токены приходят кусками произвольной
    длины и граница может оказаться внутри одного куска.
    """

    def __init__(self, first_max_chars: int = 50, min_chars: int = 24,
                 max_chars: int = 220, first_min_chars: int = 5):
        self.first_max_chars = first_max_chars
        # У первой клаузы порог ниже общего: ей можно и нужно быть короткой.
        # Но не вырожденной — «Ну…» в начале реплики это заминка, а не
        # предложение, и отдельным куском синтеза она звучит как обрубок.
        self.first_min_chars = first_min_chars
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.buf = ""
        self.emitted = 0

    def push(self, token: str) -> list[Clause]:
        """Добавить токен, вернуть готовые клаузы (обычно ноль или одну)."""
        self.buf += token
        out = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            text, self.buf = self.buf[:cut].strip(), self.buf[cut:].lstrip()
            if text:
                out.append(Clause(self.emitted, text, self.emitted == 0))
                self.emitted += 1
        return out

    def flush(self) -> list[Clause]:
        """Хвост после конца потока."""
        text, self.buf = self.buf.strip(), ""
        if not text:
            return []
        c = Clause(self.emitted, text, self.emitted == 0)
        self.emitted += 1
        return [c]

    def _find_cut(self) -> int | None:
        buf = self.buf
        limit = self.first_max_chars if self.emitted == 0 else self.max_chars

        # Граница предложения — лучшее место для разреза: там и так пауза.
        floor = self.first_min_chars if self.emitted == 0 else self.min_chars
        for i, ch in enumerate(buf):
            if ch in SENTENCE_END:
                # Многоточие и «?!» режем целиком, а не по первому знаку.
                j = i
                while j + 1 < len(buf) and buf[j + 1] in SENTENCE_END:
                    j += 1
                if j + 1 >= len(buf):
                    return None          # знак может быть не последним, ждём
                if j + 1 > limit:
                    # Граница есть, но она за лимитом длины: ждать её значит
                    # отдать первый звук слишком поздно. Дальше по длине.
                    break
                if j + 1 >= floor:
                    return j + 1

        if len(buf) < limit:
            return None

        # Предложение длиннее лимита: режем по ближайшей внутренней границе,
        # чтобы не рвать посреди слова — рваная клауза слышна в синтезе.
        # Для ПЕРВОЙ клаузы окно строго по лимиту: она задаёт время до первого
        # звука, и перебор в сорок символов съедает как раз то, ради чего её
        # и режут коротко.
        window = buf[:limit] if self.emitted == 0 else buf[:limit + 40]
        best = max((window.rfind(ch) for ch in CLAUSE_END), default=-1)
        if best >= self.min_chars:
            return best + 1
        space = window.rfind(" ", self.min_chars)
        if space > 0:
            return space + 1
        return None


def split_text(text: str, **kw) -> list[Clause]:
    """Разрезать готовый текст. Для тестов и для нестримового пути."""
    sp = ClauseSplitter(**kw)
    out = sp.push(text)
    return out + sp.flush()
