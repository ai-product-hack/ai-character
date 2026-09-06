"""Заглушка модели для тестов и для прогона движка без сети.

Не пытается быть умной: её задача — вести сценарий по прямой и, по требованию
теста, возвращать мусор вместо действия, чтобы проверить откат.
"""
from __future__ import annotations

import itertools
import json
import re


class StubLLM:
    """Проходит сценарий насквозь: пара реплик на этап, потом дальше.

    `turns_per_stage` управляет тем, сколько раз агент остаётся на этапе
    прежде чем попросить следующий.
    """

    def __init__(self, turns_per_stage: int = 1, evaluate_every: int = 2,
                 garbage_at: set[int] | None = None):
        self.turns_per_stage = max(1, turns_per_stage)
        self.evaluate_every = evaluate_every
        self.garbage_at = garbage_at or set()
        self.calls = 0
        self._on_stage = 0
        self._last_stage = None
        self._counter = itertools.count(1)

    def chat(self, system: str, messages: list[dict]) -> str:
        """Массив ролей — в текст: заглушке важен только последний ход."""
        return self(system, "\n".join(
            m["content"] if isinstance(m["content"], str)
            else " ".join(b.get("text", "") for b in m["content"])
            for m in messages))

    def __call__(self, system: str, prompt: str) -> str:
        self.calls += 1
        n = self.calls

        if n in self.garbage_at:
            # Ровно то, что делает живая модель, когда ей не повезло:
            # обрыв на полуслове, лишний текст, сломанный JSON.
            return 'Хорошо, продолжим. {"action": "next_st'

        stage = self._stage_of(prompt)
        if stage != self._last_stage:
            self._last_stage = stage
            self._on_stage = 0
        self._on_stage += 1

        last_stage = "Это последний этап сценария" in prompt
        text = f"Реплика агента номер {next(self._counter)} на этапе «{stage}»."

        if self.evaluate_every and n % self.evaluate_every == 0:
            key = self._first_criterion(prompt)
            if key:
                return (f'{text}\n{{"action": "evaluate", "criterion": "{key}", '
                        f'"score": 4, "note": "ответ по существу"}}')

        if self._on_stage >= self.turns_per_stage:
            action = "finish" if last_stage else "next_stage"
            return f'{text}\n{{"action": "{action}"}}'
        return f'{text}\n{{"action": "evaluate", "criterion": "нет_такого", "note": "держим этап"}}'

    @staticmethod
    def _stage_of(prompt: str) -> str:
        m = re.search(r"ЭТАП \d+ из \d+: (.+)", prompt)
        return m.group(1).strip() if m else "?"

    @staticmethod
    def _first_criterion(prompt: str) -> str | None:
        m = re.search(r"^- ([a-z_]+): ", prompt, re.M)
        return m.group(1) if m else None


class ScriptedLLM:
    """Отдаёт заранее заданные ответы по списку. Для точечных проверок."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, system: str, messages: list[dict]) -> str:
        """Массив ролей — в текст: заглушке важен только последний ход."""
        return self(system, "\n".join(
            m["content"] if isinstance(m["content"], str)
            else " ".join(b.get("text", "") for b in m["content"])
            for m in messages))

    def __call__(self, system: str, prompt: str) -> str:
        self.calls += 1
        return self.replies[min(self.calls - 1, len(self.replies) - 1)]
