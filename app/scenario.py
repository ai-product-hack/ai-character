"""Формат сценария и его загрузка.

Сценарий пишет методист, поэтому формат должен быть читаемым руками и
переживать неполноту: отсутствующая подсказка не должна ронять движок.

Четыре сущности:

    Persona    кто ведёт разговор: роль, тон, строгость, давление, эмоция
    Stage      этап: цель, подсказка модели, условие перехода, бюджет ходов
    Criterion  критерий оценки со шкалой и якорями
    Scenario   всё вместе

Условие перехода — это ТЕКСТ для модели, а не предикат. Сопоставление ответа
со сценарием семантическое: строковое сравнение не работает на свободных
формулировках, и это требование зафиксировано ещё в R-фазе (DECISION.md, п. 3).

Персона раньше была одной строкой. Строка годится для промпта и никуда не
годится для редактирования: методист не может поправить «строгость», не
переписав абзац, а эмоциональному слою неоткуда взять стартовое состояние.
Поля разложены, но `str(persona)` по-прежнему даёт тот же абзац — старые
сценарии со строкой в поле `persona` читаются без конвертации.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

from .emotion_tags import DEFAULT as DEFAULT_EMOTION, EMOTIONS


@dataclass(frozen=True)
class Persona:
    """Собеседник, выведенный из текста методиста.

    `start_emotion` берётся из того же белого списка, что и разметка `[emo:...]`
    — эмоциональный слой аватара получает ровно те пять состояний, что и
    раньше, и никакого нового контракта здесь не появляется.
    """
    role: str = ""
    tone: str = ""
    strictness: str = ""       # насколько придирчив, словами
    pressure: str = ""         # чем давит: темпом, недоверием, ценой, эмоцией
    start_emotion: str = DEFAULT_EMOTION

    @staticmethod
    def from_any(x) -> "Persona":
        """Строка или словарь — одинаково. Старые сценарии хранят строку."""
        if isinstance(x, Persona):
            return x
        if isinstance(x, str):
            return Persona(role=x.strip())
        if not isinstance(x, dict):
            return Persona()
        emo = str(x.get("start_emotion", "") or "").strip().lower()
        return Persona(
            role=str(x.get("role", "") or "").strip(),
            tone=str(x.get("tone", "") or "").strip(),
            strictness=str(x.get("strictness", "") or "").strip(),
            pressure=str(x.get("pressure", "") or "").strip(),
            start_emotion=emo if emo in EMOTIONS else DEFAULT_EMOTION,
        )

    def to_dict(self) -> dict:
        return {"role": self.role, "tone": self.tone, "strictness": self.strictness,
                "pressure": self.pressure, "start_emotion": self.start_emotion}

    def prompt_block(self) -> str:
        """Абзац для системного промпта сессии."""
        parts = [self.role or "собеседник в тренировочном диалоге"]
        if self.tone:
            parts.append(f"Тон: {self.tone}.")
        if self.strictness:
            parts.append(f"Строгость: {self.strictness}.")
        if self.pressure:
            parts.append(f"Давление: {self.pressure}.")
        return " ".join(parts)

    def __str__(self) -> str:
        return self.prompt_block()

    @property
    def empty(self) -> bool:
        return not any((self.role, self.tone, self.strictness, self.pressure))


@dataclass(frozen=True)
class Stage:
    id: str
    goal: str
    hint: str = ""            # подсказка модели: что делать на этом этапе
    advance_when: str = ""    # при каком ответе пора дальше, словами
    opening: str = ""         # реплика-затравка, если методист её задал
    # Бюджет ходов ЭТОГО этапа. None — общий бюджет диалога. Задаётся
    # генератором: знакомство закрывается одним ответом, разбор инцидента — нет.
    max_turns: int | None = None

    @property
    def has_opening(self) -> bool:
        return bool(self.opening.strip())


@dataclass(frozen=True)
class Criterion:
    key: str
    title: str
    scale: str = "1-5"
    anchor_1: str = ""
    anchor_5: str = ""

    @property
    def bounds(self) -> tuple[int, int]:
        """(мин, макс) из строки шкалы. По умолчанию 1-5."""
        try:
            lo, hi = self.scale.split("-")
            return int(lo), int(hi)
        except (ValueError, AttributeError):
            return 1, 5


@dataclass
class Scenario:
    id: str
    title: str
    type: str
    persona: Persona = field(default_factory=Persona)
    stages: list[Stage] = field(default_factory=list)
    criteria: list[Criterion] = field(default_factory=list)
    # Откуда сценарий взялся: "file" — лежит на диске, "generated" — собран из
    # текста методиста, "template" — фолбэк, когда модель не справилась.
    source: str = "file"
    # Исходный текст, из которого сценарий сгенерирован. Нужен и для отчёта, и
    # для чат-слоя: перегенерация части артефакта без него невозможна.
    source_text: str = ""

    def stage(self, index: int) -> Stage | None:
        return self.stages[index] if 0 <= index < len(self.stages) else None

    def criterion(self, key: str) -> Criterion | None:
        return next((c for c in self.criteria if c.key == key), None)

    @property
    def criteria_keys(self) -> list[str]:
        return [c.key for c in self.criteria]

    def validate(self) -> list[str]:
        """Список проблем. Пустой — сценарий пригоден."""
        problems = []
        if not self.stages:
            problems.append("нет этапов")
        if not self.criteria:
            problems.append("нет критериев оценки")
        if self.persona.empty:
            problems.append("персона пустая: некому вести разговор")
        seen = set()
        for s in self.stages:
            if s.id in seen:
                problems.append(f"повторяющийся id этапа: {s.id}")
            seen.add(s.id)
            if not s.goal:
                problems.append(f"этап {s.id}: нет цели")
            if s.max_turns is not None and not (1 <= s.max_turns <= 8):
                problems.append(f"этап {s.id}: бессмысленный бюджет ходов {s.max_turns}")
        keys = set()
        for c in self.criteria:
            if c.key in keys:
                problems.append(f"повторяющийся ключ критерия: {c.key}")
            keys.add(c.key)
            if not c.key.isascii() or not c.key.replace("_", "").isalnum():
                problems.append(f"ключ критерия не латиницей: {c.key}")
            lo, hi = c.bounds
            if lo >= hi:
                problems.append(f"критерий {c.key}: бессмысленная шкала «{c.scale}»")
        return problems

    @staticmethod
    def from_dict(d: dict) -> "Scenario":
        return Scenario(
            id=d["id"],
            title=d.get("title", d["id"]),
            type=d.get("type", "generic"),
            persona=Persona.from_any(d.get("persona", d.get("agent_persona", ""))),
            stages=[Stage(
                id=s.get("id") or f"stage_{i + 1}",
                goal=s.get("goal", ""),
                hint=s.get("hint", ""),
                advance_when=s.get("advance_when", s.get("expect", "")),
                opening=s.get("opening", s.get("agent", "")),
                max_turns=s.get("max_turns"),
            ) for i, s in enumerate(d.get("stages", d.get("steps", [])))],
            criteria=[Criterion(
                key=c["key"], title=c.get("title", c["key"]),
                scale=c.get("scale", "1-5"),
                anchor_1=c.get("anchor_1", ""), anchor_5=c.get("anchor_5", ""),
            ) for c in d.get("criteria", [])],
            source=d.get("source", "file"),
            source_text=d.get("source_text", ""),
        )

    def to_dict(self) -> dict:
        d = {
            "id": self.id, "title": self.title, "type": self.type,
            "persona": self.persona.to_dict(),
            "stages": [{"id": s.id, "goal": s.goal, "hint": s.hint,
                        "advance_when": s.advance_when, "opening": s.opening,
                        **({"max_turns": s.max_turns} if s.max_turns else {})}
                       for s in self.stages],
            "criteria": [{"key": c.key, "title": c.title, "scale": c.scale,
                          "anchor_1": c.anchor_1, "anchor_5": c.anchor_5}
                         for c in self.criteria],
        }
        if self.source != "file":
            d["source"] = self.source
        if self.source_text:
            d["source_text"] = self.source_text
        return d

    def copy(self) -> "Scenario":
        """Свежая копия. Правки методиста не должны течь в следующую сессию."""
        return Scenario.from_dict(self.to_dict())


def load(path) -> Scenario:
    s = Scenario.from_dict(json.loads(pathlib.Path(path).read_text(encoding="utf-8")))
    problems = s.validate()
    if problems:
        raise ValueError(f"сценарий {path}: " + "; ".join(problems))
    return s


def load_all(directory) -> list[Scenario]:
    return [load(p) for p in sorted(pathlib.Path(directory).glob("*.json"))]
