"""Формат сценария и его загрузка.

Сценарий пишет методист, поэтому формат должен быть читаемым руками и
переживать неполноту: отсутствующая подсказка не должна ронять движок.

Три сущности:

    Stage      этап: цель, подсказка модели, условие перехода
    Criterion  критерий оценки со шкалой и якорями
    Scenario   всё вместе плюс персона агента

Условие перехода — это ТЕКСТ для модели, а не предикат. Сопоставление ответа
со сценарием семантическое: строковое сравнение не работает на свободных
формулировках, и это требование зафиксировано ещё в R-фазе (DECISION.md, п. 3).
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Stage:
    id: str
    goal: str
    hint: str = ""            # подсказка модели: что делать на этом этапе
    advance_when: str = ""    # при каком ответе пора дальше, словами
    opening: str = ""         # реплика-затравка, если методист её задал

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
    persona: str
    stages: list[Stage] = field(default_factory=list)
    criteria: list[Criterion] = field(default_factory=list)

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
        seen = set()
        for s in self.stages:
            if s.id in seen:
                problems.append(f"повторяющийся id этапа: {s.id}")
            seen.add(s.id)
            if not s.goal:
                problems.append(f"этап {s.id}: нет цели")
        keys = set()
        for c in self.criteria:
            if c.key in keys:
                problems.append(f"повторяющийся ключ критерия: {c.key}")
            keys.add(c.key)
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
            persona=d.get("persona", d.get("agent_persona", "")),
            stages=[Stage(
                id=s.get("id") or f"stage_{i + 1}",
                goal=s.get("goal", ""),
                hint=s.get("hint", ""),
                advance_when=s.get("advance_when", s.get("expect", "")),
                opening=s.get("opening", s.get("agent", "")),
            ) for i, s in enumerate(d.get("stages", d.get("steps", [])))],
            criteria=[Criterion(
                key=c["key"], title=c.get("title", c["key"]),
                scale=c.get("scale", "1-5"),
                anchor_1=c.get("anchor_1", ""), anchor_5=c.get("anchor_5", ""),
            ) for c in d.get("criteria", [])],
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id, "title": self.title, "type": self.type,
            "persona": self.persona,
            "stages": [{"id": s.id, "goal": s.goal, "hint": s.hint,
                        "advance_when": s.advance_when, "opening": s.opening}
                       for s in self.stages],
            "criteria": [{"key": c.key, "title": c.title, "scale": c.scale,
                          "anchor_1": c.anchor_1, "anchor_5": c.anchor_5}
                         for c in self.criteria],
        }


def load(path) -> Scenario:
    s = Scenario.from_dict(json.loads(pathlib.Path(path).read_text(encoding="utf-8")))
    problems = s.validate()
    if problems:
        raise ValueError(f"сценарий {path}: " + "; ".join(problems))
    return s


def load_all(directory) -> list[Scenario]:
    return [load(p) for p in sorted(pathlib.Path(directory).glob("*.json"))]
