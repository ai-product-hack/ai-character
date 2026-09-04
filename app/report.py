"""Отчёт по диалогу.

Собирается инкрементально: замечания копятся по ходу разговора, а к моменту
`finish` остаётся только сложить их в структуру. Фоновая оценка моделью
(отдельная сессия, вне критического пути) появится следующим шагом и будет
дописывать сюда обоснования; здесь — каркас и сведение того, что уже есть.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field

from .dialogue import DialogueState


@dataclass
class CriterionResult:
    key: str
    title: str
    scale: str
    score: float | None = None
    rationale: str = ""
    observations: list[str] = field(default_factory=list)

    @property
    def evaluated(self) -> bool:
        return self.score is not None


@dataclass
class Report:
    scenario_id: str
    scenario_title: str
    completed: bool
    finish_reason: str
    stages_reached: int
    stages_total: int
    user_turns: int
    criteria: list[CriterionResult] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def overall(self) -> float | None:
        scored = [c.score for c in self.criteria if c.score is not None]
        return round(statistics.mean(scored), 2) if scored else None

    @property
    def coverage(self) -> float:
        """Доля критериев, по которым есть хоть какая-то оценка."""
        return len([c for c in self.criteria if c.evaluated]) / len(self.criteria) \
            if self.criteria else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["overall"] = self.overall
        d["coverage"] = round(self.coverage, 3)
        return d

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def build(state: DialogueState, evaluation=None) -> Report:
    """Свести состояние в отчёт.

    Дешёвая операция без обращений к модели — поэтому её можно звать хоть после
    каждой реплики. Всё дорогое уже сделано фоновой сессией, и к моменту
    `finish` отчёт складывается мгновенно.

    Источников оценки два: `evaluate` от самого агента по ходу разговора и
    фоновый оценщик. Второй важнее — живая модель за действием `evaluate`
    почти не тянется (замерено: 0 вызовов из 59).
    """
    sc = state.scenario
    criteria = []
    for c in sc.criteria:
        obs = state.observations_for(c.key)
        scores = [o.score for o in obs if o.score is not None]
        notes = [o.note for o in obs if o.note]
        rationale = ""
        if evaluation is not None:
            bg = evaluation.for_criterion(c.key)
            scores += [a.score for a in bg]
            notes += [a.rationale for a in bg if a.rationale]
            if bg:
                # Обоснование берём последнее: оно опирается на самый полный
                # контекст разговора.
                rationale = bg[-1].rationale
        criteria.append(CriterionResult(
            key=c.key, title=c.title, scale=c.scale,
            score=round(statistics.mean(scores), 2) if scores else None,
            rationale=rationale,
            observations=notes,
        ))
    return Report(
        scenario_id=sc.id,
        scenario_title=sc.title,
        completed=state.finished,
        finish_reason=state.finish_reason,
        stages_reached=state.stage_index + 1,
        stages_total=len(sc.stages),
        user_turns=state.user_turns,
        criteria=criteria,
        transcript=[{"role": t.role, "text": t.text, "stage": t.stage_id}
                    for t in state.turns],
    )
