"""Отчёт по диалогу.

Собирается инкрементально: замечания копятся по ходу разговора, а к моменту
`finish` остаётся только сложить их в структуру. Дорогое уже сделано фоновой
сессией, поэтому отчёт показывается мгновенно.

Главное здесь — цитаты. «Коммуникация 3 из 5» — мнение модели, и спорить с ним
можно только на уровне «а мне кажется, четыре». «3 из 5, вот ответ, где
кандидат поплыл» — разбор: методист открывает реплику и видит, о чём речь.
Поэтому оценка хранит НОМЕР реплики, а не её копию: по номеру интерфейс
прокручивает транскрипт к нужному месту, а копия рассинхронизировалась бы с
ним при первой же правке.

Шаблон один на все виды тренировок. Отдельные формы под собеседование и под
продажи выглядели бы разными продуктами, а механика везде одна: критерии,
цитаты, вывод — меняется только шапка.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import time
from dataclasses import asdict, dataclass, field

from .dialogue import DialogueState


@dataclass
class Citation:
    """Оценка со ссылкой на реплику, которая её обосновала."""
    turn: int                 # индекс в `transcript`; -1 — ссылки нет
    score: float | None
    rationale: str
    stage: str = ""
    source: str = "background"        # background | agent


@dataclass
class CriterionResult:
    key: str
    title: str
    scale: str
    # Якоря шкалы едут в отчёт вместе с оценкой. Без них «3» — число без
    # единиц: непонятно ни из скольких, ни что считается хорошим ответом.
    # Методист их уже написал, они лежат в сценарии и до экрана не доходили.
    anchor_1: str = ""
    anchor_5: str = ""
    lo: int = 1
    hi: int = 5
    score: float | None = None
    rationale: str = ""
    observations: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)

    @property
    def evaluated(self) -> bool:
        return self.score is not None

    @property
    def ratio(self) -> float | None:
        """Оценка долей от своей шкалы, 0..1. Низ шкалы — дно, а не доля.

        Нужно, чтобы сводить критерии с РАЗНЫМИ шкалами. Средний балл по
        критериям с потолками 5 и 10 — число без смысла.
        """
        if self.score is None or self.hi <= self.lo:
            return None
        return max(0.0, min(1.0, (self.score - self.lo) / (self.hi - self.lo)))


@dataclass
class Report:
    scenario_id: str
    scenario_title: str
    scenario_type: str
    persona: dict
    completed: bool
    finish_reason: str
    stages_reached: int
    stages_total: int
    user_turns: int
    criteria: list[CriterionResult] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)
    typing: dict | None = None
    conclusion: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    session_id: str = ""

    @property
    def duration_s(self) -> float | None:
        """Длительность разговора: от первой реплики до последней."""
        if self.started_at is None:
            return None
        return round(max(0.0, self.created_at - self.started_at), 1)

    @property
    def overall(self) -> float | None:
        scored = [c.score for c in self.criteria if c.score is not None]
        return round(statistics.mean(scored), 2) if scored else None

    @property
    def coverage(self) -> float:
        """Доля критериев, по которым есть хоть какая-то оценка."""
        return len([c for c in self.criteria if c.evaluated]) / len(self.criteria) \
            if self.criteria else 0.0

    @property
    def scale_max(self) -> int | None:
        """Общий потолок шкалы, если он у всех критериев один.

        Только тогда «2.5 из 5» — правда. Разные потолки сводятся долей.
        """
        tops = {c.hi for c in self.criteria}
        return tops.pop() if len(tops) == 1 else None

    @property
    def scale_min(self) -> int | None:
        bottoms = {c.lo for c in self.criteria}
        return bottoms.pop() if len(bottoms) == 1 else None

    @property
    def overall_ratio(self) -> float | None:
        """Итог долей от шкалы, 0..1 — единственная честная сводка.

        Средний балл считается по критериям с разными потолками одинаково
        охотно и одинаково бессмысленно; доля сводит их корректно.
        """
        vals = [c.ratio for c in self.criteria if c.ratio is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    @property
    def cited(self) -> int:
        """Сколько критериев подкреплено ссылкой на реплику."""
        return len([c for c in self.criteria
                    if any(q.turn >= 0 for q in c.citations)])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["overall"] = self.overall
        d["coverage"] = round(self.coverage, 3)
        d["duration_s"] = self.duration_s
        d["cited"] = self.cited
        d["scale_max"] = self.scale_max
        d["scale_min"] = self.scale_min
        d["overall_ratio"] = self.overall_ratio
        return d

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def build(state: DialogueState, evaluation=None, session_id: str = "",
          conclusion: str = "") -> Report:
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
        citations = [Citation(turn=o.turn_index, score=o.score, rationale=o.note,
                              stage=o.stage_id, source="agent")
                     for o in obs if o.note]
        rationale = ""
        if evaluation is not None:
            bg = evaluation.for_criterion(c.key)
            scores += [a.score for a in bg]
            notes += [a.rationale for a in bg if a.rationale]
            citations += [Citation(turn=a.quote_turn, score=a.score,
                                   rationale=a.rationale, stage=a.stage_id)
                          for a in bg]
            if bg:
                # Обоснование берём последнее: оно опирается на самый полный
                # контекст разговора.
                rationale = bg[-1].rationale
        lo, hi = c.bounds
        criteria.append(CriterionResult(
            key=c.key, title=c.title, scale=c.scale,
            anchor_1=c.anchor_1, anchor_5=c.anchor_5, lo=lo, hi=hi,
            score=round(statistics.mean(scores), 2) if scores else None,
            rationale=rationale,
            observations=notes,
            citations=citations,
        ))
    return Report(
        scenario_id=sc.id,
        scenario_title=sc.title,
        scenario_type=sc.type,
        persona=sc.persona.to_dict(),
        completed=state.finished,
        finish_reason=state.finish_reason,
        stages_reached=state.stage_index + 1,
        stages_total=len(sc.stages),
        user_turns=state.user_turns,
        criteria=criteria,
        transcript=[{"i": i, "role": t.role, "text": t.text, "stage": t.stage_id,
                     "at": t.at, "interrupted": t.interrupted,
                     **({"typing": t.typing} if t.typing else {})}
                    for i, t in enumerate(state.turns)],
        typing=_typing_summary(state),
        conclusion=conclusion,
        started_at=state.turns[0].at if state.turns else None,
        session_id=session_id,
    )


def _typing_summary(state: DialogueState) -> dict | None:
    """Как человек печатал: сводка по всем ответам.

    Не оценка, а сигнал методисту. Ответ, набранный за девять секунд с двумя
    переписываниями, и ответ той же длины, набранный без пауз, читаются в
    расшифровке одинаково — разницу видно только здесь.
    """
    signals = [t.typing for t in state.turns if t.role == "user" and t.typing]
    signals = [s for s in signals if s.get("confidence") != "нет данных"]
    if not signals:
        return None
    firsts = [s["first_key_ms"] for s in signals if s.get("first_key_ms") is not None]
    labels = [s["confidence"] for s in signals]
    return {
        "answers": len(signals),
        "median_first_key_ms": round(statistics.median(firsts)) if firsts else None,
        "median_longest_pause_ms": round(statistics.median(
            [s["longest_pause_ms"] for s in signals])),
        "median_rewrite_ratio": round(statistics.median(
            [s["rewrite_ratio"] for s in signals]), 3),
        "confidence": {k: labels.count(k) for k in dict.fromkeys(labels)},
        "hesitant_answers": sum(1 for x in labels if x != "уверенно"),
    }


# ------------------------------------------------------------------ на диск

HISTORY = pathlib.Path(__file__).resolve().parents[1] / "data" / "reports" / "history.jsonl"


def save(report: dict, path=None) -> str:
    """Дописать отчёт одной строкой. История прогонов без базы данных.

    Одна строка на прогон: файл читается `jq`, растёт линейно и не требует ни
    миграций, ни сервера. Сбой записи гасится — отчёт уже на экране, и падать
    из-за журнала после успешного диалога незачем.
    """
    p = pathlib.Path(path or HISTORY)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")
        return str(p)
    except OSError as e:
        return f"не записан: {e}"
