"""Состояние диалога.

Состояние принадлежит движку, а не модели. Модель получает его в промпте
целиком на каждом ходу и ничего не «помнит» между вызовами: при перебивании
запрос отменяется на полуслове, при перезапуске спекулятивной генерации
контекст пересобирается заново, и любая опора на память модели тут же
разъезжается с тем, что видел пользователь.

Хранится ровно три вещи: где мы в сценарии, что было сказано, и что уже
замечено по критериям методиста.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

from .scenario import Scenario


@dataclass
class Turn:
    """Одна реплика. `generation_id` нужен, чтобы отменённые ответы не попадали
    ни в историю, ни в отчёт."""
    role: str                  # "agent" | "user"
    text: str
    stage_id: str
    generation_id: str | None = None
    at: float = field(default_factory=time.time)


@dataclass
class Observation:
    """Замечание по критерию: чем агент обосновывает будущую оценку."""
    criterion: str
    note: str
    score: int | None = None
    stage_id: str = ""
    turn_index: int = -1


@dataclass
class DialogueState:
    scenario: Scenario
    stage_index: int = 0
    turns: list[Turn] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    finished: bool = False
    finish_reason: str = ""
    # Бюджет реплик на этап. Без него диалог не заканчивается: измерено на живой
    # модели — после того как в протоколе появился явный «остаться на этапе»,
    # DeepSeek выбрала его в 60 ходах из 66 и не перешла ни разу. Модель всегда
    # найдёт, что ещё уточнить; ограничение принадлежит движку, а не промпту.
    max_turns_per_stage: int = 3
    forced_advances: int = 0
    _stage_started_at: int = 0

    # ------------------------------------------------------------------ этап

    @property
    def stage(self):
        return self.scenario.stage(self.stage_index)

    @property
    def stage_id(self) -> str:
        s = self.stage
        return s.id if s else "—"

    @property
    def is_last_stage(self) -> bool:
        return self.stage_index >= len(self.scenario.stages) - 1

    @property
    def turns_on_stage(self) -> int:
        """Сколько раз пользователь ответил на текущем этапе."""
        return sum(1 for t in self.turns[self._stage_started_at:] if t.role == "user")

    @property
    def stage_budget_spent(self) -> bool:
        return self.turns_on_stage >= self.max_turns_per_stage

    def advance(self, forced: bool = False) -> bool:
        """Следующий этап. Возвращает False, если этапы кончились."""
        if self.is_last_stage:
            return False
        self.stage_index += 1
        self._stage_started_at = len(self.turns)
        if forced:
            self.forced_advances += 1
        return True

    def finish(self, reason: str = "") -> None:
        self.finished = True
        self.finish_reason = reason or "сценарий пройден"

    # --------------------------------------------------------------- реплики

    def add_user(self, text: str) -> Turn:
        t = Turn("user", text, self.stage_id)
        self.turns.append(t)
        return t

    def add_agent(self, text: str, generation_id: str | None = None) -> Turn:
        t = Turn("agent", text, self.stage_id, generation_id)
        self.turns.append(t)
        return t

    def drop_generation(self, generation_id: str) -> int:
        """Убрать из истории всё, что принадлежит отменённой генерации.

        Перебитая реплика не должна попасть ни в контекст следующего запроса,
        ни в отчёт: пользователь её не дослушал, и агент не вправе считать,
        что она прозвучала.
        """
        before = len(self.turns)
        self.turns = [t for t in self.turns if t.generation_id != generation_id]
        return before - len(self.turns)

    # ------------------------------------------------------------ наблюдения

    def observe(self, criterion: str, note: str, score: int | None = None) -> Observation | None:
        """Записать замечание. Неизвестный критерий отбрасывается: модель
        периодически придумывает ключи, которых методист не задавал."""
        if not self.scenario.criterion(criterion):
            return None
        o = Observation(criterion, note, score, self.stage_id, len(self.turns) - 1)
        self.observations.append(o)
        return o

    def observations_for(self, key: str) -> list[Observation]:
        return [o for o in self.observations if o.criterion == key]

    # ----------------------------------------------------------------- прочее

    @property
    def user_turns(self) -> int:
        return sum(1 for t in self.turns if t.role == "user")

    def transcript(self, limit: int | None = None) -> list[Turn]:
        return self.turns[-limit:] if limit else list(self.turns)

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario.id,
            "stage_index": self.stage_index,
            "stage_id": self.stage_id,
            "finished": self.finished,
            "finish_reason": self.finish_reason,
            "turns": [asdict(t) for t in self.turns],
            "observations": [asdict(o) for o in self.observations],
        }
