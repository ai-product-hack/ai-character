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
    # Как этот ответ печатали: время до первого нажатия, паузы, правки.
    # Только у реплик пользователя и только в вебе — в скриптовых прогонах None.
    typing: dict | None = None
    # Ход состоялся не полностью: агента перебили, и он не задал свой вопрос
    # до конца. Такой ход не списывает бюджет этапа — иначе каждое перебивание
    # приближало бы сценарий к принудительному завершению, а перебивание это
    # нормальная часть разговора, а не потраченный ход.
    counted: bool = True
    # Реплика прозвучала лишь частично — пользователь перебил на середине.
    interrupted: bool = False


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
    # Последний этап — тот, где прощаются, и ему нужно больше места: обрубать
    # прощание тем же лимитом, что и промежуточный вопрос, значит заканчивать
    # на полуслове. None — на ход больше общего бюджета; так лимит остаётся
    # осмысленным и когда общий задан снаружи.
    last_stage_max_turns: int | None = None
    # Предохранитель на весь диалог. Перебитые ходы не списывают бюджет ЭТАПА
    # — иначе живой разговор наказывался бы, — но списывают этот. Без него
    # собеседник, перебивающий каждую реплику, не дал бы сценарию закончиться
    # никогда, а «диалог обязан заканчиваться» важнее справедливости к ходу.
    # None — считается от размера сценария при первом обращении.
    max_user_turns: int | None = None
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
        """Полноценных обменов на текущем этапе.

        Перебитые не в счёт: агент не успел договорить свой вопрос, так что
        обмен не состоялся. Считать их значило бы наказывать пользователя за
        живость разговора.
        """
        return sum(1 for t in self.turns[self._stage_started_at:]
                   if t.role == "user" and t.counted)

    def budget_of(self, index: int) -> int:
        """Бюджет этапа по номеру.

        Приоритет у бюджета, заданного самим этапом: знакомство закрывается
        одним ответом, разбор инцидента — нет, и генератор это различает.
        Дальше — общий лимит, а у последнего этапа он на ход больше: это этап,
        на котором разговор сворачивают, и обрубать его тем же лимитом, что и
        промежуточный, значит заканчивать на полуслове.
        """
        st = self.scenario.stage(index)
        if st is not None and st.max_turns:
            return st.max_turns
        if index < len(self.scenario.stages) - 1:
            return self.max_turns_per_stage
        return (self.last_stage_max_turns if self.last_stage_max_turns is not None
                else self.max_turns_per_stage + 1)

    @property
    def stage_max_turns(self) -> int:
        """Бюджет текущего этапа."""
        return self.budget_of(self.stage_index)

    @property
    def stage_budget_spent(self) -> bool:
        return self.turns_on_stage >= self.stage_max_turns

    @property
    def dialogue_max_turns(self) -> int:
        """Потолок ходов на весь сценарий, включая перебитые.

        По умолчанию — сколько нужно, чтобы пройти все этапы по полному
        бюджету, плюс запас на перебивания. Считается от сценария, а не
        константой: этапов у сценариев от шести до семи.
        """
        if self.max_user_turns is not None:
            return self.max_user_turns
        # Сумма бюджетов этапов, а не формула от их числа: у сгенерированных
        # сценариев бюджеты разные. Для сценария без собственных бюджетов сумма
        # совпадает со старой формулой ровно, поэтому метрика завершения на
        # существующих пяти сценариях не меняется.
        return sum(self.budget_of(i) for i in range(len(self.scenario.stages))) + 6

    @property
    def dialogue_budget_spent(self) -> bool:
        return self.user_turns >= self.dialogue_max_turns

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

    def add_user(self, text: str, typing: dict | None = None) -> Turn:
        t = Turn("user", text, self.stage_id, typing=typing)
        self.turns.append(t)
        return t

    def add_interrupted_agent(self, text: str, generation_id: str | None = None) -> Turn:
        """Записать то, что агент успел произнести до перебивания.

        Пользователь это СЛЫШАЛ. Выбрасывать прозвучавшее — значит заставлять
        агента отвечать так, будто он молчал: он повторит вопрос, на который
        уже получил ответ. В историю идёт ровно озвученная часть, помеченная
        как оборванная, чтобы модель понимала, почему фраза без конца.
        """
        t = Turn("agent", text, self.stage_id, generation_id, interrupted=True)
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
