"""Действия агента и разбор ответа модели.

Действий ровно три: `next_stage`, `finish`, `evaluate`. Четвёртое поведение —
остаться на этапе — НЕ действие, а откат: именно им заканчивается любой ответ,
из которого не удалось вытащить корректное действие. Так молчание, обрыв
соединения и галлюцинация дают одно и то же безопасное поведение вместо
исключения посреди диалога.

Формат ответа модели: сначала реплика, ПОТОМ управляющий JSON последней
строкой. Порядок принципиален и продиктован конвейером — реплика уходит в
синтез по мере поступления токенов, поэтому всё, что не предназначено для
произнесения, обязано идти после неё.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

NEXT_STAGE = "next_stage"
FINISH = "finish"
EVALUATE = "evaluate"
STAY = "stay"

# Действий, меняющих ход диалога, ровно три. `stay` — объявленный ПУСТОЙ ход:
# модель говорит «остаюсь на этапе» вслух, вместо того чтобы молчать.
#
# Разница не косметическая. В первой версии `stay` был только откатом при
# неразобранном ответе, и модель, которой нечего было объявить, просто не
# выводила JSON. Измерено на живой модели: 22% ходов уходили в откат, а на
# длинном прогоне доходило до 83% — и в этом шуме утонуло бы настоящее
# нарушение протокола. Теперь молчание означает поломку, и только поломку.
ACTIONS = {NEXT_STAGE, FINISH, EVALUATE}
VALID = ACTIONS | {STAY}

# Управляющий блок: последний JSON-объект верхнего уровня в ответе.
_JSON_OBJ = re.compile(r"\{[^{}]*\}", re.S)
# Модели любят завернуть JSON в ```json ... ```
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


@dataclass
class AgentAction:
    """`fell_back=True` означает, что действие не разобрано вообще: пустой
    ответ, обрыв, сломанный JSON. Осознанный пустой ход — это `stay` с
    `fell_back=False`."""
    action: str
    criterion: str | None = None
    score: int | None = None
    note: str = ""
    raw: str = ""
    fell_back: bool = False       # действие не разобрано, откатились на STAY

    @property
    def advances(self) -> bool:
        return self.action == NEXT_STAGE

    @property
    def finishes(self) -> bool:
        return self.action == FINISH


@dataclass
class AgentReply:
    """Что агент скажет вслух и что при этом делает."""
    text: str
    action: AgentAction

    @property
    def speakable(self) -> str:
        return self.text.strip()


def _coerce_score(v) -> int | None:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n


def parse_action(blob: str) -> AgentAction:
    """Вытащить действие из хвоста ответа. Мусор -> STAY."""
    if not blob:
        return AgentAction(STAY, raw="", fell_back=True)

    candidates = [m.group(1) for m in _FENCE.finditer(blob)]
    candidates += [m.group(0) for m in _JSON_OBJ.finditer(blob)]
    # Управляющий блок идёт последним, поэтому и разбираем с конца.
    for chunk in reversed(candidates):
        try:
            d = json.loads(chunk)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        name = str(d.get("action", "")).strip().lower()
        if name not in VALID:
            continue
        return AgentAction(
            action=name,
            criterion=(str(d["criterion"]).strip() if d.get("criterion") else None),
            score=_coerce_score(d.get("score")),
            note=str(d.get("note", "")).strip(),
            raw=chunk,
        )
    return AgentAction(STAY, raw=blob[-200:], fell_back=True)


def strip_control(blob: str) -> str:
    """Убрать управляющий блок из того, что пойдёт в синтез и в субтитры."""
    if not blob:
        return ""
    out = _FENCE.sub("", blob)
    # Срезаем только ХВОСТОВЫЕ json-объекты: фигурные скобки могут встретиться
    # и в самой реплике, и вырезать их из середины нельзя.
    while True:
        stripped = out.rstrip()
        m = _JSON_OBJ.search(stripped)
        if m and m.end() == len(stripped) and _looks_like_control(m.group(0)):
            out = stripped[:m.start()]
            continue
        break
    return out.strip()


def _looks_like_control(chunk: str) -> bool:
    try:
        d = json.loads(chunk)
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(d, dict) and "action" in d


def parse_reply(blob: str) -> AgentReply:
    """Разобрать полный ответ модели на реплику и действие."""
    return AgentReply(text=strip_control(blob), action=parse_action(blob))
