"""Действия агента и разбор ответа модели.

Действий два: `next_stage` и `finish`, плюс объявленный пустой ход `stay`.
Третье поведение — остаться на этапе при неразобранном ответе — НЕ действие,
а откат: именно им заканчивается любой ответ,
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
STAY = "stay"

# Действие `evaluate` убрано. Живая модель не выбрала его НИ РАЗУ из 59 вызовов,
# а отчёт по критериям наполняет фоновая сессия — та, что и должна этим
# заниматься, вне критического пути. Мёртвая ветка в промпте только отнимала у
# модели внимание. Константа оставлена: старые записи в состоянии диалога могут
# на неё ссылаться, и разбирать их надо без падения.
EVALUATE = "evaluate"

# Действий, меняющих ход диалога, ровно три. `stay` — объявленный ПУСТОЙ ход:
# модель говорит «остаюсь на этапе» вслух, вместо того чтобы молчать.
#
# Разница не косметическая. В первой версии `stay` был только откатом при
# неразобранном ответе, и модель, которой нечего было объявить, просто не
# выводила JSON. Измерено на живой модели: 22% ходов уходили в откат, а на
# длинном прогоне доходило до 83% — и в этом шуме утонуло бы настоящее
# нарушение протокола. Теперь молчание означает поломку, и только поломку.
ACTIONS = {NEXT_STAGE, FINISH}
VALID = ACTIONS | {STAY, EVALUATE}

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
    repaired: bool = False        # действие восстановлено вторым разбором

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
        """Только произносимое: без управляющего JSON и без тегов эмоций.

        Теги вырезаются и здесь, а не только перед синтезом. Первая версия
        снимала их лишь в конвейере, и в озвучку они действительно не попадали
        — зато оставались в истории на экране, в расшифровке отчёта и в
        контексте следующего запроса. «Произносимое» обязано означать
        произносимое во всех трёх местах.
        """
        from .emotion_tags import parse as _parse_emotions
        return _parse_emotions(self.text).text


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


# --------------------------------------------------------------- починка

REPAIR_SYSTEM = (
    "Ты разбираешь реплику интервьюера. Отвечай ОДНИМ словом из списка: "
    "next_stage — интервьюер закончил с текущей темой и переходит к следующей; "
    "finish — интервью окончено, интервьюер прощается; "
    "stay — тема не закрыта, разговор продолжается. "
    "Ничего, кроме одного слова."
)


def build_repair_prompt(reply_text: str, stage_title: str = "",
                        is_last_stage: bool = False) -> str:
    lines = []
    if stage_title:
        lines.append(f"Текущая тема: {stage_title}")
    if is_last_stage:
        lines.append("Это последняя тема сценария.")
    lines.append(f"Реплика интервьюера:\n{reply_text}")
    lines.append("Одно слово:")
    return "\n".join(lines)


def repair_action(ask, reply_text: str, stage_title: str = "",
                  is_last_stage: bool = False) -> AgentAction | None:
    """Второй разбор для реплик, пришедших без управляющей строки.

    Не спасение от галлюцинаций, а закрытие дыры в протоколе: примерно каждая
    десятая реплика приходит без последней строки, и молчание модели сейчас
    неотличимо от объявленного `stay`. Разница существенная — из-за неё
    сценарий может простоять на этапе до принудительного перехода по бюджету.

    Спрашиваем по УЖЕ СКАЗАННОМУ тексту, одним словом, с крошечным бюджетом
    токенов. Реплика к этому моменту звучит, так что на первый звук это не
    влияет; влияет только на момент обновления этапа.

    `ask` — вызываемое (system, prompt) -> str. Любая ошибка означает «не
    починили»: возвращаем None, и остаётся обычный откат на STAY.
    """
    if not reply_text.strip():
        return None
    try:
        out = ask(REPAIR_SYSTEM,
                  build_repair_prompt(reply_text, stage_title, is_last_stage))
    except Exception:                                      # noqa: BLE001
        return None
    word = re.sub(r"[^a-z_]", "", str(out or "").strip().lower())
    for name in (NEXT_STAGE, FINISH, STAY):
        if name in word:
            return AgentAction(action=name, raw=str(out)[:80], repaired=True)
    return None
