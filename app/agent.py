"""Агент: собирает промпт из состояния, применяет действие к состоянию.

Модель не помнит ничего между ходами — весь контекст приходит в промпте.
Это не перестраховка: спекулятивная генерация с перезапуском (S1) отменяет и
пересобирает запрос на полуслове, и любая опора на серверную память модели
разъехалась бы с тем, что реально услышал пользователь.
"""
from __future__ import annotations

from .actions import EVALUATE, FINISH, NEXT_STAGE, STAY, AgentReply, parse_reply
from .dialogue import DialogueState

SYSTEM = """Ты ведёшь тренировочный диалог по сценарию. Твоя роль описана ниже.

Правила:
- Говори живой разговорной репликой на 1-3 предложения. Без списков и разметки.
- Не подсказывай собеседнику правильный ответ и не оценивай его вслух.
- Держись своей роли и цели текущего этапа.
- Реагируй на то, что собеседник сказал НА САМОМ ДЕЛЕ, а не на то, что ждал сценарий.

После реплики ОБЯЗАТЕЛЬНО выведи отдельной последней строкой управляющий JSON.
Ровно один объект, всегда, без исключений:

  {"action": "stay"}         — остаёшься на этом же этапе, переспрашиваешь
  {"action": "next_stage"}   — ответ закрывает цель этапа, пора дальше
  {"action": "finish"}       — сценарий пройден или продолжать бессмысленно

Строку с JSON нельзя пропускать, даже когда ничего не меняется. JSON идёт
ПОСЛЕ реплики, никогда до неё и никогда внутри.

Про переходы: этап закрывается, как только собеседник дал ответ по существу.
Идеального ответа ждать не надо — уточнить одну деталь можно, но сценарий
должен двигаться. Если на этапе уже был обмен репликами и ответ в целом получен,
выводи "next_stage"."""


def scenario_block(state: DialogueState) -> str:
    """Неизменная часть контекста: роль, сценарий, критерии.

    Отделена от остального нарочно. Весь разговор она одна и та же, значит
    может быть началом кешируемого префикса — а всё, что меняется по ходу,
    висит на последней реплике.
    """
    sc = state.scenario
    lines = [f"РОЛЬ: {sc.persona.prompt_block()}", f"СЦЕНАРИЙ: {sc.title}"]
    if sc.criteria:
        # Критерии говорят агенту, ЧТО важно вытянуть из собеседника. Оценивать
        # их он не должен — это делает фоновая сессия.
        lines += ["", "НА ЧТО СМОТРИМ (оценивать вслух не надо):"]
        for c in sc.criteria:
            lines.append(f"- {c.title}: от «{c.anchor_1}» до «{c.anchor_5}».")
    return "\n".join(lines)


def stage_block(state: DialogueState) -> str:
    """Меняющаяся часть: где мы в сценарии прямо сейчас.

    Прикрепляется к ПОСЛЕДНЕЙ реплике пользователя, а не в начало разговора:
    так всё, что было раньше, остаётся байт в байт тем же и попадает в кеш.
    """
    sc = state.scenario
    stage = state.stage
    lines = [f"[ЭТАП {state.stage_index + 1} из {len(sc.stages)}: "
             f"{stage.goal if stage else '—'}]"]
    if stage and stage.hint:
        lines.append(stage.hint)
    if stage and stage.advance_when:
        lines.append(f"Переходить дальше, когда: {stage.advance_when}.")
    if state.is_last_stage:
        lines.append("Это последний этап сценария: закончив его, заверши диалог.")
    if state.closing_turn:
        lines.append("ЭТО ТВОЯ ПОСЛЕДНЯЯ РЕПЛИКА В РАЗГОВОРЕ. Заверши его сам: "
                     "коротко подведи черту и попрощайся, одной-двумя фразами. "
                     "Новых вопросов не задавай — отвечать на них будет уже "
                     "некому.")
    spent = state.turns_on_stage
    if spent:
        left = max(0, state.stage_max_turns - spent)
        lines.append(f"На этом этапе уже {spent} обмен(ов) репликами; "
                     f"осталось {left} до принудительного перехода.")
    return "\n".join(lines)


# Требование про JSON повторяется в конце КАЖДОЙ реплики пользователя, а не
# один раз в системном сообщении: замерено на DeepSeek — управляющая строка
# терялась примерно в каждом шестом ходу, причём не из-за обрыва по токенам.
CONTROL_REMINDER = (
    "Ответь репликой, а затем ОБЯЗАТЕЛЬНО последней строкой — управляющим "
    'JSON. Даже если ничего не меняется, это должно быть {"action": "stay"}. '
    "Ответ без последней строки считается ошибкой."
)


def build_messages(state: DialogueState, user_text: str | None = None) -> list[dict]:
    """Разговор как настоящий чат: массив ролей вместо склеенного текста.

    Зачем это вместо одного `user`-сообщения со всей историей внутри:

    1. **История перестаёт теряться.** Склеенный промпт нёс последние 12 ходов,
       и на записанных прогонах обрезка срабатывала в 10 случаях из 10 — в
       длинном сценарии агент не помнил, что человек сказал в начале.
    2. **Появляется стабильный префикс.** Всё до последней реплики не меняется
       от хода к ходу, значит читается из кеша. Замерено: вход дорожает на 8%,
       а с кешем выходит в 2.4 раза дешевле нынешнего.

    Опоры на память провайдера здесь по-прежнему нет: массив собирается заново
    на каждый запрос из состояния движка. Перебитая реплика приходит сюда уже
    обрезанной по услышанному, спекуляция строит тот же массив с черновиком
    вместо последней реплики.
    """
    opening = scenario_block(state)
    hist = state.transcript()
    # Последний ход пользователя уже лежит в истории (его добавляют ДО сборки
    # промпта). Он и есть «свежая» часть — собирается отдельно, вместе с
    # состоянием этапа, чтобы всё до него осталось байт в байт прежним.
    if user_text is not None and hist and hist[-1].role == "user":
        hist = hist[:-1]

    messages: list[dict] = []
    opening_used = False
    for turn in hist:
        if turn.role == "agent":
            if not messages:
                # Массив обязан начинаться с user: разговор открывал агент, и
                # блоку сценария больше некуда деться.
                messages.append({"role": "user", "content": opening})
                opening_used = True
            # Управляющая строка возвращается в историю. Без неё модель не
            # видит собственного формата и перестаёт его выводить: замерено,
            # 63.7% реплик без JSON против 0-5% в прежней склейке.
            text = turn.text
            if turn.action and not turn.interrupted:
                text = f'{text}\n{{"action": "{turn.action}"}}'
            _append(messages, "assistant", text)
        else:
            text = turn.text
            if not opening_used:
                text = f"{opening}\n\n{text}"
                opening_used = True
            _append(messages, "user", text)

    tail = [] if opening_used else [opening, ""]
    tail.append(stage_block(state))
    if user_text is not None:
        tail.append(user_text)
    elif not hist:
        tail.append("Разговор начинается. Открой первый этап.")
    tail += ["", CONTROL_REMINDER]
    _append(messages, "user", "\n".join(tail))
    return messages


def _append(messages: list[dict], role: str, text: str) -> None:
    """Добавить сообщение, склеивая подряд идущие одинаковые роли.

    Два `user` подряд бывают: если агента перебили до первого звука, его ход
    не оставляет следа в истории, и две реплики человека оказываются рядом.
    Провайдеры такое либо отвергают, либо обрабатывают по-разному — надёжнее
    склеить самим.
    """
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"] += "\n" + text
        return
    messages.append({"role": role, "content": text})


def build_prompt(state: DialogueState, user_text: str | None = None) -> str:
    """Промпт одного хода: роль, где мы, что было, что делать."""
    sc = state.scenario
    stage = state.stage
    lines = [
        f"РОЛЬ: {sc.persona.prompt_block()}",
        f"СЦЕНАРИЙ: {sc.title}",
        "",
        f"ЭТАП {state.stage_index + 1} из {len(sc.stages)}: {stage.goal if stage else '—'}",
    ]
    if stage and stage.hint:
        lines.append(stage.hint)
    if stage and stage.advance_when:
        lines.append(f"Переходить дальше, когда: {stage.advance_when}.")
    if state.is_last_stage:
        lines.append("Это последний этап сценария: закончив его, заверши диалог.")
    if state.closing_turn:
        lines.append("ЭТО ТВОЯ ПОСЛЕДНЯЯ РЕПЛИКА В РАЗГОВОРЕ. Заверши его сам: "
                     "коротко подведи черту и попрощайся, одной-двумя фразами. "
                     "Новых вопросов не задавай — отвечать на них будет уже "
                     "некому.")
    spent = state.turns_on_stage
    if spent:
        left = max(0, state.stage_max_turns - spent)
        lines.append(f"На этом этапе уже {spent} обмен(ов) репликами; "
                     f"осталось {left} до принудительного перехода.")

    if sc.criteria:
        # Критерии остаются в промпте: они говорят агенту, ЧТО важно вытянуть из
        # собеседника. Оценивать их он не должен — это делает фоновая сессия.
        lines += ["", "НА ЧТО СМОТРИМ (оценивать вслух не надо):"]
        for c in sc.criteria:
            lines.append(f"- {c.title}: от «{c.anchor_1}» до «{c.anchor_5}».")

    hist = state.transcript(limit=12)
    if hist:
        lines += ["", "ХОД РАЗГОВОРА:"]
        for t in hist:
            who = "Ты" if t.role == "agent" else "Собеседник"
            lines.append(f"{who}: {t.text}")

    if state.observations:
        lines += ["", "УЖЕ ЗАМЕЧЕНО:"]
        for o in state.observations[-8:]:
            score = f" ({o.score})" if o.score is not None else ""
            lines.append(f"- {o.criterion}{score}: {o.note}")

    if user_text is not None:
        lines += ["", f"Собеседник только что сказал: {user_text}"]
    elif not hist:
        lines += ["", "Разговор начинается. Открой первый этап."]

    # Требование про JSON повторяется в самом конце промпта, а не только в
    # системном сообщении: измерено на DeepSeek — управляющая строка терялась
    # примерно в каждом шестом ходу, причём не из-за обрыва по токенам
    # (реплики были по 46-239 символов), а просто пропускалась.
    lines += [
        "",
        "Ответь репликой, а затем ОБЯЗАТЕЛЬНО последней строкой — управляющим JSON.",
        'Даже если ничего не меняется, это должно быть {"action": "stay"}.',
        "Ответ без последней строки считается ошибкой.",
    ]
    return "\n".join(lines)


def apply(state: DialogueState, reply: AgentReply, generation_id: str | None = None) -> dict:
    """Применить разобранный ответ к состоянию. Возвращает, что произошло."""
    a = reply.action
    happened = {"action": a.action, "fell_back": a.fell_back, "advanced": False,
                "finished": False, "observed": False, "forced": False}

    if reply.speakable:
        state.add_agent(reply.speakable, generation_id, a.action)

    if a.action == EVALUATE and a.criterion:
        happened["observed"] = state.observe(a.criterion, a.note, a.score) is not None

    if a.action == NEXT_STAGE:
        if state.is_last_stage:
            # Последний этап и просьба идти дальше — значит сценарий пройден.
            # Требовать от модели отдельного finish здесь было бы лишним ходом,
            # который пользователь увидел бы как повисшую паузу.
            state.finish("модель попросила следующий этап на последнем")
            happened["finished"] = True
        else:
            happened["advanced"] = state.advance()

    if a.action == FINISH:
        state.finish("модель завершила диалог")
        happened["finished"] = True

    # Предохранитель на весь диалог. Перебитые ходы не списывают бюджет этапа,
    # и без этого потолка собеседник, перебивающий каждую реплику, не дал бы
    # сценарию закончиться вовсе.
    if not state.finished and state.dialogue_budget_spent:
        state.finish("бюджет диалога исчерпан")
        happened["finished"] = True
        happened["forced"] = True
        return happened

    # Бюджет этапа. Модель всегда найдёт, что ещё уточнить, поэтому право
    # двигать сценарий принадлежит движку, а не только модели.
    if not state.finished and not happened["advanced"] and state.stage_budget_spent:
        if state.is_last_stage:
            state.finish("бюджет последнего этапа исчерпан")
            happened["finished"] = True
        else:
            happened["advanced"] = state.advance(forced=True)
        happened["forced"] = True

    return happened


class Agent:
    """Один ход агента поверх произвольного источника текста.

    `llm` — вызываемое: (system, prompt) -> str. Синхронное здесь намеренно:
    потоковая версия появится в конвейере реплики, а движку для тестов и для
    фоновой оценки достаточно целого ответа.
    """

    def __init__(self, llm, system: str = SYSTEM):
        self.llm = llm
        self.system = system

    def step(self, state: DialogueState, user_text: str | None = None,
             generation_id: str | None = None) -> tuple[AgentReply, dict]:
        if user_text is not None:
            state.add_user(user_text)
        messages = build_messages(state, user_text)
        raw = self.llm.chat(self.system, messages)
        reply = parse_reply(raw)
        happened = apply(state, reply, generation_id)
        return reply, happened
