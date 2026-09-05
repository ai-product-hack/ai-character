"""Панели, которые вызывает сам агент.

Механизм не новый: тот же приём, что у тегов эмоций. Модель ставит разметку
внутри реплики, разметка вырезается ДО синтеза, смещение превращается в
`pts_ms` теми же посимвольными таймкодами, а событие уходит в интерфейс.
Значит панель наследует общий PTS и отмену по `generation_id` — своих
механизмов у неё нет.

Разметка:
    [panel:material]            материал этапа: график, документ, код
    [panel:scenario]            этапы разговора
    [panel:close]               убрать панель

Главная — `material`: то, О ЧЁМ агент спрашивает. График, к которому просят
прокомментировать динамику; фрагмент кода на ревью; пункт регламента, по
которому идёт проверка. Панель становится частью упражнения, а не отчётом о нём.

Раньше главной была `skills` — карта накопленных оценок. Её убрали после
живого прогона, и правильно: показывать человеку его же оценку посреди
разговора — это подсказка на экзамене, а не материал. Данные для неё
по-прежнему считаются (они нужны отчёту), просто агент их больше не
показывает.

Содержимое материала лежит В СЦЕНАРИИ, у этапа. Дополнительной генерации по
ходу разговора не требуется: методист задал материал заранее или его придумал
генератор вместе с этапом. Ноль задержки, ноль второго запроса.

Парсер строгий по той же причине, что и у эмоций: прецедент с управляющим JSON
уже был — модель приклеивала его к последней клаузе, и агент буквально
произносил «фигурная скобка action next stage». Ни один тег не должен
просочиться в синтез.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Белый список. Больше четырёх типов на демонстрации не нужно, а каждый лишний
# — это ещё одна ветка, которую модель может вызвать невпопад.
PANELS = ("material", "code", "scenario", "close")

# Русские написания: модель предлагает их сама.
ALIASES = {
    "материал": "material", "показать": "material", "график": "material",
    "документ": "material", "регламент": "material", "данные": "material",
    "код": "code", "листинг": "code",
    "сценарий": "scenario", "этапы": "scenario",
    "убрать": "close", "закрыть": "close", "скрыть": "close",
}

# Аргумент необязателен: `[panel:code:python]` -> ("code", "python").
_TAG = re.compile(r"\[\s*panel\s*:\s*([a-zA-Zа-яёА-ЯЁ_]+)"
                  r"(?:\s*:\s*([a-zA-Z0-9_+#.-]{1,24}))?\s*\]", re.I)
# Развалившаяся разметка — «[panel:» без закрытия, «[panel]» без типа. Вырезаем
# тоже: озвученная квадратная скобка звучит как поломка, а она и есть поломка.
_BROKEN = re.compile(r"\[\s*panel\b[^\]]*\]?", re.I)


@dataclass
class PanelMark:
    """Вызов панели: смещение в очищенном тексте, тип и аргумент."""
    char_index: int
    panel: str
    arg: str = ""


@dataclass
class Parsed:
    text: str                       # то, что уйдёт в синтез и в субтитры
    marks: list[PanelMark]
    dropped: int = 0                # сколько тегов отброшено как неизвестные


def parse(text: str) -> Parsed:
    """Вырезать разметку панелей, запомнив позиции.

    Позиции считаются в ОЧИЩЕННОМ тексте: дальше они превращаются в `pts_ms` по
    таймкодам произнесённого, а произносится именно очищенный текст.
    """
    if not text:
        return Parsed("", [], 0)

    out: list[str] = []
    marks: list[PanelMark] = []
    dropped = 0
    pos = 0
    length = 0
    for m in _TAG.finditer(text):
        out.append(text[pos:m.start()])
        length += m.start() - pos
        pos = m.end()
        name = (m.group(1) or "").lower()
        name = ALIASES.get(name, name)
        if name in PANELS:
            marks.append(PanelMark(length, name, (m.group(2) or "").lower()))
        else:
            dropped += 1
    out.append(text[pos:])
    clean = "".join(out)

    # Второй проход: то, что осталось от развалившейся разметки. Считаем
    # отброшенным — модель пыталась вызвать панель и не смогла.
    repaired, n = _BROKEN.subn("", clean)
    dropped += n
    return Parsed(re.sub(r"[ \t]{2,}", " ", repaired).strip(), marks, dropped)


def to_timeline(marks: list[PanelMark], chars: list[dict],
                clause_start_ms: float = 0.0, text_offset: int = 0) -> list[dict]:
    """Символьные смещения -> `pts_ms`. Те же таймкоды, что у висем и субтитров."""
    if not marks:
        return []
    out = []
    n = len(chars)
    for mk in marks:
        local = mk.char_index - text_offset
        if local < 0 or n == 0:
            continue
        out.append({
            "pts_ms": clause_start_ms + chars[min(local, n - 1)]["ms"],
            "panel": mk.panel,
            "arg": mk.arg,
        })
    return out


def material_payload(stage) -> dict:
    """Материал текущего этапа. Ничего не генерируется — берём из сценария."""
    m = getattr(stage, "material", None) if stage is not None else None
    if not m:
        return {}
    return {"kind": m.get("kind", "text"), "title": m.get("title", ""),
            "body": m.get("body", ""), "lang": m.get("lang", ""),
            "series": m.get("series") or []}


def skills_payload(report: dict) -> dict:
    """Карта навыков из уже накопленного отчёта.

    Никакой генерации: фоновая сессия считает оценки по ходу разговора, панель
    только показывает то, что уже есть.
    """
    criteria = report.get("criteria") or []
    rows = [{"key": c.get("key"), "title": c.get("title"),
             "score": c.get("score"), "scale": c.get("scale"),
             "rationale": c.get("rationale") or ""}
            for c in criteria]
    scored = [r["score"] for r in rows if r["score"] is not None]
    return {
        "rows": rows,
        "overall": report.get("overall"),
        "coverage": report.get("coverage"),
        "evaluated": len(scored),
        "total": len(rows),
    }


SYSTEM_HINT = """
Если у текущего этапа есть материал — график, документ, фрагмент кода, — ты
можешь показать его собеседнику, поставив тег прямо в тексте реплики:
[panel:material]. Панель покажет то, что приложено к этапу; придумывать
содержимое не надо и нельзя.

Показывай материал ТОГДА, КОГДА СПРАШИВАЕШЬ О НЁМ: «взгляните на эти цифры —
что здесь произошло в марте?», «вот фрагмент, что бы вы поправили?». Панель —
часть вопроса, а не иллюстрация к ней.

Ещё есть [panel:scenario] — этапы разговора, и [panel:close] — убрать панель.

Это делается ОДНИМ тегом в начале реплики, никаких пояснений про панель вслух
не нужно. Не чаще одного раза в несколько реплик: панель поверх разговора
отвлекает.
""".strip()
