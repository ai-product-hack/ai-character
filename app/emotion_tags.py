"""Теги эмоций внутри текста ответа.

Не отдельный вызов и не тул: при TTFT в 2.25 с лишний round trip недопустим, а
тул к тому же не знает позиции внутри реплики. Модель размечает текст, разметка
вырезается ДО синтеза, а символьные смещения превращаются в `pts_ms` теми же
посимвольными таймкодами, по которым уже едут висемы и субтитры.

Значит эмоция наследует общий PTS и отмену по `generation_id`. Новых механизмов
не появляется.

Разметка:
    [emo:skeptical] в начале — эмоция всей реплики
    [emo:warming]   по ходу  — смена с этого места

Парсер строгий: белый список, неизвестное отбрасывается, при развале разметки —
нейтральная эмоция. Ни один тег не должен просочиться в синтез.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Белый список — он же контракт со слоем эмоции аватара
# (`avatar/expression.config.json`, `EMOTIONS` в `avatar/src/avatar.js`).
# Первые пять лежат на оси «холодно → тепло» и выводятся из накопленных оценок;
# `angry` и `anxious` на ней не лежат — их назначает персона сценария или сама
# модель разметкой. Пока их не было, разгневанный клиент получал `pressing`, а
# тревожный пациент — тот же `pressing`, и второе просто неверно: тревога это
# не давление.
EMOTIONS = ("neutral", "skeptical", "pressing", "warming", "impressed",
            "angry", "anxious")
DEFAULT = "neutral"

# Допускаем и русские написания: модель их предлагает сама.
ALIASES = {
    "скепсис": "skeptical", "скептично": "skeptical", "сомнение": "skeptical",
    "давление": "pressing", "напор": "pressing",
    "тепло": "warming", "теплее": "warming", "одобрение": "warming",
    "интерес": "impressed", "впечатление": "impressed", "уважение": "impressed",
    "нейтрально": "neutral",
    "гнев": "angry", "ярость": "angry", "злость": "angry", "зло": "angry",
    "раздражение": "angry",
    "тревога": "anxious", "волнение": "anxious", "беспокойство": "anxious",
    "растерянность": "anxious", "испуг": "anxious",
}

_TAG = re.compile(r"\[\s*emo\s*:\s*([a-zA-Zа-яёА-ЯЁ_]+)\s*\]", re.I)
# Мусор вида «[emo:» без закрытия или «[emo]» — вырезаем тоже, чтобы не озвучить.
_BROKEN = re.compile(r"\[\s*emo\b[^\]]*\]?", re.I)


@dataclass
class EmotionMark:
    """Смена эмоции: смещение в очищенном тексте и сама эмоция."""
    char_index: int
    emotion: str
    intensity: float = 1.0


@dataclass
class Parsed:
    text: str                       # то, что уйдёт в синтез и в субтитры
    marks: list[EmotionMark]
    dropped: int = 0                # сколько тегов отброшено как неизвестные

    @property
    def opening(self) -> str:
        """Эмоция реплики: первая метка, если она в самом начале."""
        if self.marks and self.marks[0].char_index == 0:
            return self.marks[0].emotion
        return DEFAULT


def normalize(name: str) -> str | None:
    n = (name or "").strip().lower()
    if n in EMOTIONS:
        return n
    return ALIASES.get(n)


def parse(raw: str) -> Parsed:
    """Вырезать теги, запомнив их позиции в ОЧИЩЕННОМ тексте."""
    if not raw:
        return Parsed("", [], 0)

    out, marks, dropped = [], [], 0
    pos = 0
    for m in _TAG.finditer(raw):
        out.append(raw[pos:m.start()])
        emo = normalize(m.group(1))
        if emo is None:
            dropped += 1                       # неизвестный тег просто исчезает
        else:
            marks.append(EmotionMark(sum(len(p) for p in out), emo))
        pos = m.end()
    out.append(raw[pos:])
    text = "".join(out)

    # Развалившаяся разметка: «[emo:» без закрывающей скобки. Вырезаем, чтобы
    # она не ушла в синтез, и считаем как отброшенный тег.
    cleaned = _BROKEN.sub("", text)
    if cleaned != text:
        dropped += 1
        # Позиции меток после вырезанного мусора сдвигаются; проще пересчитать
        # их по ближайшей допустимой границе, чем терять эмоцию целиком.
        shift = len(text) - len(cleaned)
        marks = [EmotionMark(max(0, mk.char_index - (shift if mk.char_index else 0)),
                             mk.emotion, mk.intensity) for mk in marks]
        text = cleaned

    # Схлопываем пробелы, оставшиеся от вырезанных тегов.
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    marks = [mk for mk in marks if 0 <= mk.char_index <= len(text)]
    return Parsed(text, marks, dropped)


def to_timeline(marks: list[EmotionMark], chars: list[dict],
                clause_start_ms: float = 0.0,
                text_offset: int = 0) -> list[dict]:
    """Символьные смещения -> `pts_ms` по таймкодам выравнивания.

    Те же таймкоды, по которым едут висемы и субтитры: отдельного механизма нет.
    Смещение `text_offset` — сколько символов реплики пришлось на предыдущие
    клаузы, потому что таймкоды приходят по клаузам.
    """
    if not marks:
        return []
    # Таймкоды идут по символам расшифровки; пробелы в ней есть, а пунктуации
    # нет, поэтому берём ближайший по порядковому номеру символа.
    out = []
    n = len(chars)
    for mk in marks:
        local = mk.char_index - text_offset
        if local < 0 or n == 0:
            continue
        idx = min(local, n - 1)
        out.append({
            "pts_ms": clause_start_ms + chars[idx]["ms"],
            "emotion": mk.emotion,
            "intensity": mk.intensity,
        })
    return out


SYSTEM_HINT = """
КАЖДУЮ реплику обязательно начинай ровно с одного тега эмоции:
[emo:neutral], [emo:skeptical], [emo:pressing], [emo:warming], [emo:impressed],
[emo:angry] или [emo:anxious]. Если по ходу реплики отношение меняется,
поставь тег в этом месте ещё раз.

Пять первых — это шкала отношения к собеседнику, от холодного к тёплому.
[emo:angry] — открытый гнев, а не усиленное давление: подходит, когда твоя роль
разозлена по существу. [emo:anxious] — тревога и растерянность, а не давление:
подходит, когда роль волнуется или боится.

Теги — единственная разметка, которая разрешена. Никаких других скобок,
звёздочек и пометок в тексте быть не должно: всё, кроме тегов, будет
произнесено вслух.
""".strip()
