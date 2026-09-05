"""Эмоция из накопленной оценки, а не из щедрости модели.

Разметку `[emo:...]` модель ставит скупо: замерено 4–5 тегов на 80 реплик. На
показе лицо будет эмоционально ровным, и никто не узнает, что эмоции вообще
есть.

Надёжный источник уже работает рядом: фоновая сессия непрерывно оценивает
ответы по критериям. Отвечает хорошо — интервьюер теплеет, плывёт — холодеет.
Эмоция появляется от механики продукта, а не от настроения модели, и дуга за
сессию видна гарантированно.

Разметка от модели остаётся сверху как уточнение: если она сказала
`[emo:impressed]`, это точнее среднего балла, и спорить с ней незачем.
"""
from __future__ import annotations

from dataclasses import dataclass

from .emotion_tags import EMOTIONS

# Порядок важен: от холодного к тёплому. Это ОСЬ ОЦЕНОК — то, что выводится из
# накопленных баллов, и только это. Палитра шире: `angry` и `anxious` на оси не
# лежат и сюда не входят, потому что их не из чего вывести баллами. Их
# назначает персона сценария или разметка модели.
COLD_TO_WARM = ("skeptical", "pressing", "neutral", "warming", "impressed")


@dataclass
class Mood:
    """Эмоция, выведенная из оценок, и почему именно она."""
    emotion: str
    intensity: float
    ratio: float | None          # средняя доля от максимума, 0..1
    scored: int                  # по скольким оценкам судим
    trend: float                 # насколько свежие оценки лучше ранних

    def to_dict(self) -> dict:
        return {"emotion": self.emotion, "intensity": round(self.intensity, 2),
                "ratio": None if self.ratio is None else round(self.ratio, 3),
                "scored": self.scored, "trend": round(self.trend, 3)}


def _ratio(assessments, bounds: dict) -> list[float]:
    """Оценки в долях от своей шкалы: критерии бывают с разными потолками.

    Границы берутся у самого критерия. Шкала в сценарии записана строкой
    («1-5»), и первая версия принимала её за число — падало на живых данных,
    хотя юнит-тесты с int-шкалой проходили.
    """
    out = []
    for a in assessments:
        lo, hi = bounds.get(a.criterion) or (1, 5)
        if a.score is None or hi <= lo:
            continue
        # Низ шкалы — это дно, 0.0, а не доля от потолка. Иначе худший ответ
        # выглядел бы как пятая часть отличного.
        out.append(max(0.0, min(1.0, (a.score - lo) / (hi - lo))))
    return out


def mood_from(log, criteria, min_scored: int = 2, recent: int = 4,
              baseline: str = "neutral") -> Mood:
    """Настроение интервьюера по накопленным оценкам.

    Пока оценок мало, судить о человеке нечестно — но лицо всё равно должно
    что-то выражать, и здесь его задаёт персона. Разгневанный клиент начинает
    разговор разгневанным, а не нейтральным: стартовая эмоция персоны и есть
    начало дуги, ради которой упражнение существует. Дальше её вытесняют
    оценки.

    Белый список эмоций тот же, что у разметки `[emo:...]`, — контракт с
    эмоциональным слоем аватара не меняется.
    """
    vals = _ratio(log.snapshot(), {c.key: c.bounds for c in criteria})
    if len(vals) < min_scored:
        # Проверяем по всей палитре, а не по оси оценок: разгневанный клиент
        # начинает разговор разгневанным, и вывести это из баллов нельзя —
        # баллов ещё нет.
        if baseline in EMOTIONS and baseline != "neutral":
            # Стартовая эмоция звучит заметно, но не на максимуме: ей ещё
            # предстоит меняться, и начинать с потолка значит не оставить хода.
            return Mood(baseline, 0.6, None, len(vals), 0.0)
        return Mood("neutral", 0.0, None, len(vals), 0.0)

    tail = vals[-recent:]
    ratio = sum(tail) / len(tail)
    # Тренд: свежие против всех предыдущих. Растёт — теплеем быстрее, чем
    # следует из одного среднего; проседает — наоборот.
    head = vals[:-recent] or vals
    trend = ratio - sum(head) / len(head)

    if ratio >= 0.75:
        emotion = "impressed"
    elif ratio >= 0.55:
        emotion = "warming"
    elif ratio >= 0.40:
        emotion = "neutral"
    elif ratio >= 0.25:
        emotion = "pressing"
    else:
        emotion = "skeptical"

    # Интенсивность — насколько далеко от середины. Ровно посередине лицо
    # нейтрально, и подмешивать туда нечего.
    intensity = min(1.0, abs(ratio - 0.5) * 2 + abs(trend))
    return Mood(emotion, intensity, ratio, len(vals), trend)
