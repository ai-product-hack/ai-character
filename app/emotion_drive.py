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

# Порядок важен: от холодного к тёплому. Совпадает с белым списком эмоций.
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


def _ratio(assessments, scales: dict) -> list[float]:
    """Оценки в долях от своей шкалы: критерии бывают с разными потолками."""
    out = []
    for a in assessments:
        scale = scales.get(a.criterion) or 5
        if a.score is None or scale <= 1:
            continue
        # Шкала начинается с единицы, а не с нуля: балл 1 из 5 — это дно, 0.0,
        # а не 0.2. Иначе худший ответ выглядел бы как пятая часть отличного.
        out.append(max(0.0, min(1.0, (a.score - 1) / (scale - 1))))
    return out


def mood_from(log, criteria, min_scored: int = 2, recent: int = 4) -> Mood:
    """Настроение интервьюера по накопленным оценкам.

    Пока оценок мало, настроения нет: судить о человеке по одному ответу
    нечестно и на лице читается как случайность.
    """
    scales = {c.key: c.scale for c in criteria}
    vals = _ratio(log.snapshot(), scales)
    if len(vals) < min_scored:
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
