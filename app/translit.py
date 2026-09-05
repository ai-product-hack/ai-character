"""Латинские ключи из русских названий.

Методист пишет «Работа с возражениями», а движку нужен `rabota_s_vozrazheniyami`:
ключ уходит в JSON отчёта, в промпт оценщика и в сравнение с ответом модели.
Придумывать идентификаторы руками HR не должен — их генерирует эта таблица.

Таблица переехала сюда из `convert_scenarios.py`: тем же кодом теперь считаются
и id этапов, и ключи критериев, сгенерированных из произвольного текста.
"""
from __future__ import annotations

import re

TABLE = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def slug(text: str, fallback: str = "", limit: int = 40) -> str:
    """«Глубина роли» -> `glubina_roli`. Пустой результат заменяется fallback."""
    out = "".join(TABLE.get(ch, ch if ch.isalnum() else "_") for ch in (text or "").lower())
    out = re.sub(r"_+", "_", out).strip("_")[:limit].strip("_")
    # Ключ, начинающийся с цифры, читается как ошибка в любом языке — префикс.
    if out and out[0].isdigit():
        out = "k_" + out
    return out or fallback


def unique_slugs(titles, prefix: str = "k") -> list[str]:
    """Ключи для списка названий: без повторов и без пустых.

    Повтор ключа роняет сопоставление оценок с критериями — оценка уходит не
    туда, и в отчёте это выглядит как «модель придумала балл».
    """
    seen: dict[str, int] = {}
    out = []
    for i, t in enumerate(titles):
        key = slug(t, f"{prefix}_{i + 1}")
        if key in seen:
            seen[key] += 1
            key = f"{key}_{seen[key]}"
        else:
            seen[key] = 1
        out.append(key)
    return out
