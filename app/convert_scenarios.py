#!/usr/bin/env python3
"""Приведение сценариев R-фазы к формату движка.

Старый формат описывал реплики агента дословно (`agent`) и ожидание словами
(`expect`). Движку нужно другое: этап задаёт ЦЕЛЬ и ПОДСКАЗКУ, а реплику
сочиняет модель — иначе агент не сможет ни переспросить, ни отреагировать на
неожиданный ответ, а это половина смысла тренажёра.

Дословная реплика сохраняется как `opening`: она годится затравкой этапа и как
образец тона для модели, но перестаёт быть единственным, что агент умеет сказать.

    python3 app/convert_scenarios.py
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "scenarios"


def slug(goal: str, n: int) -> str:
    """id этапа из цели: «глубина роли» -> «glubina_roli»."""
    table = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
        "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
        "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
        "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
        "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    }
    out = "".join(table.get(ch, ch if ch.isalnum() else "_") for ch in goal.lower())
    out = re.sub(r"_+", "_", out).strip("_")
    return out or f"stage_{n}"


def hint_for(step: dict, persona: str) -> str:
    """Подсказка модели. Собирается из того, что методист уже написал."""
    parts = [f"Цель этапа: {step['goal']}."]
    if step.get("expect"):
        parts.append(f"От собеседника ждём: {step['expect']}.")
    if step.get("trigger"):
        parts.append(f"Этот этап особенно уместен, когда {step['trigger']}.")
    if step.get("agent"):
        parts.append(f"Тон и направление задаёт реплика: «{step['agent']}» — "
                     f"её можно сказать дословно или своими словами.")
    return " ".join(parts)


def convert(old: dict) -> dict:
    steps = old.get("steps", [])
    return {
        "id": old["id"],
        "title": old["title"],
        "type": old.get("type", "generic"),
        "persona": old.get("agent_persona", ""),
        "stages": [{
            "id": slug(s.get("goal", ""), s.get("n", i + 1)),
            "goal": s.get("goal", ""),
            "hint": hint_for(s, old.get("agent_persona", "")),
            # Условие перехода — текст для модели. Сопоставление семантическое:
            # строковое сравнение не работает на свободных формулировках.
            "advance_when": s.get("expect", "собеседник ответил по существу"),
            "opening": s.get("agent", ""),
        } for i, s in enumerate(steps)],
        "criteria": old.get("criteria", []),
    }


def main():
    sys.path.insert(0, str(ROOT))
    from app.scenario import Scenario

    files = sorted(SRC.glob("*.json"))
    if not files:
        raise SystemExit(f"нет сценариев в {SRC}")
    for f in files:
        old = json.loads(f.read_text(encoding="utf-8"))
        if "stages" in old:
            print(f"  {f.name}: уже в новом формате, пропускаю")
            continue
        new = convert(old)
        problems = Scenario.from_dict(new).validate()
        if problems:
            raise SystemExit(f"{f.name}: " + "; ".join(problems))
        f.write_text(json.dumps(new, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  {f.name}: {len(new['stages'])} этапов, "
              f"{len(new['criteria'])} критериев")
    print(f"готово: {len(files)} сценариев")


if __name__ == "__main__":
    main()
