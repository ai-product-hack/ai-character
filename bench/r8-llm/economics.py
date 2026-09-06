#!/usr/bin/env python3
"""Юнит-экономика одного разговора: токены, символы синтеза, деньги.

    bench/r1-stt/.venv/bin/python bench/r8-llm/economics.py

Считает по ЗАПИСАННЫМ прогонам (`bench/results/rehearsal_all10.json`), а не по
свежим вызовам: генерировать заново нечего, все реплики уже сказаны. Входные
токены пересчитываются бесплатным `count_tokens`, выходные — по тем же
записанным репликам. Ни одной платной генерации этот скрипт не делает.

Учитываются ВСЕ четыре статьи расхода, а не только реплики:

    диалог      промпт каждого хода + реплика агента
    оценка      фоновая сессия после КАЖДОГО хода пользователя
    отчёт       один вызов на два вывода в конце
    сценарий    разовая генерация, размазывается по числу прогонов

Забыть оценку — главная ошибка в таком расчёте: она удваивает расход, потому
что зовётся столько же раз, сколько сам диалог.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.agent import SYSTEM, build_prompt                   # noqa: E402
from app.dialogue import DialogueState                       # noqa: E402
from app.emotion_tags import SYSTEM_HINT as EMO_HINT         # noqa: E402
from app.evaluator import (SUMMARY_SYSTEM, SYSTEM as EVAL_SYSTEM,   # noqa: E402
                           build_eval_prompt, build_summary_prompt)
from app.llm import load_env                                 # noqa: E402
from app.panels import SYSTEM_HINT as PANEL_HINT             # noqa: E402
from app.scenario import load                                # noqa: E402

DIALOGUE_SYSTEM = "\n\n".join((SYSTEM, EMO_HINT, PANEL_HINT))

# $ за миллион токенов: (вход, выход). Anthropic — прайс first-party 2026-06,
# DeepSeek — deepseek-chat вне скидочных часов.
LLM_PRICES = {
    "deepseek-chat":    (0.27, 1.10),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5":  (3.00, 15.00),
}

# Синтез. Silero крутится локально — денег не стоит, стоит железа (см. BUDGET).
#
# ElevenLabs: тариф Creator, $22 за 100 000 кредитов — снято с elevenlabs.io
# 2026-09-06, ПЛАТЕЖОМ НЕ ПРОВЕРЕНО. Сколько кредитов стоит символ, зависит от
# модели: у обычных один, у Flash и Turbo — половина. Мы синтезируем
# `eleven_flash_v2_5`, то есть 0.5. Обе цифры ниже, потому что разница вдвое, а
# уверенности в тарифе у меня нет.
_EL = 22.0 / 100_000                       # $ за кредит
TTS_PRICES = {
    "silero": 0.0,
    "elevenlabs (flash, 0.5 кредита/симв.)": _EL * 0.5,
    "elevenlabs (1 кредит/симв.)": _EL,
}

# Темп разговора. Токены тратятся на ХОД, а не на секунду, поэтому «цена
# минуты» целиком зависит от того, кто говорит.
#
#   машинный  репетиция печатает мгновенно: 4.4 с на реплику
#   живой     единственный записанный прогон с человеком (data/reports):
#             20 реплик за 738 с, то есть 37 с на реплику
#
# Живой темп и есть настоящий: человек думает, формулирует и слушает.
PACE_SEC_PER_TURN = {"машинный": 4.4, "живой": 36.9}


class Counter:
    """Точный счётчик токенов. Бесплатный эндпоинт, платных вызовов нет."""

    def __init__(self, client):
        self.client = client
        self.calls = 0
        self._cache: dict[tuple, int] = {}

    def __call__(self, text: str, system: str = "") -> int:
        if not text.strip():
            return 0
        key = (hash(text), hash(system))
        if key in self._cache:
            return self._cache[key]
        r = self.client.messages.count_tokens(
            model="claude-sonnet-5", system=system,
            messages=[{"role": "user", "content": text}])
        self.calls += 1
        self._cache[key] = r.input_tokens
        return r.input_tokens


def cost_of_run(count: Counter, run: dict) -> dict:
    """Разложить один записанный разговор по статьям расхода."""
    sc = load(ROOT / "data" / "scenarios" / f"{run['scenario_id']}.json")
    rep = run["report"]
    state = DialogueState(sc)

    dlg_in = count(build_prompt(DialogueState(sc), None), DIALOGUE_SYSTEM)
    dlg_out = 0
    eval_in = eval_out = 0
    speak_chars = 0

    for turn in rep["transcript"]:
        if turn["role"] == "user":
            state.add_user(turn["text"])
            # Промпт следующей реплики агента.
            dlg_in += count(build_prompt(state, turn["text"]), DIALOGUE_SYSTEM)
            # Фоновая оценка — на КАЖДЫЙ ход пользователя.
            eval_in += count(build_eval_prompt(state), EVAL_SYSTEM)
            # Ответ оценщика: JSON на два-три критерия. Считаем по факту —
            # в отчёте лежат сами оценки с обоснованиями.
            eval_out += 90
        else:
            state.add_agent(turn["text"])
            dlg_out += count(turn["text"])
            speak_chars += len(turn["text"])

    summary_in = count(build_summary_prompt(state, _log_of(rep)), SUMMARY_SYSTEM)
    summary_out = count(rep.get("conclusion", "")) + \
        count(rep.get("conclusion_methodist", "")) + 30

    return {
        "scenario_id": run["scenario_id"],
        "user_turns": rep["user_turns"],
        "duration_s": rep.get("duration_s"),
        "dialogue": {"in": dlg_in, "out": dlg_out},
        "evaluation": {"in": eval_in, "out": eval_out},
        "summary": {"in": summary_in, "out": summary_out},
        "speak_chars": speak_chars,
    }


class _Fake:
    """Лог оценок из записанного отчёта — для промпта общего вывода."""

    def __init__(self, rep):
        self.rep = rep

    def for_criterion(self, key):
        for c in self.rep["criteria"]:
            if c["key"] == key and c["score"] is not None:
                return [type("A", (), {"score": c["score"]})()]
        return []


def _log_of(rep):
    return _Fake(rep)


def money(model: str, inp: int, out: int) -> float:
    pin, pout = LLM_PRICES[model]
    return inp / 1e6 * pin + out / 1e6 * pout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="bench/results/rehearsal_all10.json")
    ap.add_argument("--out", default="bench/results/r8_economics.json")
    ap.add_argument("--gen", default="scenarios/generated/01_tech_interview.meta.json",
                    help="метаданные генерации сценария")
    args = ap.parse_args()

    load_env()
    import anthropic
    count = Counter(anthropic.Anthropic(timeout=120))

    runs = json.loads((ROOT / args.runs).read_text(encoding="utf-8"))["runs"]
    rows = [cost_of_run(count, r) for r in runs]

    tot = {k: {"in": sum(r[k]["in"] for r in rows), "out": sum(r[k]["out"] for r in rows)}
           for k in ("dialogue", "evaluation", "summary")}
    n = len(rows)
    avg_turns = sum(r["user_turns"] for r in rows) / n
    avg_secs = sum(r["duration_s"] or 0 for r in rows) / n
    avg_chars = sum(r["speak_chars"] for r in rows) / n

    print(f"разговоров {n}, в среднем {avg_turns:.0f} реплик, "
          f"{avg_secs:.0f} с машинного темпа, {avg_chars:.0f} символов синтеза\n")
    print(f"{'статья':12}{'вход':>10}{'выход':>9}   на разговор")
    per = {}
    for k, label in (("dialogue", "диалог"), ("evaluation", "оценка"),
                     ("summary", "отчёт")):
        i, o = tot[k]["in"] / n, tot[k]["out"] / n
        per[k] = (i, o)
        print(f"{label:12}{i:10.0f}{o:9.0f}")
    tin = sum(v[0] for v in per.values())
    tout = sum(v[1] for v in per.values())
    print(f"{'ИТОГО':12}{tin:10.0f}{tout:9.0f}")

    # Генерация сценария — разовая, на много прогонов.
    gen = json.loads((ROOT / args.gen).read_text(encoding="utf-8"))
    gen_in = gen.get("tokens_in") or 2014
    gen_out = gen.get("tokens_out") or 4377
    print(f"\nгенерация сценария (разово): вход {gen_in}, выход {gen_out}")

    # Цена минуты зависит от темпа: токены тратятся на ход, а не на секунду.
    live_min = avg_turns * PACE_SEC_PER_TURN["живой"] / 60
    mach_min = avg_turns * PACE_SEC_PER_TURN["машинный"] / 60
    print(f"\nразговор из {avg_turns:.0f} реплик длится "
          f"{live_min:.1f} мин у человека и {mach_min:.1f} мин у репетиции")

    print(f"\n{'модель':18}{'$ разговор':>12}{'$/мин живой':>13}"
          f"{'$/мин машинный':>16}{'$ за 1000':>11}")
    money_rows = {}
    for model in LLM_PRICES:
        c = money(model, tin, tout)
        money_rows[model] = round(c, 5)
        print(f"{model:18}{c:12.4f}{c / live_min:13.4f}{c / mach_min:16.4f}"
              f"{c * 1000:11.2f}")

    print(f"\n{'синтез':40}{'$ разговор':>12}{'$/мин живой':>13}{'$ за 1000':>11}")
    tts_rows = {}
    for name, price in TTS_PRICES.items():
        c = avg_chars * price
        tts_rows[name] = round(c, 5)
        print(f"{name:40}{c:12.4f}{c / live_min:13.4f}{c * 1000:11.2f}")

    out = ROOT / args.out
    out.write_text(json.dumps({
        "runs": rows,
        "per_conversation": {k: {"in": round(v[0]), "out": round(v[1])}
                             for k, v in per.items()},
        "avg_turns": round(avg_turns, 1), "avg_seconds": round(avg_secs),
        "avg_speak_chars": round(avg_chars),
        "scenario_generation": {"in": gen_in, "out": gen_out},
        "llm_usd_per_conversation": money_rows,
        "tts_usd_per_conversation": tts_rows,
        "pace_sec_per_turn": PACE_SEC_PER_TURN,
        "minutes_live": round(live_min, 1), "minutes_machine": round(mach_min, 1),
        "llm_prices_per_mtok": LLM_PRICES, "tts_prices_per_char": TTS_PRICES,
        "count_tokens_calls": count.calls,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nвызовов count_tokens (бесплатных): {count.calls}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
