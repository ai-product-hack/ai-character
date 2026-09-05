#!/usr/bin/env python3
"""Кто отвечает быстрее: DeepSeek или Anthropic. И во что обходится сценарий.

    bench/r1-stt/.venv/bin/python bench/r8-llm/run.py
    bench/r1-stt/.venv/bin/python bench/r8-llm/run.py --repeats 2

Меряется TTFT — время до ПЕРВОГО токена, а не общее время ответа. Для
тренажёра важно только оно: первый звук человек слышит, когда готова первая
клауза, а она готова вскоре после первого токена. Общее время ответа не влияет
ни на что, потому что синтез идёт клаузами и обгоняет генерацию.

Нагрузка настоящая: промпты собираются движком (`build_prompt`) из настоящего
сценария на трёх глубинах разговора — первый ход, середина, конец. Промпт растёт
с историей, и на длинном контексте модели ведут себя иначе, чем на коротком.

Замер намеренно маленький: Anthropic платный. Три промпта на модель, ответ
ограничен теми же 300 токенами, что и в бою. Расчёт стоимости сценария при этом
НЕ оценочный — токены считаются точно, по записи настоящего прогона диалога,
бесплатным `count_tokens`.

Мышление у Anthropic выключено намеренно. В голосовом тренажёре бюджет 3000 мс
на весь путь, и модель, которая думает перед ответом, в него не помещается ни
при каком качестве.
"""
import argparse
import json
import pathlib
import random
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.agent import SYSTEM, build_prompt                   # noqa: E402
from app.dialogue import DialogueState                       # noqa: E402
from app.emotion_tags import SYSTEM_HINT as EMO_HINT         # noqa: E402
from app.llm import load_env                                 # noqa: E402
from app.panels import SYSTEM_HINT as PANEL_HINT             # noqa: E402
from app.scenario import load                                # noqa: E402

FULL_SYSTEM = "\n\n".join((SYSTEM, EMO_HINT, PANEL_HINT))
MAX_TOKENS = 300            # столько же, сколько в бою

# Цены за миллион токенов, доллары. Anthropic — прайс first-party на 2026-06.
# DeepSeek — deepseek-chat, цена со скидкой вне часов пик не учитывается.
PRICES = {
    "deepseek-chat":     (0.27, 1.10),
    "claude-haiku-4-5":  (1.00, 5.00),
    "claude-sonnet-5":   (3.00, 15.00),
    "claude-opus-5":     (5.00, 25.00),
}

ANTHROPIC = [m for m in PRICES if m.startswith("claude")]


# ------------------------------------------------------------------ нагрузка

def prompts_at_depths(scenario_id: str = "s1_interview_backend") -> list[tuple[str, str]]:
    """Промпты движка на трёх глубинах: пустая история, середина, конец.

    Берём настоящий сценарий и настоящие реплики прогона, а не выдуманный
    текст: длина и форма промпта — половина ответа на вопрос «кто быстрее».
    """
    from app.run_dialogue import USER_LINES
    sc = load(ROOT / "data" / "scenarios" / f"{scenario_id}.json")
    lines = USER_LINES.get(scenario_id, [])
    state = DialogueState(sc)
    out = [("первый ход", build_prompt(state, None))]
    for i, line in enumerate(lines):
        state.add_agent(f"Реплика агента номер {i + 1}, обычной длины для этого "
                        f"разговора: вопрос и уточнение к нему.")
        state.add_user(line)
        if i == 2:
            out.append(("середина", build_prompt(state, line)))
        if i == min(6, len(lines) - 1):
            out.append(("конец", build_prompt(state, line)))
            break
    return out


# -------------------------------------------------------------------- замеры

def measure_deepseek(system: str, prompt: str) -> dict:
    """TTFT потокового DeepSeek тем же клиентом, что в бою."""
    import urllib.request

    load_env()
    import os
    key = os.environ["DEEPSEEK_API_KEY"]
    body = json.dumps({
        "model": "deepseek-chat", "stream": True, "max_tokens": MAX_TOKENS,
        "temperature": 0.7, "stream_options": {"include_usage": True},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})

    t0 = time.perf_counter()
    ttft = None
    usage = {}
    text = []
    with urllib.request.urlopen(req, timeout=90) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    text.append(piece)
    return {"ttft_ms": ttft, "total_ms": (time.perf_counter() - t0) * 1000,
            "in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens"),
            "chars": len("".join(text))}


def warmup(client, model: str) -> float:
    """Прогреть соединение до замера.

    Первый вызов к каждой модели платит установку соединения, и в сыром
    прогоне это било по той модели, которая шла первой: у Haiku выходило
    751/821/807 мс на прогретом и 2363/2132/3374 на холодном — то есть замер
    сравнивал не модели, а порядок вызовов. Ответ на один токен стоит доли
    цента и снимает артефакт целиком.
    """
    t0 = time.perf_counter()
    kw = {"thinking": {"type": "disabled"}} if model in (
        "claude-opus-5", "claude-sonnet-5") else {}
    try:
        client.messages.create(model=model, max_tokens=1,
                               messages=[{"role": "user", "content": "."}], **kw)
    except Exception:                                        # noqa: BLE001
        pass
    return (time.perf_counter() - t0) * 1000


def measure_anthropic(client, model: str, system: str, prompt: str) -> dict:
    kw = {}
    # Мышление выключаем там, где это принимает модель. У Haiku 4.5 его и так
    # нет: оно включается только явным бюджетом.
    if model in ("claude-opus-5", "claude-sonnet-5"):
        kw["thinking"] = {"type": "disabled"}
    t0 = time.perf_counter()
    ttft = None
    with client.messages.stream(model=model, max_tokens=MAX_TOKENS, system=system,
                                messages=[{"role": "user", "content": prompt}],
                                **kw) as stream:
        for event in stream:
            if event.type == "content_block_delta" and ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
        final = stream.get_final_message()
    text = "".join(b.text for b in final.content if b.type == "text")
    return {"ttft_ms": ttft, "total_ms": (time.perf_counter() - t0) * 1000,
            "in": final.usage.input_tokens, "out": final.usage.output_tokens,
            "chars": len(text)}


# --------------------------------------------------- токены за весь сценарий

def tokens_per_scenario(client, run: dict) -> dict:
    """Точный расход за ОДИН записанный сценарий, без единой генерации.

    Промпт каждого хода пересобирается движком по записанной расшифровке, и
    входные токены считаются `count_tokens` — он бесплатный. Выходные берутся
    из тех же записанных реплик. Оценивать тут нечего, всё считается.
    """
    sc = load(ROOT / "data" / "scenarios" / f"{run['scenario_id']}.json")
    state = DialogueState(sc)
    turns = run["report"]["transcript"]
    prompts, replies = [], []
    for t in turns:
        if t["role"] == "user":
            prompts.append(build_prompt(state, t["text"]))
            state.add_user(t["text"])
        else:
            replies.append(t["text"])
            state.add_agent(t["text"])
    # Первый ход агента идёт по пустой истории — его промпт тоже считаем.
    prompts.insert(0, build_prompt(DialogueState(sc), None))

    def count(text: str, as_output: bool = False) -> int:
        r = client.messages.count_tokens(
            model="claude-sonnet-5",
            system="" if as_output else FULL_SYSTEM,
            messages=[{"role": "user", "content": text}])
        return r.input_tokens

    inp = sum(count(p) for p in prompts)
    out = sum(count(r, as_output=True) for r in replies)
    # Сколько из входа — это системный промпт, пересылаемый на КАЖДОМ ходу.
    # Ровно его и срезает кеширование префикса, поэтому число полезное.
    # Считаем его как обычное сообщение: пустое содержимое API не принимает.
    system_once = count(FULL_SYSTEM, as_output=True)
    return {"scenario_id": run["scenario_id"], "turns": len(prompts),
            "in": inp, "out": out,
            "system_repeated": system_once * len(prompts)}


def money(model: str, inp: int, out: int) -> float:
    pin, pout = PRICES[model]
    return inp / 1e6 * pin + out / 1e6 * pout


# ---------------------------------------------------------------------- ход

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1,
                    help="прогонов на каждый промпт; больше — дороже")
    ap.add_argument("--scenario", default="s1_interview_backend")
    ap.add_argument("--out", default="bench/results/r8_llm.json")
    ap.add_argument("--seed", type=int, default=7, help="порядок вызовов")
    ap.add_argument("--only-tokens", action="store_true",
                    help="только бесплатный подсчёт токенов, без генераций")
    ap.add_argument("--skip-tokens", action="store_true",
                    help="не считать расход за сценарий (он и так бесплатный)")
    args = ap.parse_args()

    load_env()
    import anthropic
    client = anthropic.Anthropic(timeout=120)

    work = prompts_at_depths(args.scenario)
    print(f"промпты движка: " +
          ", ".join(f"{name} {len(p)} симв." for name, p in work))
    print(f"моделей 4, прогонов на промпт {args.repeats} — "
          f"{4 * len(work) * args.repeats} вызовов")
    print("прогрев соединений: " + "  ".join(
        f"{m.split('-', 1)[1]} {warmup(client, m):.0f} мс" for m in ANTHROPIC))
    # DeepSeek прогревается тем же способом — своим первым же вызовом.
    measure_deepseek("Отвечай одним словом.", "Скажи «да».")
    print()

    # Порядок вызовов перемешивается. Сгруппированный (все прогоны одной
    # модели подряд) давал систематическую ошибку: первый вызов в группе
    # стабильно медленнее остальных, и страдала та модель, что шла первой.
    # Прогрев соединения это не снимает — значит дело не в нём, а в порядке.
    plan = [] if args.only_tokens else [(name, prompt, model)
            for name, prompt in work
            for model in ["deepseek-chat"] + ANTHROPIC
            for _ in range(args.repeats)]
    random.Random(args.seed).shuffle(plan)

    rows = []
    if True:
        if True:
            for name, prompt, model in plan:
                try:
                    r = (measure_deepseek(FULL_SYSTEM, prompt) if model == "deepseek-chat"
                         else measure_anthropic(client, model, FULL_SYSTEM, prompt))
                except Exception as e:                       # noqa: BLE001
                    print(f"  {model:18} {name:10} ОШИБКА {type(e).__name__}: {e}")
                    continue
                r.update(model=model, depth=name)
                rows.append(r)
                print(f"  {model:18} {name:10} TTFT {r['ttft_ms']:6.0f} мс  "
                      f"всего {r['total_ms']:6.0f} мс  "
                      f"токенов {r['in']}/{r['out']}  {r['chars']} симв.")

    print("\n================== TTFT, мс ==================")
    print(f"{'модель':20}{'медиана':>9}{'мин':>7}{'макс':>7}   по глубинам")
    summary = {}
    for model in ["deepseek-chat"] + ANTHROPIC:
        got = [r for r in rows if r["model"] == model and r["ttft_ms"]]
        if not got:
            continue
        t = [r["ttft_ms"] for r in got]
        by_depth = "  ".join(f"{r['depth']}: {r['ttft_ms']:.0f}" for r in got)
        summary[model] = {"ttft_median": round(statistics.median(t)),
                          "ttft_min": round(min(t)), "ttft_max": round(max(t)),
                          "total_median": round(statistics.median(
                              [r["total_ms"] for r in got]))}
        print(f"{model:20}{statistics.median(t):9.0f}{min(t):7.0f}{max(t):7.0f}   {by_depth}")

    per_scenario = None
    if not args.skip_tokens:
        runs = json.loads((ROOT / "bench" / "results" / "rehearsal_all10.json")
                          .read_text(encoding="utf-8"))["runs"]
        run = next(r for r in runs if r["scenario_id"] == args.scenario)
        per_scenario = tokens_per_scenario(client, run)
        print(f"\n========= один сценарий целиком: {per_scenario['scenario_id']} =========")
        share = per_scenario["system_repeated"] / per_scenario["in"] * 100
        print(f"ходов {per_scenario['turns']}, "
              f"входных токенов {per_scenario['in']}, "
              f"выходных {per_scenario['out']}")
        print(f"из входных {per_scenario['system_repeated']} ({share:.0f}%) — "
              f"системный промпт, пересланный на каждом ходу")
        print(f"\n{'модель':20}{'$ за сценарий':>15}{'$ за 100':>11}")
        for model in PRICES:
            c = money(model, per_scenario["in"], per_scenario["out"])
            print(f"{model:20}{c:15.4f}{c * 100:11.2f}")
            summary.setdefault(model, {})["usd_per_scenario"] = round(c, 5)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "max_tokens": MAX_TOKENS, "repeats": args.repeats,
        "scenario": args.scenario, "prices_per_mtok": PRICES,
        "per_scenario_tokens": per_scenario,
        "summary": summary, "calls": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
