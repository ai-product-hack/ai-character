#!/usr/bin/env python3
"""Прогон сценария через движок. Дымовой тест протокола на живой модели.

    python3 app/run_dialogue.py --provider stub
    python3 app/run_dialogue.py --provider deepseek --scenario s1_interview_backend

Пользователя изображает список заготовленных ответов: цель не в качестве
диалога, а в том, выдержит ли модель контракт действий на всей длине сценария.
"""
import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import llm as llm_mod, report as report_mod       # noqa: E402
from app.agent import Agent                                # noqa: E402
from app.dialogue import DialogueState                     # noqa: E402
from app.scenario import load_all                          # noqa: E402

# Ответы «пользователя» — свои для каждого сценария. Общий список был ошибкой
# замера: в сценарии про холодный звонок собеседник отвечал репликами про
# архитектуру сервиса, агент законно не переходил дальше, и это выглядело как
# поломка движка.
USER_LINES = {
    "s1_interview_backend": [
        "Я backend-разработчик, последние два года делал сервис выдачи рекомендаций.",
        "Лично я держал слой кеширования и переписал выборку с синхронной на батчевую.",
        "Три сервиса: приём событий, счётчик и выдача. Разделили из-за разной нагрузки.",
        "Пик около двух тысяч запросов в секунду, задержка сто миллисекунд на p95.",
        "Упал кеш, вся нагрузка ушла в базу. Добавили предохранитель и прогрев.",
        "Одной очередью не вышло бы: у выдачи и счётчика разные требования по задержке.",
        "Да, как у вас устроено ревью и кто принимает архитектурные решения?",
    ],
    "s2_interview_stress": [
        "Я рассчитываю на вилку от двухсот шестидесяти до двухсот девяноста на руки.",
        "Понимаю, что выше вашей вилки. Готов обсуждать, но не ниже двухсот пятидесяти.",
        "Цифра из двух офферов на руках и из того, что я закрываю и бэкенд, и инфраструктуру.",
        "Обучение полезно, но оно не закрывает разницу в деньгах. Мне важен оклад.",
        "Я подожду, пока вы сформулируете. Мне нечего добавить к сказанному.",
        "До завтра решение принять готов, если получу письменный оффер сегодня.",
    ],
    "s3_sales_cold": [
        "Пётр, компания «Тензор». Мы сокращаем сроки согласования закупок примерно вдвое.",
        "Понимаю. Уточню один момент: сколько у вас в среднем занимает согласование заявки?",
        "Мы работаем с производственными компаниями от пятисот человек, у нас сорок внедрений.",
        "От трёхсот тысяч в год за отдел. Точная цифра зависит от числа согласующих.",
        "Пришлю. Чтобы письмо было по делу, скажите, что для вас важнее: сроки или контроль?",
        "Предлагаю двадцать минут в четверг: покажу, как это выглядит на ваших заявках.",
    ],
    "s4_sales_objections": [
        "Понял. А что именно вошло в их расчёт — там та же глубина интеграции?",
        "Тогда давайте сравним по пунктам: у них есть маршрутизация и аудит изменений?",
        "Внедрение ведём мы, от вашей команды нужен один человек на два часа в неделю.",
        "Хорошо, что нужно финансовому директору, чтобы он сказал да? Цифры окупаемости?",
        "В следующем квартале цена будет другая. Но давайте зафиксирую текущую до конца месяца.",
        "Да, есть пилот на один отдел, месяц, без обязательств по продлению.",
        "От вас — контакт ответственного и доступ к тестовому контуру. Мы делаем остальное.",
    ],
    "s5_knowledge_check": [
        "ФИО, паспортные данные, адрес, телефон, электронная почта.",
        "Ещё биометрия, данные о здоровье и сведения о доходах.",
        "Принимаю заявление письменно, проверяю личность, передаю ответственному за ПДн.",
        "Тридцать дней с момента получения обращения.",
        "Отказываю. Выгрузка передаётся только по служебной почте и с обоснованием.",
        "Точно так же отказываю и фиксирую обращение у ответственного за ПДн.",
        "Сообщаю ответственному за ПДн и фиксирую инцидент, дальше по регламенту.",
    ],
}
FALLBACK_LINES = ["Понимаю, спасибо.", "Мне здесь добавить нечего."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="stub")
    ap.add_argument("--scenario", default=None, help="id; по умолчанию все")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--out", default=None, help="куда сложить отчёты и сырьё")
    args = ap.parse_args()

    scenarios = load_all(ROOT / "data" / "scenarios")
    if args.scenario:
        scenarios = [s for s in scenarios if s.id == args.scenario] or scenarios
    model = llm_mod.build(args.provider)

    runs = []
    for sc in scenarios:
        state = DialogueState(sc)
        agent = Agent(model)
        t0 = time.perf_counter()
        stats = {"fell_back": 0, "advanced": 0, "observed": 0, "calls": 0,
                 "stay": 0, "next_stage": 0, "finish": 0, "evaluate": 0, "forced": 0}

        reply, happened = agent.step(state)
        stats["calls"] += 1
        stats["fell_back"] += happened["fell_back"]
        print(f"\n=== {sc.id} | {sc.title}")
        print(f"[{state.stage_id}] агент: {reply.speakable[:150]}")

        for i in range(args.max_turns):
            if state.finished:
                break
            lines = USER_LINES.get(sc.id, FALLBACK_LINES)
            user = lines[i] if i < len(lines) else FALLBACK_LINES[i % len(FALLBACK_LINES)]
            print(f"           польз.: {user[:110]}")
            reply, happened = agent.step(state, user)
            stats["calls"] += 1
            for k in ("fell_back", "advanced", "observed", "forced"):
                stats[k] += bool(happened[k])
            stats[happened["action"]] = stats.get(happened["action"], 0) + 1
            mark = " [откат]" if happened["fell_back"] else ""
            act = happened["action"]
            print(f"[{state.stage_id}] агент{mark} ({act}): {reply.speakable[:150]}")

        rep = report_mod.build(state)
        runs.append({
            "scenario_id": sc.id, "finished": state.finished,
            "stages_reached": rep.stages_reached, "stages_total": rep.stages_total,
            "user_turns": rep.user_turns, "coverage": round(rep.coverage, 2),
            "forced_advances": state.forced_advances,
            "overall": rep.overall, "wall_s": round(time.perf_counter() - t0, 1),
            **stats,
        })
        print(f"--- {sc.id}: finished={state.finished} "
              f"этапов {rep.stages_reached}/{rep.stages_total} "
              f"откатов {stats['fell_back']}/{stats['calls']} "
              f"покрытие критериев {rep.coverage:.0%}")

        if args.out:
            d = pathlib.Path(args.out)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{sc.id}.report.json").write_text(rep.to_json(), encoding="utf-8")

    print("\n=== итог ===")
    done = sum(r["finished"] for r in runs)
    print(f"дошли до finish: {done}/{len(runs)}")
    falls = sum(r["fell_back"] for r in runs)
    calls = sum(r["calls"] for r in runs)
    print(f"нарушений протокола (ответ без действия): {falls} из {calls} вызовов "
          f"({100 * falls / calls if calls else 0:.0f}%)")
    for a in ("stay", "next_stage", "evaluate", "finish"):
        n = sum(r.get(a, 0) for r in runs)
        print(f"  {a:<11} {n}")
    if args.out:
        p = pathlib.Path(args.out) / "runs.json"
        p.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {p}")
    return 0 if done == len(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
