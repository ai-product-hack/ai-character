#!/usr/bin/env python3
"""Прогон тринадцати входов через собственную панель.

    bench/r1-stt/.venv/bin/python app/run_generation.py            # через сервер
    bench/r1-stt/.venv/bin/python app/run_generation.py --direct   # без сервера

Цель — не положить в репозиторий тринадцать JSON, а доказать, что генератор
общий: один и тот же вызов делает и техническое интервью, и разгневанного
клиента, и аттестацию по регламенту. Поэтому входы различаются МЕХАНИКОЙ
диалога, а не темой — тринадцать сценариев про продажи доказывали бы только
то, что модель умеет говорить про продажи.

По умолчанию идёт через HTTP тем же путём, что браузер: `/api/generate` и
опрос `/api/generate?id=`. Прогон мимо сервера проверял бы не панель, а
библиотеку.

Кладёт результаты в `scenarios/generated/` и печатает таблицу: за сколько,
со скольких попыток, сколько этапов и критериев, и что в сценарии вызывает
сомнения. Оценка качества — отдельно, руками, в `scenarios/REPORT.md`:
автоматически «шаблонную ерунду» от осмысленного результата не отличить.
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.scenario import Scenario                          # noqa: E402
from app import generator as gen_mod                       # noqa: E402

PROMPTS = ROOT / "scenarios" / "prompts"
OUT = ROOT / "scenarios" / "generated"


def via_http(base: str, text: str, kind: str, timeout: float = 180) -> dict:
    """Тем же путём, что браузер: запуск задания и опрос прогресса."""
    req = urllib.request.Request(
        f"{base}/api/generate",
        data=json.dumps({"text": text, "kind": kind}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        job = json.loads(r.read().decode())
    if "error" in job:
        raise RuntimeError(job["error"])
    deadline = time.time() + timeout
    while job.get("state") == "running" and time.time() < deadline:
        time.sleep(1.0)
        with urllib.request.urlopen(f"{base}/api/generate?id={job['id']}",
                                    timeout=30) as r:
            job = json.loads(r.read().decode())
    return job


def via_library(llm, text: str, kind: str) -> dict:
    return gen_mod.generate(llm, text, kind).to_dict()


def suspicious(sc: Scenario, source: str) -> list[str]:
    """Механические признаки шаблонной выдачи.

    Это НЕ оценка качества: сценарий может пройти все проверки и остаться
    пустым, а может зацепить замечание и быть отличным. Здесь ловится только
    то, что видно без чтения, — чтобы при разборе руками было с чего начать.
    """
    notes = []
    words = set()
    for w in source.lower().replace(",", " ").replace(".", " ").split():
        if len(w) > 5:
            words.add(w[:6])
    # Пересекаются ли слова критериев с исходным текстом: критерии обязаны
    # быть вытащены ИЗ него, а не взяты из общего списка добродетелей.
    hits = sum(1 for c in sc.criteria
               if any(w[:6] in words for w in c.title.lower().split() if len(w) > 5))
    if sc.criteria and hits == 0:
        notes.append("ни один критерий не пересекается со словами исходного текста")
    if not any(s.opening for s in sc.stages):
        notes.append("ни у одного этапа нет реплики-затравки")
    if len({s.max_turns for s in sc.stages}) == 1:
        notes.append("бюджет ходов одинаков у всех этапов")
    if len(sc.persona.pressure) < 15:
        notes.append("манера давления не описана")
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8010")
    ap.add_argument("--direct", action="store_true",
                    help="мимо сервера, прямым вызовом генератора")
    ap.add_argument("--only", default="", help="подстрока имени файла")
    args = ap.parse_args()

    index = json.loads((PROMPTS / "index.json").read_text(encoding="utf-8"))
    items = [p for p in index["prompts"] if args.only in p["file"]]
    OUT.mkdir(parents=True, exist_ok=True)
    llm = gen_mod.AnthropicGenerator() if args.direct else None

    rows, failures = [], 0
    for item in items:
        text = (PROMPTS / item["file"]).read_text(encoding="utf-8")
        t0 = time.perf_counter()
        try:
            job = (via_library(llm, text, item["kind"]) if args.direct
                   else via_http(args.base, text, item["kind"]))
        except (urllib.error.URLError, RuntimeError, TimeoutError) as e:
            print(f"  {item['file']}: НЕ ПРОШЁЛ — {type(e).__name__}: {e}")
            failures += 1
            continue
        wall = time.perf_counter() - t0
        sc = Scenario.from_dict(job["scenario"])
        problems = sc.validate()
        notes = suspicious(sc, text)
        stem = item["file"].removesuffix(".txt")
        (OUT / f"{stem}.json").write_text(
            json.dumps(job["scenario"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        (OUT / f"{stem}.meta.json").write_text(json.dumps({
            "mechanic": item["mechanic"], "kind": item["kind"],
            "fallback": job["fallback"], "message": job.get("message", ""),
            "attempts": job.get("attempts", []), "total_ms": job.get("total_ms"),
            "wall_s": round(wall, 1), "problems": problems, "notes": notes,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append({
            "file": stem, "mechanic": item["mechanic"],
            "fallback": job["fallback"], "wall_s": round(wall, 1),
            "attempts": len(job.get("attempts", [])),
            "stages": len(sc.stages), "criteria": len(sc.criteria),
            "emotion": sc.persona.start_emotion,
            "problems": problems, "notes": notes,
        })
        flag = "ШАБЛОН" if job["fallback"] else ("ok" if not notes else "?")
        print(f"  {stem:24} {wall:5.1f} с  попыток {len(job.get('attempts', []))}"
              f"  этапов {len(sc.stages)}  критериев {len(sc.criteria)}"
              f"  {sc.persona.start_emotion:10} {flag}")
        for n in notes:
            print(f"       — {n}")

    (OUT / "runs.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = sum(1 for r in rows if not r["fallback"] and not r["problems"])
    print(f"\nпрошло без фолбэка и без проблем схемы: {ok} из {len(items)}"
          f"; не выполнено: {failures}")
    if rows:
        print(f"медиана времени: {sorted(r['wall_s'] for r in rows)[len(rows) // 2]} с")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
