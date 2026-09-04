#!/usr/bin/env python3
"""Замер отмены на реальном пути конвейера.

Перебивание в трёх точках, как требует задание: до первого звука, в середине
первой клаузы, на стыке клауз. Меряется не «когда сервер узнал», а когда
перестало приходить хоть что-то с чужим generation_id.

    bench/r1-stt/.venv/bin/python app/run_cancel.py
"""
import argparse
import json
import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import llm as llm_mod                              # noqa: E402
from app.agent import SYSTEM, build_prompt                  # noqa: E402
from app.dialogue import DialogueState                      # noqa: E402
from app.generation import GenerationRegistry               # noqa: E402
from app.media import GigaAMAligner, build_tts, stream_deepseek, tts_config  # noqa: E402
from app.pipeline import ReplyPipeline                      # noqa: E402
from app.scenario import load_all                           # noqa: E402
from app.visemes_bridge import VisemeBridge                 # noqa: E402

POINTS = ("до первого звука", "в середине первой клаузы", "на стыке клауз")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="s1_interview_backend")
    # Голос по умолчанию берётся из app/config.json — того же, по которому
    # живёт сервер. Прибитый сюда «eugene» пережил смену Silero на v5, где
    # такого голоса нет, и скрипт падал на подъёме моделей.
    ap.add_argument("--voice", default=None, help="по умолчанию из app/config.json")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default="bench/results/cancel_pipeline.json")
    args = ap.parse_args()

    tts = build_tts({**tts_config(), **({"voice": args.voice} if args.voice else {})})
    aligner = GigaAMAligner()
    bridge = VisemeBridge()
    llm = llm_mod.DeepSeekLLM()

    sc = [s for s in load_all(ROOT / "data" / "scenarios") if s.id == args.scenario][0]
    state = DialogueState(sc)
    state.add_agent("Здравствуйте! Расскажите о себе и последнем проекте.")
    state.add_user("Я backend-разработчик, делал сервис выдачи рекомендаций.")
    prompt = build_prompt(state, "Лично я держал слой кеширования.")

    rows = []
    for point in POINTS:
        for rep in range(args.repeats):
            registry = GenerationRegistry()
            stream = stream_deepseek(llm)
            pipe = ReplyPipeline(stream, tts, aligner, bridge, registry)
            gen = registry.start()

            delivered = []
            after_cancel = []
            t_cancel = {"at": None}
            lock = threading.Lock()

            def on_result(r):
                with lock:
                    delivered.append((time.perf_counter(), r))
                    # Всё, что пришло ПОСЛЕ отмены с чужим id, — нарушение.
                    if t_cancel["at"] is not None and r.generation_id == gen.id:
                        after_cancel.append((time.perf_counter(), r))

            def cancel_when_ready():
                if point == "до первого звука":
                    time.sleep(0.05)                 # ещё идёт TTFT
                elif point == "в середине первой клаузы":
                    while not delivered:
                        if time.perf_counter() - t0 > 20:
                            return
                        time.sleep(0.002)
                    time.sleep(delivered[0][1].audio_ms / 1000 * 0.4)
                else:                                # на стыке клауз
                    while len(delivered) < 2:
                        if time.perf_counter() - t0 > 20:
                            break
                        time.sleep(0.002)
                t_cancel["at"] = time.perf_counter()
                registry.cancel(gen.id)
                stream.cancel()          # гасим и запрос к модели

            t0 = time.perf_counter()
            th = threading.Thread(target=cancel_when_ready, daemon=True)
            th.start()
            pipe.run(SYSTEM, prompt, gen, on_result=on_result)
            th.join(timeout=5)
            t_end = time.perf_counter()

            if t_cancel["at"] is None:
                print(f"  {point}: отмена не сработала (реплика кончилась раньше)")
                continue

            # Два РАЗНЫХ числа, и путать их нельзя.
            #
            # stop_ms — когда перестаёт приходить хоть что-то с чужим
            # generation_id. Это то, что доходит до пользователя, и именно его
            # сравнивают с порогом кейса в 300 мс.
            #
            # cleanup_ms — когда умирает брошенный запрос к модели. До первого
            # токена поток стоит внутри urlopen, закрывать ещё нечего, и запрос
            # доживает свой TTFT. Пользователь этого не слышит: кадры с чужим
            # id отбрасываются до всякой доставки. Платим соединением и
            # токенами, не задержкой.
            leak_ms = ((max(t for t, _ in after_cancel) - t_cancel["at"]) * 1000
                       if after_cancel else 0.0)
            cleanup_ms = (t_end - t_cancel["at"]) * 1000
            rows.append({
                "point": point, "repeat": rep,
                "clauses_before_cancel": len(delivered),
                "leaked_after_cancel": len(after_cancel),
                "stop_ms": round(leak_ms),
                "cleanup_ms": round(cleanup_ms),
            })
            print(f"  {point} #{rep}: клауз до отмены {len(delivered)}, "
                  f"протекло после {len(after_cancel)}, "
                  f"остановка {leak_ms:.0f} мс, "
                  f"брошенный запрос умирал {cleanup_ms:.0f} мс")

    print("\n=== итог ===")
    for point in POINTS:
        sub = [r for r in rows if r["point"] == point]
        if not sub:
            continue
        stop = sorted(r["stop_ms"] for r in sub)
        clean = sorted(r["cleanup_ms"] for r in sub)
        print(f"{point:<26} остановка {stop[len(stop)//2]} мс, "
              f"брошенный запрос {clean[len(clean)//2]} мс, "
              f"протечек {sum(r['leaked_after_cancel'] for r in sub)}")
    total_leak = sum(r["leaked_after_cancel"] for r in rows)
    worst = max((r["stop_ms"] for r in rows), default=0)
    print(f"\nпротечек старой генерации: {total_leak} (порог задания — ноль)")
    print(f"худшая остановка: {worst} мс (порог кейса 300 мс)")
    print(f"худшая смерть брошенного запроса: "
          f"{max((r['cleanup_ms'] for r in rows), default=0)} мс — "
          f"не слышно, но соединение и токены оплачены")

    out = ROOT / args.out
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    bridge.close()


if __name__ == "__main__":
    sys.exit(main())
