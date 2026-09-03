#!/usr/bin/env python3
"""Пересчёт порога Deepgram Flux по уже собранным событиям, без новых запросов.

В каждом TurnInfo сохранён end_of_turn_confidence, поэтому поведение при другом
eot_threshold восстанавливается офлайн. Симуляция на 0.7 сверяется с реальным
прогоном на 0.7 — если сходится, остальным порогам можно верить.
"""
import json, pathlib, statistics as st, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "bench" / "results" / "r2_endpoint_synth.jsonl"


def simulate(rec, thr, rearm=0.3):
    """Одно решение на паузу: сработали — ждём падения уверенности, иначе
    один и тот же затухающий хвост даст десяток срабатываний."""
    fires, armed = [], True
    for e in rec.get("flux_events", []):
        c = e.get("conf")
        if c is None:
            continue
        if armed and c >= thr:
            fires.append(e["t_ms"] / 1000)
            armed = False
        elif not armed and c < rearm:
            armed = True
    return fires


def score(rec, fires):
    wins = rec.get("forbidden_windows_s", [])
    end = rec["speech_end_s"]
    fc = [f for f in fires if any(a - 0.02 <= f <= b + 0.02 for a, b in wins)]
    valid = [f for f in fires if f >= end - 0.02]
    return len(fc), (round((valid[0] - end) * 1000) if valid else None)


def main():
    rows = [json.loads(l) for l in SRC.read_text().splitlines()]
    flux = [r for r in rows if r.get("spec") == "deepgram-flux"]
    if not flux:
        sys.exit("нет строк deepgram-flux")
    print(f"клипов: {len(flux)}\n")
    print("| порог | фраз испорчено | всего срезов | задержка на терминальных, мс | "
          "задержка на заминках, мс | пропущено терминальных |")
    print("|---|---|---|---|---|---|")
    for thr in (0.5, 0.7, 0.8, 0.9, 0.95):
        bad = fc_tot = miss = 0
        hes = term = 0
        lt, lh = [], []
        for r in flux:
            fires = simulate(r, thr)
            nfc, lat = score(r, fires)
            if r["set"] == "r2_hesitations":
                hes += 1; fc_tot += nfc; bad += 1 if nfc else 0
                if lat is not None:
                    lh.append(lat)
            else:
                term += 1
                if lat is None:
                    miss += 1
                else:
                    lt.append(lat)
        print(f"| {thr} | **{bad}/{hes}** ({100*bad/max(1,hes):.0f}%) | {fc_tot} | "
              f"{round(st.median(lt)) if lt else '—'} | {round(st.median(lh)) if lh else '—'} | "
              f"{miss}/{term} |")
    print("\nсверка: строка 0.7 должна совпасть с реальным прогоном "
          "(18 срезов в 13/20 фразах, 760 мс на терминальных)")


if __name__ == "__main__":
    main()
