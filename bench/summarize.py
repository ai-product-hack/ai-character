#!/usr/bin/env python3
"""Turns raw bench JSONL into the tables that go into research/*.md.
   python3 bench/summarize.py r1 | r2 | r6"""
import json, pathlib, statistics as st, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "bench" / "results"


def rows(name):
    p = RES / name
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def pct(x, n): return f"{100*x/n:.0f}%" if n else "—"
def med(v): return round(st.median(v)) if v else None


def r1(fname="r1_stt_synth.jsonl"):
    rs = rows(fname)
    if not rs:
        return print(f"нет {fname}")
    engines = {}
    for r in rs:
        engines.setdefault(r["engine"], []).append(r)
    print(f"\n## R1 — потоковый STT ({fname}, {len(rs)} прогонов)\n")
    print("| движок | режим | 1-й партиал, мс | финал после конца речи, мс | "
          "worst RTF | max lag, мс | WER рус | термины exact/variant/miss |")
    print("|---|---|---|---|---|---|---|---|")
    for name, g in engines.items():
        fp = [r["t_first_partial_ms"] for r in g if r["t_first_partial_ms"] is not None]
        fin = [r["t_final_after_speech_end_ms"] for r in g]
        rtf = [r["rtf_worst_chunk"] for r in g if r.get("rtf_worst_chunk")]
        lag = [r["max_lag_ms"] for r in g]
        wr = [r["wer"] for r in g]
        t = {"exact": 0, "variant": 0, "miss": 0}
        for r in g:
            for v in (r.get("terms") or {}).values():
                t[v] += 1
        print(f"| {name} | {g[0]['mode']} | {med(fp)} | {med(fin)} | "
              f"{max(rtf) if rtf else '—'} | {max(lag)} | {st.mean(wr):.3f} | "
              f"{t['exact']}/{t['variant']}/{t['miss']} |")
    print("\n### Английские термины внутри русской фразы\n")
    print("| термин | " + " | ".join(engines) + " |")
    print("|---" * (len(engines) + 1) + "|")
    terms = {}
    for r in rs:
        for k, v in (r.get("terms") or {}).items():
            terms.setdefault(k, {}).setdefault(r["engine"], []).append(v)
    for term, per in sorted(terms.items()):
        cells = []
        for e in engines:
            vs = per.get(e, [])
            cells.append("/".join(sorted(set(vs))) if vs else "—")
        print(f"| `{term}` | " + " | ".join(cells) + " |")


def r2(fname="r2_endpoint_synth.jsonl"):
    rs = rows(fname)
    if not rs:
        return print(f"нет {fname}")
    dets = {}
    for r in rs:
        dets.setdefault(r["detector"], []).append(r)
    out = []
    for name, g in dets.items():
        hes = [r for r in g if r["set"] == "r2_hesitations"]
        term = [r for r in g if r["set"] == "r2_terminals"]
        bad = sum(1 for r in hes if r["false_cuts"])
        lt = [r["endpoint_latency_ms"] for r in term if r["endpoint_latency_ms"] is not None]
        lh = [r["endpoint_latency_ms"] for r in hes if r["endpoint_latency_ms"] is not None]
        inf = [r["infer_ms_p50"] for r in g if r.get("infer_ms_p50")]
        out.append((bad / max(1, len(hes)), name, g[0]["kind"], bad, len(hes),
                    sum(r["false_cuts"] for r in hes), med(lt), med(lh),
                    sum(1 for r in term if r["endpoint_latency_ms"] is None), len(term),
                    round(st.median(inf), 1) if inf else None))
    out.sort()
    print(f"\n## R2 — определение конца хода ({fname})\n")
    print("| детектор | тип | фраз испорчено | всего срезов | задержка на терминальных, мс | "
          "задержка на заминках, мс | пропущено терминальных | инференс p50, мс |")
    print("|---|---|---|---|---|---|---|---|")
    for f, name, kind, bad, nh, fc, lt, lh, miss, nt, inf in out:
        print(f"| {name} | {kind} | **{bad}/{nh}** ({f*100:.0f}%) | {fc} | {lt} | {lh} | "
              f"{miss}/{nt} | {inf or '—'} |")


def flux_stt(fname="r2_endpoint_synth.jsonl"):
    """Латентность Deepgram Flux как STT — извлекается из уже сохранённых
    событий TurnInfo, без дополнительных запросов к API."""
    rs = [r for r in rows(fname) if r.get("spec") == "deepgram-flux"]
    if not rs:
        return print("нет строк deepgram-flux")
    fp, fin = [], []
    for r in rs:
        evs = r.get("flux_events", [])
        first = next((e for e in evs if (e.get("text") or "").strip()), None)
        if first:
            fp.append(first["t_ms"] - r.get("speech_onset_s", 0.403) * 1000)
        # Первое EndOfTurn часто оказывается ложным срезом ДО конца речи;
        # брать его — значит мерить ошибку, а не задержку. Нужен первый после.
        eot = next((e for e in evs if e["event"] == "EndOfTurn"
                    and e["t_ms"] >= r["speech_end_s"] * 1000 - 20), None)
        if eot:
            fin.append(eot["t_ms"] - r["speech_end_s"] * 1000)
    print(f"\n## R1 — Deepgram Flux как STT (из событий {len(rs)} клипов)\n")
    print(f"первый партиал от начала речи: медиана {med(fp)} мс")
    print(f"EndOfTurn с транскриптом от конца речи: медиана {med(fin)} мс")
    print("(вторая величина включает решение об окончании хода, а не только распознавание)")


def r6(fname="r6_echo.jsonl"):
    rs = [r for r in rows(fname) if r.get("kind") != "barge_in"]
    if not rs:
        return print("нет данных R6 — стенд запускается руками: bench/r6-echo/run.sh")
    print(f"\n## R6 — эхо в браузере ({len(rs)} ячеек)\n")
    print("| вывод | AEC | путь TTS | шум dBFS | эхо dBFS | утечка dB | ложных VAD | % времени |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rs:
        print(f"| {r['device']} | {r['echo_cancellation_actual']} | {r['output_path']} | "
              f"{r['floor_dbfs']} | {r['echo_mean_dbfs']} | **{r['leak_db']}** | "
              f"{r['vad_events']} | {100*r['vad_hot_ratio']:.0f}% |")


if __name__ == "__main__":
    for a in (sys.argv[1:] or ["r1", "r2", "r6"]):
        globals()[a](*(sys.argv[2:3] or []))
