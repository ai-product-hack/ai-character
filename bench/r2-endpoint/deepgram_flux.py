#!/usr/bin/env python3
"""R2 — Deepgram Flux: встроенный end-of-turn, замер по той же разметке.

Flux — живой API, поэтому аудио подаётся по стенным часам чанками 80 мс (как
рекомендует документация). Время срабатывания = время от начала потока, что при
подаче в реальном времени совпадает с аудио-временем.

    DEEPGRAM_API_KEY=... ../r1-stt/.venv/bin/python deepgram_flux.py --eot 0.7
"""
import argparse, asyncio, json, os, pathlib, statistics as st, time
import numpy as np
import soundfile as sf
import websockets

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench" / "results" / "r2_endpoint_synth.jsonl"
SR, CHUNK_MS = 16000, 80


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def score(rec, fires):
    wins = rec.get("forbidden_windows_s", [])
    end = rec["speech_end_s"]
    fc = [f for f in fires if any(a - 0.02 <= f <= b + 0.02 for a, b in wins)]
    valid = [f for f in fires if f >= end - 0.02]
    return {"n_fires": len(fires), "false_cuts": len(fc),
            "cut_in_speech": len([f for f in fires if f < end - 0.02 and f not in fc]),
            "endpoint_latency_ms": round((valid[0] - end) * 1000) if valid else None,
            "fires_s": [round(f, 3) for f in fires]}


async def run_clip(rec, eot, eager, key):
    pcm, sr = sf.read(ROOT / rec["wav"], dtype="float32")
    assert sr == SR
    i16 = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
    step = int(SR * CHUNK_MS / 1000) * 2

    url = ("wss://api.deepgram.com/v2/listen?model=flux-general-multi"
           f"&encoding=linear16&sample_rate={SR}&language_hint=ru"
           f"&eot_threshold={eot}")
    if eager is not None:
        url += f"&eager_eot_threshold={eager}"

    fires, events, t0 = [], [], None
    async with websockets.connect(url, additional_headers={"Authorization": f"Token {key}"}) as ws:
        async def send():
            nonlocal t0
            t0 = time.perf_counter()
            for n, off in enumerate(range(0, len(i16), step)):
                await ws.send(i16[off:off + step])
                d = t0 + (n + 1) * CHUNK_MS / 1000 - time.perf_counter()
                if d > 0:
                    await asyncio.sleep(d)
            await asyncio.sleep(1.2)          # дать досказать финальные события

        async def recv():
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    m = json.loads(raw)
                    if m.get("type") != "TurnInfo":
                        continue
                    t = time.perf_counter() - t0
                    ev = m.get("event")
                    events.append({"t_ms": round(t * 1000), "event": ev,
                                   "conf": m.get("end_of_turn_confidence"),
                                   "text": (m.get("transcript") or "")[:70]})
                    if ev == "EndOfTurn":
                        fires.append(t)
            except websockets.ConnectionClosed:
                pass

        r = asyncio.create_task(recv())
        await send()
        await ws.close()
        await asyncio.wait_for(r, timeout=5)

    return fires, events


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eot", type=float, default=0.7)
    ap.add_argument("--eager", type=float, default=None)
    a = ap.parse_args()
    load_env()
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise SystemExit("нет DEEPGRAM_API_KEY")

    recs = [json.loads(l) for l in
            (ROOT / "data" / "audio" / "manifest_synth.jsonl").read_text().splitlines()
            if json.loads(l)["set"].startswith("r2_")]
    name = f"deepgram-flux-multi(eot={a.eot}" + (f",eager={a.eager}" if a.eager else "") + ")"
    print(f"### {name} — {len(recs)} клипов, подача в реальном времени\n")

    agg = {"fc": 0, "bad": 0, "hes": 0, "lat_t": [], "lat_h": [], "miss": 0, "term": 0}
    with OUT.open("a") as f:
        for rec in recs:
            fires, events = await run_clip(rec, a.eot, a.eager, key)
            s = score(rec, fires)
            row = {"detector": name, "kind": "cloud", "spec": "deepgram-flux",
                   "clip": rec["id"], "set": rec["set"], "source": rec.get("source"),
                   "text": rec["text"], "speech_end_s": rec["speech_end_s"],
                   "forbidden_windows_s": rec.get("forbidden_windows_s", []),
                   **s, "flux_events": events,
                   "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            if rec["set"] == "r2_hesitations":
                agg["hes"] += 1; agg["fc"] += s["false_cuts"]
                agg["bad"] += 1 if s["false_cuts"] else 0
                if s["endpoint_latency_ms"] is not None:
                    agg["lat_h"].append(s["endpoint_latency_ms"])
            else:
                agg["term"] += 1
                if s["endpoint_latency_ms"] is None:
                    agg["miss"] += 1
                else:
                    agg["lat_t"].append(s["endpoint_latency_ms"])
            print(f"  {rec['id']:5} срабатываний={s['n_fires']} ложных={s['false_cuts']} "
                  f"задержка={s['endpoint_latency_ms']} мс")

    print(f"\nложных срезов: {agg['fc']} в {agg['bad']}/{agg['hes']} фразах "
          f"({100*agg['bad']/max(1,agg['hes']):.0f}%)")
    print(f"задержка на терминальных p50: "
          f"{round(st.median(agg['lat_t'])) if agg['lat_t'] else '—'} мс | "
          f"на заминках p50: {round(st.median(agg['lat_h'])) if agg['lat_h'] else '—'} мс | "
          f"пропущено терминальных: {agg['miss']}/{agg['term']}")


if __name__ == "__main__":
    asyncio.run(main())
