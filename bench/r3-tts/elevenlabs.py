#!/usr/bin/env python3
"""R3/R4 — ElevenLabs Flash v2.5: TTFB и пословный alignment.

Два пути меряются отдельно, потому что дают разное:
  HTTP  /stream        — TTFB до первого аудиобайта, без таймингов;
  WS    /stream-input  — TTFB плюс alignment с посимвольными таймкодами,
                         то есть то, ради чего он вообще нужен для R4.

    ../r1-stt/.venv/bin/python elevenlabs.py
"""
import asyncio, json, os, pathlib, statistics as st, time
import httpx, websockets

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench" / "results" / "r3_tts_elevenlabs.jsonl"
MODEL = "eleven_flash_v2_5"
PHRASES = [
    "Здравствуйте! Расскажите, пожалуйста, о вашем последнем проекте.",
    "Понятно. А что именно делали лично вы?",
    "Вы сказали, что держали задержку на уровне ста миллисекунд. За счёт чего?",
    "Хорошо, давайте перейдём к следующему вопросу.",
    "Спасибо за ответ. У меня есть уточнение по поводу вашей роли в команде.",
]


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


async def pick_voice(key):
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": key})
        r.raise_for_status()
        vs = r.json()["voices"]
    for v in vs:
        labels = " ".join(str(x) for x in (v.get("labels") or {}).values()).lower()
        if "multilingual" in labels or "multilingual" in (v.get("description") or "").lower():
            return v["voice_id"], v["name"]
    return vs[0]["voice_id"], vs[0]["name"]


async def http_ttfb(key, voice, text):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream?output_format=pcm_24000"
    body = {"text": text, "model_id": MODEL}
    t0 = time.perf_counter()
    first, total = None, 0
    async with httpx.AsyncClient(timeout=60) as c:
        async with c.stream("POST", url, headers={"xi-api-key": key}, json=body) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                if chunk and first is None:
                    first = (time.perf_counter() - t0) * 1000
                total += len(chunk)
    return first, total, (time.perf_counter() - t0) * 1000


async def ws_align(key, voice, text):
    url = (f"wss://api.elevenlabs.io/v1/text-to-speech/{voice}/stream-input"
           f"?model_id={MODEL}&output_format=pcm_24000&sync_alignment=true")
    t0 = time.perf_counter()
    first_audio, chars, starts, durs, nbytes = None, [], [], [], 0
    async with websockets.connect(url, additional_headers={"xi-api-key": key}) as ws:
        await ws.send(json.dumps({"text": " ", "voice_settings": {"stability": 0.5,
                                                                  "similarity_boost": 0.8}}))
        await ws.send(json.dumps({"text": text + " ", "flush": True}))
        await ws.send(json.dumps({"text": ""}))
        try:
            async for raw in ws:
                m = json.loads(raw)
                if m.get("audio"):
                    import base64
                    nbytes += len(base64.b64decode(m["audio"]))
                    if first_audio is None:
                        first_audio = (time.perf_counter() - t0) * 1000
                a = m.get("normalizedAlignment") or m.get("alignment")
                if a:
                    chars += a.get("chars", [])
                    starts += a.get("charStartTimesMs", [])
                    durs += a.get("charDurationsMs", [])
                if m.get("isFinal"):
                    break
        except websockets.ConnectionClosed:
            pass
    return first_audio, chars, starts, durs, nbytes, (time.perf_counter() - t0) * 1000


async def main():
    load_env()
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise SystemExit("нет ELEVENLABS_API_KEY")
    voice, vname = await pick_voice(key)
    print(f"### ElevenLabs {MODEL}, голос «{vname}» ({voice[:8]}…)\n")
    rows = []
    for i, text in enumerate(PHRASES):
        h_first, h_bytes, h_total = await http_ttfb(key, voice, text)
        w_first, chars, starts, durs, w_bytes, w_total = await ws_align(key, voice, text)
        audio_s = w_bytes / 2 / 24000
        step = (st.median([b - a for a, b in zip(starts, starts[1:])])
                if len(starts) > 2 else None)
        row = {"engine": f"elevenlabs-{MODEL}", "voice": vname, "phrase_id": i, "text": text,
               "http_ttfb_ms": round(h_first) if h_first else None,
               "http_total_ms": round(h_total),
               "ws_ttfb_ms": round(w_first) if w_first else None,
               "ws_total_ms": round(w_total),
               "audio_s": round(audio_s, 3),
               "align_chars": len(chars), "align_step_ms": step,
               "has_alignment": bool(chars),
               "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        rows.append(row)
        print(f"  [{i}] HTTP TTFB={row['http_ttfb_ms']:5} мс | WS TTFB={row['ws_ttfb_ms']:5} мс | "
              f"аудио {audio_s:.2f} с | alignment: {len(chars)} символов, шаг {step} мс")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = [r["http_ttfb_ms"] for r in rows if r["http_ttfb_ms"]]
    w = [r["ws_ttfb_ms"] for r in rows if r["ws_ttfb_ms"]]
    print(f"\nHTTP TTFB: медиана {round(st.median(h))} мс, макс {max(h)} мс")
    print(f"WS   TTFB: медиана {round(st.median(w))} мс, макс {max(w)} мс")
    print(f"alignment есть: {sum(1 for r in rows if r['has_alignment'])}/{len(rows)}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
