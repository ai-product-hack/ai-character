#!/usr/bin/env python3
"""ElevenLabs через потоковый websocket, как это работало бы в бою.

Прошлые числа (медиана 1831 мс на пяти запросах) сняты с холодным соединением:
в них входит установка TCP, TLS и рукопожатие websocket. В продукте соединение
держится открытым между репликами, и мерить надо именно так.

Второе отличие: текст подаётся КУСКАМИ, по мере того как их выдаёт модель, а не
одной посылкой. Ради этого потоковый режим и существует.

    bench/r1-stt/.venv/bin/python bench/r3-tts/round2_elevenlabs.py
"""
import asyncio, base64, json, os, pathlib, statistics, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench" / "r3-tts"))
from compare_voices import REPLIES                              # noqa: E402
sys.path.insert(0, str(ROOT))
from app.clauses import split_text                              # noqa: E402

RESULTS = ROOT / "bench" / "results"
OUT = ROOT / "bench" / "r3-tts" / "round2" / "elevenlabs"
MODEL = "eleven_flash_v2_5"
SR = 24000


def load_env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


async def pick_voice(key):
    import httpx
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get("https://api.elevenlabs.io/v1/voices",
                        headers={"xi-api-key": key})
        r.raise_for_status()
        voices = r.json()["voices"]
    # Берём первый голос с русским в labels, иначе просто первый.
    for v in voices:
        lab = json.dumps(v.get("labels", {}), ensure_ascii=False).lower()
        if "russian" in lab or "рус" in lab:
            return v["voice_id"], v["name"]
    return voices[0]["voice_id"], voices[0]["name"]


async def one_request(ws, text, save_to=None):
    """Одна реплика в уже открытое соединение, текст подаётся клаузами."""
    t0 = time.perf_counter()
    first_audio = None
    nbytes = 0
    chunks = []
    chars, starts = [], []

    clauses = [c.text for c in split_text(text)]
    for i, c in enumerate(clauses):
        await ws.send(json.dumps({"text": c + " ",
                                  "flush": i == len(clauses) - 1}))
    await ws.send(json.dumps({"text": ""}))

    import websockets
    try:
        async for raw in ws:
            m = json.loads(raw)
            if m.get("audio"):
                b = base64.b64decode(m["audio"])
                nbytes += len(b)
                chunks.append((time.perf_counter() - t0) * 1000)
                if first_audio is None:
                    first_audio = (time.perf_counter() - t0) * 1000
            a = m.get("normalizedAlignment") or m.get("alignment")
            if a:
                chars += a.get("chars", [])
                starts += a.get("charStartTimesMs", [])
            if m.get("isFinal"):
                break
    except websockets.ConnectionClosed:
        pass

    total = (time.perf_counter() - t0) * 1000
    gaps = [b - a for a, b in zip(chunks, chunks[1:])] if len(chunks) > 1 else []
    return {
        "ttfb_ms": round(first_audio) if first_audio else None,
        "total_ms": round(total),
        "audio_s": round(nbytes / 2 / SR, 3),
        "chunks": len(chunks),
        "gap_max_ms": round(max(gaps)) if gaps else 0,
        "gap_p50_ms": round(statistics.median(gaps)) if gaps else 0,
        "align_chars": len(chars),
        "clauses_sent": len(clauses),
    }


async def main():
    load_env()
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise SystemExit("нет ELEVENLABS_API_KEY")
    import websockets

    voice, name = await pick_voice(key)
    print(f"голос: {name} ({voice}), модель {MODEL}")
    url = (f"wss://api.elevenlabs.io/v1/text-to-speech/{voice}/stream-input"
           f"?model_id={MODEL}&output_format=pcm_{SR}&sync_alignment=true")

    rows = []

    # Холодное соединение: столько стоит первый запрос после простоя.
    for rep in REPLIES[:3]:
        t0 = time.perf_counter()
        async with websockets.connect(url, additional_headers={"xi-api-key": key}) as ws:
            connect_ms = (time.perf_counter() - t0) * 1000
            await ws.send(json.dumps({"text": " ",
                                      "voice_settings": {"stability": 0.5,
                                                         "similarity_boost": 0.8}}))
            r = await one_request(ws, rep["text"])
        r.update({"mode": "cold", "reply_id": rep["id"], "connect_ms": round(connect_ms)})
        rows.append(r)
        print(f"  холодное {rep['id']}: соединение {r['connect_ms']} мс, "
              f"TTFB {r['ttfb_ms']} мс")

    # Соединение живёт РОВНО ОДНУ реплику: пустая посылка закрывает контекст,
    # и следующий запрос в тот же сокет получает ConnectionClosedOK. Значит
    # переиспользовать его нельзя, и остаётся открывать заранее — тогда
    # установка уходит с критического пути, как её и убрал бы пул соединений.
    async def preopened():
        ws = await websockets.connect(url, additional_headers={"xi-api-key": key})
        await ws.send(json.dumps({"text": " ",
                                  "voice_settings": {"stability": 0.5,
                                                     "similarity_boost": 0.8}}))
        return ws

    nxt = await preopened()
    for rnd in range(2):
        for rep in REPLIES:
            ws, nxt = nxt, None
            # Следующее соединение открывается ПАРАЛЛЕЛЬНО текущему запросу.
            task = asyncio.create_task(preopened())
            try:
                r = await one_request(ws, rep["text"])
            except Exception as e:                       # noqa: BLE001
                print(f"  заранее {rep['id']}: {type(e).__name__}: {str(e)[:80]}")
                r = None
            finally:
                try:
                    await ws.close()
                except Exception:                        # noqa: BLE001
                    pass
                nxt = await task
            if r is None:
                continue
            r.update({"mode": "preopened", "reply_id": rep["id"], "round": rnd,
                      "connect_ms": 0})
            rows.append(r)
            print(f"  заранее {rep['id']} #{rnd}: TTFB {r['ttfb_ms']} мс, "
                  f"кусков {r['chunks']}, разрыв макс {r['gap_max_ms']} мс")
    try:
        await nxt.close()
    except Exception:                                    # noqa: BLE001
        pass

    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / "r3_tts_round2_elevenlabs.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                 encoding="utf-8")

    def stat(mode, field):
        v = [r[field] for r in rows if r["mode"] == mode and r.get(field) is not None]
        if not v:
            return "—"
        v = sorted(v)
        return (f"медиана {v[len(v)//2]}, p90 {v[int(len(v)*0.9)]}, "
                f"мин {v[0]}, макс {v[-1]}, n={len(v)}")

    print(f"\n=== TTFB ===")
    print(f"холодное соединение: {stat('cold', 'ttfb_ms')}")
    print(f"заранее открытое:    {stat('preopened', 'ttfb_ms')}")
    print(f"разрывы потока:      {stat('preopened', 'gap_max_ms')}")
    print(f"установка соединения: {stat('cold', 'connect_ms')}")
    print(f"-> {p}")


if __name__ == "__main__":
    asyncio.run(main())
