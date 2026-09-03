#!/usr/bin/env python3
"""Браузерный диктофон для живых записей. Обходит проблему доступа к микрофону
у терминала: разрешение запрашивает Chrome, а не iTerm.

    python3 bench/data-gen/record-web/server.py     # http://localhost:8090

Клиент шлёт сырой Float32 с частотой AudioContext; сервер приводит к
mono/16 кГц через ffmpeg и кладёт в data/audio/live/<набор>/<id>.wav.
"""
import http.server, json, os, pathlib, socketserver, struct, subprocess, tempfile, wave

_REC = None


def gap_check(path):
    """Самая длинная пауза между словами. Для фраз с заминками это и есть то,
    ради чего запись делается: если человек прочитал бегло, паузы нет, и клип
    для R2 бесполезен — лучше сказать об этом сразу, чем после 20 фраз."""
    global _REC
    try:
        import numpy as np, sherpa_onnx, soundfile as sf
        from huggingface_hub import hf_hub_download
        if _REC is None:
            repo = "csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16"
            _REC = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=hf_hub_download(repo, "model.int8.onnx"),
                tokens=hf_hub_download(repo, "tokens.txt"),
                num_threads=4, sample_rate=16000, feature_dim=80,
                decoding_method="greedy_search")
        pcm, sr = sf.read(path, dtype="float32")
        pcm = (pcm / max(1e-9, float(np.max(np.abs(pcm)))) * 0.7).astype("float32")
        st = _REC.create_stream(); st.accept_waveform(sr, pcm); _REC.decode_stream(st)
        toks, ts = list(st.result.tokens or []), list(st.result.timestamps or [])
        ends, starts, cur = [], [], ""
        for t, tm in zip(toks, ts):
            if t == " ":
                if cur:
                    ends.append(tm); cur = ""
            else:
                if not cur:
                    starts.append(tm)
                cur += t
        gaps = [s - e for e, s in zip(ends, starts[1:])]
        return {"max_gap_ms": round(max(gaps) * 1000) if gaps else 0,
                "text": st.result.text}
    except Exception as e:
        return {"max_gap_ms": None, "text": f"(проверка недоступна: {e})"}

ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
LIVE = ROOT / "data" / "audio" / "live"
PORT = int(os.environ.get("PORT", "8090"))
SETS = ["r2_hesitations", "r2_terminals", "r1_terms"]


def phrases():
    out = []
    for name in SETS:
        src = ROOT / "data" / f"{name}.jsonl"
        for line in src.read_text().splitlines():
            r = json.loads(line)
            dst = LIVE / name / f"{r['id']}.wav"
            out.append({"set": name, "id": r["id"], "text": r["text"],
                        "recorded": dst.exists()})
    return out


def save(setname, cid, sr, floats: bytes):
    """Float32 -> WAV на исходной частоте -> ffmpeg -> mono 16 кГц."""
    n = len(floats) // 4
    vals = struct.unpack(f"<{n}f", floats)
    peak = max((abs(v) for v in vals), default=0.0)
    rms = (sum(v * v for v in vals) / max(1, n)) ** 0.5
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp = pathlib.Path(tf.name)
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(struct.pack(f"<{n}h",
                                  *(max(-32768, min(32767, int(v * 32767))) for v in vals)))
    dst = LIVE / setname / f"{cid}.wav"
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                    "-ac", "1", "-ar", "16000", str(dst)], check=True)
    tmp.unlink(missing_ok=True)
    pad_tail(dst)
    return {"duration_s": round(n / sr, 3), "rms": round(rms, 5), "peak": round(peak, 4),
            "path": str(dst.relative_to(ROOT))}


def pad_tail(dst, tail_s=2.0):
    """Доклеивает хвост из собственного фона клипа.

    Человек жмёт «стоп» сразу после фразы, и хвоста остаётся ~400 мс. Детектор,
    которому нужно 700–1500 мс тишины, при этом не может сработать в принципе —
    и это выглядело бы как его недостаток, хотя виновата длина записи.
    Клеится собственный фон, а не цифровая тишина: она уронила бы шумовой пол
    и сделала задачу детектору легче, чем в жизни.
    """
    import array
    with wave.open(str(dst)) as w:
        sr, n = w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    a = array.array("h"); a.frombytes(raw)
    win = int(0.1 * sr)
    if len(a) < win * 2:
        return
    quiet = min((sum(x * x for x in a[i:i + win]), i)
                for i in range(0, len(a) - win, win))[1]
    amb = a[quiet:quiet + win]
    need = int(tail_s * sr)
    tail = array.array("h")
    while len(tail) < need:
        tail.extend(amb)
    a.extend(tail[:need])
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(a.tobytes())


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/phrases":
            return self._json(phrases())
        if self.path == "/manifest":
            return self._json({"written": write_manifest()})
        return super().do_GET()

    def do_POST(self):
        if self.path != "/clip":
            return self.send_error(404)
        meta = json.loads(self.headers["X-Meta"])
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        try:
            info = save(meta["set"], meta["id"], int(meta["sampleRate"]), body)
        except Exception as e:
            return self._json({"error": str(e)}, 500)
        if meta["set"] == "r2_hesitations":
            info.update(gap_check(ROOT / info["path"]))
        flag = "ТИШИНА" if info["rms"] < 0.001 else "ok"
        print(f"[rec] {meta['id']:5} {info['duration_s']:6.2f} с  RMS {info['rms']:.4f}  {flag}",
              flush=True)
        self._json(info)


def write_manifest():
    rows, n = [], 0
    for name in SETS:
        src = ROOT / "data" / f"{name}.jsonl"
        for line in src.read_text().splitlines():
            r = json.loads(line)
            dst = LIVE / name / f"{r['id']}.wav"
            if not dst.exists():
                continue
            with wave.open(str(dst)) as w:
                d = w.getnframes() / w.getframerate()
            r.update(set=name, wav=str(dst.relative_to(ROOT)),
                     duration_s=round(d, 3), source="live")
            rows.append(r); n += 1
    mf = ROOT / "data" / "audio" / "manifest_live.jsonl"
    mf.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"[rec] манифест: {n} записей -> {mf.relative_to(ROOT)}", flush=True)
    return n


class Threaded(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    with Threaded(("127.0.0.1", PORT), H) as s:
        print(f"Диктофон: http://localhost:{PORT}/   (файлы -> {LIVE.relative_to(ROOT)})",
              flush=True)
        s.serve_forever()
