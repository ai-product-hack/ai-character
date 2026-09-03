#!/usr/bin/env python3
"""S3 — cancellation by generation_id. Stdlib only.

The server streams "TTS" audio FASTER than real time, exactly as a real TTS
does. That is the whole point of the spike: the client accumulates a buffer,
and the buffer is where cancellation latency hides.

Frame wire format: [u32 json_len][u32 pcm_len][json][pcm int16le]
"""
import http.server, json, pathlib, socketserver, struct, sys, threading, time, wave

ROOT = pathlib.Path(__file__).resolve().parent
WAV = ROOT.parents[1] / "bench" / "r6-echo" / "assets" / "agent_ru.wav"
LOG = ROOT.parents[1] / "bench" / "results" / "s3_cancel_server.jsonl"
PORT = int(__import__("os").environ.get("PORT", "8070"))
FRAME_MS = 20

CANCELLED: dict[int, float] = {}      # gen -> monotonic time the cancel arrived
LOCK = threading.Lock()


def pcm16():
    with wave.open(str(WAV)) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1
        return w.getframerate(), w.readframes(w.getnframes())


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/cancel":
            with LOCK:
                CANCELLED[int(body["gen"])] = time.monotonic()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "t_server": time.monotonic()}).encode())
        elif self.path == "/report":
            LOG.parent.mkdir(parents=True, exist_ok=True)
            body["logged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            with LOG.open("a") as f:
                f.write(json.dumps(body, ensure_ascii=False) + "\n")
            print(f"[s3] {body.get('mode'):12} stop={body.get('stop_latency_ms')} мс "
                  f"buffered={body.get('buffered_ms_at_cancel')} мс "
                  f"last|amp|={body.get('last_sample_abs')}", flush=True)
            self.send_response(204); self.end_headers()
        else:
            self.send_error(404)

    def do_GET(self):
        if not self.path.startswith("/stream"):
            return super().do_GET()
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        gen = int(q.get("gen", ["0"])[0])
        speed = float(q.get("speed", ["3"])[0])       # TTS runs ahead of playback
        sr, data = pcm16()
        step = int(sr * FRAME_MS / 1000) * 2

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("X-Sample-Rate", str(sr))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        t0 = time.monotonic()
        sent = 0
        try:
            for seq, off in enumerate(range(0, len(data), step)):
                with LOCK:
                    if gen in CANCELLED:
                        dt = (time.monotonic() - CANCELLED[gen]) * 1000
                        print(f"[s3] gen={gen} сервер остановил поток через "
                              f"{dt:.1f} мс после /cancel, отправлено {sent} кадров", flush=True)
                        return
                chunk = data[off:off + step]
                hdr = json.dumps({"gen": gen, "seq": seq,
                                  "pts_ms": round(seq * FRAME_MS, 1)}).encode()
                self.wfile.write(struct.pack("<II", len(hdr), len(chunk)) + hdr + chunk)
                self.wfile.flush()
                sent += 1
                target = t0 + (seq + 1) * (FRAME_MS / 1000) / speed
                d = target - time.monotonic()
                if d > 0:
                    time.sleep(d)
        except (BrokenPipeError, ConnectionResetError):
            print(f"[s3] gen={gen} клиент отвалился после {sent} кадров", flush=True)


class Threaded(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    if not WAV.exists():
        sys.exit(f"нет {WAV} — сначала bench/r6-echo/make-assets.sh")
    with Threaded(("127.0.0.1", PORT), H) as s:
        print(f"S3: http://localhost:{PORT}/  (лог -> {LOG})", flush=True)
        s.serve_forever()
