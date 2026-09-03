#!/usr/bin/env python3
"""S2 — one timeline for audio and face. Stdlib only.

Two tracks with PTS relative to the start of generation:
  audio      20 ms PCM frames
  blendshape 30 FPS frames

The blendshapes here are derived from the audio envelope. That is deliberate:
this spike measures TRANSPORT AND SYNC, not animation quality. Swapping in
Audio2Face output changes the producer, not the timeline contract.

Both tracks are sent ahead of real time, as a real TTS+A2F pair would.
"""
import http.server, json, math, os, pathlib, socketserver, struct, time, wave

ROOT = pathlib.Path(__file__).resolve().parent
WAV = ROOT.parents[1] / "bench" / "r6-echo" / "assets" / "agent_ru.wav"
LOG = ROOT.parents[1] / "bench" / "results" / "s2_timeline.jsonl"
PORT = int(os.environ.get("PORT", "8080"))
FRAME_MS, FACE_FPS = 20, 30


def load():
    with wave.open(str(WAV)) as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    return sr, raw


def blendshapes(sr, raw):
    """jawOpen from the envelope; a couple of correlates so interpolation has
    more than one channel to prove itself on."""
    import array
    s = array.array("h"); s.frombytes(raw)
    step = int(sr / FACE_FPS)
    out = []
    for i in range(0, len(s), step):
        w = s[i:i + step]
        if not w:
            break
        rms = math.sqrt(sum(x * x for x in w) / len(w)) / 32768
        peak = max(abs(x) for x in w) / 32768
        jaw = min(1.0, rms * 6.0)
        out.append({
            "pts_ms": round(i / sr * 1000, 1),
            "v": {"jawOpen": round(jaw, 4),
                  "mouthFunnel": round(min(1.0, max(0.0, peak - rms * 2) * 3), 4),
                  "mouthClose": round(max(0.0, 1 - jaw * 2), 4)},
        })
    return out


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        LOG.parent.mkdir(parents=True, exist_ok=True)
        body["logged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with LOG.open("a") as f:
            f.write(json.dumps(body, ensure_ascii=False) + "\n")
        print(f"[s2] compensate={body.get('compensate')} "
              f"drift p50={body.get('drift_p50_ms')} мс p95={body.get('drift_p95_ms')} мс "
              f"| rAF p50={body.get('raf_p50_ms')} мс | подвисаний={body.get('underruns')}", flush=True)
        self.send_response(204); self.end_headers()

    def do_GET(self):
        if not self.path.startswith("/stream"):
            return super().do_GET()
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        speed = float(q.get("speed", ["3"])[0])
        sr, raw = load()
        face = blendshapes(sr, raw)
        step = int(sr * FRAME_MS / 1000) * 2

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("X-Sample-Rate", str(sr))
        self.send_header("X-Face-Fps", str(FACE_FPS))
        self.end_headers()

        t0 = time.monotonic()
        fi = 0
        try:
            for seq, off in enumerate(range(0, len(raw), step)):
                pts = seq * FRAME_MS
                # Face frames are emitted interleaved, slightly AHEAD of the
                # audio they belong to, so the renderer always has a bracket to
                # interpolate inside instead of extrapolating off the end.
                while fi < len(face) and face[fi]["pts_ms"] <= pts + FRAME_MS * 2:
                    self._send({"kind": "face", **face[fi]}, b"")
                    fi += 1
                self._send({"kind": "audio", "pts_ms": pts}, raw[off:off + step])
                d = t0 + (seq + 1) * (FRAME_MS / 1000) / speed - time.monotonic()
                if d > 0:
                    time.sleep(d)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send(self, hdr, payload):
        h = json.dumps(hdr).encode()
        self.wfile.write(struct.pack("<II", len(h), len(payload)) + h + payload)
        self.wfile.flush()


class Threaded(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    with Threaded(("127.0.0.1", PORT), H) as s:
        print(f"S2: http://localhost:{PORT}/  (лог -> {LOG})", flush=True)
        s.serve_forever()
