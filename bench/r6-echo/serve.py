#!/usr/bin/env python3
"""R6 echo/barge-in bench: static server + JSONL result sink. Stdlib only."""
import http.server, json, os, socketserver, sys, datetime, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT.parent / "results" / "r6_echo.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("PORT", "8060"))

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def do_POST(self):
        if self.path != "/result":
            self.send_error(404); return
        n = int(self.headers.get("Content-Length", 0))
        rec = json.loads(self.rfile.read(n) or b"{}")
        rec["logged_at"] = datetime.datetime.now().astimezone().isoformat()
        with OUT.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[r6] {rec.get('cell','?')}: leak={rec.get('leak_db')} dB "
              f"vad_fp={rec.get('vad_false_frames')}/{rec.get('vad_total_frames')}", flush=True)
        self.send_response(204); self.end_headers()

    def log_message(self, *a):
        pass

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), H) as s:
        print(f"R6 bench: http://localhost:{PORT}/  -> results append to {OUT}", flush=True)
        try:
            s.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
