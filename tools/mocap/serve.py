#!/usr/bin/env python3
"""Local-only authoring server for the facial mocap page."""

import http.server
import json
import os
import pathlib
import re
import socketserver
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CLIPS = ROOT / "avatar" / "clips"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8092
SAFE_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".glb": "model/gltf-binary",
        ".json": "application/json",
        ".task": "application/octet-stream",
        ".wasm": "application/wasm",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        if self.path != "/tools/mocap/api/save-clip":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 20 * 1024 * 1024:
                raise ValueError("invalid payload size")
            payload = json.loads(self.rfile.read(size))
            name = payload.get("name", "")
            clip = payload.get("clip")
            if not SAFE_NAME.fullmatch(name) or not isinstance(clip, dict):
                raise ValueError("invalid clip")
            CLIPS.mkdir(parents=True, exist_ok=True)
            destination = CLIPS / f"{name}.json"
            destination.write_text(json.dumps(clip, ensure_ascii=False, indent=2) + "\n")
            body = json.dumps({"ok": True, "path": str(destination.relative_to(ROOT))}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_error(400, str(exc))

    def log_message(self, fmt, *args):
        if "404" in (fmt % args):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    os.chdir(ROOT)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as server:
        print(f"http://localhost:{PORT}/tools/mocap/")
        server.serve_forever()
