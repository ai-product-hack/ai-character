#!/usr/bin/env python3
"""Dev-сервер модуля аватара. Стдлиб, как в спайках.

    python3 avatar/dev/serve.py        # http://localhost:8090/avatar/dev/

Раздаёт корень репозитория, чтобы со страницы были видны и /data/model.glb,
и /avatar/node_modules/three. Больше он ничего не делает: модуль обязан
разрабатываться без бэкенда, источник данных на странице фейковый.
"""
import functools, http.server, os, pathlib, socketserver, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".glb": "model/gltf-binary",
                      ".json": "application/json", ".wasm": "application/wasm"}

    def end_headers(self):
        # Никакого кеша: конфиг и шейдеры правятся и перезагружаются постоянно.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "404" in (fmt % args):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    os.chdir(ROOT)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"http://localhost:{PORT}/avatar/dev/")
        httpd.serve_forever()
