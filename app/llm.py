"""Адаптеры моделей.

Синхронные: движку и фоновой оценке нужен целый ответ, а не поток. Потоковый
вариант появится в конвейере реплики — там он нужен ради первого звука.

Переключение провайдера — одна строка в конфиге, как и было заложено в S1.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_env(path=None) -> None:
    """Подтянуть ключи из .env, если их нет в окружении."""
    p = pathlib.Path(path or ROOT / ".env")
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class DeepSeekLLM:
    """OpenAI-совместимый API. Стдлиб, без httpx: движку хватает одного POST."""

    def __init__(self, model: str = "deepseek-chat", max_tokens: int = 300,
                 temperature: float = 0.7, timeout: float = 40):
        load_env()
        self.key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.key:
            raise SystemExit("нет DEEPSEEK_API_KEY в окружении или .env")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.calls = 0
        self.total_ms = 0.0
        self.last_ms = 0.0

    def __call__(self, system: str, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.deepseek.com/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"DeepSeek {e.code}: {e.read()[:200].decode(errors='replace')}")
        self.last_ms = (time.perf_counter() - t0) * 1000
        self.total_ms += self.last_ms
        self.calls += 1
        return data["choices"][0]["message"]["content"]


def build(provider: str = "stub", **kw):
    if provider == "deepseek":
        return DeepSeekLLM(**kw)
    if provider == "stub":
        from .stub_llm import StubLLM
        return StubLLM(**kw)
    raise SystemExit(f"провайдер '{provider}' не реализован")
