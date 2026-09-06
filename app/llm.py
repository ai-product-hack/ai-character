"""Адаптеры моделей.

Синхронные: движку и фоновой оценке нужен целый ответ, а не поток. Потоковый
нужен конвейеру реплики — там он определяет первый звук.

Провайдеров два, переключаются строкой в `app/config.json`:

    deepseek    дешевле в 4-19 раз, но TTFT 2330 мс
    anthropic   claude-haiku-4-5 (772 мс) или claude-sonnet-5 (941 мс)

Числа замерены, не взяты с потолка: `research/R8-llm-latency.md`. Разница в
задержке решает, укладывается ли реплика в бюджет 3000 мс, разница в цене — во
что обходится минута разговора; и то и другое посчитано в `BUDGET.md`.

У обоих провайдеров одинаковый утиный интерфейс, чтобы движок не знал, с кем
разговаривает:

    llm(system, prompt) -> str        целый ответ
    llm.make_stream() -> поток        генератор токенов с .cancel()
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
        return self.chat(system, [{"role": "user", "content": prompt}])

    def chat(self, system: str, messages: list[dict]) -> str:
        """Разговор массивом ролей. Кеш у DeepSeek автоматический, кода не просит."""
        body = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system}] + list(messages),
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

    def make_stream(self, max_tokens: int | None = None, temperature: float | None = None):
        from .media import stream_deepseek
        return stream_deepseek(self, max_tokens or self.max_tokens,
                               self.temperature if temperature is None else temperature)


_EPHEMERAL = {"type": "ephemeral"}


def cacheable_system(system: str):
    """Системный блок с точкой кеширования.

    Он не меняется весь разговор и составляет заметную долю входа — держать
    его некешируемым значит платить за одно и то же на каждом ходу.
    """
    if not system:
        return system
    return [{"type": "text", "text": system, "cache_control": _EPHEMERAL}]


def with_cache_breakpoint(messages: list[dict]) -> list[dict]:
    """Отметить конец стабильного префикса — всё, кроме последней реплики.

    Последнее сообщение меняется каждый ход (в нём состояние этапа и свежая
    реплика), а всё до него от хода к ходу совпадает байт в байт. Точка
    ставится ровно на границе, иначе кеш промахивался бы каждый раз.

    Точек всего две на запрос: здесь и в системном блоке. Лимит провайдера —
    четыре, запас есть.

    ЗАМЕРЕНО, и результат неочевидный: у моделей разный минимальный размер
    кешируемого префикса, и наши разговоры попадают по разные стороны порога.

        claude-sonnet-5    кеш работает с первого хода: вход упал с 2028 до
                           336 токенов, остальное прочиталось из кеша
        claude-haiku-4-5   при префиксе 3811 токенов НЕ кешируется, при 5431 —
                           кешируется; наши разговоры дают 1700-3200, то есть
                           кеш не срабатывает ни разу за диалог

    Отметки всё равно ставятся всегда: вреда от них нет (провайдер просто
    игнорирует префикс ниже порога), а на длинных сценариях и на Sonnet они
    экономят кратно.
    """
    if len(messages) < 2:
        return list(messages)
    out = [dict(m) for m in messages]
    last_stable = out[-2]
    content = last_stable["content"]
    if isinstance(content, str):
        last_stable["content"] = [{"type": "text", "text": content,
                                   "cache_control": _EPHEMERAL}]
    return out


# Модели Anthropic, пригодные для диалога. Opus в списке нет намеренно: по
# TTFT он неотличим от Sonnet (949 против 941 мс), а стоит вдвое дороже —
# замерено в R8.
ANTHROPIC_MODELS = ("claude-haiku-4-5", "claude-sonnet-5")


class AnthropicLLM:
    """Anthropic под тем же интерфейсом, что DeepSeekLLM.

    Мышление выключено принудительно и не настраивается. В голосовом тренажёре
    весь бюджет 3000 мс, и модель, которая думает перед ответом, в него не
    помещается ни при каком качестве.

    `temperature` не принимается: в SDK 1.x его нет в сигнатуре вовсе, а на
    новых моделях сэмплирование удалено на стороне API. Аргумент проглатывается
    молча, чтобы вызывающий код не расходился между провайдерами.
    """

    def __init__(self, model: str = "claude-haiku-4-5", max_tokens: int = 300,
                 temperature: float | None = None, timeout: float = 40):
        load_env()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            # RuntimeError, а не SystemExit: адаптер живёт в потоке сервера, а
            # SystemExit проходит мимо `except Exception` насквозь.
            raise RuntimeError("нет ANTHROPIC_API_KEY в окружении или .env")
        import anthropic                     # импорт здесь: пакет нужен только тут
        self.client = anthropic.Anthropic(timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.calls = 0
        self.total_ms = 0.0
        self.last_ms = 0.0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cache_write = 0
        self.cache_read = 0

    def __call__(self, system: str, prompt: str) -> str:
        return self.chat(system, [{"role": "user", "content": prompt}])

    def chat(self, system: str, messages: list[dict]) -> str:
        t0 = time.perf_counter()
        r = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=cacheable_system(system),
            messages=with_cache_breakpoint(messages),
            thinking={"type": "disabled"})
        self.last_ms = (time.perf_counter() - t0) * 1000
        self.total_ms += self.last_ms
        self.calls += 1
        self.tokens_in += r.usage.input_tokens
        self.tokens_out += r.usage.output_tokens
        # Сколько ушло в кеш и сколько прочиталось оттуда. Без этих двух чисел
        # нельзя понять, работает кеширование или молча не срабатывает:
        # `cache_read` в нуле при повторных ходах и есть тот самый признак.
        self.cache_write += getattr(r.usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read += getattr(r.usage, "cache_read_input_tokens", 0) or 0
        return "".join(b.text for b in r.content if b.type == "text")

    def make_stream(self, max_tokens: int | None = None, temperature=None):
        return AnthropicStream(self, max_tokens or self.max_tokens)


class AnthropicStream:
    """Потоковый Anthropic, который можно оборвать снаружи.

    Та же задача, что у `CancellableStream` для DeepSeek: пока не пришёл первый
    токен, поток стоит в чтении сокета, и закрыть его может только другой
    поток. Без этого отмена не освобождала соединение ровно на величину TTFT, а
    токены всё равно оплачивались.
    """

    def __init__(self, llm: AnthropicLLM, max_tokens: int = 300):
        self.llm = llm
        self.max_tokens = max_tokens
        self._stream = None
        self._closed = False

    def cancel(self) -> None:
        """Оборвать соединение. Зовётся из другого потока."""
        self._closed = True
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.close()
            except Exception:                              # noqa: BLE001
                pass

    def __call__(self, system: str, prompt):
        """`prompt` — строка или уже готовый массив сообщений."""
        messages = ([{"role": "user", "content": prompt}]
                    if isinstance(prompt, str) else list(prompt))
        try:
            manager = self.llm.client.messages.stream(
                model=self.llm.model, max_tokens=self.max_tokens,
                system=cacheable_system(system),
                messages=with_cache_breakpoint(messages),
                thinking={"type": "disabled"})
            stream = manager.__enter__()
        except Exception:                                  # noqa: BLE001
            if self._closed:
                return
            raise
        self._stream = stream
        try:
            for event in stream:
                if self._closed:
                    break
                if event.type == "content_block_delta" and \
                        getattr(event.delta, "type", "") == "text_delta":
                    yield event.delta.text
        except Exception:                                  # noqa: BLE001
            # Оборванное соединение — это и есть отмена, а не сбой.
            if not self._closed:
                raise
        finally:
            self._stream = None
            try:
                manager.__exit__(None, None, None)
            except Exception:                              # noqa: BLE001
                pass


def build(provider: str = "stub", **kw):
    if provider == "deepseek":
        return DeepSeekLLM(**kw)
    if provider == "anthropic":
        return AnthropicLLM(**kw)
    if provider == "stub":
        from .stub_llm import StubLLM
        return StubLLM(**kw)
    raise SystemExit(f"провайдер '{provider}' не реализован")


def build_dialogue(cfg: dict | None = None) -> dict:
    """Три клиента диалога по конфигу: реплика, второй разбор, вывод отчёта.

    Бюджеты токенов у них разные и не случайные. Реплика — 300, как и было
    замерено. Второй разбор возвращает одно слово, и держать под него бюджет
    реплики незачем. Вывод отчёта — два текста по 2-4 предложения, и на 300
    токенах он обрывался на середине JSON (поймано на живом прогоне).
    """
    cfg = cfg or {}
    provider = cfg.get("provider", "deepseek")
    model = cfg.get("model")
    if provider == "anthropic" and model and model not in ANTHROPIC_MODELS:
        raise SystemExit(f"модель «{model}» не в списке для диалога: "
                         f"{', '.join(ANTHROPIC_MODELS)}")
    kw = {"model": model} if model else {}
    return {
        "llm": build(provider, max_tokens=cfg.get("max_tokens", 300), **kw),
        "repair_llm": build(provider, max_tokens=8, temperature=0, **kw),
        "summary_llm": build(provider, max_tokens=1200, **kw),
        "provider": provider,
    }


def make_stream(llm, max_tokens: int = 300, temperature: float = 0.7):
    """Поток для любого провайдера. Конвейер не должен знать, с кем говорит."""
    return llm.make_stream(max_tokens, temperature)
