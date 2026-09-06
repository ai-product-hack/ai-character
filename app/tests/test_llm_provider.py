"""Переключение провайдера диалога. Сети здесь нет.

Проверяется контракт, а не качество ответов: движок не должен знать, с кем
разговаривает, а конфиг — переключать провайдера одной строкой. Живые вызовы
стоят денег и живут в `bench/r8-llm/`, не в тестах.
"""
import contextlib
import json
import os
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app import llm as llm_mod                              # noqa: E402


class FakeAnthropic:
    """Заглушка SDK: считает вызовы и отдаёт заранее заданный текст."""

    def __init__(self, text="Реплика агента.", tokens=(120, 40)):
        self.text = text
        self.tokens = tokens
        self.calls = []
        self.messages = types.SimpleNamespace(create=self._create, stream=self._stream)

    def _create(self, **kw):
        self.calls.append(kw)
        block = types.SimpleNamespace(type="text", text=self.text)
        usage = types.SimpleNamespace(input_tokens=self.tokens[0],
                                      output_tokens=self.tokens[1])
        return types.SimpleNamespace(content=[block], usage=usage)

    def _stream(self, **kw):
        self.calls.append(kw)
        words = self.text.split()
        outer = self

        class Manager:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                outer.closed = True
                return False

            def __iter__(self):
                for w in words:
                    yield types.SimpleNamespace(
                        type="content_block_delta",
                        delta=types.SimpleNamespace(type="text_delta", text=w + " "))

            def close(self):
                outer.closed = True

        return Manager()


def anthropic_llm(fake=None, **kw):
    """AnthropicLLM с подменённым клиентом: ключ и сеть не нужны."""
    with fake_key():
        llm = llm_mod.AnthropicLLM(**kw)
    llm.client = fake or FakeAnthropic()
    return llm


@contextlib.contextmanager
def fake_key(value: str = "тест"):
    """Подставить ключ на время и вернуть окружение как было.

    `setdefault` здесь не годится: если другой тест оставил после себя ПУСТУЮ
    строку, ключ формально есть, и умолчание не сработает. Ровно на этом
    набор падал в полном прогоне и проходил поодиночке.
    """
    saved = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = value
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = saved


class SameInterfaceForBoth(unittest.TestCase):
    """Утиный контракт: движок зовёт одно и то же у любого провайдера."""

    def test_call_returns_text(self):
        llm = anthropic_llm()
        self.assertEqual(llm("система", "промпт"), "Реплика агента.")

    def test_both_providers_expose_what_the_engine_uses(self):
        for cls in (llm_mod.DeepSeekLLM, llm_mod.AnthropicLLM):
            for attr in ("model", "max_tokens", "calls", "total_ms",
                         "last_ms", "make_stream", "__call__"):
                self.assertTrue(hasattr(cls, attr) or attr in
                                cls.__init__.__code__.co_names, f"{cls.__name__}.{attr}")

    def test_stats_accumulate(self):
        llm = anthropic_llm()
        llm("с", "п")
        llm("с", "п")
        self.assertEqual(llm.calls, 2)
        self.assertEqual((llm.tokens_in, llm.tokens_out), (240, 80))
        self.assertGreater(llm.total_ms, 0)


class ThinkingIsOff(unittest.TestCase):
    """Мышление гасится всегда: бюджет реплики 3000 мс на всё."""

    def test_sync_call_disables_thinking(self):
        fake = FakeAnthropic()
        anthropic_llm(fake)("с", "п")
        self.assertEqual(fake.calls[0]["thinking"], {"type": "disabled"})

    def test_stream_disables_thinking(self):
        fake = FakeAnthropic()
        list(anthropic_llm(fake).make_stream()("с", "п"))
        self.assertEqual(fake.calls[0]["thinking"], {"type": "disabled"})

    def test_temperature_is_never_sent(self):
        """В SDK 1.x его нет в сигнатуре, на новых моделях API отвечает 400."""
        fake = FakeAnthropic()
        anthropic_llm(fake, temperature=0.7)("с", "п")
        self.assertNotIn("temperature", fake.calls[0])


class Streaming(unittest.TestCase):
    def test_tokens_arrive_one_by_one(self):
        llm = anthropic_llm(FakeAnthropic(text="раз два три"))
        self.assertEqual([t.strip() for t in llm.make_stream()("с", "п")],
                         ["раз", "два", "три"])

    def test_cancel_stops_the_stream(self):
        llm = anthropic_llm(FakeAnthropic(text="раз два три четыре пять"))
        stream = llm.make_stream()
        got = []
        for token in stream("с", "п"):
            got.append(token)
            if len(got) == 2:
                stream.cancel()
        self.assertEqual(len(got), 2, "после отмены поток обязан прекратиться")

    def test_cancel_before_start_yields_nothing(self):
        llm = anthropic_llm()
        stream = llm.make_stream()
        stream.cancel()

        def boom(**kw):
            raise RuntimeError("соединение закрыто")

        llm.client.messages.stream = boom
        self.assertEqual(list(stream("с", "п")), [],
                         "отмена до первого токена — не ошибка")

    def test_stream_has_cancel_for_both_providers(self):
        """Конвейер зовёт .cancel() не глядя на провайдера."""
        self.assertTrue(hasattr(anthropic_llm().make_stream(), "cancel"))
        from app.media import CancellableStream
        self.assertTrue(hasattr(CancellableStream, "cancel"))


class BuildFromConfig(unittest.TestCase):
    def test_default_is_deepseek(self):
        cfg = json.loads((ROOT / "app" / "config.json").read_text(encoding="utf-8"))
        self.assertIn("dialogue", cfg)
        self.assertEqual(cfg["dialogue"]["provider"], "deepseek")

    def test_three_clients_with_their_own_budgets(self):
        with fake_key():
            d = llm_mod.build_dialogue({"provider": "anthropic",
                                        "model": "claude-haiku-4-5"})
        self.assertEqual(d["llm"].max_tokens, 300)
        self.assertEqual(d["repair_llm"].max_tokens, 8)
        self.assertEqual(d["summary_llm"].max_tokens, 1200)
        self.assertEqual(d["provider"], "anthropic")

    def test_model_travels_to_every_client(self):
        with fake_key():
            d = llm_mod.build_dialogue({"provider": "anthropic",
                                        "model": "claude-sonnet-5"})
        for key in ("llm", "repair_llm", "summary_llm"):
            self.assertEqual(d[key].model, "claude-sonnet-5", key)

    def test_unknown_model_is_refused_loudly(self):
        """Опечатка в конфиге не должна выясняться первой репликой на показе."""
        with self.assertRaises(SystemExit):
            llm_mod.build_dialogue({"provider": "anthropic", "model": "claude-opus-5"})

    def test_opus_is_out_of_the_dialogue_list(self):
        """По TTFT неотличим от Sonnet (949 против 941), а вдвое дороже — R8."""
        self.assertNotIn("claude-opus-5", llm_mod.ANTHROPIC_MODELS)

    def test_unknown_provider_is_refused(self):
        with self.assertRaises(SystemExit):
            llm_mod.build_dialogue({"provider": "openai"})


class MissingKey(unittest.TestCase):
    def test_absent_key_raises_catchable_error(self):
        """SystemExit прошёл бы мимо `except Exception` в потоке сервера."""
        with fake_key(""):
            with self.assertRaises(Exception) as ctx:
                llm_mod.AnthropicLLM()
            self.assertNotIsInstance(ctx.exception, SystemExit)


if __name__ == "__main__":
    unittest.main()
