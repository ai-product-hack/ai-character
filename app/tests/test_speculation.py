"""Тесты спекуляции на недопечатанном сообщении.

Механика перенесена из S1, где запрос уходил по растущему транскрипту речи.
Проверяется то же, что там: перезапуск по росту буфера, один запрос в полёте,
и главное — правило переиспользования. Взять результат от промпта, который не
является префиксом финального, значит ответить не на тот вопрос.
"""
import pathlib
import sys
import threading
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.speculation import InFlight, Speculator            # noqa: E402


class FakeStream:
    """Поток токенов с управляемой задержкой и отменой."""

    made = 0

    def __init__(self, text="Ответ модели на вопрос.", delay=0.0):
        self.text = text
        self.delay = delay
        self.cancelled = False
        FakeStream.made += 1

    def cancel(self):
        self.cancelled = True

    def __call__(self, system, prompt):
        for i in range(0, len(self.text), 5):
            if self.cancelled:
                return
            if self.delay:
                time.sleep(self.delay)
            yield self.text[i:i + 5]


def make_spec(**cfg):
    streams = []

    def factory():
        s = FakeStream(delay=cfg.pop("delay", 0.0) if False else 0.002)
        streams.append(s)
        return s

    sp = Speculator(make_stream=factory,
                    build_prompt=lambda t: f"промпт({t})",
                    system="sys", cfg=cfg)
    return sp, streams


class Launching(unittest.TestCase):
    def test_short_text_does_not_launch(self):
        sp, streams = make_spec(min_chars=15)
        self.assertFalse(sp.on_typing("Коротко"))
        self.assertEqual(streams, [])

    def test_launches_after_min_chars(self):
        sp, streams = make_spec(min_chars=15)
        self.assertTrue(sp.on_typing("Я backend-разработчик"))
        self.assertEqual(len(streams), 1)
        self.assertEqual(sp.stats.launches, 1)

    def test_small_growth_does_not_relaunch(self):
        sp, streams = make_spec(min_chars=10, relaunch_growth=0.4)
        sp.on_typing("Двадцать символов тут")
        sp.on_typing("Двадцать символов тут!")     # прирост около 5%
        self.assertEqual(len(streams), 1, "каждый символ не должен родить запрос")
        self.assertEqual(sp.stats.relaunches, 0)

    def test_big_growth_relaunches(self):
        sp, streams = make_spec(min_chars=10, relaunch_growth=0.4)
        sp.on_typing("Двадцать символов")
        sp.on_typing("Двадцать символов и ещё столько же сверху")
        self.assertEqual(len(streams), 2)
        self.assertEqual(sp.stats.relaunches, 1)

    def test_only_freshest_stays_in_flight(self):
        sp, streams = make_spec(min_chars=10, relaunch_growth=0.3)
        sp.on_typing("Первый вариант текста")
        sp.on_typing("Первый вариант текста и продолжение подлиннее")
        self.assertTrue(streams[0].cancelled, "старый запрос обязан быть отменён")
        self.assertFalse(streams[1].cancelled)

    def test_disabled_never_launches(self):
        sp, streams = make_spec(enabled=False, min_chars=5)
        self.assertFalse(sp.on_typing("Длинный текст сообщения"))
        self.assertEqual(streams, [])


class Reuse(unittest.TestCase):
    def test_exact_text_is_a_hit(self):
        sp, _ = make_spec(min_chars=10, reuse_cover=0.6)
        text = "Я держал слой кеширования в этом сервисе"
        sp.on_typing(text)
        flight, cover = sp.take(text)
        self.assertIsNotNone(flight)
        self.assertEqual(sp.stats.hits, 1)
        self.assertAlmostEqual(cover, 1.0, places=6)

    def test_prefix_with_enough_cover_is_a_hit(self):
        sp, _ = make_spec(min_chars=10, reuse_cover=0.6)
        sp.on_typing("Я держал слой кеширования")            # 25 символов
        flight, cover = sp.take("Я держал слой кеширования и переписал")  # 37
        self.assertIsNotNone(flight, f"покрытие {cover:.2f} должно хватать")
        self.assertGreater(cover, 0.6)

    def test_prefix_with_low_cover_is_a_miss(self):
        sp, streams = make_spec(min_chars=10, reuse_cover=0.6)
        sp.on_typing("Я держал слой")
        flight, cover = sp.take(
            "Я держал слой кеширования и переписал выборку на батчевую целиком")
        self.assertIsNone(flight, "ранний промпт покрывает лишь начало вопроса")
        self.assertLess(cover, 0.6)
        self.assertTrue(streams[0].cancelled)
        self.assertEqual(sp.stats.misses, 1)

    def test_text_rewritten_is_a_miss(self):
        """Человек стёр начало и написал другое — брать нельзя ни при каком покрытии."""
        sp, _ = make_spec(min_chars=10, reuse_cover=0.3)
        sp.on_typing("Я держал слой кеширования")
        flight, _ = sp.take("Совсем другой ответ на вопрос")
        self.assertIsNone(flight)

    def test_take_without_flight_is_a_miss(self):
        sp, _ = make_spec(min_chars=10)
        flight, cover = sp.take("Любой текст")
        self.assertIsNone(flight)
        self.assertEqual(sp.stats.misses, 1)

    def test_flight_consumed_once(self):
        sp, _ = make_spec(min_chars=10)
        text = "Достаточно длинный текст"
        sp.on_typing(text)
        self.assertIsNotNone(sp.take(text)[0])
        self.assertIsNone(sp.take(text)[0], "повторно тот же запрос не отдаётся")


class Replay(unittest.TestCase):
    def test_replay_yields_all_tokens(self):
        stream = FakeStream("Полный ответ модели без потерь.", delay=0.001)
        f = InFlight(stream, "sys", "prompt", "текст")
        got = "".join(f.replay())
        self.assertEqual(got, stream.text)

    def test_replay_includes_tokens_buffered_before_read(self):
        """Токены, накопившиеся до того как их начали читать, не теряются —
        в этом весь смысл: запрос ушёл раньше, чем нажали Enter."""
        stream = FakeStream("Ранние токены и поздние тоже.", delay=0.001)
        f = InFlight(stream, "sys", "prompt", "текст")
        f.done.wait(timeout=3)
        self.assertGreater(len(f.tokens), 0, "должны были накопиться")
        self.assertEqual("".join(f.replay()), stream.text)

    def test_cancel_stops_replay(self):
        stream = FakeStream("А" * 200, delay=0.01)
        f = InFlight(stream, "sys", "prompt", "текст")
        time.sleep(0.05)
        f.cancel()
        got = "".join(f.replay())
        self.assertLess(len(got), 200, "после отмены поток обязан оборваться")

    def test_first_token_time_recorded(self):
        f = InFlight(FakeStream("Ответ", delay=0.005), "sys", "p", "t")
        f.done.wait(timeout=3)
        self.assertIsNotNone(f.t_first_token)
        self.assertGreater(f.t_first_token, 0)


class Lifecycle(unittest.TestCase):
    def test_drop_cancels_in_flight(self):
        sp, streams = make_spec(min_chars=10)
        sp.on_typing("Достаточно длинный текст")
        sp.drop()
        self.assertTrue(streams[0].cancelled, "перебивание гасит и спекуляцию")
        self.assertIsNone(sp.flight)

    def test_drop_is_not_called_on_ordinary_send(self):
        """Отправка сообщения не должна гасить спекуляцию под это же сообщение.

        Первая версия сервера звала drop() из обработчика перебивания, который
        срабатывает на КАЖДОЕ сообщение, и убивала ровно тот запрос, который
        собиралась использовать: попаданий было ноль при четырёх запусках.
        """
        sp, streams = make_spec(min_chars=10)
        text = "Достаточно длинный текст сообщения"
        sp.on_typing(text)
        # так это делает сервер: сначала гасит текущую реплику агента…
        self.assertIsNotNone(sp.flight, "перебивание не трогает спекуляцию")
        flight, cover = sp.take(text)
        self.assertIsNotNone(flight)
        self.assertEqual(sp.stats.hits, 1)

    def test_stats_summary_shape(self):
        sp, _ = make_spec(min_chars=10)
        sp.on_typing("Достаточно длинный текст")
        sp.take("Достаточно длинный текст")
        sp.note_ttft(2000, hit=False)
        sp.note_ttft(1100, hit=True)
        d = sp.stats.summary()
        for k in ("launches", "relaunches", "hits", "misses", "hit_rate",
                  "ttft_hit_ms", "ttft_miss_ms"):
            self.assertIn(k, d)
        self.assertEqual(d["hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
