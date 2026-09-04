"""Динамика набора как сигнал.

Проверяем не «считает ли», а свойства, на которые опирается отчёт: метка
уверенности должна различать три разных способа написать один и тот же текст.
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.dialogue import DialogueState                      # noqa: E402
from app.report import build                                # noqa: E402
from app.scenario import load_all                           # noqa: E402
from app.typing_signal import TypingTracker                 # noqa: E402


def type_out(tracker, steps):
    """steps: [(момент в мс, длина текста), ...]"""
    for at, ln in steps:
        tracker.observe(at, ln)
    return tracker.summarise(final_length=steps[-1][1])


class Confidence(unittest.TestCase):
    def test_steady_typing_reads_as_confident(self):
        t = TypingTracker()
        sig = type_out(t, [(300 * i, 5 * i) for i in range(1, 15)])
        self.assertEqual(sig.confidence, "уверенно")
        self.assertEqual(sig.deletions, 0)
        self.assertEqual(sig.rewrite_ratio, 0.0)

    def test_long_thinking_pause_shows_up(self):
        t = TypingTracker()
        # Начал бодро, замер на пять секунд, дописал.
        sig = type_out(t, [(200, 5), (500, 12), (5800, 18), (6100, 30)])
        self.assertGreater(sig.longest_pause_ms, 3000)
        self.assertNotEqual(sig.confidence, "уверенно")

    def test_rewriting_counts_deleted_chars_not_events(self):
        """Одно длинное стирание должно весить больше одного короткого."""
        t = TypingTracker()
        sig = type_out(t, [(200, 40), (600, 5), (900, 45)])
        self.assertEqual(sig.deletions, 1)
        self.assertEqual(sig.deleted_chars, 35)
        self.assertGreater(sig.rewrite_ratio, 0.25)

    def test_three_signals_together_read_as_unsure(self):
        t = TypingTracker()
        sig = type_out(t, [(6000, 30), (12000, 4), (12500, 25)])
        self.assertEqual(sig.confidence, "неуверенно")

    def test_single_sample_gives_no_verdict(self):
        """Один замер — не сигнал. Молчать честнее, чем угадывать."""
        t = TypingTracker()
        t.observe(400, 12)
        self.assertEqual(t.summarise(final_length=12).confidence, "нет данных")

    def test_reset_between_answers(self):
        t = TypingTracker()
        type_out(t, [(100, 5), (400, 20)])
        t.reset()
        self.assertEqual(t.summarise(final_length=0).samples, 0)


class InReport(unittest.TestCase):
    def setUp(self):
        self.state = DialogueState(load_all(ROOT / "data" / "scenarios")[0])

    def test_absent_when_nobody_measured(self):
        """Скриптовые прогоны не печатают. Пустой раздел лучше нулей."""
        self.state.add_user("ответ")
        self.assertIsNone(build(self.state).typing)

    def test_summary_counts_hesitant_answers(self):
        t = TypingTracker()
        calm = type_out(t, [(200 * i, 4 * i) for i in range(1, 10)])
        t.reset()
        rough = type_out(t, [(6000, 30), (12000, 4), (12500, 25)])

        self.state.add_user("быстрый", typing=calm.to_dict())
        self.state.add_user("трудный", typing=rough.to_dict())
        rep = build(self.state).typing

        self.assertEqual(rep["answers"], 2)
        self.assertEqual(rep["hesitant_answers"], 1)
        self.assertEqual(rep["confidence"]["уверенно"], 1)

    def test_signal_travels_with_the_turn(self):
        t = TypingTracker()
        sig = type_out(t, [(100, 5), (400, 20)])
        self.state.add_user("ответ", typing=sig.to_dict())
        line = build(self.state).transcript[-1]
        self.assertEqual(line["text"], "ответ")
        self.assertIn("confidence", line["typing"])


if __name__ == "__main__":
    unittest.main()


class Origin(unittest.TestCase):
    """Нуль отсчёта ставится при подведении итога, а не при наблюдении."""

    def test_answering_before_agent_finished_reads_negative(self):
        t = TypingTracker()
        # Часы абсолютные: агент договорил на отметке 10 000, а человек начал
        # печатать на 8 500 — не дослушав.
        for at, ln in [(8500, 6), (9200, 20), (10400, 38)]:
            t.observe(at, ln)
        sig = t.summarise(final_length=38, origin_ms=10000)
        self.assertLess(sig.first_key_ms, 0)
        self.assertEqual(sig.confidence, "уверенно")

    def test_pauses_do_not_depend_on_origin(self):
        t = TypingTracker()
        for at, ln in [(1000, 5), (6000, 12), (6300, 30)]:
            t.observe(at, ln)
        a = t.summarise(final_length=30, origin_ms=0)
        b = t.summarise(final_length=30, origin_ms=5000)
        self.assertEqual(a.longest_pause_ms, b.longest_pause_ms)
        self.assertEqual(a.rewrite_ratio, b.rewrite_ratio)
        self.assertNotEqual(a.first_key_ms, b.first_key_ms)
