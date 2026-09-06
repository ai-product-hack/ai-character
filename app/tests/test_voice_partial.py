"""Распознавание на лету: слова в интерфейсе и корм для спекуляции.

Живая сессия на 12 минут показала первый звук 4584 мс медианы и 16928 мс
максимума при нуле попаданий спекуляции. Здесь проверяется механика починки;
числа задержки мерятся стендом, а не тестами.
"""
import pathlib
import sys
import threading
import time
import types
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.server import Session                                # noqa: E402
from app.voice import VoiceInput                              # noqa: E402


class Snapshot(unittest.TestCase):
    """Копия накопленного звука, не трогающая сам буфер."""

    def _voice(self):
        v = VoiceInput(endpointer=None)
        v.enabled = True
        return v

    def test_snapshot_returns_what_was_pushed(self):
        v = self._voice()
        v.push(np.ones(100, dtype=np.float32))
        v.push(np.ones(50, dtype=np.float32) * 2)
        snap = v.snapshot()
        self.assertEqual(len(snap), 150)
        self.assertEqual(snap[-1], 2)

    def test_snapshot_does_not_consume_the_buffer(self):
        """Иначе конец хода получил бы обрезанную реплику."""
        v = self._voice()
        v.push(np.ones(100, dtype=np.float32))
        v.snapshot()
        v.snapshot()
        self.assertEqual(len(v.snapshot()), 100)

    def test_empty_buffer_gives_nothing(self):
        self.assertIsNone(self._voice().snapshot())

    def test_snapshot_is_a_copy(self):
        v = self._voice()
        v.push(np.ones(10, dtype=np.float32))
        snap = v.snapshot()
        v.push(np.ones(10, dtype=np.float32))
        self.assertEqual(len(snap), 10, "снимок не должен расти вслед за буфером")


class PartialFeedsSpeculation(unittest.TestCase):
    """Растущая расшифровка — это и есть «человек печатает»."""

    def _session(self, text="я бекенд разработчик последние два года"):
        s = Session.__new__(Session)
        s.models = {"aligner": lambda pcm, sr: [{"ch": c, "ms": 0} for c in text],
                    "voice_cfg": {"partial_every_ms": 0}}
        # Эндпоинтер-заглушка: конец хода тут не проверяется, но `_maybe_partial`
        # требует, чтобы он вообще был — без распознавания голос не включается.
        s.voice = VoiceInput(endpointer=types.SimpleNamespace(
            speech_ms=0, push=lambda pcm: False, reset=lambda: None))
        s.voice.enabled = True
        s.partial_text = ""
        s._partial_sent = ""
        s._partial_at = 0.0
        s._partial_cost = 0.0
        s._turn = 0
        s._partial_thread = None
        s._partial_lock = threading.Lock()
        s.marks = []
        s.typed = []
        s.spec = types.SimpleNamespace(on_typing=s.typed.append)
        return s

    def _settle(self, s):
        """Дождаться фонового разбора. В бою его никто не ждёт — в том и смысл."""
        if s._partial_thread is not None:
            s._partial_thread.join(timeout=5)

    def _partial(self, s):
        """Разбор и его результат. В бою они приходят разными вызовами: опрос
        не ждёт разбора. Здесь заглушка успевает за тот же вызов, поэтому
        берём первый непустой ответ из двух."""
        first = s._maybe_partial()
        self._settle(s)
        return first if first is not None else s._maybe_partial()

    def test_partial_reaches_speculation(self):
        s = self._session()
        s.voice.push(np.ones(16000, dtype=np.float32) * 0.1)
        got = self._partial(s)
        self.assertIn("бекенд", got)
        # Спекуляции достаётся текст без последнего слова: хвост распознавания
        # ещё перепишется, и запрос по нему не пройдёт проверку префикса.
        self.assertEqual(s.typed, [got.rsplit(" ", 1)[0]])

    def test_intake_does_not_wait_for_recognition(self):
        """Главное свойство починки: приём звука не ждёт разбор никогда.

        Раньше разбор шёл прямо в обработчике `/api/audio`, а он на восьми
        секундах речи занимает 703 мс. Куски идут раз в 100 мс и строго по
        одному — очередь на клиенте упиралась в потолок и теряла звук.
        """
        s = self._session()
        started = threading.Event()
        release = threading.Event()

        def slow(pcm, sr):
            started.set()
            release.wait(5)
            return [{"ch": c, "ms": 0} for c in "поздно"]

        s.models["aligner"] = slow
        s.voice.push(np.ones(16000, dtype=np.float32) * 0.1)
        t0 = time.perf_counter()
        s._maybe_partial()
        self.assertLess((time.perf_counter() - t0) * 1000, 100,
                        "приём звука заблокировался разбором")
        self.assertTrue(started.wait(5), "разбор не начался вовсе")
        # И пока разбор идёт, следующие куски тоже проходят насквозь.
        t0 = time.perf_counter()
        s._maybe_partial()
        self.assertLess((time.perf_counter() - t0) * 1000, 100)
        release.set()
        self._settle(s)

    def test_late_result_does_not_leak_into_the_next_turn(self):
        """Ход мог кончиться, пока мы разбирали: чужой текст не показываем."""
        s = self._session()
        started, release = threading.Event(), threading.Event()

        def slow(pcm, sr):
            started.set()
            release.wait(5)
            return [{"ch": c, "ms": 0} for c in "из прошлой реплики"]

        s.models["aligner"] = slow
        s.voice.push(np.ones(16000, dtype=np.float32) * 0.1)
        s._maybe_partial()
        self.assertTrue(started.wait(5))
        s._turn += 1                      # ход кончился, пока шёл разбор
        release.set()
        self._settle(s)
        self.assertEqual(s.partial_text, "")
        self.assertEqual(s.typed, [])

    def test_too_short_audio_is_skipped(self):
        """Меньше полусекунды разбирать нечего — только жечь процессор."""
        s = self._session()
        s.voice.push(np.ones(4000, dtype=np.float32) * 0.1)
        self.assertIsNone(self._partial(s))
        self.assertEqual(s.typed, [])

    def test_unchanged_text_does_not_relaunch(self):
        s = self._session()
        s.voice.push(np.ones(16000, dtype=np.float32) * 0.1)
        self._partial(s)
        self._partial(s)
        self.assertEqual(len(s.typed), 1, "тот же текст не должен пускать второй запрос")

    def test_recognizer_failure_does_not_break_audio_intake(self):
        """Разбор на лету — подсказка, а не источник правды."""
        s = self._session()
        def boom(pcm, sr):
            raise RuntimeError("распознаватель упал")
        s.models["aligner"] = boom
        s.voice.push(np.ones(16000, dtype=np.float32) * 0.1)
        self.assertIsNone(self._partial(s))
        self.assertTrue(s.marks and "partial_error" in s.marks[0])

    def test_no_recognizer_means_no_partial(self):
        s = self._session()
        s.models["aligner"] = None
        s.voice.push(np.ones(16000, dtype=np.float32) * 0.1)
        self.assertIsNone(self._partial(s))

    def test_rate_limit_holds(self):
        s = self._session()
        s.models["voice_cfg"] = {"partial_every_ms": 100000}
        s.voice.push(np.ones(16000, dtype=np.float32) * 0.1)
        self._partial(s)
        s._partial_thread = None
        self.assertIsNone(self._partial(s), "чаще заданного разбирать незачем")

    def test_pause_scales_with_how_long_recognition_took(self):
        """На длинной реплике разбор дорожает: 30 с звука — 3 с работы.

        Без этой оговорки ядро было бы занято распознаванием непрерывно.
        """
        s = self._session()
        s.voice.push(np.ones(16000, dtype=np.float32) * 0.1)
        s._partial_cost = 100000.0        # прошлый разбор был очень долгим
        s._partial_at = time.perf_counter() * 1000
        s._maybe_partial()
        self.assertIsNone(s._partial_thread, "разбор пустился слишком рано")


class OpeningHasNoFiller(unittest.TestCase):
    """«Понятно… Здравствуйте!» — понимать было ещё нечего."""

    def test_emit_loop_takes_the_answering_flag(self):
        import inspect
        sig = inspect.signature(Session._emit_loop)
        self.assertIn("answering", sig.parameters)
        self.assertIs(sig.parameters["answering"].default, True)

    def test_opening_turn_passes_false(self):
        src = (ROOT / "app" / "server.py").read_text(encoding="utf-8")
        self.assertIn("args=(gen, t0, pending, DONE, em, user_text is not None)", src)

    def test_thinking_covers_the_silence(self):
        """Без заполнителя лицо не должно сидеть в позе слушателя."""
        src = (ROOT / "app" / "server.py").read_text(encoding="utf-8")
        i = src.index("if first is None and not use_bc")
        self.assertIn('"face": "thinking"', src[i:i + 500])


if __name__ == "__main__":
    unittest.main()
