"""Голосовой ввод: накопление, конец хода, защита от собственного голоса.

VAD здесь подменяется: настоящий проверен в R2, и гонять его в юнит-тестах
значило бы мерить Silero, а не нашу обвязку.
"""
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.voice import SR_ASR, VoiceInput                     # noqa: E402


class FakeEndpointer:
    """Срабатывает через заданное число кусков после начала речи."""

    def __init__(self, fire_after=3, speech_ms=800):
        self.fire_after = fire_after
        self._speech_ms = speech_ms
        self.pushes = 0
        self.resets = 0

    @property
    def speech_ms(self):
        return self._speech_ms

    def reset(self):
        self.resets += 1
        self.pushes = 0

    def push(self, pcm):
        self.pushes += 1
        return self.pushes >= self.fire_after


def chunk(ms=100):
    return np.zeros(int(SR_ASR * ms / 1000), dtype=np.float32)


class Turns(unittest.TestCase):
    def setUp(self):
        self.ep = FakeEndpointer()
        self.v = VoiceInput(self.ep)
        self.v.enabled = True
        self.v.start()

    def test_utterance_returned_on_endpoint(self):
        self.assertIsNone(self.v.push(chunk()))
        self.assertIsNone(self.v.push(chunk()))
        utt = self.v.push(chunk())
        self.assertIsNotNone(utt)
        self.assertAlmostEqual(utt.duration_ms, 300, delta=1)
        self.assertEqual(self.v.stats["utterances"], 1)

    def test_next_turn_starts_clean(self):
        """Хвост прошлой реплики не должен попасть в следующую."""
        for _ in range(3):
            self.v.push(chunk())
        self.assertGreater(self.ep.resets, 0)
        self.ep.fire_after = 2
        for _ in range(2):
            utt = self.v.push(chunk())
        self.assertAlmostEqual(utt.duration_ms, 200, delta=1)

    def test_disabled_input_is_ignored(self):
        """Тумблер выключен — текстовый путь не должен ничего заметить."""
        self.v.enabled = False
        for _ in range(5):
            self.assertIsNone(self.v.push(chunk()))

    def test_agent_voice_does_not_start_a_turn(self):
        """Пока агент говорит, микрофон не слушаем: на колонках VAD услышит его."""
        self.v.muted_by_agent = True
        for _ in range(5):
            self.assertIsNone(self.v.push(chunk()))
        self.assertEqual(self.v.stats["muted_chunks"], 5)
        self.v.muted_by_agent = False
        for _ in range(3):
            utt = self.v.push(chunk())
        self.assertIsNotNone(utt)

    def test_silence_only_is_not_a_turn(self):
        """Тишина не должна начинать ход: иначе агент отвечает на кашель."""
        self.ep._speech_ms = 0
        for _ in range(3):
            out = self.v.push(chunk())
        self.assertIsNone(out)
        self.assertEqual(self.v.stats["dropped_short"], 1)

    def test_long_stream_is_cut_by_ceiling(self):
        """Зависший микрофон не должен копить память до конца сессии."""
        v = VoiceInput(FakeEndpointer(fire_after=10**6), max_ms=500)
        v.enabled = True
        v.start()
        out = None
        for _ in range(6):
            out = out or v.push(chunk())
        self.assertIsNotNone(out)


class PushToTalk(unittest.TestCase):
    def test_flush_ends_the_turn(self):
        v = VoiceInput(FakeEndpointer(fire_after=10**6))
        v.enabled = True
        v.start()
        v.push(chunk())
        v.push(chunk())
        utt = v.flush()
        self.assertIsNotNone(utt)
        self.assertAlmostEqual(utt.duration_ms, 200, delta=1)

    def test_flush_trusts_the_human_about_speech(self):
        """Кнопку отпустили осознанно — не нам решать, была ли там речь."""
        ep = FakeEndpointer(fire_after=10**6, speech_ms=0)
        v = VoiceInput(ep)
        v.enabled = True
        v.start()
        v.push(chunk())
        self.assertIsNotNone(v.flush())

    def test_flush_on_empty_gives_nothing(self):
        v = VoiceInput(FakeEndpointer())
        v.enabled = True
        v.start()
        self.assertIsNone(v.flush())


if __name__ == "__main__":
    unittest.main()
