"""Переключатель синтезаторов и офлайновый запасной.

Настоящие модели не поднимаем: проверяем логику выбора и, главное, то, что
падение сетевого синтеза не роняет реплику.
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.media import FallbackTTS, build_tts                # noqa: E402


class FakeTTS:
    def __init__(self, name, sr=24000, fail=False):
        self.name = name
        self.sr = sr
        self.fail = fail
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("сеть недоступна")
        return [0.0] * 100, self.sr

    def describe(self):
        return {"name": self.name}


class Fallback(unittest.TestCase):
    def test_silent_while_primary_works(self):
        primary, backup = FakeTTS("net"), FakeTTS("local")
        tts = FallbackTTS(primary, backup)
        for _ in range(3):
            tts("реплика")
        self.assertFalse(tts.failed_over)
        self.assertEqual(backup.calls, [])
        self.assertEqual(tts.describe()["active"]["name"], "net")

    def test_survives_network_failure(self):
        primary, backup = FakeTTS("net", fail=True), FakeTTS("local")
        tts = FallbackTTS(primary, backup)
        pcm, sr = tts("реплика")
        self.assertTrue(pcm)                     # реплика не потеряна
        self.assertEqual(sr, 24000)
        self.assertTrue(tts.failed_over)
        self.assertEqual(backup.calls, ["реплика"])

    def test_does_not_flap_back(self):
        """Голос не должен меняться посреди диалога, даже если сеть вернулась."""
        primary, backup = FakeTTS("net", fail=True), FakeTTS("local")
        tts = FallbackTTS(primary, backup)
        tts("первая")
        primary.fail = False                     # сеть «починилась»
        tts("вторая")
        self.assertEqual(primary.calls, ["первая"])   # к сетевому больше не ходим
        self.assertEqual(backup.calls, ["первая", "вторая"])
        self.assertEqual(len(tts.failures), 1)

    def test_sample_rate_follows_active(self):
        tts = FallbackTTS(FakeTTS("net", sr=16000, fail=True), FakeTTS("local", sr=24000))
        self.assertEqual(tts.sr, 16000)
        tts("реплика")
        self.assertEqual(tts.sr, 24000)


class Selection(unittest.TestCase):
    def test_unknown_provider_is_loud(self):
        with self.assertRaises(SystemExit) as cm:
            build_tts({"provider": "выдуманный"})
        self.assertIn("выдуманный", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
