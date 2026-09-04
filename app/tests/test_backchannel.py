"""Тесты заполнителя.

Главное свойство: заполнитель — не отдельный тракт, а клауза 0 той же
генерации. Значит он гасится теми же правилами и едет по тому же PTS.
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.backchannel import Backchannel                    # noqa: E402


class FakePCM(list):
    pass


def fake_tts(text):
    """Длительность пропорциональна тексту: заполнители разной длины."""
    samples = 24000 * len(text) * 60 // 1000
    return FakePCM([0.0] * samples), 24000


def fake_align(pcm, sr):
    n = max(1, int(len(pcm) / sr * 1000 / 40))
    return [{"ch": "а", "ms": i * 40} for i in range(n)]


def fake_visemes(chars):
    return [{"pts_ms": c["ms"], "viseme": "AA"} for c in chars]


def warm(**kw):
    return Backchannel(seed=1, **kw).warm(fake_tts, fake_align, fake_visemes)


class Preparation(unittest.TestCase):
    def test_everything_ready_after_warm(self):
        bc = warm()
        self.assertTrue(bc.ready)
        self.assertGreaterEqual(len(bc.fillers), 4, "нужно 4-6 вариантов")
        self.assertLessEqual(len(bc.fillers), 6)
        for f in bc.fillers:
            self.assertGreater(f.audio_ms, 0)
            self.assertGreater(len(f.visemes), 0,
                               "трек висем считается заранее, а не по Enter")
            self.assertTrue(f.text)

    def test_not_ready_before_warm(self):
        self.assertFalse(Backchannel().ready)
        self.assertIsNone(Backchannel().pick())

    def test_fillers_are_short(self):
        for f in warm().fillers:
            self.assertLess(f.audio_ms, 2000,
                            f"«{f.text}» длиной {f.audio_ms:.0f} мс — это не заполнитель")


class Selection(unittest.TestCase):
    def test_no_immediate_repeat(self):
        bc = warm()
        prev = None
        for _ in range(200):
            f = bc.pick()
            self.assertIsNot(f, prev, "одно и то же дважды подряд слышно как заедание")
            prev = f

    def test_all_variants_eventually_used(self):
        bc = warm()
        seen = {bc.pick().text for _ in range(400)}
        self.assertEqual(seen, set(bc.texts), "должны звучать все варианты")

    def test_single_variant_degrades_gracefully(self):
        bc = warm(texts=["Угу."])
        self.assertEqual(bc.pick().text, "Угу.")
        self.assertEqual(bc.pick().text, "Угу.", "с одним вариантом повтор неизбежен")


class Determinism(unittest.TestCase):
    def test_same_seed_same_order(self):
        a = [f.text for f in (lambda b: [b.pick() for _ in range(10)])(warm())]
        b = [f.text for f in (lambda b: [b.pick() for _ in range(10)])(warm())]
        self.assertEqual(a, b, "с одним сидом порядок должен воспроизводиться")

    def test_describe_is_serialisable(self):
        import json
        json.dumps(warm().describe())


if __name__ == "__main__":
    unittest.main(verbosity=2)
