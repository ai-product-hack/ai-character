"""Тесты конвейера реплики на подставных TTS и выравнивании.

Настоящие Silero и GigaAM здесь не нужны: проверяется склейка, а не качество
синтеза. Подставки детерминированы, поэтому тест не зависит ни от сети, ни от
модели.
"""
import pathlib
import sys
import threading
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.generation import GenerationRegistry               # noqa: E402
from app.pipeline import ReplyPipeline, subtitle_cues       # noqa: E402


class FakePCM(list):
    """Минимальная замена массива: нужна только длина."""


def fake_tts(ms_per_char=20, sr=24000, delay=0.0):
    def tts(text):
        if delay:
            time.sleep(delay)
        n = int(sr * len(text) * ms_per_char / 1000)
        return FakePCM([0.0] * n), sr
    return tts


def fake_align(step_ms=40):
    def align(pcm, sr):
        n = max(1, int(len(pcm) / sr * 1000 / step_ms))
        return [{"ch": "а" if i % 4 else " ", "ms": i * step_ms} for i in range(n)]
    return align


def fake_visemes(chars):
    return [{"pts_ms": c["ms"], "viseme": "SIL" if c["ch"] == " " else "AA"}
            for c in chars]


def token_stream(text, chunk=4, delay=0.0):
    def gen(system, prompt):
        for i in range(0, len(text), chunk):
            if delay:
                time.sleep(delay)
            yield text[i:i + chunk]
    return gen


REPLY = ("Спасибо, картина понятна. Расскажите, что именно делали лично вы? "
         "И какие решения принимали сами?")


def build(text=REPLY, **kw):
    reg = GenerationRegistry()
    p = ReplyPipeline(token_stream(text, **kw.pop("stream", {})),
                      kw.pop("tts", fake_tts()), kw.pop("align", fake_align()),
                      fake_visemes, reg, kw.pop("splitter_kw", None))
    return reg, p


class Slicing(unittest.TestCase):
    def test_reply_becomes_several_clauses(self):
        reg, p = build()
        g = reg.start()
        res = p.run("sys", "prompt", g)
        self.assertGreaterEqual(len(res), 2, "реплика должна резаться на клаузы")
        self.assertTrue(res[0].first)
        self.assertFalse(any(r.first for r in res[1:]))

    def test_first_clause_arrives_first(self):
        reg, p = build()
        g = reg.start()
        seen = []
        p.run("sys", "prompt", g, on_result=lambda r: seen.append(r.index))
        self.assertEqual(seen, sorted(seen), "клаузы должны приходить по порядку")
        self.assertEqual(seen[0], 0)


class Timeline(unittest.TestCase):
    def test_pts_is_continuous_across_clauses(self):
        reg, p = build()
        g = reg.start()
        res = p.run("sys", "prompt", g)
        for a, b in zip(res, res[1:]):
            self.assertAlmostEqual(a.start_ms + a.audio_ms, b.start_ms, places=6,
                                   msg="клаузы должны стыковаться без дыр и нахлёста")

    def test_visemes_shifted_to_generation_time(self):
        reg, p = build()
        g = reg.start()
        res = p.run("sys", "prompt", g)
        self.assertEqual(res[0].visemes[0]["pts_ms"], 0)
        for r in res[1:]:
            self.assertGreaterEqual(r.visemes[0]["pts_ms"], r.start_ms,
                                    "висемы второй клаузы не могут начинаться с нуля")

    def test_offset_follows_actual_audio_length(self):
        """Смещение считается по фактическому звуку, а не по длине текста."""
        reg, p = build(tts=fake_tts(ms_per_char=35))
        g = reg.start()
        res = p.run("sys", "prompt", g)
        for a, b in zip(res, res[1:]):
            self.assertAlmostEqual(b.start_ms, a.start_ms + a.audio_ms, places=6)
        total = sum(r.audio_ms for r in res)
        self.assertAlmostEqual(p.last_timeline.total_ms, total, places=6)

    def test_monotonic_visemes_over_whole_reply(self):
        reg, p = build()
        g = reg.start()
        res = p.run("sys", "prompt", g)
        pts = [v["pts_ms"] for r in res for v in r.visemes]
        self.assertEqual(pts, sorted(pts), "таймкоды всей реплики обязаны расти")


class Cancellation(unittest.TestCase):
    def test_cancel_before_first_clause_yields_nothing(self):
        reg, p = build(stream={"chunk": 2, "delay": 0.004})
        g = reg.start()
        reg.cancel(g.id)
        res = p.run("sys", "prompt", g)
        self.assertEqual(res, [], "после отмены не должно родиться ни одной клаузы")

    def test_cancel_mid_stream_stops_delivery(self):
        reg, p = build(stream={"chunk": 2, "delay": 0.003})
        g = reg.start()
        delivered = []

        def on_result(r):
            delivered.append(r)
            reg.cancel(g.id)          # перебиваем на первой же готовой клаузе

        p.run("sys", "prompt", g, on_result=on_result)
        self.assertEqual(len(delivered), 1,
                         "после отмены дальше первой клаузы уйти не должно")

    def test_cancel_during_synthesis_drops_clause(self):
        """Клауза, начатая до отмены, не должна доехать до плеера."""
        reg = GenerationRegistry()
        started = threading.Event()

        def slow_tts(text):
            started.set()
            time.sleep(0.05)
            return FakePCM([0.0] * 2400), 24000

        p = ReplyPipeline(token_stream(REPLY), slow_tts, fake_align(),
                          fake_visemes, reg)
        g = reg.start()
        delivered = []

        def canceller():
            started.wait(timeout=2)
            reg.cancel(g.id)

        threading.Thread(target=canceller, daemon=True).start()
        p.run("sys", "prompt", g, on_result=delivered.append)
        self.assertEqual(delivered, [],
                         "синтез уже улетевшей вперёд клаузы обязан быть отброшен")

    def test_new_generation_invalidates_previous(self):
        reg, p = build()
        g1 = reg.start()
        reg.start()                    # пользователь отправил новое сообщение
        res = p.run("sys", "prompt", g1)
        self.assertEqual(res, [], "старая генерация не имеет права договорить")

    def test_results_carry_generation_id(self):
        reg, p = build()
        g = reg.start()
        res = p.run("sys", "prompt", g)
        self.assertTrue(all(r.generation_id == g.id for r in res))


class Subtitles(unittest.TestCase):
    def test_cues_cover_every_word_of_the_clause(self):
        reg, p = build()
        g = reg.start()
        res = p.run("sys", "prompt", g)
        for r in res:
            cues = subtitle_cues(r)
            self.assertEqual([c["text"] for c in cues], r.text.split(),
                             "субтитр должен показывать исходный текст, а не расшифровку")

    def test_cues_are_on_the_generation_timeline(self):
        reg, p = build()
        g = reg.start()
        res = p.run("sys", "prompt", g)
        allc = [c for r in res for c in subtitle_cues(r)]
        pts = [c["pts_ms"] for c in allc]
        self.assertEqual(pts, sorted(pts), "субтитры едут по тому же PTS, что и висемы")
        self.assertGreaterEqual(allc[0]["pts_ms"], res[0].start_ms)

    def test_cues_carry_generation_id(self):
        reg, p = build()
        g = reg.start()
        res = p.run("sys", "prompt", g)
        for r in res:
            for c in subtitle_cues(r):
                self.assertEqual(c["generation_id"], g.id,
                                 "строка субтитров тоже гасится по generation_id")

    def test_cues_survive_empty_alignment(self):
        reg, p = build(align=lambda pcm, sr: [])
        g = reg.start()
        res = p.run("sys", "prompt", g)
        for r in res:
            cues = subtitle_cues(r)
            self.assertEqual(len(cues), 1, "без таймкодов клауза показывается целиком")


class Timings(unittest.TestCase):
    def test_first_audio_before_stream_end(self):
        """Смысл нарезки: первый звук готов, пока модель ещё договаривает."""
        reg, p = build(stream={"chunk": 3, "delay": 0.004})
        g = reg.start()
        t0 = time.perf_counter()
        p.run("sys", "prompt", g)
        total_ms = (time.perf_counter() - t0) * 1000
        self.assertIsNotNone(p.last_stats["t_first_audio"])
        self.assertLess(p.last_stats["t_first_audio"], total_ms * 0.8,
                        "первый звук должен появиться заметно раньше конца потока")

    def test_per_clause_timings_recorded(self):
        reg, p = build()
        g = reg.start()
        res = p.run("sys", "prompt", g)
        for r in res:
            self.assertIn("tts_ms", r.timings)
            self.assertIn("align_ms", r.timings)


if __name__ == "__main__":
    unittest.main(verbosity=2)
