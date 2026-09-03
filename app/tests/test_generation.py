"""Тесты сквозного generation_id и склейки PTS.

Проверяется то, что названо в задании: отмена гасит всю цепочку безусловно, а
таймкоды клауз считаются от начала генерации по ФАКТИЧЕСКОЙ длине аудио.
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.generation import GenerationRegistry, PTSTimeline      # noqa: E402


class Registry(unittest.TestCase):
    def test_ids_are_unique_and_sequential(self):
        r = GenerationRegistry()
        ids = [r.start().id for _ in range(5)]
        self.assertEqual(len(set(ids)), 5)
        self.assertEqual(ids, [f"gen-{i}" for i in range(1, 6)])

    def test_new_generation_kills_previous(self):
        r = GenerationRegistry()
        a = r.start()
        b = r.start()
        self.assertFalse(a.alive, "старая генерация должна умереть сама")
        self.assertTrue(b.alive)
        self.assertIn(a.id, r.cancelled_ids)

    def test_cancel_current(self):
        r = GenerationRegistry()
        g = r.start()
        self.assertEqual(r.cancel(), g.id)
        self.assertFalse(g.alive)
        self.assertIsNone(r.current_id)

    def test_cancel_is_idempotent(self):
        r = GenerationRegistry()
        g = r.start()
        r.cancel(g.id)
        self.assertFalse(g.alive)
        r.cancel(g.id)          # второй раз не должен ломаться
        self.assertFalse(g.alive)

    def test_cancel_unknown_id_is_harmless(self):
        r = GenerationRegistry()
        g = r.start()
        self.assertIsNone(r.cancel("gen-999"))
        self.assertTrue(g.alive, "чужой id не должен трогать текущую генерацию")

    def test_alive_check_is_the_gate(self):
        """Проверка принадлежности делается на каждом шаге, а не только на входе:
        клауза, начатая до отмены, доедет до выравнивания уже после неё."""
        r = GenerationRegistry()
        g = r.start()
        self.assertTrue(g.check())
        r.cancel(g.id)
        self.assertFalse(g.check(), "после отмены каждый следующий шаг обязан отвалиться")

    def test_alive_by_id(self):
        r = GenerationRegistry()
        a = r.start()
        b = r.start()
        self.assertFalse(r.alive(a.id))
        self.assertTrue(r.alive(b.id))
        self.assertFalse(r.alive(None))
        self.assertFalse(r.alive("нет такого"))


class Timeline(unittest.TestCase):
    def test_pts_counted_from_generation_start(self):
        tl = PTSTimeline()
        _, v1 = tl.add_clause(1000, [{"pts_ms": 0, "viseme": "AA"},
                                     {"pts_ms": 500, "viseme": "MBP"}], "первая")
        start2, v2 = tl.add_clause(800, [{"pts_ms": 0, "viseme": "OU"},
                                         {"pts_ms": 400, "viseme": "SIL"}], "вторая")
        self.assertEqual([v["pts_ms"] for v in v1], [0, 500])
        self.assertEqual(start2, 1000)
        self.assertEqual([v["pts_ms"] for v in v2], [1000, 1400],
                         "вторая клауза обязана лечь за первой, а не начаться с нуля")
        self.assertEqual(tl.total_ms, 1800)

    def test_offset_uses_actual_audio_not_expected(self):
        """Смещение берётся из фактической длины аудио.

        Если считать по ожидаемой (из длины текста), ошибка копится с каждой
        клаузой — это и есть главный источник накапливающегося дрейфа.
        """
        tl = PTSTimeline()
        # Текст одинаковой длины, а звучание разное: так и бывает у синтеза.
        tl.add_clause(900, [{"pts_ms": 0, "viseme": "AA"}], "текст один")
        tl.add_clause(1400, [{"pts_ms": 0, "viseme": "EE"}], "текст два")
        start3, v3 = tl.add_clause(600, [{"pts_ms": 0, "viseme": "OH"}], "текст три")
        self.assertEqual(start3, 2300)
        self.assertEqual(v3[0]["pts_ms"], 2300)

    def test_no_drift_over_many_clauses(self):
        tl = PTSTimeline()
        durations = [random_ms for random_ms in (713, 1244, 508, 1907, 322, 1150)]
        for i, d in enumerate(durations):
            tl.add_clause(d, [{"pts_ms": 0, "viseme": "AA"}], f"клауза {i}")
        self.assertEqual(tl.total_ms, sum(durations))
        last = tl.clauses[-1]
        self.assertEqual(last["start_ms"], sum(durations[:-1]))

    def test_visemes_inside_clause_keep_relative_order(self):
        tl = PTSTimeline()
        tl.add_clause(500, [{"pts_ms": 0, "viseme": "A"}], "первая")
        _, v = tl.add_clause(500, [{"pts_ms": 0, "viseme": "A"},
                                   {"pts_ms": 120, "viseme": "B"},
                                   {"pts_ms": 300, "viseme": "C"}], "вторая")
        self.assertEqual([x["pts_ms"] for x in v], [500, 620, 800])

    def test_original_visemes_not_mutated(self):
        """Сдвиг не должен портить исходный трек: он может понадобиться ещё раз."""
        tl = PTSTimeline()
        src = [{"pts_ms": 0, "viseme": "AA"}, {"pts_ms": 100, "viseme": "MBP"}]
        tl.add_clause(500, src, "первая")
        tl.add_clause(500, src, "вторая")
        self.assertEqual([v["pts_ms"] for v in src], [0, 100])

    def test_gap_between_clauses(self):
        tl = PTSTimeline()
        tl.add_clause(1000, [], "первая")
        tl.add_clause(500, [], "вторая")
        # Клаузы стыкуются вплотную: пауза берётся из самого аудио, а не
        # добавляется искусственно.
        self.assertEqual(tl.gap_before(1), 0.0)
        self.assertEqual(tl.gap_before(0), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
