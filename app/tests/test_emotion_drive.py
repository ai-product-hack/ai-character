"""Эмоция из накопленной оценки.

Модель размечает реплики скупо — 4–5 тегов на 80. Проверяется, что механика
продукта даёт эмоцию сама и что дуга за сессию действительно читается.
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.emotion_drive import COLD_TO_WARM, mood_from        # noqa: E402
from app.evaluator import Assessment, EvaluationLog          # noqa: E402
from app.scenario import Criterion                           # noqa: E402

# Шкала — СТРОКА, как в файлах сценариев. Первая версия тестов писала сюда
# int, и из-за этого разошлась с боевыми данными: `mood_from` падал на живом
# сценарии, а тесты были зелёные.
CRITS = [Criterion(key="a", title="A", scale="1-5"),
         Criterion(key="b", title="B", scale="1-10")]


def log_of(*scores, key="a"):
    log = EvaluationLog()
    for s in scores:
        log.add(Assessment(criterion=key, score=s, rationale=""))
    return log


class Mood(unittest.TestCase):
    def test_weak_answers_read_cold(self):
        self.assertEqual(mood_from(log_of(1, 2, 1, 2), CRITS).emotion, "skeptical")

    def test_strong_answers_read_warm(self):
        self.assertEqual(mood_from(log_of(5, 5, 4, 5), CRITS).emotion, "impressed")

    def test_middle_is_neutral_and_silent(self):
        """Ровно посередине подмешивать в лицо нечего."""
        m = mood_from(log_of(3, 3, 3, 3), CRITS)
        self.assertEqual(m.emotion, "neutral")
        self.assertEqual(m.intensity, 0.0)

    def test_one_score_is_not_a_verdict(self):
        """Судить о человеке по одному ответу нечестно."""
        m = mood_from(log_of(5), CRITS)
        self.assertEqual(m.emotion, "neutral")
        self.assertEqual(m.intensity, 0.0)

    def test_scales_are_normalised(self):
        """Потолок пятибалльной и десятибалльной шкалы значит одно и то же."""
        top5 = mood_from(log_of(5, 5, 5, 5, key="a"), CRITS)
        top10 = mood_from(log_of(10, 10, 10, 10, key="b"), CRITS)
        self.assertEqual(top5.ratio, top10.ratio, 1.0)
        self.assertEqual(top5.emotion, top10.emotion)
        low5 = mood_from(log_of(1, 1, 1, 1, key="a"), CRITS)
        low10 = mood_from(log_of(1, 1, 1, 1, key="b"), CRITS)
        self.assertEqual(low5.ratio, low10.ratio)

    def test_bottom_of_scale_is_zero_not_a_fifth(self):
        """Балл 1 из 5 — это дно, а не пятая часть отличного."""
        self.assertEqual(mood_from(log_of(1, 1, 1, 1), CRITS).ratio, 0.0)

    def test_arc_is_visible_across_a_session(self):
        """Провалился и выправился — лицо обязано это отразить."""
        up = mood_from(log_of(1, 1, 2, 2, 4, 5, 5, 5), CRITS)
        down = mood_from(log_of(5, 5, 4, 4, 2, 1, 1, 2), CRITS)
        order = COLD_TO_WARM.index
        self.assertGreater(order(up.emotion), order(down.emotion))
        self.assertGreater(up.trend, 0)
        self.assertLess(down.trend, 0)

    def test_unscored_assessments_are_ignored(self):
        log = EvaluationLog()
        for _ in range(5):
            log.add(Assessment(criterion="a", score=None, rationale="без оценки"))
        self.assertEqual(mood_from(log, CRITS).scored, 0)

    def test_every_outcome_is_a_known_emotion(self):
        for scores in ([1, 1], [2, 2], [3, 3], [4, 4], [5, 5]):
            with self.subTest(scores=scores):
                self.assertIn(mood_from(log_of(*scores), CRITS).emotion, COLD_TO_WARM)


if __name__ == "__main__":
    unittest.main()


class RealScenarios(unittest.TestCase):
    """На настоящих сценариях, а не на выдуманных критериях.

    Шкала там записана строкой, и именно на этом расхождении первая версия
    падала в бою при зелёных тестах.
    """

    def test_works_on_every_shipped_scenario(self):
        from app.scenario import load_all
        for sc in load_all(ROOT / "data" / "scenarios"):
            with self.subTest(scenario=sc.id):
                log = EvaluationLog()
                for i, c in enumerate(sc.criteria):
                    log.add(Assessment(criterion=c.key, score=2 + i % 3,
                                       rationale=""))
                m = mood_from(log, sc.criteria)
                self.assertIn(m.emotion, COLD_TO_WARM)
                self.assertGreaterEqual(m.ratio, 0.0)
                self.assertLessEqual(m.ratio, 1.0)

    def test_weak_answers_on_real_criteria_read_cold(self):
        from app.scenario import load_all
        sc = load_all(ROOT / "data" / "scenarios")[0]
        log = EvaluationLog()
        for c in sc.criteria:
            log.add(Assessment(criterion=c.key, score=1, rationale=""))
        m = mood_from(log, sc.criteria)
        self.assertEqual(m.emotion, "skeptical")
        self.assertGreater(m.intensity, 0)


class BaselineOutsideTheAxis(unittest.TestCase):
    """Стартовая эмоция персоны берётся из всей палитры, а не только с оси.

    Оси оценок `angry` и `anxious` не принадлежат: вывести их из баллов нельзя,
    баллов в начале разговора ещё нет. Ровно поэтому проверка членства идёт по
    белому списку, а не по `COLD_TO_WARM`.
    """

    def test_angry_persona_starts_angry(self):
        m = mood_from(log_of(), CRITS, baseline="angry")
        self.assertEqual(m.emotion, "angry")
        self.assertGreater(m.intensity, 0, "иначе лицо ровное с первого кадра")

    def test_anxious_persona_starts_anxious(self):
        self.assertEqual(mood_from(log_of(), CRITS, baseline="anxious").emotion,
                         "anxious")

    def test_scores_push_the_baseline_out(self):
        """Дуга принадлежит оценкам: стартовая эмоция — только начало."""
        m = mood_from(log_of(5, 5, 5), CRITS, baseline="angry")
        self.assertIn(m.emotion, COLD_TO_WARM)
        self.assertNotEqual(m.emotion, "angry")

    def test_unknown_baseline_is_ignored(self):
        self.assertEqual(mood_from(log_of(), CRITS, baseline="ликование").emotion,
                         "neutral")

    def test_score_axis_never_yields_the_two_new_states(self):
        """Оценки не должны выводить гнев: он не про качество ответов."""
        for scores in ([1, 1, 1], [3, 3], [5, 5, 5], [1, 5, 3, 4]):
            m = mood_from(log_of(*scores), CRITS)
            self.assertIn(m.emotion, COLD_TO_WARM, scores)
