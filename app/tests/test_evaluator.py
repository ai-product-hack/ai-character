"""Тесты фоновой оценки.

Главное свойство: оценщик не на критическом пути. Он не может ни задержать
диалог, ни уронить его — что бы ни вернула модель.
"""
import pathlib
import sys
import threading
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app import report as report_mod                        # noqa: E402
from app.dialogue import DialogueState                      # noqa: E402
from app.evaluator import (BackgroundEvaluator, EvaluationLog,   # noqa: E402
                           build_eval_prompt, parse_scores)
from app.scenario import load_all                            # noqa: E402

SCENARIOS = load_all(ROOT / "data" / "scenarios")
SC = SCENARIOS[0]


def state_with_turns():
    st = DialogueState(SC)
    st.add_agent("Расскажите про нагрузку.")
    st.add_user("Две тысячи запросов в секунду, задержка сто миллисекунд.")
    return st


def scorer(payload):
    def llm(system, prompt):
        return payload
    return llm


class Parsing(unittest.TestCase):
    def test_valid_scores(self):
        key = SC.criteria[0].key
        out = parse_scores(
            f'{{"scores":[{{"criterion":"{key}","score":4,"rationale":"есть числа"}}]}}',
            SC.criteria)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].criterion, key)
        self.assertEqual(out[0].score, 4.0)

    def test_prose_around_json_survives(self):
        key = SC.criteria[0].key
        out = parse_scores(
            f'Вот оценка:\n{{"scores":[{{"criterion":"{key}","score":3,"rationale":"ок"}}]}}\nГотово.',
            SC.criteria)
        self.assertEqual(len(out), 1)

    def test_garbage_yields_nothing(self):
        for blob in ["", "совсем не json", "{сломан", "null", '{"scores": "не список"}']:
            with self.subTest(blob=blob[:20]):
                self.assertEqual(parse_scores(blob, SC.criteria), [])

    def test_unknown_criterion_dropped(self):
        out = parse_scores('{"scores":[{"criterion":"выдуманный","score":5}]}', SC.criteria)
        self.assertEqual(out, [], "критерий, которого методист не задавал, не берём")

    def test_score_outside_scale_dropped(self):
        key = SC.criteria[0].key
        for bad in (0, 9, -3, "много"):
            with self.subTest(score=bad):
                out = parse_scores(
                    f'{{"scores":[{{"criterion":"{key}","score":{bad!r}}}]}}'.replace("'", '"'),
                    SC.criteria)
                self.assertEqual(out, [], f"оценка {bad} вне шкалы — не оценка")


class Prompt(unittest.TestCase):
    def test_prompt_has_criteria_and_last_exchange(self):
        p = build_eval_prompt(state_with_turns())
        for c in SC.criteria:
            self.assertIn(c.key, p)
        self.assertIn("Две тысячи запросов", p)
        self.assertIn(SC.title, p)


class OffCriticalPath(unittest.TestCase):
    def test_submit_does_not_block(self):
        """Оценщик не имеет права задерживать диалог."""
        slow = threading.Event()

        def slow_llm(system, prompt):
            slow.wait(timeout=5)
            return '{"scores":[]}'

        ev = BackgroundEvaluator(slow_llm)
        st = state_with_turns()
        t0 = time.perf_counter()
        for _ in range(5):
            ev.submit(st)
        elapsed = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed, 50, f"постановка в очередь заняла {elapsed:.0f} мс")
        slow.set()
        ev.close()

    def test_llm_failure_does_not_propagate(self):
        def broken(system, prompt):
            raise RuntimeError("оценщик упал")

        ev = BackgroundEvaluator(broken)
        st = state_with_turns()
        ev.submit(st)
        ev.drain(timeout=3)
        self.assertGreaterEqual(len(ev.log.errors), 1, "ошибка должна попасть в лог")
        self.assertEqual(ev.log.assessments, [])
        ev.close()

    def test_garbage_from_model_does_not_propagate(self):
        ev = BackgroundEvaluator(scorer("я не буду отвечать json"))
        ev.submit(state_with_turns())
        ev.drain(timeout=3)
        self.assertEqual(ev.log.assessments, [])
        self.assertEqual(ev.log.errors, [], "мусор — это не ошибка, а пустая оценка")
        ev.close()


class Incremental(unittest.TestCase):
    def test_report_ready_without_extra_calls(self):
        """К моменту finish отчёт складывается без обращений к модели."""
        key = SC.criteria[0].key
        ev = BackgroundEvaluator(scorer(
            f'{{"scores":[{{"criterion":"{key}","score":4,"rationale":"назвал числа"}}]}}'))
        st = state_with_turns()
        for _ in range(3):
            ev.submit(st)
        self.assertTrue(ev.drain(timeout=5))
        calls_before = ev.log.calls

        t0 = time.perf_counter()
        rep = report_mod.build(st, ev.log)
        build_ms = (time.perf_counter() - t0) * 1000

        self.assertEqual(ev.log.calls, calls_before, "сборка отчёта не зовёт модель")
        self.assertLess(build_ms, 50, f"сборка отчёта заняла {build_ms:.0f} мс")
        got = next(c for c in rep.criteria if c.key == key)
        self.assertEqual(got.score, 4.0)
        self.assertEqual(got.rationale, "назвал числа")
        ev.close()

    def test_scores_accumulate_across_turns(self):
        key = SC.criteria[0].key
        payloads = iter([
            f'{{"scores":[{{"criterion":"{key}","score":2,"rationale":"общие слова"}}]}}',
            f'{{"scores":[{{"criterion":"{key}","score":4,"rationale":"появились числа"}}]}}',
        ])
        ev = BackgroundEvaluator(lambda s, p: next(payloads))
        st = state_with_turns()
        ev.submit(st)
        ev.drain(timeout=5)
        ev.submit(st)
        ev.drain(timeout=5)
        rep = report_mod.build(st, ev.log)
        got = next(c for c in rep.criteria if c.key == key)
        self.assertEqual(got.score, 3.0, "оценки по критерию усредняются")
        self.assertEqual(got.rationale, "появились числа",
                         "обоснование берётся последнее, оно на полном контексте")
        ev.close()

    def test_report_without_evaluation_still_builds(self):
        rep = report_mod.build(state_with_turns())
        self.assertIsNone(rep.overall)
        self.assertEqual(rep.coverage, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
