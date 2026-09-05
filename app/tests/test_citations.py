"""Цитаты в отчёте: оценка со ссылкой на реплику, которая её обосновала.

Разница принципиальная. «Коммуникация 3 из 5» — мнение модели, спорить с ним
можно только на уровне «а мне кажется, четыре». «3 из 5, вот реплика» — разбор.
Поэтому хранится НОМЕР реплики, а не её копия: по номеру интерфейс
прокручивает транскрипт, копия рассинхронизировалась бы с ним.
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app import report as report_mod                      # noqa: E402
from app.dialogue import DialogueState                    # noqa: E402
from app.evaluator import (Assessment, EvaluationLog,      # noqa: E402
                           build_eval_prompt, build_summary_prompt,
                           fallback_conclusion, parse_scores)
from app.scenario import Criterion, Persona, Scenario, Stage   # noqa: E402

CRITERIA = [
    Criterion("structure", "Структурность", "1-5", "рассыпается", "построен"),
    Criterion("numbers", "Конкретность", "1-5", "общие слова", "числа"),
]


def scenario():
    return Scenario(
        id="t", title="Тест", type="interview",
        persona=Persona(role="интервьюер", start_emotion="skeptical"),
        stages=[Stage("a", "первый"), Stage("b", "второй")],
        criteria=list(CRITERIA))


def state_with_talk():
    st = DialogueState(scenario())
    st.add_agent("Расскажите о себе.")
    st.add_user("Я делал сервис выдачи.")
    st.add_agent("А какая была нагрузка?")
    st.add_user("Ну, много.")
    return st


class PromptNumbersTurns(unittest.TestCase):
    def test_turns_are_numbered_through_the_whole_history(self):
        """Номера сквозные: окно скользит, а транскрипт нумеруется от нуля."""
        st = state_with_talk()
        p = build_eval_prompt(st, window=2)
        self.assertIn("[2] Агент:", p)
        self.assertIn("[3] Собеседник:", p)
        self.assertNotIn("[0]", p, "окно короче истории — ранних номеров быть не может")

    def test_prompt_asks_for_the_turn(self):
        self.assertIn("turn", build_eval_prompt(state_with_talk()))


class ParseCitations(unittest.TestCase):
    def test_turn_is_parsed(self):
        out = parse_scores('{"scores":[{"criterion":"numbers","score":2,'
                           '"turn":3,"rationale":"«много» — не число"}]}',
                           CRITERIA, turns=4)
        self.assertEqual(out[0].quote_turn, 3)

    def test_turn_out_of_range_is_dropped(self):
        """Клик по цитате, ведущий в пустоту, хуже отсутствующей цитаты."""
        out = parse_scores('{"scores":[{"criterion":"numbers","score":2,'
                           '"turn":99,"rationale":"x"}]}', CRITERIA, turns=4)
        self.assertEqual(out[0].quote_turn, -1)

    def test_missing_turn_does_not_drop_the_score(self):
        out = parse_scores('{"scores":[{"criterion":"numbers","score":2,'
                           '"rationale":"x"}]}', CRITERIA, turns=4)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].quote_turn, -1)

    def test_garbage_turn_does_not_raise(self):
        out = parse_scores('{"scores":[{"criterion":"numbers","score":2,'
                           '"turn":"третья","rationale":"x"}]}', CRITERIA, turns=4)
        self.assertEqual(out[0].quote_turn, -1)


class ReportCitations(unittest.TestCase):
    def setUp(self):
        self.state = state_with_talk()
        self.log = EvaluationLog()
        self.log.add(Assessment("numbers", 2.0, "«много» — не число",
                                turn_index=3, quote_turn=3, stage_id="a"))
        self.log.add(Assessment("structure", 4.0, "ответ построен",
                                turn_index=3, quote_turn=1, stage_id="a"))

    def test_each_criterion_carries_its_citation(self):
        rep = report_mod.build(self.state, self.log).to_dict()
        by_key = {c["key"]: c for c in rep["criteria"]}
        self.assertEqual(by_key["numbers"]["citations"][0]["turn"], 3)
        self.assertEqual(by_key["structure"]["citations"][0]["turn"], 1)

    def test_citation_points_at_a_real_turn(self):
        rep = report_mod.build(self.state, self.log).to_dict()
        for c in rep["criteria"]:
            for q in c["citations"]:
                if q["turn"] >= 0:
                    self.assertLess(q["turn"], len(rep["transcript"]))
                    self.assertTrue(rep["transcript"][q["turn"]]["text"])

    def test_citation_stores_index_not_a_copy_of_the_text(self):
        rep = report_mod.build(self.state, self.log).to_dict()
        blob = str(rep["criteria"])
        self.assertNotIn("Ну, много.", blob,
                         "копия текста разъедется с транскриптом при первой правке")

    def test_cited_counts_criteria_with_a_live_link(self):
        rep = report_mod.build(self.state, self.log).to_dict()
        self.assertEqual(rep["cited"], 2)

    def test_agent_observations_become_citations_too(self):
        self.state.observe("structure", "замечание агента", 3)
        rep = report_mod.build(self.state, self.log).to_dict()
        by_key = {c["key"]: c for c in rep["criteria"]}
        sources = {q["source"] for q in by_key["structure"]["citations"]}
        self.assertEqual(sources, {"agent", "background"})

    def test_transcript_is_indexed_for_scrolling(self):
        rep = report_mod.build(self.state, self.log).to_dict()
        self.assertEqual([t["i"] for t in rep["transcript"]], [0, 1, 2, 3])

    def test_header_carries_persona_and_duration(self):
        """Шапка одна на все тренировки — меняется её содержимое, не шаблон."""
        rep = report_mod.build(self.state, self.log).to_dict()
        self.assertEqual(rep["persona"]["role"], "интервьюер")
        self.assertEqual(rep["scenario_type"], "interview")
        self.assertIsNotNone(rep["duration_s"])


class Conclusion(unittest.TestCase):
    def test_fallback_conclusion_names_best_and_worst(self):
        log = EvaluationLog()
        log.add(Assessment("numbers", 1.0, "", quote_turn=3))
        log.add(Assessment("structure", 5.0, "", quote_turn=1))
        text = fallback_conclusion(state_with_talk(), log)
        self.assertIn("Структурность", text)
        self.assertIn("Конкретность", text)

    def test_fallback_conclusion_without_scores_says_so(self):
        text = fallback_conclusion(state_with_talk(), EvaluationLog())
        self.assertIn("не накопилось", text)

    def test_summary_prompt_carries_scores_and_talk(self):
        log = EvaluationLog()
        log.add(Assessment("numbers", 2.0, "", quote_turn=3))
        p = build_summary_prompt(state_with_talk(), log)
        self.assertIn("Ну, много.", p)
        self.assertIn("Конкретность", p)
        self.assertIn("интервьюер", p)


class Persistence(unittest.TestCase):
    def test_report_is_appended_as_one_line(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "history.jsonl"
            report_mod.save({"scenario_id": "a", "text": "строка\nс переводом"}, path)
            report_mod.save({"scenario_id": "b"}, path)
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2, "один прогон — одна строка")
            self.assertEqual(json.loads(lines[0])["scenario_id"], "a")

    def test_broken_path_does_not_raise(self):
        """Диалог уже состоялся — падать из-за журнала после него нельзя."""
        out = report_mod.save({"x": 1}, pathlib.Path("/нет/такого/пути/h.jsonl"))
        self.assertIn("не записан", out)


if __name__ == "__main__":
    unittest.main()


class ScaleReachesTheScreen(unittest.TestCase):
    """«3» — число без единиц. Шкала есть в данных и до отчёта не доходила.

    Замечание с живого прогона: «набрал два с половиной балла и не знаю, что
    это значит; если максимум три — круто, если десять — плохо».
    """

    def _report(self, *criteria):
        sc = scenario()
        sc.criteria = list(criteria)
        st = DialogueState(sc)
        st.add_agent("Вопрос.")
        st.add_user("Ответ.")
        log = EvaluationLog()
        for i, c in enumerate(criteria):
            log.add(Assessment(c.key, 2.0 + i, "потому что", quote_turn=1))
        return report_mod.build(st, log).to_dict()

    def test_bounds_and_anchors_travel_with_the_score(self):
        rep = self._report(*CRITERIA)
        c = rep["criteria"][0]
        self.assertEqual((c["lo"], c["hi"]), (1, 5))
        self.assertEqual(c["anchor_1"], "рассыпается")
        self.assertEqual(c["anchor_5"], "построен")

    def test_uniform_scale_is_reported(self):
        rep = self._report(*CRITERIA)
        self.assertEqual(rep["scale_max"], 5)
        self.assertEqual(rep["scale_min"], 1)

    def test_mixed_scales_give_no_common_maximum(self):
        """Средний балл по шкалам 1-5 и 1-10 — число без смысла."""
        rep = self._report(CRITERIA[0],
                           Criterion("wide", "Широкий", "1-10", "низ", "верх"))
        self.assertIsNone(rep["scale_max"])
        self.assertIsNotNone(rep["overall_ratio"])

    def test_ratio_normalises_each_criterion_to_its_own_scale(self):
        """2 из 5 и 3 из 10 — это 0.25 и 0.22, а не 2 и 3."""
        rep = self._report(CRITERIA[0],
                           Criterion("wide", "Широкий", "1-10", "низ", "верх"))
        self.assertAlmostEqual(rep["overall_ratio"], (0.25 + 2 / 9) / 2, places=2)

    def test_bottom_of_the_scale_is_zero_not_a_fifth(self):
        rep = self._report(Criterion("k", "К", "1-5", "низ", "верх"))
        rep["criteria"][0]["score"] = 1.0
        self.assertEqual(report_mod.CriterionResult(
            key="k", title="К", scale="1-5", lo=1, hi=5, score=1.0).ratio, 0.0)


class TwoReadersTwoConclusions(unittest.TestCase):
    """Один разговор, два читателя. Механика общая, адресат разный.

    Замечание с живого прогона: «методисту приходит тот же отчёт, но он
    персонализированный под того, кто проходил; а HR как будто нужно
    по-другому — кандидат ответил так-то, обратите внимание».
    """

    def test_both_texts_are_parsed(self):
        from app.evaluator import parse_conclusions
        out = parse_conclusions('{"for_trainee": "Вы держали рамку.", '
                                '"for_methodist": "Кандидат ушёл в общие слова."}')
        self.assertEqual(out["trainee"], "Вы держали рамку.")
        self.assertEqual(out["methodist"], "Кандидат ушёл в общие слова.")

    def test_truncated_answer_is_salvaged(self):
        """Обрыв по лимиту токенов не должен стоить всего вывода."""
        from app.evaluator import parse_conclusions
        out = parse_conclusions('{ "for_trainee": "Вы хорошо начали и признали '
                                'вину, но дальше ушли в общие')
        self.assertIn("Вы хорошо начали", out["trainee"])
        self.assertNotIn("{", out["trainee"], "сырой JSON на экран не идёт")

    def test_broken_json_never_reaches_the_screen_as_prose(self):
        from app.evaluator import parse_conclusions
        self.assertEqual(parse_conclusions('{ "нечто": '), {})

    def test_plain_text_answer_goes_to_the_trainee(self):
        from app.evaluator import parse_conclusions
        out = parse_conclusions("Просто текст без JSON.")
        self.assertEqual(out["trainee"], "Просто текст без JSON.")

    def test_report_carries_both(self):
        st = state_with_talk()
        rep = report_mod.build(st, EvaluationLog(), conclusion="вам",
                               conclusion_methodist="о нём").to_dict()
        self.assertEqual(rep["conclusion"], "вам")
        self.assertEqual(rep["conclusion_methodist"], "о нём")

    def test_methodist_falls_back_to_the_shared_text(self):
        """Если второго текста нет, методист читает первый, а не пустоту."""
        st = state_with_talk()
        rep = report_mod.build(st, EvaluationLog(), conclusion="общий").to_dict()
        self.assertEqual(rep["conclusion_methodist"], "общий")

    def test_summary_prompt_asks_for_both(self):
        from app.evaluator import SUMMARY_SYSTEM
        self.assertIn("for_trainee", SUMMARY_SYSTEM)
        self.assertIn("for_methodist", SUMMARY_SYSTEM)

    def test_summary_has_its_own_token_budget(self):
        """Бюджет оценки — 300 токенов, двум выводам его не хватает."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[2] / "app" / "server.py"
               ).read_text(encoding="utf-8")
        self.assertIn("summary_llm", src)
