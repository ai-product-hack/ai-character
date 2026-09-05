"""Генератор сценария из текста и всё, что его страхует.

Живую модель здесь не зовём: проверяется не качество формулировок, а то, что
сломанный ответ не доходит до методиста. Три слоя защиты — схема, повтор с
текстом ошибки, шаблон — и каждый проверяется отдельно.
"""
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app import generator as gen                          # noqa: E402
from app import translit                                  # noqa: E402
from app.emotion_tags import EMOTIONS                     # noqa: E402
from app.scenario import Scenario                         # noqa: E402
from app.templates import TYPES, template                 # noqa: E402


def good_raw(stages=6, criteria=4):
    return {
        "title": "Собеседование на Rust-разработчика",
        "type": "interview",
        "persona": {"role": "технический лид команды системного ПО",
                    "tone": "деловой, дотошный",
                    "strictness": "высокая, не принимает общих слов",
                    "pressure": "просит обосновать каждое решение",
                    "start_emotion": "skeptical"},
        "stages": [{"goal": f"этап номер {i}",
                    "hint": "Спроси про конкретику и дожимай общие ответы.",
                    "advance_when": "прозвучал конкретный пример",
                    "opening": "Расскажите подробнее.",
                    "max_turns": 3} for i in range(stages)],
        "criteria": [{"title": f"критерий номер {i}",
                      "anchor_1": "не проявлено",
                      "anchor_5": "проявлено полностью"} for i in range(criteria)],
    }


class Translit(unittest.TestCase):
    def test_russian_title_becomes_latin_key(self):
        self.assertEqual(translit.slug("Глубина роли"), "glubina_roli")
        self.assertEqual(translit.slug("Работа с возражениями"),
                         "rabota_s_vozrazheniyami")

    def test_empty_falls_back(self):
        self.assertEqual(translit.slug("!!!", "stage_3"), "stage_3")

    def test_key_never_starts_with_digit(self):
        self.assertTrue(translit.slug("3 уровня зрелости")[0].isalpha())

    def test_duplicates_get_suffixes(self):
        keys = translit.unique_slugs(["Цена", "Цена", "Цена"])
        self.assertEqual(len(set(keys)), 3, "повтор ключа уводит оценку не туда")


class Assembly(unittest.TestCase):
    def test_ids_and_keys_generated_from_titles(self):
        sc = gen.to_scenario(good_raw())
        self.assertEqual(sc.validate(), [])
        for c in sc.criteria:
            self.assertTrue(c.key.isascii(), f"ключ не латиницей: {c.key}")
        self.assertEqual(len(set(s.id for s in sc.stages)), len(sc.stages))

    def test_model_never_supplies_identifiers(self):
        """Ключ приходит из названия, а не из ответа модели."""
        raw = good_raw()
        raw["criteria"][0]["key"] = "МУСОР ОТ МОДЕЛИ"
        raw["stages"][0]["id"] = "тоже мусор"
        sc = gen.to_scenario(raw)
        self.assertNotIn("МУСОР", sc.criteria[0].key)
        self.assertNotIn("мусор", sc.stages[0].id)

    def test_turn_budget_is_clamped(self):
        """Семь этапов по пять ходов — это диалог, который не кончится за показ."""
        raw = good_raw()
        raw["stages"][0]["max_turns"] = 99
        raw["stages"][1]["max_turns"] = 0
        raw["stages"][2]["max_turns"] = "нет"
        sc = gen.to_scenario(raw)
        self.assertEqual(sc.stages[0].max_turns, 4)
        self.assertEqual(sc.stages[1].max_turns, 2)
        self.assertIsNone(sc.stages[2].max_turns)

    def test_unknown_emotion_falls_back_to_whitelist(self):
        """Белый список эмоций — контракт с эмоциональным слоем аватара."""
        raw = good_raw()
        raw["persona"]["start_emotion"] = "яростный"
        sc = gen.to_scenario(raw)
        self.assertIn(sc.persona.start_emotion, EMOTIONS)

    def test_source_text_is_kept(self):
        sc = gen.to_scenario(good_raw(), "исходный текст методиста")
        self.assertEqual(sc.source_text, "исходный текст методиста")
        self.assertEqual(sc.source, "generated")


class Checks(unittest.TestCase):
    def test_good_scenario_passes(self):
        self.assertEqual(gen.extra_checks(gen.to_scenario(good_raw())), [])

    def test_too_few_stages_caught(self):
        self.assertTrue(gen.extra_checks(gen.to_scenario(good_raw(stages=3))))

    def test_too_many_criteria_caught(self):
        self.assertTrue(gen.extra_checks(gen.to_scenario(good_raw(criteria=9))))

    def test_empty_hint_caught(self):
        raw = good_raw()
        raw["stages"][2]["hint"] = ""
        self.assertTrue(any("подсказка" in p
                            for p in gen.extra_checks(gen.to_scenario(raw))))

    def test_repeated_goals_caught(self):
        raw = good_raw()
        raw["stages"][1]["goal"] = raw["stages"][0]["goal"]
        self.assertTrue(any("повторяются" in p
                            for p in gen.extra_checks(gen.to_scenario(raw))))


class Retries(unittest.TestCase):
    def test_first_attempt_is_used_when_valid(self):
        calls = []
        result = gen.generate(lambda p: (calls.append(p), good_raw())[1], "текст")
        self.assertFalse(result.fallback)
        self.assertEqual(len(calls), 1)

    def test_broken_answer_triggers_retry_with_error_in_prompt(self):
        """Текст ошибки обязан уйти в промпт: иначе повтор бессмыслен."""
        prompts = []

        def llm(prompt):
            prompts.append(prompt)
            return good_raw() if len(prompts) > 1 else good_raw(stages=2)

        result = gen.generate(llm, "текст")
        self.assertFalse(result.fallback)
        self.assertEqual(len(prompts), 2)
        self.assertIn("НЕ ПРОШЛА ПРОВЕРКУ", prompts[1])
        self.assertIn("этапов 2", prompts[1])

    def test_three_failures_fall_back_to_template(self):
        result = gen.generate(lambda p: good_raw(stages=2), "текст", "sales")
        self.assertTrue(result.fallback)
        self.assertEqual(len(result.attempts), 3)
        self.assertEqual(result.scenario.validate(), [])
        self.assertEqual(result.scenario.source, "template")
        self.assertTrue(result.message, "методисту нужно честное сообщение")

    def test_exception_also_falls_back(self):
        def boom(prompt):
            raise RuntimeError("сеть отвалилась")
        result = gen.generate(boom, "текст")
        self.assertTrue(result.fallback)
        self.assertEqual(result.scenario.validate(), [])

    def test_bad_request_is_not_retried(self):
        """400 повтором не чинится — незачем держать методиста втрое дольше."""
        class Bad(RuntimeError):
            status_code = 400

        calls = []

        def llm(prompt):
            calls.append(prompt)
            raise Bad("схема не та")

        result = gen.generate(llm, "текст")
        self.assertTrue(result.fallback)
        self.assertEqual(len(calls), 1)

    def test_progress_is_reported_before_every_attempt(self):
        seen = []
        gen.generate(lambda p: good_raw(stages=2), "текст",
                     on_progress=seen.append)
        self.assertEqual([s["attempt"] for s in seen], [1, 2, 3])


class JobsSurviveAnything(unittest.TestCase):
    """Задание обязано завершиться, чем бы ни кончился вызов.

    Задание, застрявшее в «идёт», — это крутящийся индикатор навсегда, и это
    хуже шаблона: методист не узнает, что генерация не состоялась.
    """

    def _finished(self, factory, timeout=3.0):
        import time
        jobs = gen.Jobs(factory)
        job = jobs.start("текст методиста, достаточно длинный для проверки", "support")
        deadline = time.time() + timeout
        while job.state == "running" and time.time() < deadline:
            time.sleep(0.02)
        return job

    def test_missing_key_yields_a_template_not_a_stuck_spinner(self):
        def factory():
            raise RuntimeError("нет ANTHROPIC_API_KEY в окружении или .env")

        job = self._finished(factory).to_dict()
        self.assertEqual(job["state"], "done")
        self.assertTrue(job["fallback"])
        self.assertEqual(Scenario.from_dict(job["scenario"]).validate(), [])

    def test_system_exit_does_not_leak_out_of_the_thread(self):
        """SystemExit — не Exception, и мимо обычного `except` проходит насквозь."""
        def factory():
            raise SystemExit("нет ключа")

        job = self._finished(factory).to_dict()
        self.assertEqual(job["state"], "done")
        self.assertTrue(job["fallback"])

    def test_generator_reports_missing_key_as_a_catchable_error(self):
        import os
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            os.environ["ANTHROPIC_API_KEY"] = ""
            with self.assertRaises(Exception) as ctx:
                gen.AnthropicGenerator()
            self.assertNotIsInstance(ctx.exception, SystemExit)
        finally:
            if saved is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved

    def test_progress_is_visible_while_running(self):
        job = self._finished(lambda: (lambda p: good_raw()))
        self.assertEqual(job.state, "done")
        self.assertFalse(job.to_dict()["fallback"])


class Templates(unittest.TestCase):
    def test_every_type_yields_a_valid_scenario(self):
        for kind in TYPES:
            sc = gen.to_scenario(template(kind), "текст", source="template")
            self.assertEqual(sc.validate(), [], kind)
            self.assertGreaterEqual(len(sc.stages), 5, kind)
            self.assertGreaterEqual(len(sc.criteria), 3, kind)

    def test_unknown_type_still_works(self):
        sc = gen.to_scenario(template("такого типа нет"), "текст")
        self.assertEqual(sc.validate(), [])


class Refine(unittest.TestCase):
    """Чат-слой: правка не должна стирать то, что методист сделал руками."""

    def setUp(self):
        self.current = gen.to_scenario(good_raw(), "исходник").to_dict()

    def _llm(self, changes):
        def llm(prompt):
            raw = good_raw()
            raw.update(changes)
            return raw
        return llm

    def test_protected_field_is_restored_after_the_model_overwrites_it(self):
        changed = {"persona": dict(good_raw()["persona"], role="что-то своё")}
        out = gen.refine(self._llm(changed), self.current, "жёстче",
                         protect=["persona.role"])
        self.assertEqual(out["scenario"]["persona"]["role"],
                         self.current["persona"]["role"])
        self.assertIn("persona.role", out["kept"])

    def test_unprotected_field_is_allowed_to_change(self):
        out = gen.refine(self._llm({"title": "новое название"}), self.current,
                         "переименуй", protect=["persona.role"])
        self.assertEqual(out["scenario"]["title"], "новое название")

    def test_shape_change_is_reported_not_silently_dropped(self):
        """Позиция ничего не значит, если этапы переставили. Честно сказать."""
        out = gen.refine(self._llm({"stages": good_raw(stages=5)["stages"]}),
                         self.current, "убери этап", protect=["stage.0.hint"])
        self.assertIn("stage.0.hint", out["lost"])
        self.assertEqual(out["kept"], [])

    def test_added_criterion_does_not_unprotect_stages(self):
        """«Добавь критерий» не должно снимать защиту с этапов.

        Списки проверяются раздельно: общий флаг сообщал методисту о потере
        правки, которая на месте, — а это хуже, чем ничего не сообщать.
        """
        changed = {"criteria": good_raw(criteria=6)["criteria"],
                   "stages": [dict(s, hint="переписано моделью")
                              for s in good_raw()["stages"]]}
        out = gen.refine(self._llm(changed), self.current,
                         "добавь критерий про английский",
                         protect=["stage.0.hint", "crit.0.title"])
        self.assertIn("stage.0.hint", out["kept"])
        self.assertIn("crit.0.title", out["lost"])
        self.assertEqual(out["scenario"]["stages"][0]["hint"],
                         self.current["stages"][0]["hint"])

    def test_current_scenario_and_request_reach_the_prompt(self):
        prompts = []

        def llm(prompt):
            prompts.append(prompt)
            return good_raw()

        gen.refine(llm, self.current, "добавь критерий про английский",
                   protect=["title"])
        self.assertIn("добавь критерий про английский", prompts[0])
        self.assertIn(self.current["stages"][0]["goal"], prompts[0])
        self.assertIn("title", prompts[0])
        self.assertIn("исходник", prompts[0])


class Identifiers(unittest.TestCase):
    """Ключи дописываются и для критериев, добавленных методистом руками."""

    def _raw(self, criteria):
        return {"id": "x", "title": "T", "type": "sales",
                "persona": {"role": "клиент, который сомневается",
                            "tone": "", "strictness": "", "pressure": "",
                            "start_emotion": "neutral"},
                "stages": [{"id": "a", "goal": "первый"},
                           {"id": "", "goal": "добавленный руками"}],
                "criteria": criteria}

    def test_empty_key_is_filled_from_the_title(self):
        raw = self._raw([{"key": "", "title": "Работа с возражениями",
                          "anchor_1": "нет", "anchor_5": "да"}])
        sc = Scenario.from_dict(gen.fill_identifiers(raw))
        self.assertEqual(sc.criteria[0].key, "rabota_s_vozrazheniyami")
        self.assertEqual(sc.validate(), [])

    def test_existing_keys_are_left_alone(self):
        """Сценарий из репозитория не меняет ключи от того, что его открыли."""
        raw = self._raw([{"key": "structure", "title": "Структурность",
                          "anchor_1": "нет", "anchor_5": "да"}])
        sc = Scenario.from_dict(gen.fill_identifiers(raw))
        self.assertEqual(sc.criteria[0].key, "structure")

    def test_duplicate_keys_are_split(self):
        raw = self._raw([{"key": "", "title": "Цена", "anchor_1": "a", "anchor_5": "b"},
                         {"key": "", "title": "Цена", "anchor_1": "a", "anchor_5": "b"}])
        sc = Scenario.from_dict(gen.fill_identifiers(raw))
        self.assertEqual(len(set(c.key for c in sc.criteria)), 2)
        self.assertEqual(sc.validate(), [])

    def test_non_latin_key_is_replaced(self):
        raw = self._raw([{"key": "цена", "title": "Цена",
                          "anchor_1": "a", "anchor_5": "b"}])
        sc = Scenario.from_dict(gen.fill_identifiers(raw))
        self.assertTrue(sc.criteria[0].key.isascii())

    def test_stage_without_id_gets_one(self):
        raw = self._raw([{"key": "k", "title": "T", "anchor_1": "a", "anchor_5": "b"}])
        sc = Scenario.from_dict(gen.fill_identifiers(raw))
        self.assertTrue(sc.stages[1].id)
        self.assertEqual(len(set(s.id for s in sc.stages)), 2)


class SchemaShape(unittest.TestCase):
    def test_schema_has_no_numeric_bounds(self):
        """Структурированный вывод отвечает 400 на minItems и minimum.

        Проверка нужна не ради схемы, а ради памяти: границы уже один раз
        положили генерацию целиком, и без этого теста их вернут обратно как
        «очевидное улучшение».
        """
        blob = json.dumps(gen.SCHEMA)
        for banned in ("minItems", "maxItems", "minimum", "maximum"):
            self.assertNotIn(banned, blob, f"схема снова содержит {banned}")

    def test_emotion_enum_matches_the_avatar_contract(self):
        enum = gen.SCHEMA["properties"]["persona"]["properties"]["start_emotion"]["enum"]
        self.assertEqual(list(enum), list(EMOTIONS))


if __name__ == "__main__":
    unittest.main()
