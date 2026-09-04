"""Тесты сценарного движка.

Главная проверка — та, что названа в задании: все пять сценариев на заглушке
модели доходят до `finish` и порождают отчёт. Остальное закрывает места, где
движок ломается тихо: разбор действия из мусора, чужой критерий, отменённая
генерация в истории.
"""
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app import report as report_mod                      # noqa: E402
from app.actions import (EVALUATE, FINISH, NEXT_STAGE, STAY,                # noqa: E402
                         parse_action, parse_reply, strip_control)
from app.agent import Agent, build_prompt                 # noqa: E402
from app.dialogue import DialogueState                    # noqa: E402
from app.scenario import Scenario, load_all               # noqa: E402
from app.stub_llm import ScriptedLLM, StubLLM             # noqa: E402

SCENARIOS = load_all(ROOT / "data" / "scenarios")


class ScenarioFormat(unittest.TestCase):
    def test_five_scenarios_load(self):
        self.assertEqual(len(SCENARIOS), 5, "ожидалось пять сценариев")

    def test_all_valid(self):
        for s in SCENARIOS:
            self.assertEqual(s.validate(), [], f"{s.id}: сценарий не прошёл проверку")

    def test_stages_and_criteria_present(self):
        for s in SCENARIOS:
            self.assertGreaterEqual(len(s.stages), 5, f"{s.id}: мало этапов")
            self.assertGreaterEqual(len(s.criteria), 3, f"{s.id}: мало критериев")
            for st in s.stages:
                self.assertTrue(st.goal, f"{s.id}/{st.id}: нет цели")
                self.assertTrue(st.hint, f"{s.id}/{st.id}: нет подсказки модели")
                self.assertTrue(st.advance_when, f"{s.id}/{st.id}: нет условия перехода")

    def test_roundtrip(self):
        for s in SCENARIOS:
            again = Scenario.from_dict(s.to_dict())
            self.assertEqual(again.to_dict(), s.to_dict(), f"{s.id}: формат не круговой")


class ActionParsing(unittest.TestCase):
    def test_plain(self):
        a = parse_action('Реплика.\n{"action": "next_stage"}')
        self.assertEqual(a.action, NEXT_STAGE)
        self.assertFalse(a.fell_back)

    def test_fenced(self):
        a = parse_action('Реплика.\n```json\n{"action": "finish"}\n```')
        self.assertEqual(a.action, FINISH)

    def test_evaluate_still_parses_for_old_records(self):
        """Действие убрано из промпта, но разбор старых записей падать не должен."""
        a = parse_action('Так.\n{"action":"evaluate","criterion":"structure",'
                         '"score":"4","note":"ок"}')
        self.assertEqual(a.action, EVALUATE)
        self.assertFalse(a.fell_back)

    def test_last_object_wins(self):
        # Модель порой рассуждает вслух и оставляет по дороге лишние объекты.
        a = parse_action('{"action": "finish"} ...передумал...\n{"action": "next_stage"}')
        self.assertEqual(a.action, NEXT_STAGE)

    def test_garbage_falls_back_to_stay(self):
        for blob in ["", "просто текст без всякого json",
                     'Реплика. {"action": "next_st',
                     '{"action": "полететь_на_луну"}',
                     "{{{}}}", 'null', '{"нет": "действия"}']:
            with self.subTest(blob=blob[:30]):
                a = parse_action(blob)
                self.assertEqual(a.action, STAY)
                self.assertTrue(a.fell_back, "неразобранный ответ — это поломка")

    def test_explicit_stay_is_not_a_fallback(self):
        """Осознанный пустой ход и поломка должны различаться.

        Пока `stay` был только откатом, модель, которой нечего объявить, просто
        не выводила JSON: на живом прогоне 22% ходов, а на длинном до 83%,
        считались нарушением протокола. В таком шуме настоящая поломка не видна.
        """
        a = parse_action('Переспрошу.\n{"action": "stay"}')
        self.assertEqual(a.action, STAY)
        self.assertFalse(a.fell_back)

    def test_explicit_stay_does_not_move_or_finish(self):
        sc = SCENARIOS[0]
        state = DialogueState(sc)
        agent = Agent(ScriptedLLM(['Переспрошу.\n{"action": "stay"}']))
        _, happened = agent.step(state, "Ответ.")
        self.assertEqual(state.stage_index, 0)
        self.assertFalse(state.finished)
        self.assertFalse(happened["advanced"])
        self.assertFalse(happened["fell_back"])

    def test_emotion_tags_stripped_from_speech(self):
        """Теги не должны оставаться ни в истории, ни в отчёте, ни в контексте.

        Вырезание только перед синтезом их из озвучки убирало, но на экране,
        в расшифровке и в промпте следующего хода они оставались.
        """
        r = parse_reply('[emo:skeptical] А цифры будут?\n{"action": "stay"}')
        self.assertEqual(r.speakable, "А цифры будут?")
        self.assertNotIn("emo", r.speakable)

    def test_broken_emotion_tag_stripped_too(self):
        r = parse_reply('Понял вас [emo:war\n{"action": "stay"}')
        self.assertNotIn("[", r.speakable)

    def test_control_stripped_from_speech(self):
        r = parse_reply('Расскажите подробнее.\n{"action": "next_stage"}')
        self.assertEqual(r.speakable, "Расскажите подробнее.")
        self.assertNotIn("action", r.speakable)

    def test_braces_inside_speech_survive(self):
        # Фигурные скобки бывают и в самой реплике — вырезать их нельзя.
        text = 'Он написал {"а": 1} в конфиге, и всё сломалось.'
        self.assertEqual(strip_control(text), text)

    def test_control_stripped_but_speech_with_braces_kept(self):
        blob = 'Он написал {"а": 1} в конфиге.\n{"action": "finish"}'
        r = parse_reply(blob)
        self.assertIn('{"а": 1}', r.speakable)
        self.assertEqual(r.action.action, FINISH)


class Progression(unittest.TestCase):
    def test_all_five_reach_finish_and_produce_report(self):
        """Главный тест задания."""
        for sc in SCENARIOS:
            with self.subTest(scenario=sc.id):
                state = DialogueState(sc)
                agent = Agent(StubLLM(turns_per_stage=2))
                agent.step(state)                      # открывающая реплика
                for i in range(200):
                    if state.finished:
                        break
                    agent.step(state, f"Ответ пользователя {i}.")
                self.assertTrue(state.finished, f"{sc.id}: не дошёл до finish")

                rep = report_mod.build(state)
                self.assertTrue(rep.completed)
                self.assertEqual(rep.stages_total, len(sc.stages))
                self.assertEqual(rep.stages_reached, len(sc.stages),
                                 f"{sc.id}: дошёл не до последнего этапа")
                self.assertGreater(len(rep.transcript), 0)
                self.assertEqual(len(rep.criteria), len(sc.criteria))
                json.loads(rep.to_json())              # отчёт сериализуем

    def test_stays_on_stage_when_action_is_garbage(self):
        sc = SCENARIOS[0]
        state = DialogueState(sc)
        agent = Agent(ScriptedLLM(['Продолжаем. {"action": "не_дейст']))
        before = state.stage_index
        _, happened = agent.step(state, "Ответ.")
        self.assertEqual(state.stage_index, before, "мусор не должен двигать этап")
        self.assertFalse(state.finished)
        self.assertTrue(happened["fell_back"])

    def test_next_stage_on_last_stage_finishes(self):
        sc = SCENARIOS[0]
        state = DialogueState(sc)
        state.stage_index = len(sc.stages) - 1
        agent = Agent(ScriptedLLM(['Ну что ж.\n{"action": "next_stage"}']))
        agent.step(state, "Ответ.")
        self.assertTrue(state.finished, "просьба идти дальше с последнего этапа = конец")

    def test_unknown_criterion_ignored(self):
        sc = SCENARIOS[0]
        state = DialogueState(sc)
        agent = Agent(ScriptedLLM(
            ['Ага.\n{"action":"evaluate","criterion":"выдуманный","score":5,"note":"н"}']))
        agent.step(state, "Ответ.")
        self.assertEqual(state.observations, [],
                         "критерий, которого методист не задавал, записывать нельзя")

    def test_known_criterion_recorded(self):
        sc = SCENARIOS[0]
        key = sc.criteria[0].key
        state = DialogueState(sc)
        agent = Agent(ScriptedLLM(
            [f'Ага.\n{{"action":"evaluate","criterion":"{key}","score":4,"note":"есть числа"}}']))
        agent.step(state, "Ответ.")
        self.assertEqual(len(state.observations), 1)
        self.assertEqual(state.observations[0].criterion, key)


class StageBudget(unittest.TestCase):
    """Бюджет реплик на этап.

    Появился по измерению, а не из принципа: как только в протоколе завёлся
    явный «остаться на этапе», DeepSeek выбрала его в 60 ходах из 66 и не
    перешла ни разу — модель всегда находит, что ещё уточнить. Право двигать
    сценарий пришлось отдать движку.
    """

    def _stuck_agent(self):
        return Agent(ScriptedLLM(['Переспрошу.\n{"action": "stay"}']))

    def test_budget_forces_advance(self):
        sc = SCENARIOS[0]
        state = DialogueState(sc, max_turns_per_stage=3)
        agent = self._stuck_agent()
        for _ in range(3):
            agent.step(state, "Ответ.")
        self.assertEqual(state.stage_index, 1, "бюджет исчерпан — этап должен смениться")
        self.assertEqual(state.forced_advances, 1)

    def test_budget_counts_per_stage_not_globally(self):
        sc = SCENARIOS[0]
        state = DialogueState(sc, max_turns_per_stage=2)
        agent = self._stuck_agent()
        for _ in range(2):
            agent.step(state, "Ответ.")
        self.assertEqual(state.stage_index, 1)
        self.assertEqual(state.turns_on_stage, 0, "счётчик обнуляется на новом этапе")
        agent.step(state, "Ответ.")
        self.assertEqual(state.stage_index, 1, "одного ответа мало для нового перехода")

    def test_budget_on_last_stage_finishes(self):
        sc = SCENARIOS[0]
        state = DialogueState(sc, max_turns_per_stage=2)
        state.stage_index = len(sc.stages) - 1
        agent = self._stuck_agent()
        for _ in range(2):
            agent.step(state, "Ответ.")
        self.assertTrue(state.finished, "на последнем этапе бюджет завершает диалог")

    def test_stuck_dialogue_always_terminates(self):
        """Модель, которая не переходит никогда, всё равно доходит до конца."""
        for sc in SCENARIOS:
            with self.subTest(scenario=sc.id):
                state = DialogueState(sc, max_turns_per_stage=3)
                agent = self._stuck_agent()
                for _ in range(len(sc.stages) * 3 + 5):
                    if state.finished:
                        break
                    agent.step(state, "Ответ ни о чём.")
                self.assertTrue(state.finished, f"{sc.id}: диалог не завершился")
                self.assertEqual(state.forced_advances, len(sc.stages) - 1)

    def test_model_advance_does_not_count_as_forced(self):
        sc = SCENARIOS[0]
        state = DialogueState(sc, max_turns_per_stage=5)
        agent = Agent(ScriptedLLM(['Дальше.\n{"action": "next_stage"}']))
        _, happened = agent.step(state, "Ответ.")
        self.assertTrue(happened["advanced"])
        self.assertFalse(happened["forced"])
        self.assertEqual(state.forced_advances, 0)


class StateOwnership(unittest.TestCase):
    def test_prompt_carries_whole_context(self):
        """Модель ничего не помнит — весь контекст обязан быть в промпте."""
        sc = SCENARIOS[0]
        state = DialogueState(sc)
        state.add_agent("Первая реплика агента.")
        state.add_user("Первый ответ пользователя.")
        state.observe(sc.criteria[0].key, "уже замечено кое-что", 3)
        p = build_prompt(state, "Свежая реплика.")
        self.assertIn(sc.persona[:24], p)
        self.assertIn(sc.stages[0].goal, p)
        self.assertIn("Первая реплика агента.", p)
        self.assertIn("Первый ответ пользователя.", p)
        self.assertIn("уже замечено кое-что", p)
        self.assertIn("Свежая реплика.", p)
        # Критерии попадают в промпт названиями, а не ключами: ключи были нужны
        # действию evaluate, которого больше нет.
        for c in sc.criteria:
            self.assertIn(c.title, p)

    def test_cancelled_generation_leaves_no_trace(self):
        sc = SCENARIOS[0]
        state = DialogueState(sc)
        state.add_user("Вопрос.")
        state.add_agent("Начал отвечать и был перебит.", generation_id="gen-1")
        state.add_user("Перебил.")
        dropped = state.drop_generation("gen-1")
        self.assertEqual(dropped, 1)
        texts = [t.text for t in state.turns]
        self.assertNotIn("Начал отвечать и был перебит.", texts)
        # И в промпт следующего хода она тоже не попадёт.
        self.assertNotIn("Начал отвечать", build_prompt(state, "Дальше."))

    def test_history_survives_other_generations(self):
        sc = SCENARIOS[0]
        state = DialogueState(sc)
        state.add_agent("Реплика первой генерации.", generation_id="gen-1")
        state.add_agent("Реплика второй генерации.", generation_id="gen-2")
        state.drop_generation("gen-1")
        self.assertEqual([t.text for t in state.turns], ["Реплика второй генерации."])


class ReportShape(unittest.TestCase):
    def test_report_covers_every_criterion(self):
        sc = SCENARIOS[0]
        state = DialogueState(sc)
        state.observe(sc.criteria[0].key, "первое", 4)
        state.observe(sc.criteria[0].key, "второе", 2)
        rep = report_mod.build(state)
        self.assertEqual(len(rep.criteria), len(sc.criteria))
        first = next(c for c in rep.criteria if c.key == sc.criteria[0].key)
        self.assertEqual(first.score, 3.0, "оценки по критерию усредняются")
        self.assertEqual(len(first.observations), 2)
        unscored = [c for c in rep.criteria if not c.evaluated]
        self.assertEqual(len(unscored), len(sc.criteria) - 1)

    def test_coverage_and_overall(self):
        sc = SCENARIOS[0]
        state = DialogueState(sc)
        for c in sc.criteria:
            state.observe(c.key, "ок", 4)
        rep = report_mod.build(state)
        self.assertEqual(rep.coverage, 1.0)
        self.assertEqual(rep.overall, 4.0)

    def test_report_without_observations_is_still_valid(self):
        rep = report_mod.build(DialogueState(SCENARIOS[0]))
        self.assertIsNone(rep.overall)
        self.assertEqual(rep.coverage, 0.0)
        json.loads(rep.to_json())


if __name__ == "__main__":
    unittest.main(verbosity=2)


class Repair(unittest.TestCase):
    """Второй разбор для реплик, пришедших без управляющей строки."""

    def test_recovers_declared_transition(self):
        from app.actions import repair_action
        a = repair_action(lambda s, p: "next_stage", "Хорошо, теперь про сроки.")
        self.assertEqual(a.action, "next_stage")
        self.assertTrue(a.repaired)
        self.assertFalse(a.fell_back)

    def test_tolerates_chatty_model(self):
        from app.actions import repair_action
        a = repair_action(lambda s, p: "  STAY.\n", "А почему именно так?")
        self.assertEqual(a.action, "stay")

    def test_nonsense_answer_leaves_fallback_alone(self):
        """Не починили — значит не починили. Угадывать хуже, чем остаться."""
        from app.actions import repair_action
        self.assertIsNone(repair_action(lambda s, p: "не знаю", "текст"))

    def test_network_failure_is_not_fatal(self):
        from app.actions import repair_action

        def boom(s, p):
            raise RuntimeError("сеть недоступна")

        self.assertIsNone(repair_action(boom, "текст"))

    def test_empty_reply_costs_no_call(self):
        """Пустую реплику разбирать нечего — и платить за это незачем."""
        from app.actions import repair_action
        calls = []
        self.assertIsNone(repair_action(lambda s, p: calls.append(1), "   "))
        self.assertEqual(calls, [])

    def test_prompt_mentions_last_stage(self):
        from app.actions import build_repair_prompt
        p = build_repair_prompt("реплика", "знакомство", is_last_stage=True)
        self.assertIn("знакомство", p)
        self.assertIn("последняя", p)
