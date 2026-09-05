"""Перебивание не должно приближать сценарий к принудительному концу.

Метрика кейса — «минимум 4 из 5 диалогов завершаются корректно», и проверять
её будут, перебивая агента. Значит перебивание обязано быть нормальной частью
разговора, а не потраченным ходом.
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.actions import AgentAction, AgentReply, STAY         # noqa: E402
from app.agent import apply                                   # noqa: E402
from app.dialogue import DialogueState                        # noqa: E402
from app.scenario import load_all                             # noqa: E402

SCENARIOS = load_all(ROOT / "data" / "scenarios")
BASE = next(s for s in SCENARIOS if s.id == "s1_interview_backend")


def stay(state, text="Ответ."):
    """Один ход, где модель остаётся на этапе."""
    state.add_user(text)
    return apply(state, AgentReply(text="Реплика.", action=AgentAction(STAY)))


class Budget(unittest.TestCase):
    def setUp(self):
        self.state = DialogueState(BASE, max_turns_per_stage=3)

    def test_interrupted_turn_does_not_spend_budget(self):
        t = self.state.add_user("Перебиваю.")
        t.counted = False
        self.state.add_user("Обычный ответ.")
        self.assertEqual(self.state.turns_on_stage, 1)

    def test_stage_survives_many_interruptions(self):
        """Три перебивания подряд не должны продвинуть этап."""
        start = self.state.stage_index
        for _ in range(3):
            self.state.add_user("Перебиваю.").counted = False
            apply(self.state, AgentReply(text="", action=AgentAction(STAY)))
        self.assertEqual(self.state.stage_index, start)
        self.assertFalse(self.state.finished)

    def test_full_exchanges_still_spend_budget(self):
        """Починка не должна отключить бюджет вообще: диалог обязан кончаться."""
        start = self.state.stage_index
        for _ in range(3):
            stay(self.state)
        self.assertGreater(self.state.stage_index, start)


class SpokenPartSurvives(unittest.TestCase):
    """Прозвучавшее нельзя терять: пользователь это слышал."""

    def setUp(self):
        self.state = DialogueState(BASE)

    def test_interrupted_reply_stays_in_history(self):
        self.state.add_user("Мой ответ.")
        self.state.add_agent("Целая реплика.", "gen-1")
        self.state.drop_generation("gen-1")
        self.state.add_interrupted_agent("Целая реп", "gen-1")
        texts = [t.text for t in self.state.turns if t.role == "agent"]
        self.assertEqual(texts, ["Целая реп"])
        self.assertTrue(self.state.turns[-1].interrupted)

    def test_interrupted_reply_reaches_the_prompt(self):
        """Иначе агент переспросит то, на что уже получил ответ."""
        from app.agent import build_prompt
        self.state.add_user("Я работал с кешем.")
        self.state.add_interrupted_agent("А что именно вы дела", "gen-1")
        self.assertIn("А что именно вы дела", build_prompt(self.state, "Проектировал."))


if __name__ == "__main__":
    unittest.main()


class DialogueCeiling(unittest.TestCase):
    """Диалог обязан заканчиваться, даже если перебивать каждую реплику."""

    def test_endless_interruptions_still_terminate(self):
        state = DialogueState(BASE)
        for _ in range(state.dialogue_max_turns + 5):
            if state.finished:
                break
            state.add_user("Перебиваю.").counted = False
            apply(state, AgentReply(text="", action=AgentAction(STAY)))
        self.assertTrue(state.finished, "перебивания не должны длиться вечно")
        self.assertIn("диалог", state.finish_reason)

    def test_ceiling_scales_with_scenario(self):
        """Этапов у сценариев разное число — потолок не может быть константой."""
        for sc in SCENARIOS:
            with self.subTest(scenario=sc.id):
                st = DialogueState(sc)
                need = (len(sc.stages) - 1) * st.max_turns_per_stage
                self.assertGreater(st.dialogue_max_turns, need,
                                   "потолок не должен резать сценарий до конца этапов")

    def test_ceiling_does_not_fire_early(self):
        """Обычный диалог должен успевать закончиться сам."""
        state = DialogueState(BASE)
        for _ in range(state.dialogue_max_turns - 1):
            if state.finished:
                break
            stay(state)
        self.assertNotIn("бюджет диалога", state.finish_reason)


class CapHoldsWhenTheLastTurnsAreInterrupted(unittest.TestCase):
    """Потолок диалога должен срабатывать и на перебивании.

    Проверка жила только в `apply`, то есть срабатывала, когда генерация
    доходила до конца. Разговор, у которого перебиты последние ходы, проезжал
    мимо: замерено на прогоне с 13 перебиваниями — 23 хода при потолке 23 и
    `finished=False`. Это ровно тот случай, ради которого потолок и заведён.
    """

    def _session(self):
        """Сессия без моделей: нужны только состояние, блокировка и кадры."""
        import threading
        import types
        from app.server import Session
        s = Session.__new__(Session)
        s.state = DialogueState(BASE)
        s.lock = threading.Lock()
        s.frames = []
        s.frame_cv = threading.Condition()
        s._emit = lambda header, payload=b"": s.frames.append(header)
        return s

    def test_budget_is_checked_on_cancel_too(self):
        s = self._session()
        while not s.state.dialogue_budget_spent:
            s.state.add_user("Ответ.")
        self.assertFalse(s.state.finished, "предусловие: движок ещё не закрыл диалог")

        self.assertTrue(s._finish_if_out_of_budget())
        self.assertTrue(s.state.finished)
        self.assertEqual(s.state.finish_reason, "бюджет диалога исчерпан")
        self.assertIn("finished", [f["kind"] for f in s.frames],
                      "клиент обязан узнать о конце, иначе отчёт не покажется")

    def test_check_is_idempotent(self):
        """Перебивания идут подряд — второй раз закрывать нечего."""
        s = self._session()
        while not s.state.dialogue_budget_spent:
            s.state.add_user("Ответ.")
        s._finish_if_out_of_budget()
        self.assertFalse(s._finish_if_out_of_budget())
        self.assertEqual([f["kind"] for f in s.frames].count("finished"), 1)

    def test_does_not_fire_early(self):
        s = self._session()
        for _ in range(s.state.dialogue_max_turns - 1):
            s.state.add_user("Ответ.")
        self.assertFalse(s._finish_if_out_of_budget())
        self.assertFalse(s.state.finished)
