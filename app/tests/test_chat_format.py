"""Диалог как массив ролей, а не склеенный текст.

Раньше весь контекст рендерился в ОДНО `user`-сообщение, а история обрезалась
до 12 ходов — на записанных прогонах обрезка срабатывала в 10 случаях из 10.
Здесь проверяется, что чат собирается корректно и что провайдер получает
ровно то, что задумано.
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.agent import (build_messages, scenario_block,        # noqa: E402
                       stage_block, CONTROL_REMINDER)
from app.dialogue import DialogueState                        # noqa: E402
from app.scenario import Criterion, Persona, Scenario, Stage  # noqa: E402


def scenario():
    return Scenario(
        id="t", title="Тест", type="interview",
        persona=Persona(role="интервьюер", tone="сухой"),
        stages=[Stage("a", "первый", hint="Спроси про опыт.",
                      advance_when="ответил по существу"),
                Stage("b", "второй", hint="Углубись.")],
        criteria=[Criterion("k", "Конкретность", "1-5", "общие слова", "числа")])


def talk(exchanges: int) -> DialogueState:
    st = DialogueState(scenario())
    st.add_agent("Открывающая реплика агента.")
    for i in range(exchanges):
        st.add_user(f"Ответ пользователя {i + 1}.")
        if i < exchanges - 1:
            st.add_agent(f"Реплика агента {i + 2}.")
    return st


class Shape(unittest.TestCase):
    """Форма массива: провайдеры к ней придирчивы."""

    def test_starts_with_user(self):
        for n in (0, 1, 3):
            st = talk(n) if n else DialogueState(scenario())
            ms = build_messages(st, f"Ответ {n}." if n else None)
            self.assertEqual(ms[0]["role"], "user", f"обменов {n}")

    def test_roles_alternate(self):
        ms = build_messages(talk(4), "Ответ пользователя 4.")
        for i in range(len(ms) - 1):
            self.assertNotEqual(ms[i]["role"], ms[i + 1]["role"],
                                f"две {ms[i]['role']} подряд на позиции {i}")

    def test_ends_with_user(self):
        ms = build_messages(talk(3), "Ответ пользователя 3.")
        self.assertEqual(ms[-1]["role"], "user")

    def test_consecutive_same_role_is_merged(self):
        """Агента перебили до первого звука — две реплики человека подряд."""
        st = talk(2)
        st.add_user("И ещё одно, сразу следом.")
        ms = build_messages(st, "И ещё одно, сразу следом.")
        for i in range(len(ms) - 1):
            self.assertNotEqual(ms[i]["role"], ms[i + 1]["role"])

    def test_opening_turn_is_a_single_message(self):
        ms = build_messages(DialogueState(scenario()), None)
        self.assertEqual(len(ms), 1)
        self.assertIn("Разговор начинается", ms[0]["content"])


class NothingIsLost(unittest.TestCase):
    """Главное, ради чего всё затевалось: история больше не обрезается."""

    def test_every_turn_reaches_the_model(self):
        st = talk(20)
        ms = build_messages(st, "Ответ пользователя 20.")
        blob = "\n".join(m["content"] for m in ms)
        for i in range(1, 20):
            self.assertIn(f"Ответ пользователя {i}.", blob, f"потерян ход {i}")
        for i in range(2, 21):
            self.assertIn(f"Реплика агента {i}.", blob, f"потеряна реплика {i}")

    def test_long_talk_keeps_the_very_first_turn(self):
        """В старом формате окно 12 ходов съедало начало разговора."""
        st = talk(30)
        ms = build_messages(st, "Ответ пользователя 30.")
        self.assertIn("Открывающая реплика агента.", ms[0]["content"] +
                      " ".join(m["content"] for m in ms))


class StablePrefix(unittest.TestCase):
    """Всё, кроме последней реплики, обязано совпадать байт в байт.

    На этом держится кеширование: сдвинься хоть один символ раньше границы —
    и кеш промахивается каждый ход, то есть не работает вовсе.
    """

    def _prefixes(self):
        st = DialogueState(scenario())
        st.add_agent("Открывающая реплика агента.")
        out = []
        for i in range(1, 4):
            text = f"Ответ пользователя {i}."
            st.add_user(text)
            out.append(build_messages(st, text)[:-1])
            st.add_agent(f"Реплика агента {i + 1}.")
        return out

    def test_each_prefix_extends_the_previous(self):
        a, b, c = self._prefixes()
        self.assertEqual(b[:len(a)], a, "префикс хода 2 разошёлся с ходом 1")
        self.assertEqual(c[:len(b)], b, "префикс хода 3 разошёлся с ходом 2")

    def test_changing_state_lives_only_in_the_last_message(self):
        """Этап и бюджет ходов — в свежей реплике, иначе кеш не соберётся."""
        st = talk(3)
        ms = build_messages(st, "Ответ пользователя 3.")
        for m in ms[:-1]:
            self.assertNotIn("[ЭТАП", m["content"])
        self.assertIn("[ЭТАП", ms[-1]["content"])

    def test_scenario_block_appears_once(self):
        ms = build_messages(talk(5), "Ответ пользователя 5.")
        blob = "\n".join(m["content"] for m in ms)
        self.assertEqual(blob.count("НА ЧТО СМОТРИМ"), 1)


class ControlLine(unittest.TestCase):
    """Требование управляющей строки повторяется на каждом ходу.

    Замерено на DeepSeek: без напоминания строка терялась примерно в каждом
    шестом ходу, и это не обрыв по токенам, а просто пропуск.
    """

    def test_reminder_is_on_the_last_message(self):
        ms = build_messages(talk(3), "Ответ пользователя 3.")
        self.assertIn(CONTROL_REMINDER, ms[-1]["content"])

    def test_reminder_is_on_the_opening_turn_too(self):
        ms = build_messages(DialogueState(scenario()), None)
        self.assertIn(CONTROL_REMINDER, ms[0]["content"])

    def test_reminder_does_not_pollute_the_prefix(self):
        """Иначе он бы повторялся в истории и раздувал кешируемую часть."""
        ms = build_messages(talk(4), "Ответ пользователя 4.")
        for m in ms[:-1]:
            self.assertNotIn(CONTROL_REMINDER, m["content"])


class Blocks(unittest.TestCase):
    def test_scenario_block_has_role_and_criteria(self):
        b = scenario_block(talk(1))
        self.assertIn("интервьюер", b)
        self.assertIn("Конкретность", b)

    def test_stage_block_has_stage_and_hint(self):
        b = stage_block(talk(1))
        self.assertIn("ЭТАП 1 из 2", b)
        self.assertIn("Спроси про опыт.", b)

    def test_closing_turn_asks_for_a_farewell(self):
        st = talk(1)
        st.stage_index = 1
        for _ in range(st.stage_max_turns):
            st.add_user("Ответ.")
        self.assertIn("ПОСЛЕДНЯЯ РЕПЛИКА", stage_block(st))


class Caching(unittest.TestCase):
    """Отметки кеша: ставятся на границе стабильного и свежего."""

    def test_breakpoint_lands_before_the_fresh_message(self):
        from app.llm import with_cache_breakpoint
        ms = with_cache_breakpoint(build_messages(talk(3), "Ответ пользователя 3."))
        self.assertIsInstance(ms[-2]["content"], list)
        self.assertIn("cache_control", ms[-2]["content"][0])
        self.assertIsInstance(ms[-1]["content"], str, "свежее не кешируем")

    def test_single_message_gets_no_breakpoint(self):
        from app.llm import with_cache_breakpoint
        ms = with_cache_breakpoint([{"role": "user", "content": "один"}])
        self.assertIsInstance(ms[0]["content"], str)

    def test_system_is_marked(self):
        from app.llm import cacheable_system
        blocks = cacheable_system("инструкция")
        self.assertIn("cache_control", blocks[0])

    def test_empty_system_is_left_alone(self):
        from app.llm import cacheable_system
        self.assertEqual(cacheable_system(""), "")


if __name__ == "__main__":
    unittest.main()


class ControlLineInHistory(unittest.TestCase):
    """Реплики агента возвращаются в историю ВМЕСТЕ с управляющей строкой.

    Замерено и стоило отдельного прогона: когда история стала чистым текстом,
    доля реплик без управляющей строки подскочила с 0-5% до **63.7%** (107 из
    168). Модель видит свои прошлые ответы как есть и, не находя в них JSON,
    перестаёт его выводить. Формат надо показывать, а не только требовать.
    """

    def _state(self):
        st = DialogueState(scenario())
        st.add_agent("Первая реплика.", action="stay")
        st.add_user("Ответ один.")
        st.add_agent("Вторая реплика.", action="next_stage")
        st.add_user("Ответ два.")
        return st

    def test_past_replies_carry_their_action(self):
        ms = build_messages(self._state(), "Ответ два.")
        assistants = [m["content"] for m in ms if m["role"] == "assistant"]
        self.assertIn('{"action": "stay"}', assistants[0])
        self.assertIn('{"action": "next_stage"}', assistants[1])

    def test_interrupted_reply_has_no_control_line(self):
        """Её оборвали на полуслове — агент до JSON не дошёл, и это правда."""
        st = self._state()
        st.add_interrupted_agent("Недоговорённая реплика")
        st.add_user("Ответ три.")
        ms = build_messages(st, "Ответ три.")
        cut = [m["content"] for m in ms if m["role"] == "assistant"][-1]
        self.assertNotIn('"action"', cut)

    def test_action_survives_apply(self):
        """Действие проставляет `apply`, иначе история его не увидит."""
        from app.actions import parse_reply
        from app.agent import apply
        st = DialogueState(scenario())
        apply(st, parse_reply('Реплика.\n{"action": "stay"}'))
        self.assertEqual(st.turns[-1].action, "stay")
