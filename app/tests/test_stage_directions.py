"""Ремарки в звёздочках: `*кивает*`, `*пауза*`.

Тот же класс беды, что у тегов эмоций и управляющего JSON: разметка, которая
доезжает до синтеза и звучит буквально. Замерено на живой сессии — 11 реплик
из 17 содержали ремарку, а одна пересказывала вслух подсказку этапа.
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.actions import AgentReply, parse_reply                      # noqa: E402
from app.stage_directions import count, strip                     # noqa: E402


class Strip(unittest.TestCase):
    def test_ремарка_внутри_реплики(self):
        self.assertEqual(strip("Понятно. *кивает* Давайте дальше."),
                         "Понятно. Давайте дальше.")

    def test_ремарка_отдельной_строкой(self):
        self.assertEqual(strip("Жду.\n\n*молчание, внимательный взгляд*"), "Жду.")

    def test_половинка_пары(self):
        # Клауза может разрезать пару пополам: озвучиваться не должна и она.
        self.assertNotIn("*", strip("Хорошо. *задумчиво"))

    def test_осмысленные_скобки_остаются(self):
        t = "Вилка сто (сто двадцать) тысяч."
        self.assertEqual(strip(t), t)

    def test_скобки_во_всю_строку_это_ремарка(self):
        self.assertEqual(strip("(задумчиво)\nА сроки?"), "А сроки?")

    def test_обычный_текст_не_трогаем(self):
        t = "Расскажите про последний проект."
        self.assertEqual(strip(t), t)

    def test_счётчик(self):
        self.assertEqual(count("*раз* и *два*"), 2)
        self.assertEqual(count("ничего такого"), 0)


class Speakable(unittest.TestCase):
    """«Произносимое» обязано означать произносимое во всех местах сразу."""

    def test_ремарка_не_доезжает_до_синтеза(self):
        # Управляющий JSON к этому месту уже снят разбором, ремарка — нет.
        r = AgentReply(action="stay", text="Понятно. *пауза* Дальше.")
        self.assertEqual(r.speakable, "Понятно. Дальше.")

    def test_через_разбор_целого_ответа(self):
        r = parse_reply('Понятно. *кивает* Дальше.\n{"action": "stay"}')
        self.assertEqual(r.speakable, "Понятно. Дальше.")

    def test_вместе_с_тегами(self):
        r = AgentReply(action="stay", text='[emo:warm]Хорошо. *улыбается* Идём дальше.')
        self.assertNotIn("*", r.speakable)
        self.assertNotIn("emo", r.speakable)


if __name__ == "__main__":
    unittest.main()
