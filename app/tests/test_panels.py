"""Панели, которые вызывает агент.

Главная проверка — та же, что стоила нам управляющего JSON: ни один тег не
должен просочиться в синтез. Модель однажды уже приклеивала разметку к
последней клаузе, и агент произносил её вслух.
"""
import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.actions import AgentAction, AgentReply, STAY         # noqa: E402
from app.panels import PANELS, parse, skills_payload, to_timeline   # noqa: E402

LEAK = re.compile(r"\[|\bpanel\b|\bemo\b", re.I)


class NothingLeaks(unittest.TestCase):
    CASES = [
        "[panel:skills] Давайте посмотрим, как вы справляетесь.",
        "Хорошо. [panel:code:python] Что делает этот код?",
        "Итог. [panel:навыки] Вот ваши оценки.",
        "Мусор [panel:выдумка] и обрывок [panel: без закрытия",
        "[panel] без типа вообще",
        "[panel:close]",
        "Два подряд [panel:skills][panel:close] и текст.",
    ]

    def test_no_tag_reaches_synthesis(self):
        for c in self.CASES:
            with self.subTest(text=c):
                self.assertIsNone(LEAK.search(parse(c).text),
                                  f"тег просочился: {parse(c).text!r}")

    def test_no_tag_reaches_history(self):
        """speakable идёт в историю, отчёт и промпт — там тегов тоже быть не должно."""
        for c in self.CASES:
            with self.subTest(text=c):
                r = AgentReply(text=c, action=AgentAction(STAY))
                self.assertIsNone(LEAK.search(r.speakable))

    def test_unknown_panel_is_dropped_not_guessed(self):
        p = parse("Тут [panel:выдумка] мусор.")
        self.assertEqual(p.marks, [])
        self.assertEqual(p.dropped, 1)

    def test_broken_markup_is_dropped(self):
        p = parse("Обрывок [panel: без закрытия")
        self.assertEqual(p.marks, [])
        self.assertGreater(p.dropped, 0)

    def test_plain_reply_is_untouched(self):
        t = "Обычная реплика без единой панели."
        self.assertEqual(parse(t).text, t)
        self.assertEqual(parse(t).marks, [])


class Marks(unittest.TestCase):
    def test_position_is_in_cleaned_text(self):
        """Смещения превращаются в pts_ms по таймкодам ПРОИЗНЕСЁННОГО."""
        p = parse("Хорошо. [panel:skills] Смотрите.")
        self.assertEqual(p.text, "Хорошо. Смотрите.")
        self.assertEqual(p.marks[0].char_index, 8)

    def test_argument_is_parsed(self):
        p = parse("[panel:code:python] Разберём.")
        self.assertEqual((p.marks[0].panel, p.marks[0].arg), ("code", "python"))

    def test_russian_aliases(self):
        for word, expect in (("навыки", "skills"), ("сценарий", "scenario"),
                             ("закрыть", "close"), ("код", "code")):
            with self.subTest(word=word):
                self.assertEqual(parse(f"[panel:{word}] текст").marks[0].panel, expect)

    def test_whitelist_is_closed(self):
        self.assertEqual(set(PANELS), {"skills", "code", "scenario", "close"})

    def test_timeline_uses_the_same_timecodes(self):
        chars = [{"ch": "а", "ms": i * 40} for i in range(20)]
        p = parse("Хорошо. [panel:skills] Смотрите.")
        cues = to_timeline(p.marks, chars, clause_start_ms=500)
        self.assertEqual(cues[0]["pts_ms"], 500 + 8 * 40)
        self.assertEqual(cues[0]["panel"], "skills")

    def test_timeline_of_nothing_is_nothing(self):
        self.assertEqual(to_timeline([], [{"ch": "а", "ms": 0}]), [])


class SkillsData(unittest.TestCase):
    """Панель рисуется из накопленного отчёта — второго запроса быть не должно."""

    def test_payload_comes_from_report(self):
        report = {
            "criteria": [
                {"key": "structure", "title": "Структура", "score": 4, "scale": 5,
                 "rationale": "отвечает по делу"},
                {"key": "depth", "title": "Глубина", "score": None, "scale": 5,
                 "rationale": ""},
            ],
            "overall": 4.0, "coverage": 0.5,
        }
        d = skills_payload(report)
        self.assertEqual(d["total"], 2)
        self.assertEqual(d["evaluated"], 1)
        self.assertEqual(d["overall"], 4.0)
        self.assertEqual(d["rows"][0]["title"], "Структура")

    def test_empty_report_does_not_crash(self):
        d = skills_payload({})
        self.assertEqual(d["rows"], [])
        self.assertEqual(d["evaluated"], 0)


if __name__ == "__main__":
    unittest.main()
