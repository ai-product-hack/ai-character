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
from app.panels import (PANELS, material_payload, parse,       # noqa: E402
                        skills_payload, to_timeline)

LEAK = re.compile(r"\[|\bpanel\b|\bemo\b", re.I)


class NothingLeaks(unittest.TestCase):
    CASES = [
        "[panel:material] Взгляните на эти цифры.",
        "Хорошо. [panel:code:python] Что делает этот код?",
        "Итог. [panel:навыки] Вот ваши оценки.",
        "Мусор [panel:выдумка] и обрывок [panel: без закрытия",
        "[panel] без типа вообще",
        "[panel:close]",
        "Два подряд [panel:material][panel:close] и текст.",
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
        p = parse("Хорошо. [panel:material] Смотрите.")
        self.assertEqual(p.text, "Хорошо. Смотрите.")
        self.assertEqual(p.marks[0].char_index, 8)

    def test_argument_is_parsed(self):
        p = parse("[panel:code:python] Разберём.")
        self.assertEqual((p.marks[0].panel, p.marks[0].arg), ("code", "python"))

    def test_russian_aliases(self):
        for word, expect in (("график", "material"), ("сценарий", "scenario"),
                             ("закрыть", "close"), ("код", "code")):
            with self.subTest(word=word):
                self.assertEqual(parse(f"[panel:{word}] текст").marks[0].panel, expect)

    def test_whitelist_is_closed(self):
        self.assertEqual(set(PANELS), {"material", "code", "scenario", "close"})

    def test_timeline_uses_the_same_timecodes(self):
        chars = [{"ch": "а", "ms": i * 40} for i in range(20)]
        p = parse("Хорошо. [panel:material] Смотрите.")
        cues = to_timeline(p.marks, chars, clause_start_ms=500)
        self.assertEqual(cues[0]["pts_ms"], 500 + 8 * 40)
        self.assertEqual(cues[0]["panel"], "material")

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


class Material(unittest.TestCase):
    """Материал этапа: то, О ЧЁМ спрашивает агент.

    Заменил карту навыков после живого прогона. Показывать человеку его же
    оценку посреди разговора — подсказка на экзамене, а не материал; график, к
    которому просят прокомментировать провал в марте, — часть упражнения.
    """

    def _stage(self, material):
        from app.scenario import Stage
        return Stage(id="a", goal="разбор", material=material)

    def test_chart_travels_whole(self):
        rows = [{"label": "янв", "value": 12}, {"label": "мар", "value": 4}]
        d = material_payload(self._stage(
            {"kind": "chart", "title": "Конверсия", "series": rows}))
        self.assertEqual(d["kind"], "chart")
        self.assertEqual(d["series"], rows)

    def test_code_carries_its_language(self):
        d = material_payload(self._stage(
            {"kind": "code", "lang": "python", "body": "def f(): pass"}))
        self.assertEqual((d["lang"], d["body"]), ("python", "def f(): pass"))

    def test_stage_without_material_gives_nothing(self):
        self.assertEqual(material_payload(self._stage(None)), {})

    def test_no_stage_at_all_does_not_raise(self):
        """Этап может кончиться раньше, чем доедет метка панели."""
        self.assertEqual(material_payload(None), {})

    def test_nothing_is_generated(self):
        """Материал берётся из сценария: второго запроса к модели быть не может."""
        import inspect
        from app import panels
        src = inspect.getsource(panels.material_payload)
        for forbidden in ("llm", "fetch", "request", "generate"):
            self.assertNotIn(forbidden, src)


class NoSelfStatsPanel(unittest.TestCase):
    """Карта своих оценок посреди разговора — подсказка на экзамене.

    Замечание с живого прогона: «не вижу смысла показывать статистику, как
    человек отвечает». Панель показывалась не только по разметке модели — её
    ПРИНУДИТЕЛЬНО выдавал сервер на каждом переходе этапа, мимо модели вообще.
    """

    def test_skills_is_not_a_panel_the_agent_can_call(self):
        self.assertNotIn("skills", PANELS)

    def test_server_no_longer_forces_the_skills_panel(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[2] / "app" / "server.py"
               ).read_text(encoding="utf-8")
        self.assertNotIn('"panel": "skills"', src)
        self.assertIn('"panel": "material"', src)

    def test_stage_panel_needs_material_to_appear(self):
        """Пустая панель на переходе хуже отсутствующей."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[2] / "app" / "server.py"
               ).read_text(encoding="utf-8")
        self.assertIn('if happened.get("advanced") and material', src)
