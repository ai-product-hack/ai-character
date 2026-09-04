"""Тесты разметки эмоций внутри ответа.

Главное требование жёсткое: ни один тег не должен просочиться в синтез. У этого
проекта уже был случай, когда управляющий JSON уезжал в озвучку, и агент чуть
не начал произносить фигурные скобки вслух.
"""
import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.emotion_tags import (DEFAULT, EMOTIONS, EmotionMark,      # noqa: E402
                              parse, to_timeline)

TAGLIKE = re.compile(r"\[\s*emo", re.I)


class Cleaning(unittest.TestCase):
    def test_opening_tag_removed(self):
        p = parse("[emo:skeptical] Понял вас. А цифры будут?")
        self.assertEqual(p.text, "Понял вас. А цифры будут?")
        self.assertEqual(p.opening, "skeptical")

    def test_inline_tag_removed(self):
        p = parse("Хорошо. [emo:warming] Это уже лучше.")
        self.assertEqual(p.text, "Хорошо. Это уже лучше.")
        self.assertEqual(len(p.marks), 1)

    def test_no_tag_survives_any_input(self):
        """Перебор кривых написаний: в тексте не должно остаться ничего от тегов."""
        for raw in [
            "[emo:skeptical] Текст.",
            "[ emo : warming ] Текст.",
            "[EMO:PRESSING] Текст.",
            "Текст [emo:impressed] ещё текст.",
            "Обрыв [emo:war",
            "Пустой [emo:] текст.",
            "[emo] Текст.",
            "[emo:выдуманная] Текст.",
            "[emo:skeptical][emo:warming] Двойной.",
        ]:
            with self.subTest(raw=raw):
                p = parse(raw)
                self.assertIsNone(TAGLIKE.search(p.text),
                                  f"в синтез ушло бы: {p.text!r}")
                self.assertNotIn("[", p.text)

    def test_unknown_emotion_dropped_text_kept(self):
        p = parse("[emo:выдуманная] Текст остаётся.")
        self.assertEqual(p.text, "Текст остаётся.")
        self.assertEqual(p.marks, [])
        self.assertEqual(p.dropped, 1)
        self.assertEqual(p.opening, DEFAULT)

    def test_broken_markup_falls_back_to_neutral(self):
        p = parse("Обрыв разметки [emo:war")
        self.assertIsNone(TAGLIKE.search(p.text))
        self.assertEqual(p.opening, DEFAULT)

    def test_plain_text_untouched(self):
        raw = "Обычная реплика без всякой разметки."
        p = parse(raw)
        self.assertEqual(p.text, raw)
        self.assertEqual(p.opening, DEFAULT)

    def test_empty_input(self):
        p = parse("")
        self.assertEqual(p.text, "")
        self.assertEqual(p.opening, DEFAULT)

    def test_russian_aliases_accepted(self):
        for alias, expected in (("тепло", "warming"), ("скепсис", "skeptical"),
                                ("напор", "pressing"), ("интерес", "impressed")):
            with self.subTest(alias=alias):
                self.assertEqual(parse(f"[emo:{alias}] Текст.").opening, expected)

    def test_all_known_emotions_parse(self):
        for e in EMOTIONS:
            self.assertEqual(parse(f"[emo:{e}] Текст.").opening, e)


class Positions(unittest.TestCase):
    def test_mark_index_is_in_cleaned_text(self):
        p = parse("Хорошо. [emo:warming] Это уже лучше.")
        self.assertEqual(p.text[p.marks[0].char_index:][:3], "Это",
                         "смещение должно указывать в ОЧИЩЕННЫЙ текст")

    def test_several_marks_keep_order(self):
        p = parse("[emo:skeptical] Раз. [emo:pressing] Два. [emo:warming] Три.")
        self.assertEqual([m.emotion for m in p.marks],
                         ["skeptical", "pressing", "warming"])
        idx = [m.char_index for m in p.marks]
        self.assertEqual(idx, sorted(idx))

    def test_opening_only_when_at_start(self):
        p = parse("Сначала текст. [emo:warming] Потом тег.")
        self.assertEqual(p.opening, DEFAULT,
                         "тег не в начале не задаёт эмоцию всей реплики")


class Timeline(unittest.TestCase):
    CHARS = [{"ch": "а", "ms": i * 40} for i in range(20)]

    def test_marks_become_pts(self):
        marks = [EmotionMark(0, "skeptical"), EmotionMark(10, "warming")]
        out = to_timeline(marks, self.CHARS)
        self.assertEqual([o["pts_ms"] for o in out], [0, 400])
        self.assertEqual([o["emotion"] for o in out], ["skeptical", "warming"])

    def test_clause_offset_applied(self):
        out = to_timeline([EmotionMark(5, "pressing")], self.CHARS,
                          clause_start_ms=1500)
        self.assertEqual(out[0]["pts_ms"], 1700)

    def test_text_offset_for_later_clauses(self):
        """Таймкоды приходят по клаузам, а смещения меток — по всей реплике."""
        out = to_timeline([EmotionMark(23, "warming")], self.CHARS,
                          clause_start_ms=1000, text_offset=20)
        self.assertEqual(out[0]["pts_ms"], 1000 + self.CHARS[3]["ms"])

    def test_mark_before_this_clause_is_skipped(self):
        out = to_timeline([EmotionMark(2, "warming")], self.CHARS, text_offset=20)
        self.assertEqual(out, [], "метка из прошлой клаузы сюда не относится")

    def test_mark_past_end_clamps_to_last(self):
        out = to_timeline([EmotionMark(999, "warming")], self.CHARS)
        self.assertEqual(out[0]["pts_ms"], self.CHARS[-1]["ms"])

    def test_no_marks_no_output(self):
        self.assertEqual(to_timeline([], self.CHARS), [])

    def test_no_chars_no_output(self):
        self.assertEqual(to_timeline([EmotionMark(0, "warming")], []), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
