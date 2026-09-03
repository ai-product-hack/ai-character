"""Тесты нарезки на клаузы.

Первая клауза — это время до первого звука, поэтому её длина проверяется
строго. Остальные клаузы важны тем, что не должны рваться посреди слова:
рваный кусок слышен в синтезе.
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.clauses import ClauseSplitter, split_text     # noqa: E402

REPLIES = [
    "Спасибо, картина в целом понятна. Вы всё время говорите «мы»: команда "
    "собрала, команда выкатила. Расскажите, что именно делали лично вы?",
    "Так, у меня ровно минута до совещания. Вы кто, откуда и что предлагаете?",
    "Понял вас.",
    "Хорошо, с архитектурой разобрались. Теперь цифры: какая была нагрузка в "
    "запросах в секунду и какую задержку вы держали на девяносто пятом перцентиле?",
]


class FirstClause(unittest.TestCase):
    def test_first_clause_is_short(self):
        for t in REPLIES:
            with self.subTest(text=t[:40]):
                first = split_text(t)[0]
                self.assertLessEqual(len(first.text), 50,
                                     f"первая клауза {len(first.text)} символов")
                self.assertTrue(first.first)

    def test_first_clause_short_even_without_punctuation(self):
        t = ("Очень длинное предложение без единого знака препинания которое "
             "всё тянется и тянется и никак не заканчивается словами")
        first = split_text(t)[0]
        self.assertLessEqual(len(first.text), 50)

    def test_first_clause_cut_at_sentence_when_it_is_early(self):
        c = split_text("Понял. Теперь давайте к цифрам, они важнее всего остального.")
        self.assertEqual(c[0].text, "Понял.")

    def test_only_first_clause_is_marked_first(self):
        cs = split_text(REPLIES[0])
        self.assertEqual([c.first for c in cs], [True] + [False] * (len(cs) - 1))


class Integrity(unittest.TestCase):
    def test_nothing_lost(self):
        for t in REPLIES:
            with self.subTest(text=t[:40]):
                joined = " ".join(c.text for c in split_text(t))
                self.assertEqual(joined.split(), t.split(),
                                 "нарезка не должна терять или менять слова")

    def test_no_cut_inside_a_word(self):
        for t in REPLIES:
            for c in split_text(t):
                self.assertFalse(c.text.startswith(" "))
                # Клауза не может начинаться с обрубка: первое слово клаузы
                # обязано целиком встречаться в исходном тексте.
                self.assertIn(c.text.split()[0], t)

    def test_indices_are_sequential(self):
        cs = split_text(REPLIES[0])
        self.assertEqual([c.index for c in cs], list(range(len(cs))))

    def test_short_hesitation_not_a_clause_of_its_own(self):
        # «Ну…» — заминка, а не предложение: отдельным куском синтеза
        # она звучит обрубком.
        cs = split_text("Ну… не знаю. Давайте попробуем.")
        self.assertEqual(cs[0].text, "Ну… не знаю.")

    def test_short_but_real_sentence_stands_alone(self):
        cs = split_text("Понял вас. Теперь к цифрам.")
        self.assertEqual(cs[0].text, "Понял вас.")

    def test_multiple_marks_kept_together(self):
        cs = split_text("Серьёзно?! Тогда объясните.")
        self.assertEqual(cs[0].text, "Серьёзно?!")


class Streaming(unittest.TestCase):
    def test_token_by_token_matches_whole_text(self):
        """Поток токенов должен дать ту же нарезку, что и готовый текст."""
        for t in REPLIES:
            with self.subTest(text=t[:40]):
                sp = ClauseSplitter()
                got = []
                for ch in t:
                    got += sp.push(ch)
                got += sp.flush()
                self.assertEqual([c.text for c in got], [c.text for c in split_text(t)])

    def test_chunked_stream_matches(self):
        """Токены приходят кусками произвольной длины — граница может попасть
        внутрь куска."""
        t = REPLIES[0]
        for size in (3, 7, 13, 40):
            with self.subTest(size=size):
                sp = ClauseSplitter()
                got = []
                for i in range(0, len(t), size):
                    got += sp.push(t[i:i + size])
                got += sp.flush()
                self.assertEqual([c.text for c in got], [c.text for c in split_text(t)])

    def test_first_clause_emitted_before_stream_ends(self):
        """Смысл всей нарезки: первая клауза уходит в синтез до конца потока."""
        t = REPLIES[0]
        sp = ClauseSplitter()
        emitted_at = None
        for i, ch in enumerate(t):
            if sp.push(ch):
                emitted_at = i
                break
        self.assertIsNotNone(emitted_at, "первая клауза не выделилась вовсе")
        self.assertLess(emitted_at, len(t) // 2,
                        "первая клауза должна выделиться задолго до конца потока")

    def test_flush_returns_tail(self):
        sp = ClauseSplitter()
        sp.push("Хвост без знака в конце")
        out = sp.flush()
        self.assertEqual([c.text for c in out], ["Хвост без знака в конце"])

    def test_flush_on_empty_is_empty(self):
        self.assertEqual(ClauseSplitter().flush(), [])

    def test_whitespace_only_produces_nothing(self):
        sp = ClauseSplitter()
        self.assertEqual(sp.push("   \n  "), [])
        self.assertEqual(sp.flush(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
