"""Нормализация перед синтезом.

Проверяется не «красиво ли звучит», а инвариант: после нормализации в тексте
не остаётся ничего, что русская модель Silero молча выбросит. Именно молчаливое
выбрасывание и есть проблема — реплика теряет смысл, а субтитры остаются
осмысленными, и расхождение видно сразу.
"""
import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.pronounce import latin_word, normalize, number_words   # noqa: E402

# То, что модель выбрасывает: латиница, цифры, знаки процента и номера.
DROPPED = re.compile(r"[A-Za-z0-9%№°]")


class Invariant(unittest.TestCase):
    def test_nothing_droppable_survives(self):
        cases = [
            "Расскажите про ваш backend и REST API.",
            "Задержка сто миллисекунд на p95.",
            "Мы используем PostgreSQL и Redis.",
            "SLA у нас 99.9 процента.",
            "Пик около 2000 запросов в секунду.",
            "В 2024 году выручка выросла на 15%.",
            "Ошибок 5xx стало меньше на 30 %.",
            "Разверните это в k8s и настройте CI/CD.",
        ]
        for c in cases:
            out = normalize(c)
            self.assertIsNone(DROPPED.search(out),
                              f"в «{out}» осталось непроизносимое (из «{c}»)")

    def test_russian_text_untouched(self):
        """Реплика без терминов не должна меняться вообще."""
        t = "Хорошо, а что именно вы делали лично? Расскажите подробнее."
        self.assertEqual(normalize(t), t)

    def test_punctuation_survives(self):
        """Клаузы режутся по знакам препинания — их терять нельзя."""
        out = normalize("Итак, 15% роста. А дальше? Дальше — API!")
        for ch in ",.?!—":
            self.assertIn(ch, out)

    def test_empty_is_empty(self):
        self.assertEqual(normalize(""), "")


class Numbers(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(number_words(0), "ноль")
        self.assertEqual(number_words(15), "пятнадцать")
        self.assertEqual(number_words(95), "девяносто пять")
        self.assertEqual(number_words(100), "сто")
        self.assertEqual(number_words(2024), "две тысячи двадцать четыре")

    def test_thousand_is_feminine(self):
        """«две тысячи», не «два тысячи» — самая заметная на слух ошибка."""
        self.assertEqual(number_words(2000), "две тысячи")
        self.assertEqual(number_words(1000), "одна тысяча")

    def test_plural_agreement(self):
        self.assertEqual(number_words(1000), "одна тысяча")
        self.assertEqual(number_words(3000), "три тысячи")
        self.assertEqual(number_words(5000), "пять тысяч")
        self.assertEqual(number_words(11000), "одиннадцать тысяч")

    def test_millions(self):
        self.assertIn("миллиона", number_words(2_000_000))
        self.assertIn("миллионов", number_words(5_000_000))

    def test_negative(self):
        self.assertTrue(number_words(-5).startswith("минус "))

    def test_percent_agrees_with_number(self):
        self.assertIn("один процент", normalize("рост 1%"))
        self.assertIn("два процента", normalize("рост 2%"))
        self.assertIn("пять процентов", normalize("рост 5%"))
        self.assertIn("одиннадцать процентов", normalize("рост 11%"))

    def test_decimal_reads_naturally(self):
        self.assertIn("девяносто девять и девять", normalize("99.9"))
        self.assertIn("два и пять", normalize("2,5"))


class Latin(unittest.TestCase):
    def test_dictionary_wins(self):
        self.assertEqual(latin_word("backend"), "бэкенд")
        self.assertEqual(latin_word("PostgreSQL"), "постгрес")

    def test_unknown_acronym_is_spelled_out(self):
        self.assertEqual(latin_word("XYZ"), "эксуайзед")

    def test_acronyms_are_written_solid(self):
        """Дефисы ломают выравнивание: «эйч-ар» даёт ноль таймкодов."""
        for w in ("XYZ", "API", "SLA", "HR", "KPI"):
            self.assertNotIn("-", latin_word(w), f"дефис в «{w}»")

    def test_capitalised_word_is_not_an_acronym(self):
        """«Slack» — слово, а не аббревиатура: читать по буквам нельзя."""
        self.assertEqual(latin_word("Slack"), latin_word("slack"))

    def test_single_letter_gets_its_name(self):
        """«p95» — это «пи девяносто пять», не «п девяносто пять»."""
        self.assertIn("пи девяносто пять", normalize("p95"))

    def test_unknown_word_is_transliterated(self):
        out = latin_word("clickhouse")
        self.assertTrue(out)
        self.assertIsNone(DROPPED.search(out))


if __name__ == "__main__":
    unittest.main()
