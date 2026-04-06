"""Tests for answer cleaning and scoring logic in gaia_submit.py.

These functions determine benchmark scores directly, so regressions here
translate 1:1 into incorrect GAIA results.
"""
import pytest

from gaia_submit import (
    _extract_answer_from_think,
    check_answer,
    clean_answer,
    normalize_for_comparison,
)


# ---------------------------------------------------------------------------
# clean_answer
# ---------------------------------------------------------------------------

class TestCleanAnswerPassthrough:
    def test_empty_string(self):
        assert clean_answer("") == ""

    def test_whitespace_only(self):
        assert clean_answer("   ") == ""

    def test_plain_word(self):
        assert clean_answer("Paris") == "Paris"

    def test_plain_number(self):
        assert clean_answer("42") == "42"

    def test_decimal(self):
        # Decimals must not be modified
        assert clean_answer("3.14") == "3.14"


class TestCleanAnswerFinalAnswer:
    def test_basic(self):
        assert clean_answer("FINAL ANSWER: Paris") == "Paris"

    def test_case_insensitive(self):
        assert clean_answer("final answer: Paris") == "Paris"

    def test_strips_square_brackets(self):
        assert clean_answer("FINAL ANSWER: [Paris]") == "Paris"

    def test_inline_with_preamble(self):
        assert clean_answer("thinking...\nFINAL ANSWER: 42") == "42"

    def test_takes_last_when_multiple(self):
        # Models sometimes emit FINAL ANSWER inside <think> and again outside;
        # the last occurrence is the intended answer.
        assert clean_answer("FINAL ANSWER: wrong\nFINAL ANSWER: right") == "right"

    def test_think_block_stripped_before_fa_extraction(self):
        # A FINAL ANSWER rehearsed inside <think> must not be captured.
        assert clean_answer("<think>FINAL ANSWER: wrong</think>Paris") == "Paris"

    def test_think_then_fa(self):
        assert clean_answer("<think>lots of reasoning</think>\nFINAL ANSWER: 42") == "42"


class TestCleanAnswerPrefixStripping:
    @pytest.mark.parametrize("prefix", [
        "The answer is",
        "Answer:",
        "answer is",
        "The final answer is",
        "So the answer is",
        "Therefore the answer is",
        "Thus the answer is",
        "The result is",
        "result:",
    ])
    def test_common_prefixes(self, prefix):
        assert clean_answer(f"{prefix} Paris") == "Paris"

    def test_chained_prefixes(self):
        # The loop must strip iteratively until no prefix remains.
        assert clean_answer("The answer is: answer: Paris") == "Paris"

    def test_leading_colon(self):
        assert clean_answer(": Paris") == "Paris"

    def test_leading_dash(self):
        assert clean_answer("- Paris") == "Paris"


class TestCleanAnswerThinkBlock:
    def test_closed_think_block(self):
        assert clean_answer("<think>I think</think>Paris") == "Paris"

    def test_think_with_prefix_outside(self):
        assert clean_answer("<think>reasoning</think>\nThe answer is Paris") == "Paris"


class TestCleanAnswerFormatting:
    def test_latex_boxed(self):
        assert clean_answer("\\boxed{42}") == "42"

    def test_trailing_period_stripped(self):
        assert clean_answer("Paris.") == "Paris"

    def test_trailing_comma_stripped(self):
        assert clean_answer("Paris,") == "Paris"

    def test_leading_dollar_stripped(self):
        assert clean_answer("$42") == "42"

    def test_double_quotes_stripped(self):
        assert clean_answer('"Paris"') == "Paris"

    def test_single_quotes_stripped(self):
        assert clean_answer("'Paris'") == "Paris"


class TestCleanAnswerNumbers:
    def test_thousands_separator(self):
        assert clean_answer("1,000") == "1000"

    def test_millions_separator(self):
        assert clean_answer("1,000,000") == "1000000"

    def test_number_with_decimal(self):
        assert clean_answer("1,000.50") == "1000.50"

    def test_comma_list_gets_spaces(self):
        # A value like "apple,banana" is not a number; commas get a space added.
        assert clean_answer("apple,banana") == "apple, banana"


# ---------------------------------------------------------------------------
# normalize_for_comparison
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercases(self):
        assert normalize_for_comparison("PARIS") == "paris"

    def test_strips_the(self):
        assert normalize_for_comparison("the cat") == "cat"

    def test_strips_a(self):
        assert normalize_for_comparison("a dog") == "dog"

    def test_strips_an(self):
        assert normalize_for_comparison("an apple") == "apple"

    def test_removes_punctuation(self):
        assert normalize_for_comparison("hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_for_comparison("  hello   world  ") == "hello world"


# ---------------------------------------------------------------------------
# check_answer
# ---------------------------------------------------------------------------

class TestCheckAnswer:
    def test_exact_match(self):
        assert check_answer("Paris", "Paris") is True

    def test_case_insensitive(self):
        assert check_answer("PARIS", "paris") is True

    def test_article_normalization(self):
        # "The Eiffel Tower" should match "Eiffel Tower"
        assert check_answer("The Eiffel Tower", "Eiffel Tower") is True

    def test_expected_substring_of_submitted(self):
        # A long submitted answer containing the expected answer (len > 2)
        assert check_answer("The city of Paris is beautiful", "Paris") is True

    def test_substring_check_requires_length_gt_2(self):
        # "12" has len == 2, so the substring path is skipped.
        # clean_answer("In 12 cases") does not produce "12", so result is False.
        assert check_answer("In 12 cases", "12") is False

    def test_clean_answer_applied(self):
        # Even if direct comparison fails, clean_answer is applied to submitted.
        assert check_answer("The answer is 42", "42") is True

    def test_number_formatting(self):
        # clean_answer strips comma separators from numbers
        assert check_answer("1,000", "1000") is True

    def test_no_match(self):
        assert check_answer("London", "Paris") is False

    def test_empty_submitted(self):
        assert check_answer("", "Paris") is False

    def test_empty_expected(self):
        # normalize("") == "", submitted also normalizes to something non-empty
        assert check_answer("Paris", "") is False


# ---------------------------------------------------------------------------
# _extract_answer_from_think
# ---------------------------------------------------------------------------

class TestExtractAnswerFromThink:
    def test_empty_returns_none(self):
        assert _extract_answer_from_think("") is None

    def test_answer_pattern(self):
        assert _extract_answer_from_think("The answer is Paris") == "Paris"

    def test_thus_answer_pattern(self):
        assert _extract_answer_from_think("Thus the answer is 42") == "42"

    def test_output_pattern(self):
        assert _extract_answer_from_think("output: 42") == "42"

    def test_respond_with_pattern(self):
        assert _extract_answer_from_think("We should output: London") == "London"

    def test_sentence_boundary_truncation(self):
        # "Claus. However..." should truncate at the sentence break.
        result = _extract_answer_from_think(
            "Thus the answer is Claus. However, Santa is also correct"
        )
        assert result == "Claus"

    def test_fallback_to_last_line(self):
        # No pattern matches → return the last non-empty line.
        result = _extract_answer_from_think(
            "I am thinking about this\nLet me consider the options\nParis"
        )
        assert result == "Paris"

    def test_last_answer_wins_in_multiline(self):
        # When multiple lines match, the last one in reverse order (first found) wins.
        result = _extract_answer_from_think(
            "The answer is wrong\nThe answer is right"
        )
        assert result == "right"
