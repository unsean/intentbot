"""Tests for the safe math evaluator and math detection."""

import pytest

from main import calculate_math, looks_like_math


class TestCalculateMath:
    @pytest.mark.parametrize(
        "expr,expected",
        [
            ("what is 25 * 4", 100),
            ("2 plus 3", 5),
            ("2 plus 3 times 4", 14),
            ("100 divided by 5", 20),
            ("divide 100 by 5", 20),
            ("multiply 6 by 7", 42),
            ("add 5 to 3", 8),
            ("subtract 5 from 20", 15),
            ("5 less than 20", 15),
            ("2 to the power of 5", 32),
            ("2 power 10", 1024),
            ("15 mod 4", 3),
            ("what is 10 percent of 200", 20),
            ("15% of 240", 36),
            ("20 percent of 150", 30),
            ("5 x 3", 15),
            ("(2 + 3) * 4", 20),
        ],
    )
    def test_expressions(self, expr, expected):
        assert str(expected) in calculate_math(expr)

    def test_division_by_zero(self):
        assert "zero" in calculate_math("10 / 0").lower()

    def test_no_expression(self):
        assert "couldn't find" in calculate_math("hello there").lower()

    def test_empty_input(self):
        assert "couldn't find" in calculate_math("").lower()

    def test_unsupported_expression(self):
        # letters that aren't math words should fail safely
        assert "couldn't compute" in calculate_math("open sesame 123").lower()


class TestLooksLikeMath:
    @pytest.mark.parametrize(
        "text",
        [
            "what is 25 * 4",
            "calculate 100 / 5",
            "multiply 6 by 7",
            "2 plus 2",
        ],
    )
    def test_math_detected(self, text):
        assert looks_like_math(text)

    @pytest.mark.parametrize(
        "text",
        [
            "hello there",
            "tell me a joke",
            "what is your name",
            "I am 25 years old",
        ],
    )
    def test_non_math(self, text):
        assert not looks_like_math(text)
