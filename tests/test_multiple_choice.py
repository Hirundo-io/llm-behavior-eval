from evaluation_utils.multiple_choice_bias_evaluator import (
    _norm,
    match_answer_to_option,
)


def test_norm():
    assert _norm("1) Foo  Bar") == "foo bar"


def test_match_answer_to_option():
    options = {"1": "red", "2": "blue"}
    assert match_answer_to_option("1) red", options) == "1"
    assert match_answer_to_option("The answer is blue", options) == "2"
    assert match_answer_to_option("unknown", options) == "Unmatched"
