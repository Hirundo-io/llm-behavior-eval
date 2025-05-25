import pytest
pytest.importorskip("torch")

from llm_behavior_eval.evaluation_utils.multiple_choice_bias_evaluator import (
    extract_options,
    match_answer_to_option,
)


def test_match_answer_to_option_with_label() -> None:
    options = {"1": "cat", "2": "dog"}
    assert match_answer_to_option("2) dog", options) == "2"


def test_match_answer_to_option_with_substring() -> None:
    options = {"1": "blue", "2": "red"}
    assert match_answer_to_option("I pick red option", options) == "2"


def test_extract_options() -> None:
    text = "Intro\n###Options:\n1) cat\n2) dog\n"
    assert extract_options(text) == {"1": "cat", "2": "dog"}
