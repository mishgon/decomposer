import pytest

from prototype import extract_answer, is_unanswerable_answer


@pytest.mark.parametrize(
    ("raw_output", "expected"),
    [
        ("  Ada Lovelace  \n", "Ada Lovelace"),
        ("<think>reasoning</think>\n  Ada Lovelace  ", "Ada Lovelace"),
        ("reasoning text</think>\n  Ada Lovelace  ", "Ada Lovelace"),
        ("New   York", "New   York"),
        ("None of the above", "None of the above"),
        ("None", "None"),
        ("  None\n", "None"),
        ("<think>reasoning</think>\nNone", "None"),
        ("Unanswerable", "Unanswerable"),
        ("  Unanswerable\n", "Unanswerable"),
    ],
)
def test_extract_answer(raw_output: str, expected: str) -> None:
    assert extract_answer(raw_output) == expected


def test_extract_answer_returns_none_for_unclosed_thinking() -> None:
    assert extract_answer("<think>still thinking") is None


@pytest.mark.parametrize(
    "answer",
    [
        "Unanswerable",
        "Unanswerable.",
        "unanswerable.",
        "  UNANSWERABLE.\n",
    ],
)
def test_is_unanswerable_answer_accepts_punctuation_variants(answer: str) -> None:
    assert is_unanswerable_answer(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "Not unanswerable.",
        "Unanswerable from the provided context.",
        "answerable",
    ],
)
def test_is_unanswerable_answer_rejects_other_text(answer: str) -> None:
    assert not is_unanswerable_answer(answer)
