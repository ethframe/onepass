import pytest

from onepass import evaluate

TEST_CASES = [
    ("1", 1),
    ("5 2 +", 7),
    ("5 2 -", 3),
    ("5 2 *", 10),
    ("5 2 /", 2),
    ("1 1 + 1 +", 3),
    ("1 1 1 + +", 3),
]

@pytest.mark.parametrize("program, result", TEST_CASES)
def test_evaluate(program, result):
    assert evaluate(program) == result
