import pytest

from onepass import BinOp, BinOpKind, Int, evaluate

TEST_CASES = [
    ([Int(1)], 1),
    ([Int(5), Int(2), BinOp(BinOpKind.add)], 7),
    ([Int(5), Int(2), BinOp(BinOpKind.sub)], 3),
    ([Int(5), Int(2), BinOp(BinOpKind.mul)], 10),
    ([Int(5), Int(2), BinOp(BinOpKind.div)], 2),
    ([Int(1), Int(1), BinOp(BinOpKind.add), Int(1), BinOp(BinOpKind.add)], 3),
    ([Int(1), Int(1), Int(1), BinOp(BinOpKind.add), BinOp(BinOpKind.add)], 3),
]


@ pytest.mark.parametrize("program, result", TEST_CASES)
def test_evaluate(program, result):
    assert evaluate(program) == result
