import subprocess

import pytest

from onepass import Assign, BinOp, BinOpKind, Int, Return, Var, translate

TEST_CASES = [
    ([Return(Int(1))], 1),
    ([Return(BinOp(BinOpKind.add, Int(5), Int(2)))], 7),
    ([Return(BinOp(BinOpKind.sub, Int(5), Int(2)))], 3),
    ([Return(BinOp(BinOpKind.mul, Int(5), Int(2)))], 10),
    ([Return(BinOp(BinOpKind.div, Int(5), Int(2)))], 2),
    ([Return(BinOp(BinOpKind.add, BinOp(BinOpKind.add, Int(1), Int(1)), Int(1)))], 3),
    ([Return(BinOp(BinOpKind.add, Int(1), BinOp(BinOpKind.add, Int(1), Int(1))))], 3),
    ([
        Assign("x", Int(1)),
        Return(Var("x"))
    ], 1),
    ([
        Assign("x", Int(1)),
        Assign("x", BinOp(BinOpKind.add, Var("x"), Int(1))),
        Return(Var("x"))
    ], 2),
    ([
        Assign("x", Int(1)),
        Assign("y", BinOp(BinOpKind.add, Var("x"), Int(1))),
        Return(Var("x"))
    ], 1),
]


@ pytest.fixture(scope="session")
def main(tmpdir_factory):
    main = tmpdir_factory.mktemp("main").join("main.c")
    main.write(r""" \
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

extern int64_t _fn();

int main() {
    printf("%" PRId64 "\n", _fn());
}
""")
    return main


@ pytest.mark.parametrize("program, result", TEST_CASES)
def test_translate(program, result, main, tmpdir):
    asm = tmpdir.join("fn.s")
    asm.write(translate(program))
    exe = tmpdir.join("a")
    assert subprocess.run(["gcc", "-o", exe, asm, main]).returncode == 0
    out = subprocess.run([exe], capture_output=True)
    assert out.returncode == 0
    assert int(out.stdout) == result
