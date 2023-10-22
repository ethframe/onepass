import pytest
import subprocess

from onepass import BinOp, BinOpKind, Int, translate

TEST_CASES = [
    (Int(1), 1),
    (BinOp(BinOpKind.add, Int(5), Int(2)), 7),
    (BinOp(BinOpKind.sub, Int(5), Int(2)), 3),
    (BinOp(BinOpKind.mul, Int(5), Int(2)), 10),
    (BinOp(BinOpKind.div, Int(5), Int(2)), 2),
    (BinOp(BinOpKind.add, BinOp(BinOpKind.add, Int(1), Int(1)), Int(1)), 3),
    (BinOp(BinOpKind.add, Int(1), BinOp(BinOpKind.add, Int(1), Int(1))), 3),
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
