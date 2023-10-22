import pytest
import subprocess

from onepass import BinOp, BinOpKind, Int, translate

TEST_CASES = [
    ([Int(1)], 1),
    ([Int(5), Int(2), BinOp(BinOpKind.add)], 7),
    ([Int(5), Int(2), BinOp(BinOpKind.sub)], 3),
    ([Int(5), Int(2), BinOp(BinOpKind.mul)], 10),
    ([Int(5), Int(2), BinOp(BinOpKind.div)], 2),
    ([Int(1), Int(1), BinOp(BinOpKind.add), Int(1), BinOp(BinOpKind.add)], 3),
    ([Int(1), Int(1), Int(1), BinOp(BinOpKind.add), BinOp(BinOpKind.add)], 3),
]


@pytest.fixture(scope="function")
def main(tmpdir):
    main = tmpdir.join("main.c")
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
