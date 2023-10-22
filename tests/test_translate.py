import subprocess

import pytest

from onepass import Assign, BinOp, BinOpKind, Int, Return, Var, translate, Program, If

TEST_CASES = [
    (Program([], [Return(Int(1))]), [], 1),
    (Program([], [Return(BinOp(BinOpKind.add, Int(5), Int(2)))]), [], 7),
    (Program([], [Return(BinOp(BinOpKind.sub, Int(5), Int(2)))]), [], 3),
    (Program([], [Return(BinOp(BinOpKind.mul, Int(5), Int(2)))]), [], 10),
    (Program([], [Return(BinOp(BinOpKind.div, Int(5), Int(2)))]), [], 2),
    (Program([], [Return(BinOp(BinOpKind.add, BinOp(BinOpKind.add, Int(1), Int(1)), Int(1)))]), [], 3),
    (Program([], [Return(BinOp(BinOpKind.add, Int(1), BinOp(BinOpKind.add, Int(1), Int(1))))]), [], 3),
    (Program([], [
        Assign("x", Int(1)),
        Return(Var("x"))
    ]), [], 1),
    (Program([], [
        Assign("x", Int(1)),
        Assign("x", BinOp(BinOpKind.add, Var("x"), Int(1))),
        Return(Var("x"))
    ]), [], 2),
    (Program([], [
        Assign("x", Int(1)),
        Assign("y", BinOp(BinOpKind.add, Var("x"), Int(1))),
        Return(Var("x"))
    ]), [], 1),
    (Program(["x"], [
        Return(Var("x"))
    ]), [1], 1),
    (Program(["x", "y"], [
        Return(Var("y"))
    ]), [1, 2], 2),
    (Program(["a", "b", "c", "d", "e", "f", "g"], [
        Return(Var("g"))
    ]), [1, 2, 3, 4, 5, 6, 7], 7),
    (Program(["x"], [
        Assign("x", BinOp(BinOpKind.add, Var("x"), Int(1))),
        Return(Var("x"))
    ]), [1], 2),
    (Program(["x"], [
        If(Var("x"), [Return(Int(0))], [Return(Int(1))])
    ]), [1], 0),
    (Program(["x"], [
        If(Var("x"), [Return(Int(0))], [Return(Int(1))])
    ]), [0], 1),
]


def get_main(tmpdir, args: list[int]):
    main = tmpdir.join("main.c")
    sig = ", ".join(["int64_t"] * len(args))
    val = ", ".join(str(i) for i in args)
    main.write(rf""" \
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

extern int64_t _fn({sig});

int main() {{
    printf("%" PRId64 "\n", _fn({val}));
}}
""")
    return main


@pytest.mark.parametrize("program, args, result", TEST_CASES)
def test_translate(program, args, result, tmpdir):
    main = get_main(tmpdir, args)
    asm = tmpdir.join("fn.s")
    asm.write(translate(program))
    exe = tmpdir.join("a")
    assert subprocess.run(["gcc", "-o", exe, asm, main]).returncode == 0
    out = subprocess.run([exe], capture_output=True)
    assert out.returncode == 0
    assert int(out.stdout) == result
