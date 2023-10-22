from dataclasses import dataclass
from enum import Enum
from io import StringIO


class Expr:
    def accept(self, visitor: "ExprVisitor") -> None:
        raise NotImplementedError()


@dataclass
class Int(Expr):
    value: int

    def accept(self, visitor: "ExprVisitor") -> None:
        visitor.visit_int(self)


class BinOpKind(Enum):
    add = "+"
    sub = "-"
    mul = "*"
    div = "/"


@dataclass
class BinOp(Expr):
    kind: BinOpKind
    lhs: Expr
    rhs: Expr

    def accept(self, visitor: "ExprVisitor") -> None:
        visitor.visit_bin_op(self)


class ExprVisitor:
    def visit_int(self, insn: Int) -> None:
        raise NotImplementedError()

    def visit_bin_op(self, insn: BinOp) -> None:
        raise NotImplementedError()


class Translator(ExprVisitor):
    def __init__(self, buffer: StringIO):
        self._buffer = buffer

    def visit_int(self, insn: Int) -> None:
        self._buffer.write(f"    pushq   ${insn.value}\n")

    def visit_bin_op(self, insn: BinOp) -> None:
        insn.lhs.accept(self)
        insn.rhs.accept(self)
        self._buffer.write(f"    popq    %rcx\n")
        self._buffer.write(f"    popq    %rax\n")
        if insn.kind == BinOpKind.add:
            self._buffer.write(f"    addq    %rcx, %rax\n")
        elif insn.kind == BinOpKind.sub:
            self._buffer.write(f"    subq    %rcx, %rax\n")
        elif insn.kind == BinOpKind.mul:
            self._buffer.write(f"    imulq   %rcx\n")
        elif insn.kind == BinOpKind.div:
            self._buffer.write(f"    movq    $0, %rdx\n")
            self._buffer.write(f"    idivq   %rcx\n")
        self._buffer.write(f"    pushq   %rax\n")


def translate(program: Expr) -> str:
    buffer = StringIO()
    buffer.write(".global _fn\n\n")
    buffer.write(".section .text\n")
    buffer.write("_fn:\n")

    translator = Translator(buffer)
    program.accept(translator)

    buffer.write("    popq    %rax\n")
    buffer.write("    ret\n")

    return buffer.getvalue()
