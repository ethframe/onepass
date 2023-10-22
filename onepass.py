from dataclasses import dataclass
from enum import Enum
from io import StringIO


class Insn:
    def accept(self, visitor: "InsnVisitor") -> None:
        raise NotImplementedError()


@dataclass
class Int(Insn):
    value: int

    def accept(self, visitor: "InsnVisitor") -> None:
        visitor.visit_int(self)


class BinOpKind(Enum):
    add = "+"
    sub = "-"
    mul = "*"
    div = "/"


@dataclass
class BinOp(Insn):
    kind: BinOpKind

    def accept(self, visitor: "InsnVisitor") -> None:
        visitor.visit_bin_op(self)


class InsnVisitor:
    def visit_int(self, insn: Int) -> None:
        raise NotImplementedError()

    def visit_bin_op(self, insn: BinOp) -> None:
        raise NotImplementedError()


Program = list[Insn]


class Translator(InsnVisitor):
    def __init__(self, buffer: StringIO):
        self._buffer = buffer

    def visit_int(self, insn: Int) -> None:
        self._buffer.write(f"    pushq   ${insn.value}\n")

    def visit_bin_op(self, insn: BinOp) -> None:
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


def translate(program: Program) -> str:
    buffer = StringIO()
    buffer.write(".global _fn\n\n")
    buffer.write(".section .text\n")
    buffer.write("_fn:\n")

    translator = Translator(buffer)
    for insn in program:
        insn.accept(translator)

    buffer.write("    popq    %rax\n")
    buffer.write("    ret\n")

    return buffer.getvalue()
