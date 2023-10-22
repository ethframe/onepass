from dataclasses import dataclass
from enum import Enum


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


class Evaluator(InsnVisitor):
    def __init__(self, stack: "Stack"):
        self._stack = stack

    def visit_int(self, insn: Int) -> None:
        self._stack.push(insn.value)

    def visit_bin_op(self, insn: BinOp) -> None:
        b = self._stack.pop()
        a = self._stack.pop()
        if insn.kind == BinOpKind.add:
            self._stack.push(a + b)
        elif insn.kind == BinOpKind.sub:
            self._stack.push(a - b)
        elif insn.kind == BinOpKind.mul:
            self._stack.push(a * b)
        elif insn.kind == BinOpKind.div:
            self._stack.push(a // b)


def evaluate(program: Program) -> int:
    stack = Stack()
    evaluator = Evaluator(stack)

    for insn in program:
        insn.accept(evaluator)

    result = stack.pop()
    if not stack.empty():
        raise RuntimeError()
    return result


class Stack:
    def __init__(self) -> None:
        self._stack: list[int] = []

    def push(self, value: int) -> None:
        self._stack.append(value)

    def pop(self) -> int:
        if not self._stack:
            raise RuntimeError("pop from empty stack")
        return self._stack.pop()

    def empty(self) -> bool:
        return not self._stack
