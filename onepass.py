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


@dataclass
class Var(Expr):
    name: str

    def accept(self, visitor: "ExprVisitor") -> None:
        visitor.visit_var(self)


class ExprVisitor:
    def visit_int(self, expr: Int) -> None:
        raise NotImplementedError()

    def visit_bin_op(self, expr: BinOp) -> None:
        raise NotImplementedError()

    def visit_var(self, expr: Var) -> None:
        raise NotImplementedError()


class Stmt:
    def accept(self, visitor: "StmtVisitor") -> None:
        raise NotImplementedError()


@dataclass
class Assign(Stmt):
    name: str
    expr: Expr

    def accept(self, visitor: "StmtVisitor") -> None:
        visitor.visit_assign(self)


@dataclass
class Return(Stmt):
    expr: Expr

    def accept(self, visitor: "StmtVisitor") -> None:
        visitor.visit_return(self)


@dataclass
class If(Stmt):
    test: Expr
    pos: list[Stmt]
    neg: list[Stmt]

    def accept(self, visitor: "StmtVisitor") -> None:
        visitor.visit_if(self)


class StmtVisitor:
    def visit_assign(self, stmt: Assign) -> None:
        raise NotImplementedError()

    def visit_return(self, stmt: Return) -> None:
        raise NotImplementedError()

    def visit_if(self, stmt: If) -> None:
        raise NotImplementedError()


@dataclass
class Program:
    args: list[str]
    body: list[Stmt]


class ExprTranslator(ExprVisitor):
    def __init__(self, variables: "Variables", buffer: StringIO):
        self._variables = variables
        self._buffer = buffer

    def visit_int(self, expr: Int) -> None:
        self._buffer.write(f"    pushq   ${expr.value}\n")

    def visit_bin_op(self, expr: BinOp) -> None:
        expr.lhs.accept(self)
        expr.rhs.accept(self)
        self._buffer.write("    popq    %rcx\n")
        self._buffer.write("    popq    %rax\n")
        if expr.kind == BinOpKind.add:
            self._buffer.write("    addq    %rcx, %rax\n")
        elif expr.kind == BinOpKind.sub:
            self._buffer.write("    subq    %rcx, %rax\n")
        elif expr.kind == BinOpKind.mul:
            self._buffer.write("    imulq   %rcx\n")
        elif expr.kind == BinOpKind.div:
            self._buffer.write("    movq    $0, %rdx\n")
            self._buffer.write("    idivq   %rcx\n")
        self._buffer.write("    pushq   %rax\n")

    def visit_var(self, expr: Var) -> None:
        offset = self._variables.get_offset(expr.name)
        self._buffer.write(f"    pushq   {offset}(%rbp)\n")


class Translator(StmtVisitor):
    def __init__(self, spill: int, variables: "Variables", buffer: StringIO):
        self._offset = -spill
        self._label = 0
        self._variables = variables
        self._buffer = buffer
        self._expr_translator = ExprTranslator(variables, buffer)

    def _get_label(self) -> str:
        label = f"l{self._label}"
        self._label += 1
        return label

    def visit_assign(self, stmt: Assign) -> None:
        if not self._variables.is_defined(stmt.name):
            self._buffer.write("    subq    $8, %rsp\n")
            self._offset -= 8
            self._variables.define_var(stmt.name, self._offset)
        stmt.expr.accept(self._expr_translator)
        offset = self._variables.get_offset(stmt.name)
        self._buffer.write(f"    popq    {offset}(%rbp)\n")

    def visit_return(self, stmt: Return) -> None:
        stmt.expr.accept(self._expr_translator)
        self._buffer.write("    popq    %rax\n")
        self._buffer.write("    movq    %rbp, %rsp\n")
        self._buffer.write("    popq    %rbp\n")
        self._buffer.write("    ret\n")

    def visit_if(self, stmt: If) -> None:
        # TODO: Handle differences in variables declarations in branches (scopes?)
        neg = self._get_label()
        end = self._get_label()
        stmt.test.accept(self._expr_translator)
        self._buffer.write("    popq    %rax\n")
        self._buffer.write("    cmpq    $0, %rax\n")
        self._buffer.write(f"    je      {neg}\n")
        for s in stmt.pos:
            s.accept(self)
        self._buffer.write(f"    jmp     {end}\n")
        self._buffer.write(f"{neg}:\n")
        for s in stmt.neg:
            s.accept(self)
        self._buffer.write(f"{end}:\n")


class Variables:
    def __init__(self) -> None:
        self._offsets: dict[str, int] = {}

    def is_defined(self, name: str) -> bool:
        return name in self._offsets

    def define_var(self, name: str, offset: int) -> None:
        self._offsets[name] = offset

    def get_offset(self, name: str) -> int:
        return self._offsets[name]


class CallingConvention:
    def get_register(self, index: int) -> str | None:
        raise NotImplementedError()

    def get_offset(self, index: int) -> int:
        raise NotImplementedError()

    def alloc_spill(self, count: int) -> int:
        raise NotImplementedError()


class MicrosoftX64(CallingConvention):
    def get_register(self, index: int) -> str | None:
        if index >= 4:
            return None
        return ["%rcx", "%rdx", "%r8", "%r9"][index]

    def get_offset(self, index: int) -> int:
        return 8 * index + 16

    def alloc_spill(self, count: int) -> int:
        return 0


class SysVAMD64(CallingConvention):
    def get_register(self, index: int) -> str | None:
        if index >= 6:
            return None
        return ["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"][index]

    def get_offset(self, index: int) -> int:
        if index >= 6:
            return 8 * (index - 6) + 16
        return -8 * index - 8

    def alloc_spill(self, count: int) -> int:
        if count > 6:
            return 8 * (count - 6)
        return 0


def translate(
        program: Program,
        convention: CallingConvention = MicrosoftX64()) -> str:
    buffer = StringIO()
    buffer.write(".global _fn\n\n")
    buffer.write(".section .text\n")
    buffer.write("_fn:\n")

    buffer.write("    pushq   %rbp\n")
    buffer.write("    movq    %rsp, %rbp\n")

    spill = convention.alloc_spill(len(program.args))
    if spill != 0:
        buffer.write("    subq    ${spill}, %rsp\n")

    variables = Variables()

    for i, name in enumerate(program.args):
        offset = convention.get_offset(i)
        variables.define_var(name, offset)
        reg = convention.get_register(i)
        if reg is not None:
            buffer.write(f"    movq    {reg}, {offset}(%rbp)\n")

    translator = Translator(spill, variables, buffer)
    for stmt in program.body:
        stmt.accept(translator)

    return buffer.getvalue()
