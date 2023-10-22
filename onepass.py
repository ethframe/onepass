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
    def __init__(self, variables: "Variables", writer: "AsmWriter"):
        self._variables = variables
        self._writer = writer

    def visit_int(self, expr: Int) -> None:
        self._writer.write(f"    pushq   ${expr.value}\n")

    def visit_bin_op(self, expr: BinOp) -> None:
        expr.lhs.accept(self)
        expr.rhs.accept(self)
        self._writer.write("    popq    %rcx\n")
        self._writer.write("    popq    %rax\n")
        if expr.kind == BinOpKind.add:
            self._writer.write("    addq    %rcx, %rax\n")
        elif expr.kind == BinOpKind.sub:
            self._writer.write("    subq    %rcx, %rax\n")
        elif expr.kind == BinOpKind.mul:
            self._writer.write("    imulq   %rcx\n")
        elif expr.kind == BinOpKind.div:
            self._writer.write("    movq    $0, %rdx\n")
            self._writer.write("    idivq   %rcx\n")
        self._writer.write("    pushq   %rax\n")

    def visit_var(self, expr: Var) -> None:
        offset = self._variables.get_offset(expr.name)
        self._writer.write(f"    pushq   {offset}(%rbp)\n")


class Translator(StmtVisitor):
    def __init__(self, variables: "Variables", writer: "AsmWriter"):
        self._variables = variables
        self._writer = writer
        self._expr_translator = ExprTranslator(variables, writer)

    def visit_assign(self, stmt: Assign) -> None:
        if not self._variables.is_defined(stmt.name):
            self._writer.write("    subq    $8, %rsp\n")
            self._variables.define_var(stmt.name, self._variables.allocate())
        stmt.expr.accept(self._expr_translator)
        offset = self._variables.get_offset(stmt.name)
        self._writer.write(f"    popq    {offset}(%rbp)\n")

    def visit_return(self, stmt: Return) -> None:
        stmt.expr.accept(self._expr_translator)
        self._writer.write("    popq    %rax\n")
        self._writer.write("    movq    %rbp, %rsp\n")
        self._writer.write("    popq    %rbp\n")
        self._writer.write("    ret\n")

    def visit_if(self, stmt: If) -> None:
        neg = self._writer.get_label()
        end = self._writer.get_label()
        stmt.test.accept(self._expr_translator)
        self._writer.write("    popq    %rax\n")
        self._writer.write("    cmpq    $0, %rax\n")
        self._writer.write(f"    je      {neg}\n")
        self._handle_if_branch(stmt.pos)
        self._writer.write(f"    jmp     {end}\n")
        self._writer.write(f"{neg}:\n")
        self._handle_if_branch(stmt.neg)
        self._writer.write(f"{end}:\n")

    def _handle_if_branch(self, stmts: list[Stmt]) -> None:
        variables = self._variables.scope()
        translator = Translator(variables, self._writer)
        for stmt in stmts:
            stmt.accept(translator)
        allocated = variables.allocated()
        if allocated > 0:
            self._writer.write(f"    addq    ${allocated}, %rsp\n")


class Variables:
    def __init__(self, offset: int, next: "Variables | None" = None) -> None:
        self._offsets: dict[str, int] = {}
        self._offset = offset
        self._next = next

    def is_defined(self, name: str) -> bool:
        return name in self._offsets or \
            self._next is not None and self._next.is_defined(name)

    def define_var(self, name: str, offset: int) -> None:
        self._offsets[name] = offset

    def get_offset(self, name: str) -> int:
        return self._offsets[name] \
            if name in self._offsets or self._next is None \
            else self._next.get_offset(name)

    def allocate(self) -> int:
        self._offset -= 8
        return self._offset

    def allocated(self) -> int:
        if self._next is not None:
            return self._next._offset - self._offset
        return -self._offset

    def scope(self) -> "Variables":
        return Variables(self._offset, self)


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


class AsmWriter:
    def __init__(self) -> None:
        self._buffer = StringIO()
        self._label = 0

    def get_label(self) -> str:
        label = f"l{self._label}"
        self._label += 1
        return label

    def write(self, line: str) -> None:
        self._buffer.write(line)

    def getvalue(self) -> str:
        return self._buffer.getvalue()


def translate(
        program: Program,
        convention: CallingConvention = MicrosoftX64()) -> str:
    writer = AsmWriter()
    writer.write(".global _fn\n\n")
    writer.write(".section .text\n")
    writer.write("_fn:\n")

    writer.write("    pushq   %rbp\n")
    writer.write("    movq    %rsp, %rbp\n")

    spill = convention.alloc_spill(len(program.args))
    if spill != 0:
        writer.write("    subq    ${spill}, %rsp\n")

    variables = Variables(-spill)

    for i, name in enumerate(program.args):
        offset = convention.get_offset(i)
        variables.define_var(name, offset)
        reg = convention.get_register(i)
        if reg is not None:
            writer.write(f"    movq    {reg}, {offset}(%rbp)\n")

    translator = Translator(variables, writer)
    for stmt in program.body:
        stmt.accept(translator)

    return writer.getvalue()
