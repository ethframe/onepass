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


@dataclass
class Call(Expr):
    name: str
    args: list[Expr]

    def accept(self, visitor: "ExprVisitor") -> None:
        visitor.visit_call(self)


class ExprVisitor:
    def visit_int(self, expr: Int) -> None:
        raise NotImplementedError()

    def visit_bin_op(self, expr: BinOp) -> None:
        raise NotImplementedError()

    def visit_var(self, expr: Var) -> None:
        raise NotImplementedError()

    def visit_call(self, expr: Call) -> None:
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
class Func:
    name: str
    args: list[str]
    body: list[Stmt]


@dataclass
class Program:
    funcs: list[Func]


class ExprTranslator(ExprVisitor):
    def __init__(
            self, cc: "CallingConvention", stack: "Stack",
            emitter: "AsmEmitter"):
        self._cc = cc
        self._stack = stack
        self._emitter = emitter

    def visit_int(self, expr: Int) -> None:
        self._emitter.emit(f"pushq   ${expr.value}")

    def visit_bin_op(self, expr: BinOp) -> None:
        expr.lhs.accept(self)
        expr.rhs.accept(self)
        self._emitter.emit("popq    %rcx")
        self._emitter.emit("popq    %rax")
        if expr.kind == BinOpKind.add:
            self._emitter.emit("addq    %rcx, %rax")
        elif expr.kind == BinOpKind.sub:
            self._emitter.emit("subq    %rcx, %rax")
        elif expr.kind == BinOpKind.mul:
            self._emitter.emit("imulq   %rcx")
        elif expr.kind == BinOpKind.div:
            self._emitter.emit("movq    $0, %rdx")
            self._emitter.emit("idivq   %rcx")
        self._emitter.emit("pushq   %rax")

    def visit_var(self, expr: Var) -> None:
        offset = self._stack.get_offset(expr.name)
        self._emitter.emit(f"pushq   {offset}(%rbp)")

    def visit_call(self, expr: Call) -> None:
        stack = self._stack.scope()
        slots: list[int] = []
        for _ in range(max(0, len(expr.args) - len(self._cc.registers))):
            slots.append(stack.allocate(self._emitter))
        regs: list[str] = []
        for i, arg in enumerate(expr.args):
            arg.accept(self)
            reg = self._cc.get_register(i)
            if reg is None:
                self._emitter.emit(f"popq    {slots.pop()}(%rbp)")
            else:
                regs.append(reg)
        while regs:
            self._emitter.emit(f"popq    {regs.pop()}")
        stack.adjust(self._cc, self._emitter)
        self._emitter.emit(f"call    {expr.name}")
        stack.free(self._emitter)
        self._emitter.emit("pushq   %rax")


class Translator(StmtVisitor):
    def __init__(
            self, convention: "CallingConvention", stack: "Stack",
            emitter: "AsmEmitter"):
        self._convention = convention
        self._stack = stack
        self._emitter = emitter
        self._expr_translator = ExprTranslator(convention, stack, emitter)

    def translate(self, stmts: list[Stmt]) -> None:
        for stmt in stmts:
            stmt.accept(self)

    def visit_assign(self, stmt: Assign) -> None:
        if not self._stack.is_defined(stmt.name):
            self._emitter.emit("subq    $8, %rsp")
            self._stack.define_var(
                stmt.name, self._stack.allocate(self._emitter))
        stmt.expr.accept(self._expr_translator)
        offset = self._stack.get_offset(stmt.name)
        self._emitter.emit(f"popq    {offset}(%rbp)")

    def visit_return(self, stmt: Return) -> None:
        stmt.expr.accept(self._expr_translator)
        self._emitter.emit("popq    %rax")
        self._emitter.emit("movq    %rbp, %rsp")
        self._emitter.emit("popq    %rbp")
        self._emitter.emit("ret")

    def visit_if(self, stmt: If) -> None:
        neg = self._emitter.get_label()
        end = self._emitter.get_label()
        stmt.test.accept(self._expr_translator)
        self._emitter.emit("popq    %rax")
        self._emitter.emit("cmpq    $0, %rax")
        self._emitter.emit(f"je      {neg}")
        stack = self._stack.scope()
        Translator(self._convention, stack, self._emitter).translate(stmt.pos)
        stack.free(self._emitter)
        self._emitter.emit(f"jmp     {end}")
        self._emitter.emit(f"{neg}:", indent=False)
        stack = self._stack.scope()
        Translator(self._convention, stack, self._emitter).translate(stmt.neg)
        stack.free(self._emitter)
        self._emitter.emit(f"{end}:", indent=False)


class Stack:
    def __init__(self, base: int, next: "Stack | None" = None) -> None:
        self._offsets: dict[str, int] = {}
        self._offset = base
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

    def adjust(self, cc: "CallingConvention", emitter: "AsmEmitter") -> None:
        offset = 8 * len(cc.registers) if cc.shadow else 0
        if cc.alignment > 0:
            offset += cc.alignment + (self._offset - offset) % cc.alignment
        self._offset -= offset
        emitter.emit(f"subq    ${offset}, %rsp")

    def allocate(self, emitter: "AsmEmitter") -> int:
        self._offset -= 8
        emitter.emit("subq    $8, %rsp")
        return self._offset

    def scope(self) -> "Stack":
        return Stack(self._offset, self)

    def free(self, emitter: "AsmEmitter") -> None:
        allocated = -self._offset if self._next is None \
            else self._next._offset - self._offset
        if allocated > 0:
            emitter.emit(f"addq    ${allocated}, %rsp")

    def align(self, alignment: int) -> int:
        if alignment > 0:
            return alignment + self._offset % alignment
        return 0


@dataclass
class CallingConvention:
    registers: list[str]
    alignment: int = 0
    shadow: bool = False

    def get_register(self, index: int) -> str | None:
        if index >= len(self.registers):
            return None
        return self.registers[index]

    def get_offset(self, index: int) -> int:
        if self.shadow:
            return 8 * index + 16
        if index >= len(self.registers):
            return 8 * (index - len(self.registers)) + 16
        return -8 * (index + 1)


MS_X64 = CallingConvention(["%rcx", "%rdx", "%r8", "%r9"], shadow=True)
SYSV_AMD64 = CallingConvention(
    ["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"], alignment=16
)


class AsmEmitter:
    def __init__(self) -> None:
        self._buffer = StringIO()
        self._label = 0

    def get_label(self) -> str:
        label = f"l{self._label}"
        self._label += 1
        return label

    def emit(self, line: str, indent: bool = True) -> None:
        if indent:
            self._buffer.write("    ")
        self._buffer.write(line)
        self._buffer.write("\n")

    def getvalue(self) -> str:
        return self._buffer.getvalue()


def translate(program: Program, cc: CallingConvention = MS_X64) -> str:
    emitter = AsmEmitter()
    emitter.emit(".section .text", indent=False)

    for func in program.funcs:
        emitter.emit("", indent=False)
        translate_func(func, emitter, cc)

    return emitter.getvalue()


def translate_func(
        func: Func, emitter: AsmEmitter, cc: CallingConvention) -> None:
    emitter.emit(f".global {func.name}", indent=False)
    emitter.emit(f"{func.name}:", indent=False)

    emitter.emit("pushq   %rbp")
    emitter.emit("movq    %rsp, %rbp")

    offset = 0
    if not cc.shadow:
        spill = 8 * len(cc.registers)
        emitter.emit(f"subq    ${spill}, %rsp")
        offset -= spill

    stack = Stack(offset)

    for i, name in enumerate(func.args):
        offset = cc.get_offset(i)
        stack.define_var(name, offset)
        reg = cc.get_register(i)
        if reg is not None:
            emitter.emit(f"movq    {reg}, {offset}(%rbp)")

    translator = Translator(cc, stack, emitter)
    for stmt in func.body:
        stmt.accept(translator)
