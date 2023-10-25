from dataclasses import dataclass
from enum import Enum, auto
from io import StringIO


from typing import Generic, TypeVar


T = TypeVar("T")


class Expr:
    def accept(self, visitor: "ExprVisitor[T]") -> T:
        raise NotImplementedError()


@dataclass
class Int(Expr):
    value: int

    def accept(self, visitor: "ExprVisitor[T]") -> T:
        return visitor.visit_int(self)


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

    def accept(self, visitor: "ExprVisitor[T]") -> T:
        return visitor.visit_bin_op(self)


@dataclass
class Var(Expr):
    name: str

    def accept(self, visitor: "ExprVisitor[T]") -> T:
        return visitor.visit_var(self)


@dataclass
class Call(Expr):
    name: str
    args: list[Expr]

    def accept(self, visitor: "ExprVisitor[T]") -> T:
        return visitor.visit_call(self)


class ExprVisitor(Generic[T]):
    def visit_int(self, expr: Int) -> T:
        raise NotImplementedError()

    def visit_bin_op(self, expr: BinOp) -> T:
        raise NotImplementedError()

    def visit_var(self, expr: Var) -> T:
        raise NotImplementedError()

    def visit_call(self, expr: Call) -> T:
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


class ValueKind(Enum):
    imm = auto()
    reg = auto()
    mem = auto()
    tos = auto()


@dataclass
class Value:
    kind: ValueKind
    value: int = 0
    offset: int = 0

    def to_tos(self, emitter: "AsmEmitter") -> "Value":
        if self.kind == ValueKind.reg:
            emitter.emit("pushq   %rax")
            return Value(ValueKind.tos)
        return self

    def to_reg(self, reg: str, emitter: "AsmEmitter") -> None:
        if self.kind == ValueKind.imm:
            emitter.emit(f"movq    ${self.value}, {reg}")
        elif self.kind == ValueKind.reg and reg != "%rax":
            emitter.emit(f"movq    %rax, {reg}")
        elif self.kind == ValueKind.mem:
            emitter.emit(f"movq    {self.offset}(%rbp), {reg}")
        elif self.kind == ValueKind.tos:
            emitter.emit(f"popq    {reg}")

    def to_mem(self, offset: int, emitter: "AsmEmitter") -> None:
        if self.kind == ValueKind.imm:
            emitter.emit(f"movq    ${self.value}, {offset}(%rbp)")
        if self.kind == ValueKind.reg:
            emitter.emit(f"movq    %rax, {offset}(%rbp)")
        elif self.kind == ValueKind.mem and self.offset != offset:
            emitter.emit(f"movq    {self.offset}(%rbp), %rax")
            emitter.emit(f"movq    %rax, {offset}(%rbp)")
        elif self.kind == ValueKind.tos:
            emitter.emit(f"popq    {offset}(%rbp)")

    def to_arg(self, reg: str, allow_imm: bool, emitter: "AsmEmitter") -> str:
        if allow_imm and self.kind == ValueKind.imm:
            return f"${self.value}"
        elif self.kind == ValueKind.mem:
            return f"{self.offset}(%rbp)"
        else:
            self.to_reg(reg, emitter)
            return reg


class ExprTranslator(ExprVisitor[Value]):
    def __init__(
            self, cc: "CallingConvention", stack: "Stack",
            emitter: "AsmEmitter"):
        self._cc = cc
        self._stack = stack
        self._emitter = emitter

    def visit_int(self, expr: Int) -> Value:
        return Value(ValueKind.imm, value=expr.value)

    def visit_bin_op(self, expr: BinOp) -> Value:
        lhs = expr.lhs.accept(self).to_tos(self._emitter)
        imm = expr.kind == BinOpKind.add or expr.kind == BinOpKind.sub
        rhs = expr.rhs.accept(self).to_arg("%rcx", imm, self._emitter)
        lhs.to_reg("%rax", self._emitter)
        if expr.kind == BinOpKind.add:
            self._emitter.emit(f"addq    {rhs}, %rax")
        elif expr.kind == BinOpKind.sub:
            self._emitter.emit(f"subq    {rhs}, %rax")
        elif expr.kind == BinOpKind.mul:
            self._emitter.emit(f"imulq   {rhs}")
        elif expr.kind == BinOpKind.div:
            self._emitter.emit("movq    $0, %rdx")
            self._emitter.emit(f"idivq   {rhs}")
        return Value(ValueKind.reg)

    def visit_var(self, expr: Var) -> Value:
        return Value(ValueKind.mem, offset=self._stack.get_offset(expr.name))

    def visit_call(self, expr: Call) -> Value:
        self._stack.new_region()
        regs = len(self._cc.registers)
        slots: list[int] = []
        if len(expr.args) > regs:
            self._stack.adjust(
                self._cc.alignment, len(expr.args) - regs, self._emitter)
            for _ in range(len(expr.args) - regs):
                slots.append(self._stack.allocate(self._emitter))
        tos: list[tuple[str, Value]] = []
        for i, arg in enumerate(expr.args):
            val = arg.accept(self)
            if i >= regs:
                val.to_mem(slots[i - regs], self._emitter)
            elif i == len(expr.args)-1:
                val.to_reg(self._cc.registers[i], self._emitter)
            else:
                tos.append((self._cc.registers[i], val.to_tos(self._emitter)))
        for reg, val in reversed(tos):
            val.to_reg(reg, self._emitter)
        self._emitter.emit(f"call    {expr.name}")
        self._stack.free_region(self._emitter)
        return Value(ValueKind.reg)


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
            self._stack.define_var(
                stmt.name, self._stack.allocate(self._emitter))
        offset = self._stack.get_offset(stmt.name)
        stmt.expr.accept(self._expr_translator).to_mem(offset, self._emitter)

    def visit_return(self, stmt: Return) -> None:
        stmt.expr.accept(self._expr_translator).to_reg("%rax", self._emitter)
        self._stack.leave(self._emitter)
        self._emitter.emit("popq    %rbp")
        self._emitter.emit("ret")

    def visit_if(self, stmt: If) -> None:
        neg = self._emitter.get_label()
        end = self._emitter.get_label()
        stmt.test.accept(self._expr_translator).to_reg("%rax", self._emitter)
        self._emitter.emit("cmpq    $0, %rax")
        self._emitter.emit(f"je      {neg}")
        self._stack.new_region()
        self.translate(stmt.pos)
        self._stack.free_region(self._emitter)
        self._emitter.emit(f"jmp     {end}")
        self._emitter.emit(f"{neg}:", indent=False)
        self._stack.new_region()
        self.translate(stmt.neg)
        self._stack.free_region(self._emitter)
        self._emitter.emit(f"{end}:", indent=False)


class Stack:
    def __init__(self, base: int, shadow: int) -> None:
        self._offsets: dict[str, int] = {}
        self._offset = base
        self._shadow = shadow
        self._next: list[tuple[dict[str, int], int]] = []

    def is_defined(self, name: str) -> bool:
        return name in self._offsets or \
            any(name in offsets for offsets, _ in reversed(self._next))

    def define_var(self, name: str, offset: int) -> None:
        self._offsets[name] = offset

    def get_offset(self, name: str) -> int:
        if name in self._offsets:
            return self._offsets[name]
        for offsets, _ in reversed(self._next):
            if name in offsets:
                return offsets[name]
        raise RuntimeError(f"undefined variable {name}")

    def adjust(self, args: int, alignment: int, emitter: "AsmEmitter") -> None:
        if alignment > 0:
            offset = alignment + (self._offset - args) % alignment
            emitter.emit(f"subq    ${offset}, %rsp")
            self._offset -= offset

    def allocate(self, emitter: "AsmEmitter") -> int:
        self._offset -= 8
        emitter.emit("subq    $8, %rsp")
        return self._offset

    def new_region(self) -> None:
        self._next.append((self._offsets, self._offset))

    def free_region(self, emitter: "AsmEmitter") -> None:
        self._offsets, offset = self._next.pop()
        allocated = offset - self._offset
        if allocated > 0:
            emitter.emit(f"addq    ${allocated}, %rsp")
        self._offset = offset

    def leave(self, emitter: "AsmEmitter") -> None:
        allocated = -self._offset + self._shadow
        if allocated > 0:
            emitter.emit(f"addq    ${allocated}, %rsp")

    def align(self, alignment: int) -> int:
        if alignment > 0:
            return alignment + self._offset % alignment
        return 0


@dataclass
class CallingConvention:
    registers: list[str]
    alignment: int = 16
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
SYSV_AMD64 = CallingConvention(["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"])


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

    base = 0
    shadow = 0

    if cc.shadow:
        shadow = 8 * len(cc.registers)
        emitter.emit(f"subq    ${shadow}, %rsp")
    else:
        base = -8 * min(len(cc.registers), len(func.args))
        emitter.emit(f"subq    ${-base}, %rsp")

    stack = Stack(base, shadow=shadow)

    for i, name in enumerate(func.args):
        offset = cc.get_offset(i)
        stack.define_var(name, offset)
        reg = cc.get_register(i)
        if reg is not None:
            emitter.emit(f"movq    {reg}, {offset}(%rbp)")

    translator = Translator(cc, stack, emitter)
    for stmt in func.body:
        stmt.accept(translator)
