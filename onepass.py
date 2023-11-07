from dataclasses import dataclass
from enum import Enum
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


class Value:
    def to_reg(self, reg: str, emitter: "AsmEmitter") -> None:
        raise NotImplementedError()

    def to_mem(self, offset: int, emitter: "AsmEmitter") -> None:
        raise NotImplementedError()

    def store_tmp(self, emitter: "AsmEmitter") -> "Value":
        return self

    def load_arg(self, reg: str, allow_imm: bool, emitter: "AsmEmitter") -> str:
        self.to_reg(reg, emitter)
        return reg


@dataclass
class Imm(Value):
    value: int

    def to_reg(self, reg: str, emitter: "AsmEmitter") -> None:
        emitter.emit(f"movq    ${self.value}, {reg}")

    def to_mem(self, offset: int, emitter: "AsmEmitter") -> None:
        emitter.emit(f"movq    ${self.value}, {offset}(%rbp)")

    def load_arg(self, reg: str, allow_imm: bool, emitter: "AsmEmitter") -> str:
        if allow_imm:
            return f"${self.value}"
        return super().load_arg(reg, allow_imm, emitter)


class Reg(Value):
    def to_reg(self, reg: str, emitter: "AsmEmitter") -> None:
        if reg != "%rax":
            emitter.emit(f"movq    %rax, {reg}")

    def to_mem(self, offset: int, emitter: "AsmEmitter") -> None:
        emitter.emit(f"movq    %rax, {offset}(%rbp)")

    def store_tmp(self, emitter: "AsmEmitter") -> "Value":
        emitter.emit("pushq   %rax")
        return Tos()


@dataclass
class Mem(Value):
    offset: int

    def to_reg(self, reg: str, emitter: "AsmEmitter") -> None:
        emitter.emit(f"movq    {self.offset}(%rbp), {reg}")

    def to_mem(self, offset: int, emitter: "AsmEmitter") -> None:
        if self.offset != offset:
            emitter.emit(f"movq    {self.offset}(%rbp), %rax")
            emitter.emit(f"movq    %rax, {offset}(%rbp)")

    def load_arg(self, reg: str, allow_imm: bool, emitter: "AsmEmitter") -> str:
        return f"{self.offset}(%rbp)"


class Tos(Value):
    def to_reg(self, reg: str, emitter: "AsmEmitter") -> None:
        emitter.emit(f"popq    {reg}")

    def to_mem(self, offset: int, emitter: "AsmEmitter") -> None:
        emitter.emit(f"popq    {offset}(%rbp)")


class ExprTranslator(ExprVisitor[Value]):
    def __init__(
            self, cc: "CallingConvention", frame: "Frame",
            emitter: "AsmEmitter"):
        self._cc = cc
        self._frame = frame
        self._emitter = emitter

    def visit_int(self, expr: Int) -> Value:
        return Imm(expr.value)

    def visit_bin_op(self, expr: BinOp) -> Value:
        lhs = expr.lhs.accept(self).store_tmp(self._emitter)
        imm = expr.kind == BinOpKind.add or expr.kind == BinOpKind.sub
        rhs = expr.rhs.accept(self).load_arg("%rcx", imm, self._emitter)
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
        return Reg()

    def visit_var(self, expr: Var) -> Value:
        return Mem(self._frame.get_offset(expr.name))

    def visit_call(self, expr: Call) -> Value:
        self._frame.new_region()
        regs = len(self._cc.registers)
        slots: list[int] = []
        if len(expr.args) > regs:
            self._frame.adjust(
                self._cc.alignment, 8 * (len(expr.args) - regs), self._emitter)
            for _ in range(len(expr.args) - regs):
                slots.append(self._frame.allocate(8, self._emitter))
        tos: list[tuple[str, Value]] = []
        for i, arg in enumerate(expr.args):
            val = arg.accept(self)
            if i >= regs:
                val.to_mem(slots[len(expr.args) - i - 1], self._emitter)
            elif i == len(expr.args)-1:
                val.to_reg(self._cc.registers[i], self._emitter)
            else:
                tos.append(
                    (self._cc.registers[i], val.store_tmp(self._emitter)))
        for reg, val in reversed(tos):
            val.to_reg(reg, self._emitter)
        self._emitter.emit(f"call    {expr.name}")
        self._frame.free_region(self._emitter)
        return Reg()


class Translator(StmtVisitor):
    def __init__(
            self, convention: "CallingConvention", frame: "Frame",
            emitter: "AsmEmitter"):
        self._convention = convention
        self._frame = frame
        self._emitter = emitter
        self._expr_translator = ExprTranslator(convention, frame, emitter)

    def translate(self, stmts: list[Stmt]) -> None:
        for stmt in stmts:
            stmt.accept(self)

    def visit_assign(self, stmt: Assign) -> None:
        if not self._frame.is_defined(stmt.name):
            self._frame.define_var(
                stmt.name, self._frame.allocate(8, self._emitter))
        offset = self._frame.get_offset(stmt.name)
        stmt.expr.accept(self._expr_translator).to_mem(offset, self._emitter)

    def visit_return(self, stmt: Return) -> None:
        stmt.expr.accept(self._expr_translator).to_reg("%rax", self._emitter)
        self._frame.leave(self._emitter)
        self._emitter.emit("popq    %rbp")
        self._emitter.emit("ret")

    def visit_if(self, stmt: If) -> None:
        neg = self._emitter.get_label()
        end = self._emitter.get_label()
        stmt.test.accept(self._expr_translator).to_reg("%rax", self._emitter)
        self._emitter.emit("cmpq    $0, %rax")
        self._emitter.emit(f"je      {neg}")
        self._frame.new_region()
        self.translate(stmt.pos)
        self._frame.free_region(self._emitter)
        self._emitter.emit(f"jmp     {end}")
        self._emitter.emit(f"{neg}:", indent=False)
        self._frame.new_region()
        self.translate(stmt.neg)
        self._frame.free_region(self._emitter)
        self._emitter.emit(f"{end}:", indent=False)


class Frame:
    def __init__(self) -> None:
        self._offset = 0
        self._offsets: dict[str, int] = {}
        self._next: list[tuple[int, dict[str, int]]] = []
        self._shadow = 0

    def is_defined(self, name: str) -> bool:
        return name in self._offsets or \
            any(name in offsets for _, offsets in reversed(self._next))

    def define_var(self, name: str, offset: int) -> None:
        self._offsets[name] = offset

    def get_offset(self, name: str) -> int:
        if name in self._offsets:
            return self._offsets[name]
        for _, offsets in reversed(self._next):
            if name in offsets:
                return offsets[name]
        raise RuntimeError(f"undefined variable {name}")

    def adjust(self, size: int, alignment: int, emitter: "AsmEmitter") -> None:
        if alignment > 0:
            offset = alignment + (self._offset - size) % alignment
            emitter.emit(f"subq    ${offset}, %rsp")
            self._offset -= offset

    def allocate(self, size: int, emitter: "AsmEmitter") -> int:
        if size > 0:
            self._offset -= size
            emitter.emit(f"subq    ${size}, %rsp")
        return self._offset

    def free(self, size: int, emitter: "AsmEmitter") -> None:
        if size > 0:
            emitter.emit(f"addq    ${size}, %rsp")

    def new_region(self) -> None:
        self._next.append((self._offset, self._offsets))

    def free_region(self, emitter: "AsmEmitter") -> None:
        offset, self._offsets = self._next.pop()
        self.free(offset - self._offset, emitter)
        self._offset = offset

    def shadow(self, shadow: int, emitter: "AsmEmitter") -> None:
        self._shadow += shadow
        emitter.emit(f"subq    ${shadow}, %rsp")

    def leave(self, emitter: "AsmEmitter") -> None:
        self.free(-self._offset + self._shadow, emitter)


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

    frame = create_frame(func.args, emitter, cc)

    translator = Translator(cc, frame, emitter)
    for stmt in func.body:
        stmt.accept(translator)


def create_frame(
        args: list[str], emitter: AsmEmitter, cc: CallingConvention) -> Frame:
    frame = Frame()

    if cc.shadow:
        frame.shadow(8 * len(cc.registers), emitter)
    else:
        frame.allocate(8 * min(len(cc.registers), len(args)), emitter)

    for i, name in enumerate(args):
        offset = cc.get_offset(i)
        frame.define_var(name, offset)
        reg = cc.get_register(i)
        if reg is not None:
            emitter.emit(f"movq    {reg}, {offset}(%rbp)")

    return frame
