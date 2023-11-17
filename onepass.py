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
    def accept(self, visitor: "StmtVisitor[T]") -> T:
        raise NotImplementedError()


@dataclass
class Assign(Stmt):
    name: str
    expr: Expr

    def accept(self, visitor: "StmtVisitor[T]") -> T:
        return visitor.visit_assign(self)


@dataclass
class Return(Stmt):
    expr: Expr

    def accept(self, visitor: "StmtVisitor[T]") -> T:
        return visitor.visit_return(self)


@dataclass
class If(Stmt):
    test: Expr
    pos: list[Stmt]
    neg: list[Stmt]

    def accept(self, visitor: "StmtVisitor[T]") -> T:
        return visitor.visit_if(self)


class StmtVisitor(Generic[T]):
    def visit_assign(self, stmt: Assign) -> T:
        raise NotImplementedError()

    def visit_return(self, stmt: Return) -> T:
        raise NotImplementedError()

    def visit_if(self, stmt: If) -> T:
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

    def store_tmp(self, frame: "Frame", emitter: "AsmEmitter") -> "Value":
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


@dataclass
class Reg(Value):
    value: str

    def to_reg(self, reg: str, emitter: "AsmEmitter") -> None:
        if reg != self.value:
            emitter.emit(f"movq    {self.value}, {reg}")

    def to_mem(self, offset: int, emitter: "AsmEmitter") -> None:
        emitter.emit(f"movq    {self.value}, {offset}(%rbp)")

    def store_tmp(self, frame: "Frame", emitter: "AsmEmitter") -> "Value":
        offset = frame.allocate(8, emitter)
        self.to_mem(offset, emitter)
        return Mem(offset)


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
        self._frame.enter_scope()
        lhs = expr.lhs.accept(self).store_tmp(self._frame, self._emitter)
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
        self._frame.exit_scope(self._emitter)
        return Reg("%rax")

    def visit_var(self, expr: Var) -> Value:
        return Mem(self._frame.get_offset(expr.name))

    def visit_call(self, expr: Call) -> Value:
        self._frame.enter_scope()
        slots = self._frame.allocate_args(len(expr.args), self._emitter)
        regs = len(self._cc.registers)
        tos: list[Value] = []
        for i, arg in enumerate(expr.args[:regs]):
            val = arg.accept(self)
            if i == len(expr.args) - 1:
                val.to_reg(self._cc.registers[i], self._emitter)
            else:
                tos.append(val.store_tmp(self._frame, self._emitter))
        for i, arg in enumerate(expr.args[regs:]):
            arg.accept(self).to_mem(slots[i], self._emitter)
        for i, val in reversed(list(enumerate(tos))):
            val.to_reg(self._cc.registers[i], self._emitter)
        self._emitter.emit(f"call    {expr.name}")
        self._frame.exit_scope(self._emitter)
        return Reg("%rax")


class Translator(StmtVisitor[bool]):
    def __init__(
            self, convention: "CallingConvention", frame: "Frame",
            emitter: "AsmEmitter"):
        self._convention = convention
        self._frame = frame
        self._emitter = emitter
        self._expr_translator = ExprTranslator(convention, frame, emitter)

    def translate(self, stmts: list[Stmt]) -> bool:
        for stmt in stmts:
            if stmt.accept(self):
                return True
        return False

    def visit_assign(self, stmt: Assign) -> bool:
        if not self._frame.is_defined(stmt.name):
            self._frame.define_var(
                stmt.name, self._frame.allocate(8, self._emitter))
        offset = self._frame.get_offset(stmt.name)
        stmt.expr.accept(self._expr_translator).to_mem(offset, self._emitter)
        return False

    def visit_return(self, stmt: Return) -> bool:
        stmt.expr.accept(self._expr_translator).to_reg("%rax", self._emitter)
        self._frame.epilogue(self._emitter)
        return True

    def visit_if(self, stmt: If) -> bool:
        neg = self._emitter.get_label()
        end = self._emitter.get_label()
        test = stmt.test.accept(self._expr_translator)
        arg = test.load_arg("%rax", False, self._emitter)
        self._emitter.emit(f"cmpq    $0, {arg}")
        self._emitter.emit(f"je      {neg}")
        self._frame.enter_scope()
        pos_ret = self.translate(stmt.pos)
        self._frame.exit_scope(self._emitter)
        if not pos_ret:
            self._emitter.emit(f"jmp     {end}")
        self._emitter.emit(f"{neg}:", indent=False)
        self._frame.enter_scope()
        neg_ret = self.translate(stmt.neg)
        self._frame.exit_scope(self._emitter)
        if not pos_ret:
            self._emitter.emit(f"{end}:", indent=False)
        return pos_ret and neg_ret


class Frame:
    def __init__(self, cc: "CallingConvention") -> None:
        self._cc = cc
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

    def allocate(self, size: int, emitter: "AsmEmitter") -> int:
        if size > 0:
            self._offset -= size
            emitter.emit(f"subq    ${size}, %rsp")
        return self._offset

    def allocate_args(self, args: int, emitter: "AsmEmitter") -> list[int]:
        size = stack = 8 * max(args - len(self._cc.registers), 0)
        if self._cc.alignment > 0:
            size += (self._offset - size) % self._cc.alignment
        offset = self.allocate(size, emitter)
        return [slot for slot in range(offset, offset + stack, 8)]

    def enter_scope(self) -> None:
        self._next.append((self._offset, self._offsets))
        self._offsets = {}

    def exit_scope(self, emitter: "AsmEmitter") -> None:
        offset, self._offsets = self._next.pop()
        size = offset - self._offset
        if size > 0:
            emitter.emit(f"addq    ${size}, %rsp")
        self._offset = offset

    def prologue(self, args: list[str], emitter: "AsmEmitter") -> None:
        emitter.emit("pushq   %rbp")
        emitter.emit("movq    %rsp, %rbp")

        regs = len(self._cc.registers)
        if self._cc.shadow:
            self._shadow = 8 * regs
            emitter.emit(f"subq    ${self._shadow}, %rsp")
        else:
            self.allocate(8 * min(regs, len(args)), emitter)

        for i, name in enumerate(args):
            offset = self._cc.get_offset(i)
            self.define_var(name, offset)
            reg = self._cc.get_register(i)
            if reg is not None:
                emitter.emit(f"movq    {reg}, {offset}(%rbp)")

    def epilogue(self, emitter: "AsmEmitter") -> None:
        if self._offset == 0 and self._shadow == 0:
            emitter.emit("popq    %rbp")
        else:
            emitter.emit("leave")
        emitter.emit("ret")


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

    frame = Frame(cc)
    translator = Translator(cc, frame, emitter)

    frame.prologue(func.args, emitter)
    if not translator.translate(func.body):
        frame.epilogue(emitter)
