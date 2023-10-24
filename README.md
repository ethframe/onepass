# Simple one-pass compiler

Just enough code to compile this

```python
Program([
    Func("fac", ["x"], [
        If(Var("x"), [
            Return(
                BinOp(
                    BinOpKind.mul,
                    Var("x"),
                    Call("fac", [BinOp(BinOpKind.sub, Var("x"), Int(1))])
                )
            )
        ], [
            Return(Int(1))
        ])
    ])
])
```

into something like this

```gas
.section .text

.global fac
fac:
    pushq   %rbp
    movq    %rsp, %rbp
    subq    $32, %rsp
    movq    %rcx, 16(%rbp)
    movq    16(%rbp), %rax
    cmpq    $0, %rax
    je      l0
    movq    16(%rbp), %rax
    subq    $1, %rax
    movq    %rax, %rcx
    call    fac
    movq    %rax, %rcx
    movq    16(%rbp), %rax
    imulq   %rcx
    addq    $32, %rsp
    popq    %rbp
    ret
    jmp     l1
l0:
    movq    $1, %rax
    addq    $32, %rsp
    popq    %rbp
    ret
l1:
```
