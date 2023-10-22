def evaluate(program: str) -> int:
    stack = Stack()

    for c in program:
        if "0" <= c <= "9":
            stack.push(int(c))
        elif c in "+-*/":
            b = stack.pop()
            a = stack.pop()
            if c == "+":
                stack.push(a + b)
            elif c == "-":
                stack.push(a - b)
            elif c == "*":
                stack.push(a * b)
            elif c == "/":
                stack.push(a // b)
        elif c != " ":
            raise RuntimeError()

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
