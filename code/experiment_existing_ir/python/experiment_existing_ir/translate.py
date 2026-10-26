"""
Translate a source program (the typed AST from `compiler.parser`) into MLIR text: the standard `func`, `arith` and
`scf` dialects, plus `dist.uniform` for a distribution, which the C++ tool in `cpp/` reads.

Every integer is given one fixed width, chosen by the front-end, the way a classical front-end gives every `int` its
declared width. That width is an upper bound: range analysis narrows it later. Results that do not fit wrap, as they do
in C. Change `WIDTH` (or pass `width`) to try another bound.

This covers constants, variables, `+`, `-`, `<`, a distribution over a range, a conditional and a final return. A
conditional becomes an `scf.if` that yields the variables either arm changed. Anything else is refused by name, so a
later milestone adds it deliberately.
"""
from typing import Dict, Iterator, List, Sequence

from compiler.parser.grammar import Assign, BinOp, Distribution, If, Node, Number, Return, Variable

WIDTH = 32


class TranslationError(ValueError):
    pass


class _Names:
    """Hands out the SSA names `%0`, `%1`, ... that every block of the function shares."""

    def __init__(self) -> None:
        self.count = 0

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        self.count += 1
        return f"%{self.count - 1}"


class _Translator:
    def __init__(self, width: int, names: _Names, env: Dict[str, str]) -> None:
        self.width = width
        self.type = f"i{width}"
        self.names = names
        self.lines: List[str] = []
        self.env = env  # variable name -> the SSA value that holds it now

    def emit(self, text: str) -> str:
        """Add an operation that produces one value and return that value's name."""
        name = next(self.names)
        self.lines.append(f"{name} = {text}")
        return name

    def constant(self, value: int) -> str:
        low, high = -(1 << (self.width - 1)), (1 << (self.width - 1)) - 1
        if not low <= value <= high:
            raise TranslationError(f"{value} does not fit in {self.type}")
        return self.emit(f"arith.constant {value} : {self.type}")

    def binary(self, node: BinOp) -> str:
        left, right = self.expression(node.left), self.expression(node.right)
        if node.op == "<":
            return self.emit(f"arith.cmpi slt, {left}, {right} : {self.type}")
        name = {"+": "arith.addi", "-": "arith.subi"}.get(node.op)
        if name is None:
            raise TranslationError(f"operator {node.op!r} is not supported yet")
        return self.emit(f"{name} {left}, {right} : {self.type}")

    def expression(self, node: Node) -> str:
        if isinstance(node, Number):
            if not isinstance(node.value, int):
                raise TranslationError("only integer constants are supported yet")
            return self.constant(node.value)
        if isinstance(node, Variable):
            if node.name not in self.env:
                raise TranslationError(f"{node.name} is used before it is assigned")
            return self.env[node.name]
        if isinstance(node, BinOp):
            return self.binary(node)
        if isinstance(node, Distribution):
            return self.emit(f'"dist.uniform"() <{{start = {node.start.value} : i32, stop = {node.stop.value} : i32}}> '
                             f": () -> {self.type}")
        raise TranslationError(f"{type(node).__name__} is not supported yet")

    def branch(self, statements: Sequence[Node]) -> "_Translator":
        """The statements of one arm, translated on a copy of the variables."""
        inner = _Translator(self.width, self.names, dict(self.env))
        for statement in statements:
            inner.statement(statement)
        return inner

    def conditional(self, node: If) -> None:
        condition = self.expression(node.condition)
        then, otherwise = self.branch(node.body), self.branch(node.orelse)
        changed = [name for name in self.env if then.env[name] != self.env[name] or otherwise.env[name] != self.env[name]]
        types = ", ".join([self.type] * len(changed))

        def arm(translated: "_Translator") -> List[str]:
            lines = translated.lines
            if changed:
                lines = lines + [f"scf.yield {', '.join(translated.env[name] for name in changed)} : {types}"]
            return ["  " + line for line in lines]

        if not changed:
            self.lines += [f"scf.if {condition} {{", *arm(then), "} else {", *arm(otherwise), "}"]
            return
        results = next(self.names)
        count = f":{len(changed)}" if len(changed) > 1 else ""
        self.lines += [f"{results}{count} = scf.if {condition} -> ({types}) {{", *arm(then), "} else {", *arm(otherwise), "}"]
        for index, name in enumerate(changed):
            self.env[name] = f"{results}#{index}" if len(changed) > 1 else results

    def statement(self, node: Node) -> None:
        if isinstance(node, Assign):
            self.env[node.name] = self.expression(node.value)
        elif isinstance(node, If):
            self.conditional(node)
        else:
            raise TranslationError(f"{type(node).__name__} is not supported yet")

    def program(self, program: Sequence[Node]) -> str:
        returned = None
        for statement in program:
            if isinstance(statement, Return):
                returned = self.expression(statement.value)
            else:
                self.statement(statement)
        if returned is None:
            raise TranslationError("a program must end with a return")
        body = ["  " + line for line in [*self.lines, f"func.return {returned} : {self.type}"]]
        return "\n".join([f"func.func @main() -> {self.type} {{", *body, "}"])


def translate(program: Sequence[Node], width: int = WIDTH) -> str:
    return _Translator(width, _Names(), {}).program(program)
