"""
Translate a source program (the typed AST from `compiler.parser`) into MLIR text: the standard `func`, `arith` and `scf`
dialects for definite values, and the `dist` dialect for distributions, which the C++ tool in `cpp/` reads.

Every integer is given one fixed width, chosen by the front-end, the way a classical front-end gives every `int` its
declared width. That width is an upper bound: range analysis narrows it later. Results that do not fit wrap, as they do
in C. Change `WIDTH` (or pass `width`) to try another bound.

A variable is a distribution if it was made by a range (`x = 0..4;`) or changed by an in-place update with one. For now
that is all that can be done to a distribution: it is created by a range, changed by `+=` or `-=` (the target is
consumed, the amount is only read), compared in a condition, and read at the return, which measures it. Any other
assignment involving a distribution is refused (`y = x;`, `y = x + 1;`, overwriting one), because what assignment means for
a distribution, a copy or the same value, is not settled. Reading a distribution again is fine: readers leave it as it was.
A conditional on a distribution is a `dist.select`, which computes both arms; one on a definite value is an `scf.if`
that yields the variables either arm changed. Anything else is refused by name, so a later milestone adds it deliberately.
"""
from typing import Dict, Iterator, List, NamedTuple, Sequence

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


class _Value(NamedTuple):
    ssa: str
    type: str  # "i32", "i1", "!dist.dist<i32>" or "!dist.dist<i1>"

    @property
    def dist(self) -> bool:
        return self.type.startswith("!dist.dist")


class _Translator:
    def __init__(self, width: int, names: _Names, env: Dict[str, _Value]) -> None:
        self.width = width
        self.type = f"i{width}"
        self.names = names
        self.lines: List[str] = []
        self.env = env  # variable name -> the value that holds it now

    def emit(self, text: str, type: str) -> _Value:
        """Add an operation that produces one value."""
        name = next(self.names)
        self.lines.append(f"{name} = {text}")
        return _Value(name, type)

    def distribution(self, element: str) -> str:
        return f"!dist.dist<{element}>"

    def constant(self, value: int) -> _Value:
        low, high = -(1 << (self.width - 1)), (1 << (self.width - 1)) - 1
        if not low <= value <= high:
            raise TranslationError(f"{value} does not fit in {self.type}")
        return self.emit(f"arith.constant {value} : {self.type}", self.type)

    def read(self, name: str) -> _Value:
        if name not in self.env:
            raise TranslationError(f"{name} is used before it is assigned")
        return self.env[name]

    def binary(self, node: BinOp, in_condition: bool) -> _Value:
        left, right = self.expression(node.left), self.expression(node.right)
        if left.dist or right.dist:
            if node.op != "<" or not in_condition:
                raise TranslationError(f"'{node.op}' on a distribution is only supported as an in-place update (+= or "
                                       "-=) or, for '<', in a condition")
            result = self.distribution("i1")
            return self.emit(f"dist.slt {left.ssa}, {right.ssa} : {left.type}, {right.type} -> {result}", result)
        if node.op == "<":
            return self.emit(f"arith.cmpi slt, {left.ssa}, {right.ssa} : {self.type}", "i1")
        name = {"+": "arith.addi", "-": "arith.subi"}.get(node.op)
        if name is None:
            raise TranslationError(f"operator {node.op!r} is not supported yet")
        return self.emit(f"{name} {left.ssa}, {right.ssa} : {self.type}", self.type)

    def expression(self, node: Node, in_condition: bool = False) -> _Value:
        if isinstance(node, Number):
            if not isinstance(node.value, int):
                raise TranslationError("only integer constants are supported yet")
            return self.constant(node.value)
        if isinstance(node, Variable):
            return self.read(node.name)
        if isinstance(node, BinOp):
            return self.binary(node, in_condition)
        if isinstance(node, Distribution):
            start, stop = node.start.value, node.stop.value
            if stop <= start:
                raise TranslationError(f"the range {start}..{stop} is empty")
            if stop - start == 1:  # a range one value wide is an ordinary definite value
                return self.constant(start)
            return self.emit(f"dist.uniform {start}, {stop} : {self.distribution(self.type)}", self.distribution(self.type))
        raise TranslationError(f"{type(node).__name__} is not supported yet")

    def branch(self, statements: Sequence[Node]) -> "_Translator":
        """The statements of one arm, translated on a copy of the variables."""
        inner = _Translator(self.width, self.names, dict(self.env))
        for statement in statements:
            inner.statement(statement)
        return inner

    def conditional(self, node: If) -> None:
        condition = self.expression(node.condition, in_condition=True)
        then, otherwise = self.branch(node.body), self.branch(node.orelse)
        changed = [name for name in self.env if then.env[name] != self.env[name] or otherwise.env[name] != self.env[name]]
        if condition.dist:
            self.select(condition, changed, then, otherwise)
        else:
            self.branch_on_definite(condition, changed, then, otherwise)

    def select(self, condition: _Value, changed: List[str], then: "_Translator", otherwise: "_Translator") -> None:
        """Both arms are computed, and one `dist.select` picks each changed variable's value, reading the condition once."""
        self.lines += then.lines + otherwise.lines
        if not changed:
            return
        trues, falses = [then.env[name] for name in changed], [otherwise.env[name] for name in changed]
        result_type = self.distribution(self.type)
        results = next(self.names)
        count = f":{len(changed)}" if len(changed) > 1 else ""
        join = lambda values, key: ", ".join(key(value) for value in values)
        self.lines.append(
            f"{results}{count} = dist.select {condition.ssa}, ({join(trues, lambda v: v.ssa)}), "
            f"({join(falses, lambda v: v.ssa)}) : {condition.type}, ({join(trues, lambda v: v.type)}), "
            f"({join(falses, lambda v: v.type)}) -> ({', '.join([result_type] * len(changed))})")
        for index, name in enumerate(changed):
            self.env[name] = _Value(f"{results}#{index}" if len(changed) > 1 else results, result_type)

    def branch_on_definite(self, condition: _Value, changed: List[str], then: "_Translator",
                           otherwise: "_Translator") -> None:
        if any(arm.env[name].dist or self.env[name].dist for arm in (then, otherwise) for name in changed):
            raise TranslationError("a conditional on a definite value that changes a distribution is not supported yet")
        types = ", ".join([self.type] * len(changed))

        def arm(translated: "_Translator") -> List[str]:
            lines = translated.lines
            if changed:
                lines = lines + [f"scf.yield {', '.join(translated.env[name].ssa for name in changed)} : {types}"]
            return ["  " + line for line in lines]

        if not changed:
            self.lines += [f"scf.if {condition.ssa} {{", *arm(then), "} else {", *arm(otherwise), "}"]
            return
        results = next(self.names)
        count = f":{len(changed)}" if len(changed) > 1 else ""
        self.lines += [f"{results}{count} = scf.if {condition.ssa} -> ({types}) {{", *arm(then), "} else {",
                       *arm(otherwise), "}"]
        for index, name in enumerate(changed):
            self.env[name] = _Value(f"{results}#{index}" if len(changed) > 1 else results, self.type)

    def update(self, node: Assign) -> None:
        """`name += amount;` or `name -= amount;`, which the parser gives as `name = name + amount` marked in place."""
        target, amount = self.read(node.name), self.expression(node.value.right)
        name = {"+": "add_assign", "-": "sub_assign"}[node.value.op]
        if not (target.dist or amount.dist):
            self.env[node.name] = self.emit(f"arith.{'addi' if name == 'add_assign' else 'subi'} {target.ssa}, "
                                            f"{amount.ssa} : {self.type}", self.type)
            return
        if target.ssa == amount.ssa:
            raise TranslationError(f"{node.name} {node.value.op}= {node.name} adds a distribution to itself, which is not "
                                   "supported yet")
        result = self.distribution(self.type)
        self.env[node.name] = self.emit(
            f"dist.{name} {target.ssa}, {amount.ssa} : {target.type}, {amount.type} -> {result}", result)

    def statement(self, node: Node) -> None:
        if isinstance(node, Assign) and node.inplace:
            self.update(node)
        elif isinstance(node, Assign):
            if self.env.get(node.name, _Value("", "")).dist:
                raise TranslationError(f"assigning to {node.name} would drop the distribution it holds; only an in-place "
                                       "update changes one yet")
            value = self.expression(node.value)
            if value.dist and not isinstance(node.value, Distribution):
                raise TranslationError(f"{node.name} = ... would copy or derive a distribution; only a range or an "
                                       "in-place update (+=, -=) creates or changes one yet")
            self.env[node.name] = value
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
        if returned.dist:  # reading a distribution measures it
            returned = self.emit(f"dist.measure {returned.ssa} : {returned.type} -> {self.type}", self.type)
        body = ["  " + line for line in [*self.lines, f"func.return {returned.ssa} : {self.type}"]]
        return "\n".join([f"func.func @main() -> {self.type} {{", *body, "}"])


def translate(program: Sequence[Node], width: int = WIDTH) -> str:
    return _Translator(width, _Names(), {}).program(program)
