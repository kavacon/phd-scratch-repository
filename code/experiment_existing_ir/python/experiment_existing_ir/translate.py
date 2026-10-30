"""
Translate a source program (the typed AST from `compiler.parser`) into MLIR text: the standard `func`, `arith` and
`scf` dialects for definite values, and the `dist` dialect for distributions, which the C++ tool in `cpp/` reads.

Every integer is given one fixed width, chosen by the front-end, the way a classical front-end gives every `int` its
declared width. That width is an upper bound: range analysis narrows it later. Results that do not fit wrap, as they do
in C. Change `WIDTH` (or pass `width`) to try another bound.

A variable is a distribution if it was made by a range (`x = 0..4;`) or computed from one. What assignment means for
one (see the decision "A distribution denotes a law"):
- a plain name on the right (`b = a;`) is an alias: both names are the same object, so an in-place update through either
  is seen through both. Definite values are copied, as integers are;
- a computed value (an expression involving a distribution) assigned to a new variable is marked with `dist.instance`,
  an independent distribution with the law of the expression. The translation only marks it: a pass, `resolve-instances`,
  decides how the instance is made;
- within one expression a distribution is one roll, so `a - a` is zero;
- `+=` and `-=` update the variable in place, including with another distribution;
- overwriting a distribution by a plain assignment would drop it and is refused.
A conditional on a distribution is a `dist.select`, which computes both arms; one on a definite value is an `scf.if`
that yields the variables either arm changed. A loop has literal bounds, so it is an `affine.for` that passes round the
variables its body changes, and a pass unrolls it before the lowering. Anything else is refused by name, so a later
milestone adds it deliberately.
"""
import itertools
from typing import Dict, Iterator, List, NamedTuple, Sequence, Tuple

from compiler.parser.grammar import Assign, BinOp, Distribution, ForLoop, If, Node, Number, Return, Variable

WIDTH = 32
MAX_DEPTH = 8  # loops nested in one another
MAX_ITERATIONS = 10_000  # times the innermost body runs, which is how many copies unrolling makes


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
    """
    Variables are names bound to cells, and a cell holds the current value. A definite value gets a cell of its own, so
    copying it copies the value. An alias binds a second name to a distribution's cell, so an update of the cell is seen
    through both names.
    """

    def __init__(self, width: int, names: _Names, cell_ids: Iterator[int], cells: Dict[int, _Value],
                 bindings: Dict[str, int]) -> None:
        self.width = width
        self.type = f"i{width}"
        self.names = names
        self.cell_ids = cell_ids
        self.lines: List[str] = []
        self.cells = cells
        self.bindings = bindings
        self.depth = 0  # loops this translator is inside
        self.iterations = 1  # how often its statements run, from the loops it is inside

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
        if name not in self.bindings:
            raise TranslationError(f"{name} is used before it is assigned")
        return self.cells[self.bindings[name]]

    def binary(self, node: BinOp) -> _Value:
        left, right = self.expression(node.left), self.expression(node.right)
        if left.dist or right.dist:
            name = {"+": "dist.addi", "-": "dist.subi", "<": "dist.slt"}.get(node.op)
            if name is None:
                raise TranslationError(f"operator {node.op!r} is not supported on a distribution yet")
            result = self.distribution("i1" if node.op == "<" else self.type)
            return self.emit(f"{name} {left.ssa}, {right.ssa} : {left.type}, {right.type} -> {result}", result)
        if node.op == "<":
            return self.emit(f"arith.cmpi slt, {left.ssa}, {right.ssa} : {self.type}", "i1")
        name = {"+": "arith.addi", "-": "arith.subi"}.get(node.op)
        if name is None:
            raise TranslationError(f"operator {node.op!r} is not supported yet")
        return self.emit(f"{name} {left.ssa}, {right.ssa} : {self.type}", self.type)

    def expression(self, node: Node) -> _Value:
        if isinstance(node, Number):
            if not isinstance(node.value, int):
                raise TranslationError("only integer constants are supported yet")
            return self.constant(node.value)
        if isinstance(node, Variable):
            return self.read(node.name)
        if isinstance(node, BinOp):
            return self.binary(node)
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
        inner = _Translator(self.width, self.names, self.cell_ids, dict(self.cells), dict(self.bindings))
        inner.depth, inner.iterations = self.depth, self.iterations
        for statement in statements:
            inner.statement(statement)
        return inner

    def conditional(self, node: If) -> None:
        condition = self.expression(node.condition)
        then, otherwise = self.branch(node.body), self.branch(node.orelse)
        changed = [cell for cell in self.cells if then.cells[cell] != self.cells[cell] or otherwise.cells[cell] != self.cells[cell]]
        if condition.dist:
            self.select(condition, changed, then, otherwise)
        else:
            self.branch_on_definite(condition, changed, then, otherwise)

    def select(self, condition: _Value, changed: List[int], then: "_Translator", otherwise: "_Translator") -> None:
        """Both arms are computed, and one `dist.select` picks each changed cell's value, using the condition once."""
        self.lines += then.lines + otherwise.lines
        if not changed:
            return
        trues, falses = [then.cells[cell] for cell in changed], [otherwise.cells[cell] for cell in changed]
        result_type = self.distribution(self.type)
        results = next(self.names)
        count = f":{len(changed)}" if len(changed) > 1 else ""
        join = lambda values, key: ", ".join(key(value) for value in values)
        self.lines.append(
            f"{results}{count} = dist.select {condition.ssa}, ({join(trues, lambda v: v.ssa)}), "
            f"({join(falses, lambda v: v.ssa)}) : {condition.type}, ({join(trues, lambda v: v.type)}), "
            f"({join(falses, lambda v: v.type)}) -> ({', '.join([result_type] * len(changed))})")
        for index, cell in enumerate(changed):
            self.cells[cell] = _Value(f"{results}#{index}" if len(changed) > 1 else results, result_type)

    def branch_on_definite(self, condition: _Value, changed: List[int], then: "_Translator",
                           otherwise: "_Translator") -> None:
        if any(arm.cells[cell].dist or self.cells[cell].dist for arm in (then, otherwise) for cell in changed):
            raise TranslationError("a conditional on a definite value that changes a distribution is not supported yet")
        types = ", ".join([self.type] * len(changed))

        def arm(translated: "_Translator") -> List[str]:
            lines = translated.lines
            if changed:
                lines = lines + [f"scf.yield {', '.join(translated.cells[cell].ssa for cell in changed)} : {types}"]
            return ["  " + line for line in lines]

        if not changed:
            self.lines += [f"scf.if {condition.ssa} {{", *arm(then), "} else {", *arm(otherwise), "}"]
            return
        results = next(self.names)
        count = f":{len(changed)}" if len(changed) > 1 else ""
        self.lines += [f"{results}{count} = scf.if {condition.ssa} -> ({types}) {{", *arm(then), "} else {",
                       *arm(otherwise), "}"]
        for index, cell in enumerate(changed):
            self.cells[cell] = _Value(f"{results}#{index}" if len(changed) > 1 else results, self.type)

    def loop_body(self, node: ForLoop, names: _Names, cell_ids: Iterator[int],
                  carried: Dict[int, _Value]) -> "Tuple[_Translator, str, _Value]":
        """
        The body translated once on a copy of the variables. The loop variable is a fresh cell holding the induction
        variable cast to the program's integer type, and `carried` replaces the cells the loop passes round.
        """
        inner = _Translator(self.width, names, cell_ids, dict(self.cells), dict(self.bindings))
        inner.cells.update(carried)
        inner.depth, inner.iterations = self.depth + 1, self.iterations * max(node.stop.value - node.start.value, 1)
        if inner.depth > MAX_DEPTH or inner.iterations > MAX_ITERATIONS:
            raise TranslationError(f"loops nested {MAX_DEPTH} deep or running {MAX_ITERATIONS} iterations in all cannot "
                                   "be unrolled")
        index = next(names)
        variable = inner.emit(f"arith.index_cast {index} : index to {self.type}", self.type)
        inner.bindings[node.var] = next(cell_ids)
        inner.cells[inner.bindings[node.var]] = variable
        for statement in node.body:
            inner.statement(statement)
        if inner.cells[inner.bindings[node.var]] != variable:
            raise TranslationError(f"the loop variable {node.var} cannot be assigned to inside its loop")
        return inner, index, variable

    def loop(self, node: ForLoop) -> None:
        """
        A loop with literal bounds is an `affine.for`, whose bounds must be static, so the trip count is classical. It
        passes round the variables the body changes, and a pass unrolls it before the lowering. A first translation, with
        throwaway names, finds those variables; a loop that runs no times is translated for its errors and then dropped.
        """
        start, stop = node.start.value, node.stop.value
        probe, _, _ = self.loop_body(node, _Names(), itertools.count(1 << 20), {})
        for cell, value in probe.cells.items():
            if value.dist and (cell not in self.cells or value != self.cells[cell]):
                raise TranslationError("a loop that changes or makes a distribution is not supported yet")
        carried = [cell for cell in self.cells if probe.cells[cell] != self.cells[cell]]
        if stop <= start:
            return
        results = next(self.names) if carried else ""
        arguments = {cell: _Value(next(self.names), self.type) for cell in carried}
        inner, index, _ = self.loop_body(node, self.names, self.cell_ids, arguments)
        header = f"affine.for {index} = {start} to {stop}"
        body = ["  " + line for line in inner.lines]
        if not carried:
            self.lines += [header + " {", *body, "}"]
            return
        types = ", ".join([self.type] * len(carried))
        passed = ", ".join(f"{arguments[cell].ssa} = {self.cells[cell].ssa}" for cell in carried)
        body.append(f"  affine.yield {', '.join(inner.cells[cell].ssa for cell in carried)} : {types}")
        count = f":{len(carried)}" if len(carried) > 1 else ""
        self.lines += [f"{results}{count} = {header} iter_args({passed}) -> ({types}) {{", *body, "}"]
        for position, cell in enumerate(carried):
            self.cells[cell] = _Value(f"{results}#{position}" if len(carried) > 1 else results, self.type)

    def update(self, node: Assign) -> None:
        """`name += amount;` or `name -= amount;`, which the parser gives as `name = name + amount` marked in place."""
        cell = self.bindings.get(node.name)
        if cell is None:
            raise TranslationError(f"{node.name} is used before it is assigned")
        target, amount = self.cells[cell], self.expression(node.value.right)
        name = {"+": "add_assign", "-": "sub_assign"}[node.value.op]
        if not (target.dist or amount.dist):
            self.cells[cell] = self.emit(f"arith.{'addi' if name == 'add_assign' else 'subi'} {target.ssa}, "
                                         f"{amount.ssa} : {self.type}", self.type)
            return
        if target.ssa == amount.ssa:
            raise TranslationError(f"{node.name} {node.value.op}= ... adds a distribution to itself, which is not "
                                   "supported yet")
        result = self.distribution(self.type)
        self.cells[cell] = self.emit(
            f"dist.{name} {target.ssa}, {amount.ssa} : {target.type}, {amount.type} -> {result}", result)

    def assign(self, node: Assign) -> None:
        held = self.cells.get(self.bindings.get(node.name, -1))
        if held is not None and held.dist:
            raise TranslationError(f"assigning to {node.name} would drop the distribution it holds; only an in-place "
                                   "update changes one")
        if isinstance(node.value, Variable) and self.read(node.value.name).dist:  # an alias of a distribution
            self.bindings[node.name] = self.bindings[node.value.name]
            return
        value = self.expression(node.value)
        if value.dist and not isinstance(node.value, Distribution):  # a computed value: an independent instance
            result = value.type
            value = self.emit(f"dist.instance {value.ssa} : {result} -> {result}", result)
        if node.name in self.bindings:  # a definite name owns its cell, so this overwrites its value
            self.cells[self.bindings[node.name]] = value
        else:
            self.bindings[node.name] = next(self.cell_ids)
            self.cells[self.bindings[node.name]] = value

    def statement(self, node: Node) -> None:
        if isinstance(node, Assign) and node.inplace:
            self.update(node)
        elif isinstance(node, Assign):
            self.assign(node)
        elif isinstance(node, If):
            self.conditional(node)
        elif isinstance(node, ForLoop):
            self.loop(node)
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
        if returned.dist:  # using a distribution as the result measures it
            returned = self.emit(f"dist.measure {returned.ssa} : {returned.type} -> {self.type}", self.type)
        body = ["  " + line for line in [*self.lines, f"func.return {returned.ssa} : {self.type}"]]
        return "\n".join([f"func.func @main() -> {self.type} {{", *body, "}"])


def translate(program: Sequence[Node], width: int = WIDTH) -> str:
    return _Translator(width, _Names(), itertools.count(), {}, {}).program(program)
