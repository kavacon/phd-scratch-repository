"""
The intended meaning of a program, worked out exactly from the source and nothing else.

This follows the semantics decided for distributions (the decision "A distribution denotes a law"), and shares no code with
the translator or the passes, so agreement with it says something about them:
- an integer is a value and is copied; a distribution is an object, so a plain name on the right of an assignment is an alias;
- a range makes a fresh distribution; a computed distribution assigned to a variable is an independent instance with the
  same law, made by replaying its history with fresh dice;
- within one expression a distribution is one roll, so `a - a` is 0;
- `+=` and `-=` change the variable's object in place, whoever else names it, and keep it tied to what was added;
- a conditional on a distribution computes both arms and ties every variable it changes to the condition;
- a function gets copies of integers and the caller's own object for a distribution; a function that returns a distribution it
  holds returns that object, and one that returns a computed value returns a value, which an assignment makes an instance.

A distribution is a tree over independent dice, so the outcome law of the returned value is found by running through every
combination of the dice it depends on. Programs here are small, so that is cheap and exact.
"""
import itertools
from typing import Dict, List, Optional, Sequence, Tuple, Union

from compiler.parser.grammar import (Assign, BinOp, Boolean, Call, Distribution, Equals, ExprStmt, ForLoop, FunctionDef, If,
                                     Node, Not, Number, Return, Variable)

WIDTH = 32
MAX_COMBINATIONS = 300_000

# A distribution: ("const", v), ("roll", id, low, high), ("op", symbol, left, right) or ("select", condition, then, else).
Tree = tuple
Value = Union[int, Tree]


class ReferenceError_(ValueError):
    """A program the intended semantics does not allow, or one this interpreter does not model."""


def wrap(value: int) -> int:
    half = 1 << (WIDTH - 1)
    return (value + half) % (1 << WIDTH) - half


def is_dist(value: Value) -> bool:
    return isinstance(value, tuple)


def as_tree(value: Value) -> Tree:
    return value if is_dist(value) else ("const", value)


OPERATORS = {
    "+": lambda a, b: wrap(a + b), "-": lambda a, b: wrap(a - b), "*": lambda a, b: wrap(a * b),
    "<": lambda a, b: int(a < b), ">": lambda a, b: int(a > b), "==": lambda a, b: int(a == b), "!=": lambda a, b: int(a != b),
    "&&": lambda a, b: int(bool(a) and bool(b)), "||": lambda a, b: int(bool(a) or bool(b)),
}


def binary(op: str, left: Value, right: Value) -> Value:
    """Both operands of `&&` and `||` are always evaluated, as in the circuit; each counts as true when it is not zero."""
    if op not in OPERATORS:
        raise ReferenceError_(f"operator {op!r} is not modelled")
    if not is_dist(left) and not is_dist(right):
        return OPERATORS[op](left, right)
    return ("op", op, as_tree(left), as_tree(right))


def rolls(tree: Tree, found: Optional[Dict[int, Tuple[int, int]]] = None) -> Dict[int, Tuple[int, int]]:
    found = {} if found is None else found
    if tree[0] == "roll":
        found[tree[1]] = (tree[2], tree[3])
    elif tree[0] in ("op", "select"):
        for child in tree[2:] if tree[0] == "op" else tree[1:]:
            rolls(child, found)
    return found


def evaluate(tree: Tree, dice: Dict[int, int]) -> int:
    kind = tree[0]
    if kind == "const":
        return tree[1]
    if kind == "roll":
        return dice[tree[1]]
    if kind == "select":
        return evaluate(tree[2], dice) if evaluate(tree[1], dice) else evaluate(tree[3], dice)
    left, right = evaluate(tree[2], dice), evaluate(tree[3], dice)
    return binary(tree[1], left, right)


class Cell:
    """The object a variable names. Two names for a distribution share one cell."""

    def __init__(self, value: Value) -> None:
        self.value = value


class Interpreter:
    def __init__(self, program: Sequence[Node]) -> None:
        self.functions = {node.name: node for node in program if isinstance(node, FunctionDef)}
        self.next_die = itertools.count()
        self.depth = 0

    # -- distributions ---------------------------------------------------------------------------------------------

    def fresh(self, low: int, high: int) -> Tree:
        return ("roll", next(self.next_die), low, high)

    def replay(self, tree: Tree, fresh: Optional[Dict[int, Tree]] = None) -> Tree:
        """The same law on dice of its own: every die in the history is a new one, shared dice staying shared."""
        fresh = {} if fresh is None else fresh
        kind = tree[0]
        if kind == "const":
            return tree
        if kind == "roll":
            if tree[1] not in fresh:
                fresh[tree[1]] = self.fresh(tree[2], tree[3])
            return fresh[tree[1]]
        if kind == "select":
            return ("select", self.replay(tree[1], fresh), self.replay(tree[2], fresh), self.replay(tree[3], fresh))
        return ("op", tree[1], self.replay(tree[2], fresh), self.replay(tree[3], fresh))

    # -- expressions -----------------------------------------------------------------------------------------------

    def expression(self, node: Node, scope: Dict[str, Cell]) -> Value:
        if isinstance(node, Number):
            return node.value
        if isinstance(node, Variable):
            if node.name not in scope:
                raise ReferenceError_(f"{node.name} is used before it is assigned")
            return scope[node.name].value
        if isinstance(node, BinOp):
            return binary(node.op, self.expression(node.left, scope), self.expression(node.right, scope))
        if isinstance(node, Boolean):
            return int(node.value)
        if isinstance(node, Not):
            return binary("==", self.expression(node.operand, scope), 0)
        if isinstance(node, Equals):
            values = [self.expression(operand, scope) for operand in node.operands]
            result = binary("==", values[0], values[1])
            for left, right in zip(values[1:], values[2:]):
                result = binary("&&", result, binary("==", left, right))
            return result
        if isinstance(node, Distribution):
            start, stop = node.start.value, node.stop.value
            return start if stop - start == 1 else self.fresh(start, stop)
        if isinstance(node, Call):
            kind, payload, _ = self.call(node, scope)
            return payload.value if kind == "cell" else payload
        raise ReferenceError_(f"{type(node).__name__} is not modelled")

    def call(self, node: Call, scope: Dict[str, Cell]) -> Tuple[str, Union[Cell, Value], bool]:
        """("cell", the object returned, _) or ("value", the value, whether it is a fresh range)."""
        definition = self.functions.get(node.name)
        if definition is None:
            raise ReferenceError_(f"{node.name} is not a function")
        if len(node.args) != len(definition.params):
            raise ReferenceError_(f"{node.name} takes {len(definition.params)} argument(s)")
        local: Dict[str, Cell] = {}
        for parameter, argument in zip(definition.params, node.args):
            if isinstance(argument, Variable) and argument.name in scope and is_dist(scope[argument.name].value):
                local[parameter.name] = scope[argument.name]  # the caller's own object
                continue
            value = self.expression(argument, scope)
            if is_dist(value) and not isinstance(argument, Distribution):
                value = self.replay(value)
            local[parameter.name] = Cell(value)
        if self.depth > 20:
            raise ReferenceError_(f"{node.name} calls itself")
        self.depth += 1
        try:
            *statements, final = definition.body
            for statement in statements:
                self.statement(statement, local)
            returned = final.value
            if isinstance(returned, Variable) and returned.name in local and is_dist(local[returned.name].value):
                return "cell", local[returned.name], False
            if isinstance(returned, Call):
                return self.call(returned, local)
            return "value", self.expression(returned, local), isinstance(returned, Distribution)
        finally:
            self.depth -= 1

    # -- statements ------------------------------------------------------------------------------------------------

    def statement(self, node: Node, scope: Dict[str, Cell]) -> None:
        if isinstance(node, Assign) and node.inplace:
            cell = scope.get(node.name)
            if cell is None:
                raise ReferenceError_(f"{node.name} is used before it is assigned")
            if node.name in self.readonly:
                raise ReferenceError_(f"the loop variable {node.name} cannot be assigned to")
            cell.value = binary(node.value.op, cell.value, self.expression(node.value.right, scope))
        elif isinstance(node, Assign):
            self.assign(node, scope)
        elif isinstance(node, If):
            self.conditional(node, scope)
        elif isinstance(node, ForLoop):
            self.loop(node, scope)
        elif isinstance(node, ExprStmt):
            self.expression(node.expr, scope)
        else:
            raise ReferenceError_(f"{type(node).__name__} is not modelled")

    def assign(self, node: Assign, scope: Dict[str, Cell]) -> None:
        held = scope.get(node.name)
        if held is not None and is_dist(held.value):
            raise ReferenceError_(f"assigning to {node.name} would drop the distribution it holds")
        if node.name in self.readonly:
            raise ReferenceError_(f"the loop variable {node.name} cannot be assigned to")
        value = node.value
        if isinstance(value, Variable) and value.name in scope and is_dist(scope[value.name].value):
            scope[node.name] = scope[value.name]  # an alias
            return
        if isinstance(value, Call):
            kind, payload, fresh = self.call(value, scope)
            if kind == "cell":
                scope[node.name] = payload
                return
            result = payload
        else:
            result, fresh = self.expression(value, scope), isinstance(value, Distribution)
        if is_dist(result) and not fresh:
            result = self.replay(result)
        if held is not None:
            held.value = result
        else:
            scope[node.name] = Cell(result)

    def inline(self, statements: Sequence[Node], scope: Dict[str, Cell]) -> None:
        before = set(scope)
        for statement in statements:
            self.statement(statement, scope)
        for name in [n for n in scope if n not in before]:
            del scope[name]

    def conditional(self, node: If, scope: Dict[str, Cell]) -> None:
        condition = self.expression(node.condition, scope)
        if not is_dist(condition):
            self.inline(node.body if condition else node.orelse, scope)
            return
        arms = []
        for statements in (node.body, node.orelse):
            copies: Dict[int, Cell] = {}
            arm = {name: copies.setdefault(id(cell), Cell(cell.value)) for name, cell in scope.items()}
            for statement in statements:
                self.statement(statement, arm)
            if any(arm.get(name) is not copies[id(cell)] for name, cell in scope.items()):
                raise ReferenceError_("a name is rebound inside a conditional on a distribution")
            arms.append(copies)
        for cell_id, cell in {id(c): c for c in scope.values()}.items():
            then, otherwise = arms[0][cell_id].value, arms[1][cell_id].value
            if then != cell.value or otherwise != cell.value:
                cell.value = ("select", as_tree(condition), as_tree(then), as_tree(otherwise))

    def loop(self, node: ForLoop, scope: Dict[str, Cell]) -> None:
        saved = scope.get(node.var)
        for k in range(node.start.value, node.stop.value):
            scope[node.var] = Cell(k)
            self.readonly.add(node.var)
            self.inline(node.body, scope)
            self.readonly.discard(node.var)
        if saved is not None:
            scope[node.var] = saved
        else:
            scope.pop(node.var, None)

    readonly: set = set()

    def run(self, program: Sequence[Node]) -> Value:
        self.readonly = set()
        scope: Dict[str, Cell] = {}
        returned = None
        for node in program:
            if isinstance(node, Return):
                returned = self.expression(node.value, scope)
            elif not isinstance(node, FunctionDef):
                self.statement(node, scope)
        if returned is None:
            raise ReferenceError_("a program must end with a return")
        return returned


def law(program: Sequence[Node]) -> Dict[int, float]:
    """The chance of each value the program can return, worked out exactly."""
    value = Interpreter(program).run(program)
    if not is_dist(value):
        return {value: 1.0}
    dice = rolls(value)
    count = 1
    for low, high in dice.values():
        count *= high - low
    if count > MAX_COMBINATIONS:
        raise ReferenceError_("too many combinations of dice to enumerate")
    result: Dict[int, float] = {}
    ids = sorted(dice)
    for values in itertools.product(*[range(*dice[i]) for i in ids]):
        outcome = evaluate(value, dict(zip(ids, values)))
        result[outcome] = result.get(outcome, 0) + 1 / count
    return result
