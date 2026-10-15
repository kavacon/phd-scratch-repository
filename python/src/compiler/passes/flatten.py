"""
Expression flattening, run after unroll and before balance and rename: one operation per step.

Each assignment becomes either a plain value or a single operation on plain values (numbers, booleans and
variables). Anything nested is pulled out into a temporary just before the statement that needs it:

    y = (a * 2) + 1;      becomes      $t1 = a * 2;  y = $t1 + 1;

An ISA instruction does one thing, so this is the shape the instructions will have. It also means every
intermediate is a temporary that liveness and release insertion see and release like any other.

- The outermost operation keeps the name the programmer chose, so `y` is not copied from a temporary.
- Return values, `if` conditions, call arguments and `~=` witnesses are flattened the same way. A returned
  expression is named so what it reads can be released, and a condition lives in a register of its own. A returned
  literal is named too (`return 2;` becomes `$t1 = 2; return $t1;`), so every program returns a variable and a
  later stage never has to deliver a bare constant.
- A chain `a == b == c` becomes pairwise comparisons joined with `&&`. Both sides of `&&` and `||` are
  evaluated, as they would be in a circuit.
- Temporaries are `$t1`, `$t2`, ... numbered from 1 in each function and in the top-level program. `$` cannot
  appear in a source name, so they never clash with the programmer's variables.
- Loops are expanded by the unroll pass first, so none reach this one.
"""
from typing import List, Optional, Tuple

from compiler.compiler import NODE, compile_pass
from compiler.system import System
from compiler.parser.grammar import (
    Assign,
    BinOp,
    Boolean,
    Call,
    Equals,
    ExprStmt,
    ForLoop,
    FunctionDef,
    If,
    Node,
    Not,
    Number,
    Return,
    Unassign,
    Variable,
)

_ATOMS = (Number, Boolean, Variable)


class FlattenError(Exception):
    pass


class _Flattener:
    """Flattens one scope (a function body, or the top-level program), numbering its temporaries."""

    def __init__(self) -> None:
        self.count = 0

    def fresh(self) -> str:
        self.count += 1
        return f"$t{self.count}"

    def atom(self, expr: Node, out: List[Node]) -> Node:
        """A plain value equal to `expr`. Any assignments needed to compute it are appended to `out`."""
        if isinstance(expr, _ATOMS):
            return expr
        operation = self.operation(expr, out)  # operands first, so temporaries are numbered in the order they are computed
        name = self.fresh()
        out.append(Assign(name, operation))
        return Variable(name)

    def operation(self, expr: Node, out: List[Node]) -> Node:
        """A single operation on plain values equal to `expr`, with its operands' assignments appended to `out`."""
        if isinstance(expr, BinOp):
            left, right = self.atom(expr.left, out), self.atom(expr.right, out)
            return BinOp(expr.op, left, right)
        if isinstance(expr, Not):
            return Not(self.atom(expr.operand, out))
        if isinstance(expr, Equals):
            operands = [self.atom(o, out) for o in expr.operands]
            pairs = [Equals((a, b)) for a, b in zip(operands, operands[1:], strict=False)]
            if len(pairs) == 1:
                return pairs[0]
            both = [self.atom(p, out) for p in pairs]  # each comparison in a register of its own
            result = both[0]
            for other in both[1:-1]:
                result = self.atom(BinOp("&&", result, other), out)
            return BinOp("&&", result, both[-1])
        if isinstance(expr, Call):
            return Call(expr.name, tuple(self.atom(a, out) for a in expr.args))
        return expr  # already a plain value

    def block(self, body: Tuple[Node, ...]) -> Tuple[Node, ...]:
        result: List[Node] = []
        for statement in body:
            self.statement(statement, result)
        return tuple(result)

    def statement(self, node: Node, out: List[Node]) -> None:
        if isinstance(node, Assign):
            out.append(Assign(node.name, self.operation(node.value, out), node.inplace))
        elif isinstance(node, ExprStmt):
            out.append(ExprStmt(self.operation(node.expr, out)))
        elif isinstance(node, Return):
            value = self.atom(node.value, out)
            if not isinstance(value, Variable):  # a literal has no register, so give it one
                name = self.fresh()
                out.append(Assign(name, value))
                value = Variable(name)
            out.append(Return(value))
        elif isinstance(node, Unassign):
            out.append(Unassign(node.name, self.operation(node.witness, out)))
        elif isinstance(node, If):
            condition = self.atom(node.condition, out)
            out.append(If(condition, self.block(node.body), self.block(node.orelse)))
        elif isinstance(node, FunctionDef):
            out.append(FunctionDef(node.name, node.params, node.return_type, _Flattener().block(node.body)))
        elif isinstance(node, ForLoop):
            raise FlattenError("loops must be unrolled before flattening")
        else:
            out.append(node)


@compile_pass(
    NODE, NODE,
    title="Flatten expressions",
    requires=("unroll",),
    summary="Puts every nested operation in a temporary of its own, so each intermediate value can be released.",
)
def flatten(program: List[Node], system: Optional[System] = None) -> List[Node]:
    return list(_Flattener().block(tuple(program)))
