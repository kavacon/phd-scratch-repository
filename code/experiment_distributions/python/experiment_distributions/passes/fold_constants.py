"""
Constant folding, before rename: a variable whose value is known is replaced by that value, and an expression of known
values is evaluated. The assignments left unread are dropped by the pass after this one.

A name can be assigned more than once here, so what is known is tracked as the program runs and forgotten when the name
is assigned something unknown, and, after an `if` that cannot be decided, for every name either branch assigns. A program
has no inputs, so every value that does not depend on a distribution is known, and this removes those chains before they
become registers once the unread ones are dropped. Addition, subtraction and multiplication are kept exact: wrapping at
the register width happens when the program is lowered or run, so the rewrite does not depend on the width. Comparisons
are the exception, because there the width changes the answer (a sum that wraps to a negative number is less than zero),
so they read their operands at the system's register width, as the machine does. A range one value wide is a constant.
An `if` on a known condition is replaced by the branch that runs.

Left alone: division, which is not exact; the returned variable, which stays a variable because the instruction lowering
needs a register to return; and an identity (`x = x`, which balancing gives the arm of an `if` that does not assign x),
which says the variable is unchanged and is not folded into a constant, so the lowering can treat it as no operation.
"""
import operator
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Union

from checking.simulate import normaliser
from compiler.compiler import NODE, compile_pass
from compiler.parser.grammar import (
    Assign,
    BinOp,
    Boolean,
    Distribution,
    Equals,
    ExprStmt,
    If,
    Node,
    Not,
    Number,
    Variable,
)
from compiler.passes.names import assigned
from compiler.system import System

Known = Dict[str, Union[int, bool]]
_ARITHMETIC = {"+": operator.add, "-": operator.sub, "*": operator.mul}
_COMPARISON = {"<": operator.lt, ">": operator.gt, "!=": operator.ne}
_LOGICAL = {"&&": lambda a, b: a and b, "||": lambda a, b: a or b}


def _value(node: Node) -> Optional[Union[int, bool]]:
    """The constant a node is, or None. Decimals are not folded."""
    if isinstance(node, Boolean) or (isinstance(node, Number) and isinstance(node.value, int)):
        return node.value
    return None


def _node(value: Union[int, bool]) -> Node:
    return Boolean(value) if isinstance(value, bool) else Number(value)


def _fold(node: Node, known: Known, norm) -> Node:
    if isinstance(node, Variable):
        return _node(known[node.name]) if node.name in known else node
    if isinstance(node, Distribution):
        return Number(node.start.value) if node.stop.value - node.start.value == 1 else node
    if isinstance(node, BinOp):
        left, right = _fold(node.left, known, norm), _fold(node.right, known, norm)
        a, b = _value(left), _value(right)
        if a is not None and b is not None:
            if node.op in _ARITHMETIC:
                return _node(_ARITHMETIC[node.op](a, b))
            if node.op in _COMPARISON:
                return _node(_COMPARISON[node.op](norm(a), norm(b)))
            if node.op in _LOGICAL:
                return _node(_LOGICAL[node.op](a, b))
        return replace(node, left=left, right=right)
    if isinstance(node, Not):
        operand = _fold(node.operand, known, norm)
        return _node(not _value(operand)) if _value(operand) is not None else replace(node, operand=operand)
    if isinstance(node, Equals):
        operands = tuple(_fold(o, known, norm) for o in node.operands)
        values = [_value(o) for o in operands]
        if all(v is not None for v in values):
            return _node(all(norm(x) == norm(y) for x, y in zip(values, values[1:], strict=False)))
        return replace(node, operands=operands)
    return node


def _block(body: Sequence[Node], known: Known, norm) -> List[Node]:
    out: List[Node] = []
    for statement in body:
        if isinstance(statement, Assign) and isinstance(statement.value, Variable) and statement.value.name == statement.name:
            out.append(statement)  # an identity (`x = x`) says the variable is unchanged, and is left for the lowering
        elif isinstance(statement, Assign):
            value = _fold(statement.value, known, norm)
            kept_in_place = (statement.inplace and isinstance(value, BinOp) and isinstance(value.left, Variable)
                             and value.left.name == statement.value.left.name)
            out.append(replace(statement, value=value, inplace=kept_in_place))
            if _value(value) is not None:
                known[statement.name] = _value(value)
            else:
                known.pop(statement.name, None)
        elif isinstance(statement, If):
            condition = _fold(statement.condition, known, norm)
            if _value(condition) is not None:  # decided now: only the branch that runs is kept
                out += _block(statement.body if _value(condition) else statement.orelse, known, norm)
            else:  # what a branch learns is true only inside it, and what it assigns is unknown after it
                out.append(replace(
                    statement, condition=condition,
                    body=tuple(_block(statement.body, dict(known), norm)),
                    orelse=tuple(_block(statement.orelse, dict(known), norm)),
                ))
                for name in assigned(statement.body + statement.orelse):
                    known.pop(name, None)
        elif isinstance(statement, ExprStmt):
            out.append(replace(statement, expr=_fold(statement.expr, known, norm)))
        else:
            out.append(statement)
    return out


@compile_pass(
    NODE, NODE,
    title="Fold constants",
    requires=("unroll",),
    summary=(
        "Replaces known values with their value, evaluates constant expressions exactly (comparisons at the "
        "register width), and resolves conditionals on a known value."
    ),
)
def fold_constants(program: List[Node], system: Optional[System] = None) -> List[Node]:
    return _block(program, {}, normaliser(system or System()))
