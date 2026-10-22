"""
Combine constant steps: `t = a * 3; u = t * 5;` becomes `u = a * 15;`, and `b = a + 1; d = b + 3;` becomes `d = a + 4;`,
when the intermediate is read only by the second step. The constants are combined as the integers they are, so the rewrite
says the same thing as the program did and does not depend on the register width: wrapping happens when the program is
lowered or run, and combining two adds or two multiplications commutes with it, so the result is the same. The
intermediate register is never needed. A multiplication by an odd constant is invertible, so the source can then be
released by computing it back from the result.

Only two steps of the same kind combine, because the result must still be one operation (a multiply and an add make a
two-operation expression the program does not have). Both steps must be at the top level, the source must not be assigned
again between them, and the intermediate must be assigned once and read once. Run before the update pass, which
turns the combined add into an in-place update.
"""
from dataclasses import replace
from typing import List, NamedTuple, Optional, Tuple

from compiler.compiler import NODE, compile_pass
from compiler.parser.ast_tools import walk
from compiler.parser.grammar import Assign, BinOp, Node, Number, Variable
from compiler.passes.names import assigned
from compiler.system import System


class Step(NamedTuple):
    kind: str  # "add" or "mul"
    source: str
    constant: int


def _constant(node: Node) -> Optional[int]:
    return node.value if isinstance(node, Number) and isinstance(node.value, int) and not isinstance(node.value, bool) else None


def _step(statement: Node) -> Optional[Step]:
    """`name = source + c`, `source - c`, `c + source`, or the same with `*`, as a step on `source`."""
    if not isinstance(statement, Assign) or statement.inplace or not isinstance(statement.value, BinOp):
        return None
    left, right, op = statement.value.left, statement.value.right, statement.value.op
    for variable, other, swapped in ((left, right, False), (right, left, True)):
        c = _constant(other)
        if isinstance(variable, Variable) and c is not None and variable.name != statement.name:
            if op == "+":
                return Step("add", variable.name, c)
            if op == "*":
                return Step("mul", variable.name, c)
            if op == "-" and not swapped:
                return Step("add", variable.name, -c)
    return None


def _count(body: List[Node], name: str, kind) -> int:
    """How many times `name` is read (a Variable) or assigned, anywhere in the program."""
    if kind == "read":
        return sum(1 for s in body for n in walk(s) if isinstance(n, Variable) and n.name == name)
    return sum(1 for s in body for n in walk(s) if isinstance(n, Assign) and n.name == name)


def _expression(step: Step) -> Node:
    if step.kind == "mul":
        return BinOp("*", Variable(step.source), Number(step.constant))
    if step.constant < 0:
        return BinOp("-", Variable(step.source), Number(-step.constant))
    return BinOp("+", Variable(step.source), Number(step.constant))


@compile_pass(
    NODE, NODE,
    title="Combine constant steps",
    requires=("unroll", "flatten"),
    summary="Merges two additions, or two multiplications, by constants through an intermediate nothing else reads, "
            "so the intermediate is never computed.",
)
def fuse_steps(program: List[Node], system: Optional[System] = None) -> List[Node]:
    body = list(program)
    changed = True
    while changed:
        changed = False
        for j, second in enumerate(body):
            outer = _step(second)
            if outer is None:
                continue
            t = outer.source
            first_at = next((i for i in range(j - 1, -1, -1) if isinstance(body[i], Assign) and body[i].name == t), None)
            inner = None if first_at is None else _step(body[first_at])
            if inner is None or inner.kind != outer.kind:
                continue
            if _count(body, t, "read") != 1 or _count(body, t, "assigned") != 1:
                continue
            if inner.source in assigned(body[first_at + 1:j]):
                continue
            constant = inner.constant + outer.constant if outer.kind == "add" else inner.constant * outer.constant
            body[j] = replace(second, value=_expression(Step(outer.kind, inner.source, constant)))
            del body[first_at]
            changed = True
            break
    return body
