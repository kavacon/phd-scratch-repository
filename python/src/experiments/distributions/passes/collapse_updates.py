"""
Update in place, before rename: `b = a + e;` where nothing reads `a` afterwards becomes the in-place update `a += e;`, and
every later reference to `b` becomes a reference to `a`. The program that comes out is ordinary source using the in-place
form the language already has, so rename and the lowering give it versions and registers the way they do any update and
nothing is carried between passes. It is the state `b` would be in after releasing `a` from it (`a` is determined by
`b`), without ever holding both registers.

This runs before rename, so names can be assigned more than once. The rewrite is made only when neither `a` nor `b` is
assigned again later, so that every later `b` is this `b` and every later `a` could only be a read of the old value,
which there is none. `a = a + e;` is already an update of itself and is marked in place without renaming anything. Only
add and subtract are undoable in place: `b = e + a` is the same update, `b = e - a` is not, and `b = a + a` is refused
for the usual reason.

Only the top level of the program is rewritten: a branch that updates in place must be matched by the other branch,
and what a branch's source is read by afterwards depends on the enclosing code.
"""
from dataclasses import replace
from typing import List, Optional

from compiler.compiler import NODE, compile_pass
from compiler.parser.ast_tools import rewrite
from compiler.parser.grammar import Assign, BinOp, Node, Variable
from compiler.passes.names import assigned, read, variables
from compiler.system import System


def _update(statement: Assign) -> Optional[Assign]:
    """The statement as an update of the variable it reads, or None if it is not one."""
    value = statement.value
    if not isinstance(value, BinOp) or value.op not in ("+", "-"):
        return None
    for source, other, swapped in ((value.left, value.right, False), (value.right, value.left, True)):
        if isinstance(source, Variable) and source.name not in variables(other) and not (swapped and value.op == "-"):
            return replace(statement, value=BinOp(value.op, source, other), inplace=True)
    return None


@compile_pass(
    NODE, NODE,
    title="Update in place",
    requires=("unroll",),
    summary="Turns an add or subtract whose source is not read again into an in-place update of that source, and "
            "renames the later uses of the result to it.",
)
def collapse_updates(program: List[Node], system: Optional[System] = None) -> List[Node]:
    body = list(program)
    for k, statement in enumerate(body):
        if not isinstance(statement, Assign) or statement.inplace:
            continue
        update = _update(statement)
        if update is None:
            continue
        source, target, later = update.value.left.name, statement.name, body[k + 1:]
        if source == target:
            body[k] = update
        elif source not in assigned(later) and target not in assigned(later) \
                and source not in frozenset().union(*map(read, later)):
            body[k] = replace(update, name=source)
            body[k + 1:] = [rewrite(s, lambda n: Variable(source) if isinstance(n, Variable) and n.name == target else n)
                            for s in later]
    return body
