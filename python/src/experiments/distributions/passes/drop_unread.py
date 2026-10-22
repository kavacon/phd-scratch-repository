"""
Drop unread values: an assignment whose variable nothing reads, and that is not returned, is removed, and so is whatever
that leaves unread in turn.

Such a value has no effect on the result. Computing it would only leave a register entangled with the registers it was
computed from, which is exactly what releasing it afterwards would undo, so a value that is never read behaves as if it
was never there. A value that is read at some point and dead afterwards is a different case: it has to be computed, and
releasing it is the release pass's concern.

Run after rename, so a version is assigned once and "unread" means the same everywhere.
"""
from dataclasses import replace
from typing import FrozenSet, List, Optional, Sequence

from compiler.compiler import NODE, compile_pass
from compiler.parser.grammar import Assign, If, Node
from compiler.passes.names import read
from compiler.system import System


def _prune(body: Sequence[Node], used: FrozenSet[str]) -> List[Node]:
    out: List[Node] = []
    for statement in body:
        if isinstance(statement, Assign) and statement.name not in used:
            continue
        if isinstance(statement, If):
            statement = replace(statement, body=tuple(_prune(statement.body, used)),
                                orelse=tuple(_prune(statement.orelse, used)))
        out.append(statement)
    return out


@compile_pass(
    NODE, NODE,
    title="Drop unread values",
    requires=("rename",),
    summary="Removes assignments that nothing reads and that are not returned, repeatedly.",
)
def drop_unread(program: List[Node], system: Optional[System] = None) -> List[Node]:
    body = list(program)
    while True:  # removing an assignment can leave the ones it read unread
        pruned = _prune(body, frozenset().union(*map(read, body)))
        if pruned == body:
            return pruned
        body = pruned
