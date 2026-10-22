"""
Drop unread values: an assignment whose value is never read afterwards, and that is not returned, is removed, and so is
whatever that leaves unread in turn.

Such a value has no effect on the result. Computing it would only leave a register entangled with the registers it was
computed from, which is exactly what releasing it afterwards would undo, so a value that is never read behaves as if it
was never there. A value that is read at some point and dead afterwards is a different case: it has to be computed, and
releasing it is the release pass's concern.

This is liveness run backwards over the program, so a name that is assigned more than once is judged assignment by
assignment, and it works before or after rename. An `if` is analysed branch by branch from what is live after it.
Before rename an assignment is kept, even if unread, when a later `if` assigns the same name: that is what makes the name
visible after the `if`, since a name first assigned inside a branch is local to it. After rename a name is unique and
the assignment can go, which is the second pass.
"""
from dataclasses import replace
from typing import FrozenSet, List, Optional, Sequence, Set, Tuple

from compiler.compiler import NODE, compile_pass
from compiler.parser.grammar import Assign, If, Node
from compiler.passes.names import assigned, read, variables
from compiler.system import System


def _backward(block: Sequence[Node], live_out: FrozenSet[str], declarations: bool) -> Tuple[List[Node], FrozenSet[str]]:
    """The block without its unread assignments, and what is live before it."""
    live: Set[str] = set(live_out)
    declares: Set[str] = set()  # names a later `if` assigns
    kept: List[Node] = []
    for statement in reversed(block):
        if isinstance(statement, Assign):
            if statement.name not in live and not (declarations and statement.name in declares):
                continue
            live.discard(statement.name)
            live |= variables(statement.value)  # an in-place update reads its own register, so it comes back
        elif isinstance(statement, If):
            body, live_body = _backward(statement.body, frozenset(live), declarations)
            orelse, live_orelse = _backward(statement.orelse, frozenset(live), declarations)
            if not body and not orelse:
                continue
            declares |= set(assigned(statement.body + statement.orelse))
            statement = replace(statement, body=tuple(body), orelse=tuple(orelse))
            live = set(variables(statement.condition) | live_body | live_orelse)
        else:
            live |= read(statement)
        kept.append(statement)
    return kept[::-1], frozenset(live)


@compile_pass(
    NODE, NODE,
    title="Drop unread values",
    requires=("unroll",),
    summary="Removes assignments that nothing reads afterwards and that are not returned, repeatedly.",
)
def drop_unread(program: List[Node], system: Optional[System] = None) -> List[Node]:
    return _backward(program, frozenset(), True)[0]


@compile_pass(
    NODE, NODE,
    title="Drop unread versions",
    requires=("rename",),
    summary="Removes versions that nothing reads, including the first value of a variable both branches of an if "
            "assign, which could not be judged before the names were unique.",
)
def drop_unread_versions(program: List[Node], system: Optional[System] = None) -> List[Node]:
    return _backward(program, frozenset(), False)[0]
