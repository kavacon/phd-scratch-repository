"""
Branch balancing, run before rename: make both branches of every `if` assign the same outer variables.

If only one branch assigns a variable that exists before the `if`, the other branch gets an identity
assignment, so after the `if` the variable has a new value on every path:

    if c { x = x + 5; }
becomes
    if c { x = x + 5; } else { x = x; }

Rename can then give the variable one shared version in both branches (they are mutually exclusive, so
only one ever runs), and nothing is left over that would need a conditional release afterwards. The
identity copies cost an extra register and are the kind of thing a later optimisation can remove.
This makes the branches write the same variables; it does not make them take equal numbers of gates, so
it is not a claim about the synchronisation condition in Yuan, Villanyi and Carbin (2024).

Scoping follows rename: a name first assigned inside a block is local to that block, so it is never copied.
Only `if` needs this: the unroll pass has already removed every loop.
"""
from dataclasses import replace
from typing import List, Sequence, Set

from parser.grammar import Assign, FunctionDef, If, Node, Variable
from passes.names import assigned


def _block(body: Sequence[Node], visible: Set[str]) -> tuple:
    """Balance a block. `visible` holds the names defined so far in this and enclosing scopes."""
    visible = set(visible)  # names first assigned in this block stay local to it
    result = []
    for statement in body:
        if isinstance(statement, Assign):
            visible.add(statement.name)
        elif isinstance(statement, If):
            statement = _if(statement, visible)
        elif isinstance(statement, FunctionDef):
            scope = {p.name for p in statement.params}
            statement = replace(statement, body=_block(statement.body, scope))
        result.append(statement)
    return tuple(result)


def _if(node: If, visible: Set[str]) -> If:
    body, orelse = _block(node.body, visible), _block(node.orelse, visible)
    in_then = [n for n in assigned(body) if n in visible]
    in_else = [n for n in assigned(orelse) if n in visible]
    body += tuple(Assign(n, Variable(n)) for n in in_else if n not in in_then)
    orelse += tuple(Assign(n, Variable(n)) for n in in_then if n not in in_else)
    return replace(node, body=body, orelse=orelse)


def balance(program: List[Node]) -> List[Node]:
    return list(_block(program, set()))
