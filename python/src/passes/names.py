"""Which names a statement reads and defines, including everything nested inside it."""
from dataclasses import fields
from typing import FrozenSet, Iterator

from parser.grammar import (
    Assign, ExprStmt, ForLoop, If, Node, Return, Unassign, Variable,
)

_NONE: FrozenSet[str] = frozenset()


def variables(node: Node) -> FrozenSet[str]:
    def walk(n) -> Iterator[str]:
        if isinstance(n, Variable):
            yield n.name
        for f in fields(n):
            value = getattr(n, f.name)
            for child in value if isinstance(value, tuple) else (value,):
                if isinstance(child, Node):
                    yield from walk(child)
    return frozenset(walk(node))


def read(statement: Node) -> FrozenSet[str]:
    """Every version a statement reads, including anything read inside nested blocks."""
    if isinstance(statement, Assign):
        return variables(statement.value)
    if isinstance(statement, Unassign):
        return variables(statement.witness) | {statement.name}
    if isinstance(statement, ExprStmt):
        return variables(statement.expr)
    if isinstance(statement, Return):
        return variables(statement.value)
    if isinstance(statement, If):
        inner, outer = statement.body + statement.orelse, variables(statement.condition)
    elif isinstance(statement, ForLoop):
        inner, outer = statement.body + statement.header, variables(statement.start) | variables(statement.stop)
    else:
        return _NONE  # a function definition is analysed on its own
    return outer.union(*map(read, inner))


def defined(statement: Node) -> FrozenSet[str]:
    """Every version defined by a statement or anything nested inside it."""
    if isinstance(statement, Assign):
        return frozenset({statement.name})
    if isinstance(statement, If):
        return frozenset().union(*map(defined, statement.body + statement.orelse))
    if isinstance(statement, ForLoop):
        return frozenset({statement.var}).union(*map(defined, statement.body + statement.header))
    return _NONE


def results(statement: If) -> FrozenSet[str]:
    """Versions an `if` produces: those defined in both branches (rename gives them the same name)."""
    then, orelse = (frozenset().union(*map(defined, block)) for block in (statement.body, statement.orelse))
    return then & orelse
