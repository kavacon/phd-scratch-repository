"""Which names a statement reads and defines, including everything nested inside it."""
from typing import FrozenSet, List, Sequence

from parser.ast_tools import walk
from parser.grammar import Assign, ExprStmt, If, Node, Return, Unassign, Variable

_NONE: FrozenSet[str] = frozenset()


def variables(node: Node) -> FrozenSet[str]:
    """Every variable name that appears anywhere inside `node`."""
    return frozenset(n.name for n in walk(node) if isinstance(n, Variable))


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
    else:
        return _NONE  # a function definition is analysed on its own
    return outer.union(*map(read, inner))


def defined(statement: Node) -> FrozenSet[str]:
    """Every version defined by a statement or anything nested inside it."""
    if isinstance(statement, Assign):
        return frozenset({statement.name})
    if isinstance(statement, If):
        return frozenset().union(*map(defined, statement.body + statement.orelse))
    return _NONE


def results(statement: If) -> FrozenSet[str]:
    """Versions an `if` produces: those defined in both branches (rename gives them the same name)."""
    then, orelse = (frozenset().union(*map(defined, block)) for block in (statement.body, statement.orelse))
    return then & orelse


def assigned(block: Sequence[Node]) -> List[str]:
    """Names assigned anywhere in a block, including nested blocks, in order of first assignment."""
    names: List[str] = []
    for statement in block:
        if isinstance(statement, Assign):
            names.append(statement.name)
        elif isinstance(statement, If):
            names += assigned(statement.body) + assigned(statement.orelse)
    return list(dict.fromkeys(names))
