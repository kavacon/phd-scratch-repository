"""
Loop unrolling, run first: replace each `for` loop with one copy of its body per iteration.

Loop bounds are integer literals (see the TODO in grammar_v1.lark), so every loop can be expanded at
compile time. Inside each copy the loop variable is replaced by the iteration's number. After this the rest
of the pipeline only sees straight-line code and `if`s: a variable a loop updates is just assigned again, so
rename gives it ordinary versions and nothing has to be carried round a loop.

    for i in 0 .. 2 { t = s + i; s = t * 2; }
becomes
    t@0 = s + 0; s = t@0 * 2; t@1 = s + 1; s = t@1 * 2;

A name first assigned inside the body is local to it, so each iteration gets its own suffixed copy (`t@0`,
`t@1`). Nothing after the loop can refer to those names, which keeps the rule that block-local names do not
escape. Inner loops are expanded first, so their names pick up more than one suffix.

A loop that runs no times disappears. Assigning to the loop variable is rejected, and so is a loop that would
expand to an unreasonable number of statements.
"""
from dataclasses import fields, replace
from typing import Dict, List, Sequence, Set

from parser.grammar import Assign, ForLoop, FunctionDef, If, Node, Number, Unassign, Variable
from passes.names import assigned

MAX_STATEMENTS = 20_000


class UnrollError(Exception):
    pass


def _size(block: Sequence[Node]) -> int:
    total = 0
    for statement in block:
        total += 1
        if isinstance(statement, If):
            total += _size(statement.body) + _size(statement.orelse)
    return total  # inner loops are already expanded when a body is measured


def _map(node: Node, rename: Dict[str, str], constants: Dict[str, Node]) -> Node:
    """Copy a statement, replacing the loop variable by its number and renaming the body's local names."""
    if isinstance(node, Variable):
        return constants.get(node.name) or Variable(rename.get(node.name, node.name))
    if isinstance(node, Assign):
        return Assign(rename.get(node.name, node.name), _map(node.value, rename, constants))
    if isinstance(node, Unassign):
        return Unassign(rename.get(node.name, node.name), _map(node.witness, rename, constants))
    changes = {}
    for f in fields(node):
        value = getattr(node, f.name)
        if isinstance(value, Node):
            changes[f.name] = _map(value, rename, constants)
        elif isinstance(value, tuple):
            changes[f.name] = tuple(_map(c, rename, constants) if isinstance(c, Node) else c for c in value)
    return replace(node, **changes) if changes else node


def _expand(loop: ForLoop, visible: Set[str]) -> List[Node]:
    body = _block(loop.body, visible)  # inner loops first
    names = assigned(body)
    if loop.var in names:
        raise UnrollError(f"the loop variable '{loop.var}' cannot be assigned to inside its loop")
    start, stop = loop.start.value, loop.stop.value
    if (stop - start) * _size(body) > MAX_STATEMENTS:
        raise UnrollError(f"this loop would expand to more than {MAX_STATEMENTS} statements")
    local = [n for n in names if n not in visible]  # first assigned inside the body, so local to each iteration
    result: List[Node] = []
    for k in range(start, stop):
        rename = {n: f"{n}@{k}" for n in local}
        result += [_map(s, rename, {loop.var: Number(k)}) for s in body]
    return result


def _block(body: Sequence[Node], visible: Set[str]) -> tuple:
    """Unroll the loops in a block. `visible` holds the names defined so far in this and enclosing scopes."""
    visible = set(visible)
    result: List[Node] = []
    for statement in body:
        if isinstance(statement, Assign):
            visible.add(statement.name)
            result.append(statement)
        elif isinstance(statement, If):
            result.append(replace(statement, body=_block(statement.body, visible), orelse=_block(statement.orelse, visible)))
        elif isinstance(statement, ForLoop):
            expanded = _expand(statement, visible)
            visible.update(assigned(expanded))
            result += expanded
        elif isinstance(statement, FunctionDef):
            result.append(replace(statement, body=_block(statement.body, {p.name for p in statement.params})))
        else:
            result.append(statement)
    return tuple(result)


def unroll(program: List[Node]) -> List[Node]:
    return list(_block(program, set()))
