"""
Allocation lowering: insert `~=` to release every temporary at the point liveness says it can go.

The program is expected to have been through unroll, balance and rename. Each temporary is released with its
own defining expression as the witness (the inverse of the assignment that created it), at the release point
and in the order that `passes.liveness` reports, which uses the ideas of Faro, Marino and Messina,
"Reversible Lifetime Semantics for Quantum Programs" (2026, arXiv:2603.14538). Every block (function body and
`if` branch) is handled the same way, using that block's own analysis.

- A version an `if` produces is defined differently in each branch, so its release is an `if` on the same
  condition with a `~=` in each branch. The witness there is written in terms of versions that still exist
  after the `if`: temporaries local to a branch are released at the end of the branch, so the definitions of
  those are substituted in. If a nested `if` contributed, the release nests the same way.
  TODO: avoid this extra condition block by flattening and releasing within the block, if that is possible
  (see `_release`).
- A function or program that returns an expression gets it named first (`return#1 = e; return return#1;`), so
  the versions the expression reads become temporaries that can be released.
- Parameters and returned values are never released. Versions already released with `~=` are left alone.
"""
from dataclasses import fields, replace
from typing import Dict, List, Mapping, Sequence, Tuple

from parser.grammar import Assign, FunctionDef, If, Node, Return, Unassign, Variable
from passes.liveness import Liveness, analyse_program
from passes.names import defined, read, results, variables


def _inline(node: Node, env: Mapping[str, Node]) -> Node:
    """Replace each variable that has an entry in `env` with that expression."""
    if isinstance(node, Variable):
        return env.get(node.name, node)
    changes = {}
    for f in fields(node):
        value = getattr(node, f.name)
        if isinstance(value, Node):
            changes[f.name] = _inline(value, env)
        elif isinstance(value, tuple):
            changes[f.name] = tuple(_inline(c, env) if isinstance(c, Node) else c for c in value)
    return replace(node, **changes) if changes else node


def _witness(statements: Sequence[Node], name: str, env: Mapping[str, Node]) -> Tuple[Node, ...]:
    """
    Statements that release `name`, given how it is defined along the way through `statements`. Versions defined
    on the way are substituted by their definitions, so the result only reads versions defined before the block.
    """
    for i, statement in enumerate(statements):
        rest = statements[i + 1:]
        if isinstance(statement, Assign):
            value = _inline(statement.value, env)
            if statement.name == name:
                return (Unassign(name, value),)
            env = {**env, statement.name: value}
        elif isinstance(statement, If):
            if name in defined(statement):
                tail: Sequence[Node] = ()
            elif results(statement) & frozenset().union(*map(read, rest)):
                tail = rest  # what follows depends on which branch ran
            else:
                continue
            branches = [_witness(tuple(b) + tuple(tail), name, env) for b in (statement.body, statement.orelse)]
            return (If(_inline(statement.condition, env), branches[0], branches[1]),)
    raise ValueError(f"no definition of {name} found")


def _release(statement: Node, name: str) -> Node:
    if isinstance(statement, Assign):
        return Unassign(name, statement.value)
    # TODO: avoid emitting a second `if` just to release what an `if` produced. The goal is to flatten the
    #  conditional and release inside the block instead, so the condition is not tested again and no extra
    #  condition block is needed. It is not yet clear this is possible: the version is used after the `if`,
    #  so the release cannot simply move into the branch that made it.
    branches = [_witness(b, name, {}) for b in (statement.body, statement.orelse)]
    return If(statement.condition, branches[0], branches[1])


def _released_inside(statement: Node) -> frozenset:
    if isinstance(statement, Unassign):
        return frozenset({statement.name})
    if isinstance(statement, If):
        return frozenset().union(*map(_released_inside, statement.body + statement.orelse))
    return frozenset()


class _Inserter:
    def __init__(self, analyses: Mapping[str, Liveness]):
        self.analyses = analyses

    def block(self, body: Sequence[Node], analysis: Liveness) -> Tuple[Node, ...]:
        definition: Dict[str, Node] = {}
        for statement in body:
            if isinstance(statement, Assign):
                definition[statement.name] = statement
            elif isinstance(statement, If):
                definition.update({name: statement for name in results(statement)})
        explicit = frozenset().union(*map(_released_inside, body)) & definition.keys()

        releases: Dict[int, List[Node]] = {}
        for name in analysis.release_order:
            if name not in explicit:
                releases.setdefault(analysis.reclaim_at[name], []).append(_release(definition[name], name))

        result: List[Node] = []
        for k, statement in enumerate(body):
            result += releases.get(k, [])  # point k is immediately before statement k
            result.append(self.statement(statement, analysis, k))
        result += releases.get(len(body), [])
        return tuple(result)

    def statement(self, statement: Node, analysis: Liveness, k: int) -> Node:
        if isinstance(statement, If):
            then, orelse = analysis.nested[k]
            return replace(statement, body=self.block(statement.body, then), orelse=self.block(statement.orelse, orelse))
        if isinstance(statement, FunctionDef):
            return replace(statement, body=self.block(statement.body, self.analyses[statement.name]))
        return statement


def _name_result(body: Sequence[Node]) -> Tuple[Node, ...]:
    *rest, last = body
    if isinstance(last, Return) and not isinstance(last.value, Variable) and variables(last.value):
        return (*rest, Assign("return#1", last.value), Return(Variable("return#1")))
    return tuple(body)


def _name_results(program: Sequence[Node]) -> List[Node]:
    return [
        replace(s, body=_name_result(s.body)) if isinstance(s, FunctionDef) else s
        for s in _name_result(program)
    ]


def prepare(program: Sequence[Node]) -> Tuple[List[Node], Dict[str, Liveness]]:
    """The program as the insertion pass sees it (returned expressions named), with its liveness analysis."""
    named = _name_results(program)
    return named, analyse_program(named)


def insert_releases(program: Sequence[Node]) -> List[Node]:
    """The program with a `~=` for every releasable temporary."""
    program, analyses = prepare(program)
    return list(_Inserter(analyses).block(program, analyses["<program>"]))


lower_allocation = insert_releases
