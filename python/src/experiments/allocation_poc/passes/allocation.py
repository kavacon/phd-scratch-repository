"""
Allocation lowering: insert `~=` to release every temporary at the point liveness says it can go.

The program is expected to have been through unroll, flatten, balance and rename. Each temporary is released
with its own defining expression as the witness (the inverse of the assignment that created it), at the release
point and in the order that `passes.liveness` reports, which uses the ideas of Faro, Marino and Messina,
"Reversible Lifetime Semantics for Quantum Programs" (2026, arXiv:2603.14538). Every block (function body and
`if` branch) is handled the same way, using that block's own analysis.

- Everything an `if` produces is released together, by an `if` on the same condition register that undoes, in
  each branch, the versions the `if` produced and the temporaries they were built from, latest defined first and
  one operation at a time. Those temporaries are kept alive after the `if` for this (see `Liveness.kept`), so
  every witness is a single operation on versions that still exist. A nested `if` is undone the same way.
- Parameters and returned values are never released. Versions already released with `~=` are left alone.
- TODO: avoid emitting a second `if` just to release what an `if` produced. The goal is to flatten the
  conditional and release inside the block instead, so the condition is not tested again and no extra condition
  block is needed. It is not yet clear this is possible: the versions are used after the `if`, so the release
  cannot simply move into the branch that made them. Decided to keep the second `if` for now and revisit it
  once operations have histories (the ISA work).
"""
from dataclasses import replace
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from compiler.compiler import NODE, compile_pass
from compiler.system import System
from compiler.parser.ast_tools import walk
from compiler.parser.grammar import Assign, FunctionDef, If, Node, Unassign
from experiments.allocation_poc.passes.liveness import Liveness, analyse_program
from compiler.passes.names import results


class ReleaseError(Exception):
    pass


def _released_inside(statement: Node) -> FrozenSet[str]:
    if isinstance(statement, Unassign):
        return frozenset({statement.name})
    if isinstance(statement, If):
        return frozenset().union(*map(_released_inside, statement.body + statement.orelse))
    return frozenset()


class _Inserter:
    def __init__(self, analyses: Mapping[str, Liveness]):
        self.analyses = analyses

    def block(self, body: Sequence[Node], analysis: Liveness) -> Tuple[Node, ...]:
        defined = {s.name for s in body if isinstance(s, Assign)}
        defined.update(*(results(s) for s in body if isinstance(s, If)))
        explicit = frozenset().union(*map(_released_inside, body)) & defined

        releases: List[Tuple[int, int, Node]] = []  # (point, index of the statement that defined it, release)
        for k, statement in enumerate(body):
            if isinstance(statement, Assign):
                name = statement.name
                if name in analysis.reclaim_at and name not in explicit:
                    releases.append((analysis.reclaim_at[name], k, Unassign(name, statement.value)))
            elif isinstance(statement, If):
                produced = results(statement) & analysis.reclaim_at.keys()
                if produced:
                    release = self.release_if(statement, analysis.nested[k], explicit)
                    if release is not None:  # released together, at the latest point any of it needs
                        releases.append((max(analysis.reclaim_at[name] for name in produced), k, release))
        by_point: Dict[int, List[Node]] = {}
        for point, _, release in sorted(releases, key=lambda r: (r[0], -r[1])):  # latest defined first
            by_point.setdefault(point, []).append(release)

        result: List[Node] = []
        for k, statement in enumerate(body):
            result += by_point.get(k, [])  # point k is immediately before statement k
            result.append(self.statement(statement, analysis, k))
        result += by_point.get(len(body), [])
        return tuple(result)

    def statement(self, statement: Node, analysis: Liveness, k: int) -> Node:
        if isinstance(statement, If):
            then, orelse = analysis.nested[k]
            return replace(statement, body=self.block(statement.body, then), orelse=self.block(statement.orelse, orelse))
        if isinstance(statement, FunctionDef):
            return replace(statement, body=self.block(statement.body, self.analyses[statement.name]))
        return statement

    def release_if(self, statement: If, nested: Tuple[Liveness, ...], skip: FrozenSet[str]) -> Optional[If]:
        """An `if` on the same condition that undoes, in each branch, what this `if` produced there."""
        then = self.undo(statement.body, nested[0], skip)
        orelse = self.undo(statement.orelse, nested[1], skip)
        return If(statement.condition, then, orelse) if then or orelse else None

    def undo(self, block: Sequence[Node], analysis: Liveness, skip: FrozenSet[str]) -> Tuple[Node, ...]:
        """Release what a branch keeps for the release of its `if`, latest defined first."""
        names = analysis.release_later - skip - frozenset().union(*map(_released_inside, block))
        undone: List[Node] = []
        for k in reversed(range(len(block))):
            statement = block[k]
            if isinstance(statement, Assign) and statement.name in names:
                undone.append(Unassign(statement.name, statement.value))
            elif isinstance(statement, If) and results(statement) & names:
                release = self.release_if(statement, analysis.nested[k], skip)
                if release is not None:
                    undone.append(release)
        return tuple(undone)


def prepare(program: Sequence[Node]) -> Tuple[List[Node], Dict[str, Liveness]]:
    """The program as the insertion pass sees it, with its liveness analysis."""
    program = list(program)
    return program, analyse_program(program)


@compile_pass(
    NODE, NODE,
    title="Insert releases",
    requires=("rename",),
    summary="Inserts a release (x ~= e;) for each temporary at the point liveness says it can go.",
)
def lower_allocation(program: Sequence[Node], system: Optional[System] = None) -> List[Node]:
    """The program with a `~=` for every releasable temporary."""
    if any(isinstance(n, Assign) and n.inplace for statement in program for n in walk(statement)):
        # a release re-reads the value a version was defined from, which an update has already overwritten
        raise ReleaseError("release insertion cannot undo an in-place update (+= or -=)")
    program, analyses = prepare(program)
    return list(_Inserter(analyses).block(program, analyses["<program>"]))


insert_releases = lower_allocation
