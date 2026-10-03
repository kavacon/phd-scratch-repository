"""
Liveness detection on the renamed AST. This only reports when each variable version stops being live;
it does not rewrite the program or insert any `~=`.

Liveness follows Faro, Marino and Messina, "Reversible Lifetime Semantics for Quantum Programs" (2026,
arXiv:2603.14538), which presents it for the Qutes language:
- Definition 3 (Liveness): a variable is live at a program point if a later operation uses it directly, or
  it is entangled (an edge of the entanglement graph at that point) with a variable that is itself live.
- Definition 2 (Entanglement graph): conservatively approximated by adding an edge whenever an operation
  jointly acts on variables. The paper describes operations on two variables; here an operation touching
  several variables connects all of them.
- Program points follow the paper's numbering: point k is the state immediately after the k-th statement
  (point 0 is the start of the block).

Liveness alone is not the whole algorithm. The paper's Remark and Lemma 11 note that entanglement-based
liveness is a sound but incomplete approximation: a temporary entangled with a live output can still be
reclaimed early when the subcircuit is output-isolable (Definition 5). Definition 6 (safe reclamation point)
then says when a temporary can be restored and released. `Liveness.dies_at` is the conservative lifetime
from Definitions 3 and 4 (the paper's Fig. 3 before isolability is applied), and `Liveness.reclaim_at` is
the point at which each temporary can safely be reclaimed (the shrunken lifetimes of Fig. 3).

The paper gives no static test for isolability, only that a compiler must disable reclamation when it cannot
certify it. Our stand-in is structural: after rename every operation writes only a fresh register and reads
others, so uncomputing a temporary writes only that temporary's register (reading its operands as controls)
and cannot change the state of any other register. A temporary is therefore reclaimable once nothing uses
it any more, as long as its operands are intact. This relies on the language having no in-place updates
and must be revisited if mutation or by-reference parameters are added.

Not specified by the paper, and chosen here:
- When a version is released with `~=`, it leaves the entanglement graph but its neighbours stay connected
  to each other, because they were correlated through it.
- Releases of versions that read each other happen in reverse order of creation: a version can only be
  released after every version whose release reads it (its witness operands must still be there). The paper's
  example inserts the adjoint history in reverse, but does not state this as a rule.

Scope of this section: straight-line blocks only (assignments, `~=`, expression statements and the final
return, per function and for the top-level program). Control flow (`if`, `while`, `for`) raises
NotImplementedError until it is added.

Rename must have run first, so every name is a unique version.
"""
from dataclasses import fields
from typing import Dict, FrozenSet, Iterator, NamedTuple, Optional, Sequence, Set, Tuple

from parser.grammar import Assign, ExprStmt, FunctionDef, Node, Return, Unassign, Variable

_NONE: FrozenSet[str] = frozenset()


class _Effects(NamedTuple):
    defs: FrozenSet[str] = _NONE
    uses: FrozenSet[str] = _NONE
    entangles: FrozenSet[str] = _NONE  # variables an operation on several variables makes mutually entangled
    releases: Optional[str] = None  # a version released by `~=`, which disentangles it


class Liveness(NamedTuple):
    live: Tuple[FrozenSet[str], ...]  # live[k]: the versions live at point k
    dies_at: Dict[str, int]  # conservative: first point at which each version (other than outputs) is not live
    outputs: FrozenSet[str]  # versions returned from the block; they escape, so they are never released
    reclaim_at: Dict[str, int]  # point at which each temporary can be released; parameters and outputs excluded
    release_order: Tuple[str, ...]  # temporaries ordered by release point, then reverse creation order


def _variables(node: Node) -> FrozenSet[str]:
    def walk(n) -> Iterator[str]:
        if isinstance(n, Variable):
            yield n.name
        for f in fields(n):
            value = getattr(n, f.name)
            for child in value if isinstance(value, tuple) else (value,):
                if isinstance(child, Node):
                    yield from walk(child)
    return frozenset(walk(node))


def _effects(statement: Node) -> _Effects:
    if isinstance(statement, Assign):
        operands = _variables(statement.value)
        group = operands | {statement.name} if operands else _NONE
        return _Effects(defs=frozenset({statement.name}), uses=operands, entangles=group)
    if isinstance(statement, Unassign):
        return _Effects(uses=_variables(statement.witness) | {statement.name}, releases=statement.name)
    if isinstance(statement, ExprStmt):
        used = _variables(statement.expr)
        return _Effects(uses=used, entangles=used if len(used) > 1 else _NONE)
    if isinstance(statement, Return):
        return _Effects(uses=_variables(statement.value))
    if isinstance(statement, FunctionDef):
        return _Effects()  # analysed on its own, see analyse_program
    raise NotImplementedError(f"liveness of {type(statement).__name__} (control flow) is not implemented yet")


def _closure(edges: Set[FrozenSet[str]], seeds: FrozenSet[str]) -> FrozenSet[str]:
    """Everything entangled, directly or through other variables, with a seed."""
    adjacent: Dict[str, Set[str]] = {}
    for a, b in map(tuple, edges):
        adjacent.setdefault(a, set()).add(b)
        adjacent.setdefault(b, set()).add(a)
    live, pending = set(seeds), list(seeds)
    while pending:
        for other in adjacent.get(pending.pop(), ()):
            if other not in live:
                live.add(other)
                pending.append(other)
    return frozenset(live)


def _release(edges: Set[FrozenSet[str]], released: str) -> Set[FrozenSet[str]]:
    """Drop a released version from the graph, keeping everything it connected connected to each other."""
    neighbours = sorted({v for e in edges if released in e for v in e if v != released})
    kept = {e for e in edges if released not in e}
    return kept | {frozenset((a, b)) for i, a in enumerate(neighbours) for b in neighbours[i + 1:]}


def _reclaim_points(body: Sequence[Node], effects: Sequence[_Effects], outputs: FrozenSet[str]) -> Dict[str, int]:
    """
    Point at which each temporary (a version defined in the block that is not returned) can be released:
    after its last use, and after the release of every temporary whose release reads it.
    Versions the programmer already released with `~=` keep that point.
    """
    defined_at = {name: k for k, e in enumerate(effects) for name in e.defs}
    last_use: Dict[str, int] = {}
    for k, e in enumerate(effects):
        for name in e.uses:
            last_use[name] = k
    explicit = {e.releases: k + 1 for k, e in enumerate(effects) if e.releases}

    reads: Dict[str, FrozenSet[str]] = {}  # versions that must still exist when a version is released
    for statement in body:
        if isinstance(statement, Assign):
            reads[statement.name] = _variables(statement.value)
    for statement in body:
        if isinstance(statement, Unassign):
            reads[statement.name] = _variables(statement.witness) - {statement.name}

    reclaim: Dict[str, int] = {}
    for name in sorted((n for n in defined_at if n not in outputs), key=defined_at.get, reverse=True):
        if name in explicit:
            reclaim[name] = explicit[name]
            continue
        earliest = max(last_use.get(name, defined_at[name]), defined_at[name]) + 1
        after_dependents = [p for other, p in reclaim.items() if name in reads.get(other, ())]
        reclaim[name] = max([earliest, *after_dependents])
    return reclaim


def analyse(body: Sequence[Node], params: Sequence[str] = (), entanglement: bool = True) -> Liveness:
    """
    Liveness of a straight-line block. With `entanglement=False` only direct uses count (classic liveness),
    which is not the paper's algorithm (it would let an operand die before a temporary computed from it) and
    exists only as a baseline for comparison.
    """
    effects = [_effects(s) for s in body]
    n = len(body)

    used_later: list = [_NONE] * (n + 1)  # liveness from direct uses, computed backwards
    for k in range(n - 1, -1, -1):
        used_later[k] = (used_later[k + 1] - effects[k].defs) | effects[k].uses

    live, edges = [], set()  # edges: entanglement existing at the current point, forward
    for k in range(n + 1):
        live.append(_closure(edges, used_later[k]) if entanglement else used_later[k])
        if k < n:
            group = sorted(effects[k].entangles)
            edges |= {frozenset((a, b)) for i, a in enumerate(group) for b in group[i + 1:]}
            if effects[k].releases:
                edges = _release(edges, effects[k].releases)

    outputs = frozenset().union(*(e.uses for s, e in zip(body, effects) if isinstance(s, Return)))
    defined_at = {p: 0 for p in params}
    defined_at.update({name: k + 1 for k, e in enumerate(effects) for name in e.defs})
    dies_at = {
        name: next(k for k in range(start, n + 1) if name not in live[k])
        for name, start in defined_at.items() if name not in outputs
    }
    reclaim_at = _reclaim_points(body, effects, outputs)
    order = sorted(reclaim_at, key=lambda v: (reclaim_at[v], -defined_at[v]))
    return Liveness(tuple(live), dies_at, outputs, reclaim_at, tuple(order))


def analyse_program(program: Sequence[Node], entanglement: bool = True) -> Dict[str, Liveness]:
    """Liveness of the top-level program (key "<program>") and of each function, keyed by name."""
    result = {"<program>": analyse(program, entanglement=entanglement)}
    for statement in program:
        if isinstance(statement, FunctionDef):
            params = [p.name for p in statement.params]
            result[statement.name] = analyse(statement.body, params, entanglement)
    return result
