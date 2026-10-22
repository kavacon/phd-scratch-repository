"""
Which registers of an instruction program are superposed and entangled, and which are live, at every point. This is data
for the release pass, the explorer and the tests; it changes nothing and is not a pass.

A point is the state after some number of instructions. The rules are the paper's (Faro, Marino and Messina, "Reversible
Lifetime Semantics for Quantum Programs", 2026) with one change: where the paper joins everything an operation acts on,
a result is joined only to the operands that are superposed. A value computed from definite values stays definite and
joins nothing.
- `DIST` makes a register superposed (a range of one value is definite) and alone.
- An operation's result is superposed, and joined to each of them, if any operand is superposed or it runs under a
  control that is superposed, even when it writes a constant. An in-place update keeps its register's status and joins
  it to a superposed operand or control.
- A register that is released (its inverse has run, so it is zero) leaves the graph, and what it joined stays joined to
  each other. That is conservative: they may be independent again, and are never reported so.
- A register is live at a point if a later instruction reads it, or it is entangled with a register that is live (the
  paper's Definition 3), or it is the program's result.

The analysis is sound while "superposed" is never under-reported: marking a register superposed when it is not only
joins more.
"""
from typing import Dict, FrozenSet, List, NamedTuple, Sequence, Set, Tuple

from compiler.parser.instructions import (
    Alloc,
    Ctrl,
    Dist,
    EndCtrl,
    Free,
    ICtrl,
    IDist,
    Instruction,
    IOp,
    IsaProgram,
    Op,
    Reg,
    Update,
)


def reads(instruction: Instruction) -> FrozenSet[str]:
    """The registers an instruction reads: for an update or an inverse, the register it changes too."""
    if isinstance(instruction, (Op, IOp)):
        names = {s.name for s in instruction.sources if isinstance(s, Reg)}
        return frozenset(names | ({instruction.dst} if isinstance(instruction, IOp) else set()))
    if isinstance(instruction, Update):
        return frozenset({instruction.dst} | ({instruction.source.name} if isinstance(instruction.source, Reg) else set()))
    if isinstance(instruction, IDist):
        return frozenset({instruction.dst})
    if isinstance(instruction, (Ctrl, ICtrl)):
        return frozenset({instruction.control.name} if isinstance(instruction.control, Reg) else set())
    return frozenset()


class State:
    """The superposed registers and the entanglement graph while a program runs, one instruction at a time."""

    def __init__(self) -> None:
        self.held: List[str] = []  # in the order they were allocated
        self.superposed: Set[str] = set()
        self.edges: Set[FrozenSet[str]] = set()
        self.controls: List[str] = []  # the register each open region is controlled on

    def groups(self) -> List[FrozenSet[str]]:
        """The held registers, grouped by what is entangled with what, directly or through others."""
        parent = {r: r for r in self.held}

        def find(name: str) -> str:
            while parent[name] != name:
                parent[name] = parent[parent[name]]
                name = parent[name]
            return name

        for edge in self.edges:
            a, b = tuple(edge)
            if a in parent and b in parent:
                parent[find(a)] = find(b)
        groups: Dict[str, Set[str]] = {}
        for name in self.held:
            groups.setdefault(find(name), set()).add(name)
        return [frozenset(group) for group in groups.values()]

    def group_of(self, name: str) -> FrozenSet[str]:
        return next(group for group in self.groups() if name in group)

    def _join(self, name: str, others: Sequence[str]) -> None:
        self.edges |= {frozenset((name, other)) for other in others if other != name}

    def _forget(self, name: str) -> None:
        """The register is zero now, so it is definite and alone; what it joined stays joined to each other."""
        self.superposed.discard(name)
        neighbours = sorted({x for edge in self.edges if name in edge for x in edge if x != name})
        self.edges = {e for e in self.edges if name not in e}
        self.edges |= {frozenset((a, b)) for i, a in enumerate(neighbours) for b in neighbours[i + 1:]}

    def apply(self, instruction: Instruction) -> None:
        under = [c for c in self.controls if c in self.superposed]
        if isinstance(instruction, Alloc):
            self.held.append(instruction.register)
        elif isinstance(instruction, Free):
            self._forget(instruction.register)
            self.held.remove(instruction.register)
        elif isinstance(instruction, (Ctrl, ICtrl)):
            self.controls.append(instruction.control.name if isinstance(instruction.control, Reg) else "")
        elif isinstance(instruction, EndCtrl):
            self.controls.pop()
        elif isinstance(instruction, Dist):
            if instruction.stop - instruction.start > 1 or under:
                self.superposed.add(instruction.dst)
            else:
                self.superposed.discard(instruction.dst)
            self._join(instruction.dst, under)
        elif isinstance(instruction, Op):
            joined = [s.name for s in instruction.sources if isinstance(s, Reg) and s.name in self.superposed] + under
            if joined:
                self.superposed.add(instruction.dst)
                self._join(instruction.dst, joined)
            else:
                self.superposed.discard(instruction.dst)
        elif isinstance(instruction, Update):
            joined = ([instruction.source.name] if isinstance(instruction.source, Reg) and instruction.source.name in self.superposed else []) + under
            if joined:
                self.superposed.add(instruction.dst)
                self._join(instruction.dst, joined)
        elif isinstance(instruction, (IOp, IDist)):
            self._forget(instruction.dst)


class Point(NamedTuple):
    held: Tuple[str, ...]
    superposed: FrozenSet[str]
    groups: Tuple[FrozenSet[str], ...]
    live: FrozenSet[str]


def read_after(program: IsaProgram) -> List[FrozenSet[str]]:
    """For each point, the registers read by the instructions still to run, and the result at the end."""
    result = frozenset({program.result.name}) if isinstance(program.result, Reg) else frozenset()
    after = [result] * (len(program.body) + 1)
    for k in range(len(program.body) - 1, -1, -1):
        after[k] = after[k + 1] | reads(program.body[k])
    return after


def live_registers(state: State, used: FrozenSet[str]) -> FrozenSet[str]:
    """Everything held that is read later or entangled with something that is."""
    return frozenset(r for group in state.groups() if group & used for r in group)


def analyse(program: IsaProgram) -> Tuple[Point, ...]:
    """Point k is the state after the first k instructions."""
    state, after, points = State(), read_after(program), []
    for k in range(len(program.body) + 1):
        if k:
            state.apply(program.body[k - 1])
        points.append(Point(tuple(state.held), frozenset(state.superposed), tuple(state.groups()),
                            live_registers(state, after[k])))
    return tuple(points)
