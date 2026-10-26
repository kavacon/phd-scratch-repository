"""
Data for the explorer, built from what the passes and the instruction machine produce: which registers of a program are
entangled according to the state, and which would be by the rule the POC's liveness uses, that any operation acting on
several variables entangles them.
"""
import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

from compiler.parser.instructions import Alloc, Ctrl, Dist, EndCtrl, ICtrl, IDist, IOp, IsaProgram, Op, Reg, Update

MOST_REGISTERS = 12  # every split of the registers is tried, so the work doubles with each one


def entangled_groups(names: Sequence[str], state: Dict[tuple, complex]) -> List[List[str]]:
    """
    The registers of a state vector, split into the finest groups whose state is the product of one state per group.
    A split is a product when the matrix of amplitudes, with one side's values down and the other's across, has rank one.
    A register holding one value in every entry always comes out alone. Raises ValueError for too many registers.
    """
    if len(names) > MOST_REGISTERS:
        raise ValueError(f"{len(names)} registers are too many to separate exactly (at most {MOST_REGISTERS})")

    def product(a: Tuple[int, ...], b: Tuple[int, ...]) -> bool:
        rows = {values: n for n, values in enumerate(sorted({tuple(key[i] for i in a) for key in state}))}
        cols = {values: n for n, values in enumerate(sorted({tuple(key[i] for i in b) for key in state}))}
        matrix = np.zeros((len(rows), len(cols)), dtype=complex)
        for key, amplitude in state.items():
            matrix[rows[tuple(key[i] for i in a)], cols[tuple(key[i] for i in b)]] += amplitude
        return np.linalg.matrix_rank(matrix, tol=1e-9) <= 1

    def split(indices: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        first, rest = indices[0], indices[1:]
        for size in range(len(rest)):  # the smallest side holding `first` that stands apart
            for chosen in itertools.combinations(rest, size):
                a, b = (first, *chosen), tuple(i for i in rest if i not in chosen)
                if product(a, b):
                    return split(a) + split(b)
        return [indices]

    if not names:
        return []
    return sorted(sorted(names[i] for i in part) for part in split(tuple(range(len(names)))))


def groups_by_joint_operation(program: IsaProgram) -> List[List[str]]:
    """The registers of a program grouped as the POC's rule does: whatever one operation reads and writes, with the
    controls it sits under, is in one group, whether or not anything is superposed."""
    parent: Dict[str, str] = {i.register: i.register for i in program.body if isinstance(i, Alloc)}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    controls: List[str] = []
    for instruction in program.body:
        if isinstance(instruction, (Ctrl, ICtrl)):
            controls.append(instruction.control.name if isinstance(instruction.control, Reg) else "")
            continue
        if isinstance(instruction, EndCtrl):
            controls.pop()
            continue
        if isinstance(instruction, (Op, IOp)):
            joined = [instruction.dst] + [s.name for s in instruction.sources if isinstance(s, Reg)]
        elif isinstance(instruction, Update):
            joined = [instruction.dst] + ([instruction.source.name] if isinstance(instruction.source, Reg) else [])
        elif isinstance(instruction, (Dist, IDist)):
            joined = [instruction.dst]
        else:
            continue
        joined = [n for n in joined + controls if n in parent]
        for name in joined[1:]:
            parent[find(name)] = find(joined[0])
    groups: Dict[str, List[str]] = {}
    for name in parent:
        groups.setdefault(find(name), []).append(name)
    return sorted(sorted(group) for group in groups.values())
