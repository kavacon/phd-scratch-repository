"""
Checking generated circuits against the program they came from. Nothing here is part of the compiler. It runs generated
X gates on bits (to check them against the instruction machine), reads OpenQASM back, and reports a simulator run against
the value the program should return. The Qiskit side is in `checking.qiskit_tools`.
"""
from typing import Any, Dict, List, Sequence, Tuple

from compiler.passes.qasm import Gate

# --- running gates on bits ---------------------------------------------------------------------------------------

def run_gates(gates: Sequence[Gate], qubits: int, initial: Sequence[int] = ()) -> List[int]:
    """The final state of every qubit after running the gates, starting from `initial` (default all zeros)."""
    state = list(initial) + [0] * (qubits - len(initial))
    for gate in gates:
        if all(state[q] == int(polarity) for q, polarity in gate.controls):
            state[gate.target] ^= 1
    return state


def value_of(state: Sequence[int], measured: Sequence[int]) -> int:
    """The number the measured qubits (least significant bit first) hold in `state`."""
    return sum(state[qubit] << i for i, qubit in enumerate(measured))


def leftover_qubits(state: Sequence[int], measured: Sequence[int]) -> List[int]:
    """Qubits that are not zero and are not part of the returned value."""
    returned = set(measured)
    return [q for q, bit in enumerate(state) if bit and q not in returned]


def wrap(value: Any, width: int) -> int:
    """A program's value as a register of `width` bits holds it: unsigned, modulo 2 to the width."""
    return int(value) & ((1 << width) - 1)


def to_signed(value: int, width: int) -> int:
    """The two's complement reading of a register's `width` bits: -2^(width-1) to 2^(width-1)-1."""
    return value - (1 << width) if value >= 1 << (width - 1) else value


# --- reading OpenQASM back ---------------------------------------------------------------------------------------

def _controls(statement) -> Tuple[int, int]:
    """How many positive and how many negative controls a gate statement has, from its name and modifiers."""
    from openqasm3 import ast

    positive = {"x": 0, "cx": 1, "ccx": 2}.get(statement.name.name)
    if positive is None:
        raise ValueError(f"unexpected gate {statement.name.name}")
    negative = 0
    for modifier in statement.modifiers:
        count = modifier.argument.value if modifier.argument is not None else 1
        if modifier.modifier == ast.GateModifierName.ctrl:
            positive += count
        elif modifier.modifier == ast.GateModifierName.negctrl:
            negative += count
        else:
            raise ValueError(f"unexpected modifier {modifier.modifier}")
    return positive, negative


def gates_from_text(text: str) -> Tuple[int, List[Gate]]:
    """Read the gates back out of OpenQASM text with the reference parser, so the printed text is what gets checked."""
    import openqasm3
    from openqasm3 import ast

    qubits, gates = 0, []
    for statement in openqasm3.parse(text).statements:
        if isinstance(statement, ast.QubitDeclaration):
            qubits = statement.size.value
        elif isinstance(statement, ast.QuantumGate):  # measurements do not change the state the gates leave
            wires = [wire.indices[0][0].value for wire in statement.qubits]
            positive, negative = _controls(statement)
            controls = [(q, True) for q in wires[:positive]] + [(q, False) for q in wires[positive:positive + negative]]
            gates.append(Gate(wires[-1], tuple(controls)))
    return qubits, gates


def report(outcome: Dict[str, Any], *, shots: int, expected: Any, scale: int = 1) -> Dict[str, Any]:
    """
    A run's outcome as the apps show it: the measured values (each times `scale`, for a register that holds a mantissa
    and an exponent) with their counts, any qubits left set, and whether every shot returned the expected value.
    """
    counts = {value * scale: count for value, count in outcome["counts"].items()}
    return {
        "shots": shots,
        "counts": sorted(([value, count] for value, count in counts.items()), key=lambda item: -item[1]),
        "leaked": [f"q[{q}]" for q in outcome["leaked"]],
        "expected": expected,
        "matches": expected is not None and len(counts) == 1 and next(iter(counts)) == expected,
    }
