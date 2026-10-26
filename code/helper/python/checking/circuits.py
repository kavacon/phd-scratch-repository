"""
Checking generated circuits against the program they came from. Nothing here is part of the compiler. It runs generated
X gates on bits (to check them against the instruction machine), reads OpenQASM back, and reports a simulator run against
the value the program should return. The Qiskit side is in `checking.qiskit_tools`.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def distribution_report(outcome: Dict[str, Any], *, shots: int, expected: Dict[int, float], scale: int = 1) -> Dict[str, Any]:
    """
    A run's outcome as the apps show it when the program returns a distribution: for each value, the probability the program
    should give it and the fraction of shots that returned it, and how far apart the two are (half the sum of the absolute
    differences). The run agrees if that distance is within what `shots` samples would give by chance.
    """
    counts = {value * scale: count for value, count in outcome["counts"].items()}
    values = sorted(set(expected) | set(counts))
    rows = [[v, expected.get(v, 0.0), counts.get(v, 0) / shots] for v in values]
    distance = 0.5 * sum(abs(p - q) for _, p, q in rows)
    return {
        "shots": shots,
        "distribution": rows,
        "distance": distance,
        "tolerance": 2 / shots ** 0.5,
        "leaked": [f"q[{q}]" for q in outcome["leaked"]],
        "matches": distance <= 2 / shots ** 0.5,
    }


# --- the state a circuit ends in ---------------------------------------------------------------------------------

def register_probabilities(circuit, width: int) -> Dict[Tuple[int, ...], float]:
    """
    The probability of each combination of register values (signed, in the circuit's register order) the circuit ends in,
    from its state vector with the final measurements left out. It holds every qubit in memory, so it is for small
    circuits: the instruction machine's state is what to compare it with.
    """
    from qiskit.quantum_info import Statevector

    run = circuit.remove_final_measurements(inplace=False)
    probabilities: Dict[Tuple[int, ...], float] = {}
    for index, p in enumerate(Statevector.from_instruction(run).probabilities()):
        if p < 1e-12:
            continue
        values = []
        for register in run.qregs:
            value = sum(((index >> run.find_bit(q).index) & 1) << i for i, q in enumerate(register))
            values.append(to_signed(value, width))
        key = tuple(values)
        probabilities[key] = probabilities.get(key, 0.0) + p
    return probabilities



def state_report(circuit, names: Sequence[str], *, reference: Optional[Dict[tuple, float]] = None) -> Dict[str, Any]:
    """
    The state a circuit ends in, by register: `names` gives the register each qubit belongs to and the measured qubits are the
    result. It lists each state the circuit can end in (the value of each register, and its probability), which registers
    other than the result still hold something, and for each whether the result is entangled with it: the probabilities of
    the result and of that register together are not the product of the two, which a state with no entanglement cannot give,
    and whether the result determines it (each value of the result goes with one value of the register). A held register that
    is not entangled, or is determined by the result, could have been released, the first by undoing it and the second by
    computing it back from the result. One that is entangled and not determined is garbage a unitary program cannot avoid. A `reference`, the
    same states from another source (a dictionary from the non-zero (register, value) pairs, sorted, to probability), is
    compared with the registers' states and the report says whether they agree.
    """
    from checking import qiskit_tools

    result = qiskit_tools.measured_qubits(circuit)
    kept, probabilities = qiskit_tools.final_probabilities(circuit)
    position = {q: k for k, q in enumerate(kept)}
    registers = [names[result[0]]]
    bits: Dict[str, List[int]] = {registers[0]: [position[q] for q in result]}
    for q in kept:
        if names[q] not in bits:
            registers.append(names[q])
            bits[names[q]] = []
        if q not in result:
            bits[names[q]].append(position[q])
    table: Dict[Tuple[int, ...], float] = {}
    for state, p in probabilities.items():
        values = tuple(sum(((state >> at) & 1) << i for i, at in enumerate(bits[r])) for r in registers)
        table[values] = table.get(values, 0.0) + p

    def entangled(columns: Sequence[int]) -> bool:
        joint: Dict[tuple, float] = {}
        first: Dict[int, float] = {}
        rest: Dict[tuple, float] = {}
        for values, p in table.items():
            key = tuple(values[k] for k in columns)
            joint[values[0], key] = joint.get((values[0], key), 0.0) + p
            first[values[0]] = first.get(values[0], 0.0) + p
            rest[key] = rest.get(key, 0.0) + p
        return any(abs(joint.get((a, b), 0.0) - first[a] * rest[b]) > 1e-6 for a in first for b in rest)

    def determined(k: int) -> bool:
        seen: Dict[int, int] = {}
        return all(seen.setdefault(values[0], values[k]) == values[k] for values in table)

    held = [k for k in range(1, len(registers)) if any(values[k] for values in table)]
    report = {
        "registers": registers,
        "states": sorted(([*values, p] for values, p in table.items()), key=lambda row: row[:-1]),
        "qubits": len(kept),
        "result_qubits": len(result),
        "held": {registers[k]: {"qubits": len(bits[registers[k]]), "entangled": entangled([k]), "determined": determined(k)} for k in held},
        "result_entangled": entangled(held) if held else False,
        "reference": None,
    }
    if reference is not None:
        mine = {}
        for values, p in table.items():
            key = tuple(sorted((r, v) for r, v in zip(registers, values) if v))
            mine[key] = mine.get(key, 0.0) + p
        report["reference"] = {"matches": mine.keys() == reference.keys() and all(abs(mine[k] - reference[k]) < 1e-6 for k in mine)}
    return report
