"""
Checking and measuring circuits. Nothing here is part of the compiler: it runs the generated gates on bits to check
them against the instruction machine, and uses Qiskit (an optional dependency) to read OpenQASM back, count gates and
wires, and run its own decomposition and optimisation as a black box.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from passes.qasm import C, Circuit, Gate, Q


def run_gates(gates: Sequence[Gate], qubits: int, initial: Sequence[int] = ()) -> List[int]:
    """The final state of every qubit after running the gates, starting from `initial` (default all zeros)."""
    state = list(initial) + [0] * (qubits - len(initial))
    for gate in gates:
        if all(state[q] == int(polarity) for q, polarity in gate.controls):
            state[gate.target] ^= 1
    return state


def value_of(state: Sequence[int], result: Sequence[Any]) -> int:
    return sum((state[b.index] if isinstance(b, Q) else b.value) << i for i, b in enumerate(result))


def leftover_qubits(state: Sequence[int], result: Sequence[Any]) -> List[int]:
    """Qubits that are not zero and are not part of the returned value."""
    returned = {b.index for b in result if isinstance(b, Q)}
    return [q for q, bit in enumerate(state) if bit and q not in returned]


def gates_from_text(text: str) -> Tuple[int, List[Gate]]:
    """Read the gates back out of OpenQASM text with the reference parser, so the printed text is what gets checked."""
    import openqasm3
    from openqasm3 import ast

    qubits, gates = 0, []
    for statement in openqasm3.parse(text).statements:
        if isinstance(statement, ast.QubitDeclaration):
            qubits = statement.size.value
        elif isinstance(statement, ast.QuantumMeasurementStatement):
            continue  # measurement does not change the state the gates leave
        elif isinstance(statement, ast.QuantumGate):
            wires = [w.indices[0][0].value for w in statement.qubits]
            positive = negative = 0
            if statement.name.name == "cx":
                positive = 1
            elif statement.name.name == "ccx":
                positive = 2
            elif statement.name.name != "x":
                raise ValueError(f"unexpected gate {statement.name.name}")
            for modifier in statement.modifiers:
                count = modifier.argument.value if modifier.argument is not None else 1
                if modifier.modifier == ast.GateModifierName.ctrl:
                    positive += count
                elif modifier.modifier == ast.GateModifierName.negctrl:
                    negative += count
                else:
                    raise ValueError(f"unexpected modifier {modifier.modifier}")
            controls = [(q, True) for q in wires[:positive]] + [(q, False) for q in wires[positive:positive + negative]]
            gates.append(Gate(wires[-1], tuple(controls)))
    return qubits, gates


# --- Qiskit: reading OpenQASM back, measuring, and the black-box decomposition and optimisation ---------------------

BASIS = ["u", "cx"]


def qiskit_available() -> bool:
    try:
        import qiskit  # noqa: F401
        import qiskit_qasm3_import  # noqa: F401
    except ImportError:
        return False
    return True


def read(text: str):
    from qiskit import qasm3
    return qasm3.loads(text)


def write(circuit) -> str:
    from qiskit import qasm3
    return qasm3.dumps(circuit)


def optimise(circuit):
    """
    Qiskit's own optimiser at its highest level, into `u` and `cx`. Treated as a black box: it also rewrites every
    gate into those two, so there is no separate decomposition step.
    """
    from qiskit import transpile
    return transpile(circuit, basis_gates=BASIS, optimization_level=3, seed_transpiler=0)


def measure(circuit) -> Dict[str, Any]:
    """Gate and wire counts for a Qiskit circuit."""
    touched = {circuit.find_bit(q).index for instruction in circuit.data for q in instruction.qubits}
    ops = {name: count for name, count in circuit.count_ops().items()}
    return {
        "gates": sum(count for name, count in ops.items() if name not in ("measure", "barrier")),
        "measurements": ops.get("measure", 0),
        "ops": ops,
        "two_qubit": sum(1 for i in circuit.data if len(i.qubits) >= 2),
        "depth": circuit.depth(),
        "wires": circuit.num_qubits,
        "touched": len(touched),
    }


def run_state(circuit, result: Sequence[Any], limit: int = 16):
    """
    Run a Qiskit circuit from all zeros on a state vector and read the returned value and any other qubit left
    set, or None if the circuit has more than `limit` qubits (the state vector would be too big).
    """
    if circuit.num_qubits > limit:
        return None
    from qiskit.quantum_info import Statevector
    probabilities = Statevector(circuit.remove_final_measurements(inplace=False)).probabilities_dict()
    bitstring, probability = max(probabilities.items(), key=lambda item: item[1])
    if probability < 0.999:
        return {"value": None, "leaked": ["not a single basis state"]}
    state = [int(c) for c in reversed(bitstring)]  # the string has qubit 0 last
    return {"value": value_of(state, result), "leaked": leftover_qubits(state, result)}


# --- drawing ---------------------------------------------------------------------------------------------------

MAX_LAYERS = 300  # a circuit can have thousands of layers, so only the start is drawn


def _layered(gates: Sequence[Dict[str, Any]], wires: int) -> Dict[str, Any]:
    """Place each gate in the first layer where every wire between its highest and lowest qubit is free."""
    busy = [0] * wires
    placed, layers = [], 0
    for gate in gates:
        touched = [gate["target"]] + [q for q, _ in gate["controls"]]
        low, high = min(touched), max(touched)
        layer = 1 + max(busy[low:high + 1])
        if layer > MAX_LAYERS:
            break
        for q in range(low, high + 1):
            busy[q] = layer
        placed.append({**gate, "layer": layer})
        layers = max(layers, layer)
    return {"gates": placed, "layers": layers, "total": len(gates), "shown": len(placed)}


def diagram(circuit: Circuit, rewritten=None) -> Dict[str, Any]:
    """
    Data for drawing a circuit: a label per wire, and the gates placed in layers. With `rewritten` (a Qiskit circuit)
    the gates come from it, otherwise from the generated gates.
    """
    width = circuit.register_width
    used = circuit.qubits // width - 1
    labels = [f"R{q // width}[{q % width}]" if q // width < used else f"scratch[{q % width}]" for q in range(circuit.qubits)]
    gates = []
    if rewritten is None:
        for g in circuit.gates:
            gates.append({"kind": "x", "target": g.target, "controls": [[q, p] for q, p in g.controls], "label": "X"})
        for q in circuit.measured:
            gates.append({"kind": "meter", "target": q, "controls": [], "label": "M", "detail": f"measure q[{q}]"})
    else:
        for instruction in rewritten.data:
            name = instruction.operation.name
            qubits = [rewritten.find_bit(q).index for q in instruction.qubits]
            if name == "cx":
                gates.append({"kind": "x", "target": qubits[1], "controls": [[qubits[0], True]], "label": "cx"})
            elif name == "measure":
                gates.append({"kind": "meter", "target": qubits[0], "controls": [], "label": "M", "detail": f"measure q[{qubits[0]}]"})
            else:
                params = ", ".join(f"{float(p):.3g}" for p in instruction.operation.params)
                gates.append({"kind": "box", "target": qubits[0], "controls": [], "label": name.upper(),
                              "detail": f"{name}({params})"})
    return {"wires": labels, "register_width": width, **_layered(gates, circuit.qubits)}
