"""
Data for drawing circuits, shared by the pipeline explorer and the comparison app (static/circuit.js draws it): one label
per wire, and the gates placed in layers. Gates are plain dicts with a `kind` ("x" for an X with controls, "box" for a
gate on one wire, "block" for a gate on several, "meter" for a measurement).
"""
from typing import Any, Dict, List, Optional, Sequence

from passes.qasm import Circuit

MAX_LAYERS = 300  # a circuit can have thousands of layers, so only the start is drawn


def _controlled_x(target: int, controls: Sequence[Sequence[Any]], label: str) -> Dict[str, Any]:
    return {"kind": "x", "target": target, "controls": [list(control) for control in controls], "label": label}


def _meter(qubit: int) -> Dict[str, Any]:
    return {"kind": "meter", "target": qubit, "controls": [], "label": "M", "detail": f"measure q[{qubit}]"}


def _box(name: str, qubit: int, params: Sequence[float]) -> Dict[str, Any]:
    shown = ", ".join(f"{float(p):.3g}" for p in params)
    return {"kind": "box", "target": qubit, "controls": [], "label": name.upper(), "detail": f"{name}({shown})" if shown else name}


def _block(name: str, qubits: Sequence[int]) -> Dict[str, Any]:
    return {"kind": "block", "qubits": list(qubits), "target": qubits[0], "controls": [], "label": name, "detail": name}


def _layered(gates: Sequence[Dict[str, Any]], wires: int) -> Dict[str, Any]:
    """Place each gate in the first layer where every wire between its highest and lowest qubit is free."""
    busy = [0] * wires
    placed, layers = [], 0
    for gate in gates:
        touched = gate.get("qubits") or [gate["target"], *(qubit for qubit, _ in gate["controls"])]
        low, high = min(touched), max(touched)
        layer = 1 + max(busy[low:high + 1])
        if layer > MAX_LAYERS:
            break
        for qubit in range(low, high + 1):
            busy[qubit] = layer
        placed.append({**gate, "layer": layer})
        layers = max(layers, layer)
    return {"gates": placed, "layers": layers, "total": len(gates), "shown": len(placed)}


def qiskit_diagram(circuit, labels: Optional[Sequence[str]] = None, groups: Optional[Sequence[int]] = None,
                   register_width: int = 0) -> Dict[str, Any]:
    """
    Data for drawing any Qiskit circuit (ours after Qiskit's optimiser, or one built by Qrisp). X gates with controls,
    positive or negative, are drawn as such; other single-qubit gates as a box; anything wider as one tall box.
    `groups` says which variable each wire belongs to and `register_width` how many wires make a register, so the
    viewer can colour the labels alternately.
    """
    gates: List[Dict[str, Any]] = []
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        base = getattr(operation, "base_gate", None)
        if operation.name == "measure":
            gates.append(_meter(qubits[0]))
        elif operation.name == "barrier":
            continue
        elif operation.name == "x" or (base is not None and base.name == "x"):
            count = getattr(operation, "num_ctrl_qubits", 0)
            state = getattr(operation, "ctrl_state", (1 << count) - 1)
            gates.append(_controlled_x(qubits[-1], [[qubits[i], bool((state >> i) & 1)] for i in range(count)], operation.name))
        elif len(qubits) == 1:
            gates.append(_box(operation.name, qubits[0], [p for p in operation.params if isinstance(p, (int, float))]))
        else:
            gates.append(_block(operation.name, qubits))
    picture = {
        "wires": list(labels) if labels is not None else [f"q{i}" for i in range(circuit.num_qubits)],
        "register_width": register_width,
        **_layered(gates, circuit.num_qubits),
    }
    if groups is not None:
        picture["groups"] = list(groups)
    return picture


def wire_labels(circuit: Circuit) -> List[str]:
    """One label per wire of our circuit: slot and bit, with the scratch register last."""
    width = circuit.register_width
    slots = circuit.qubits // width - 1
    return [f"R{q // width}[{q % width}]" if q // width < slots else f"scratch[{q % width}]" for q in range(circuit.qubits)]


def diagram(circuit: Circuit, rewritten=None) -> Dict[str, Any]:
    """
    Data for drawing our circuit. With `rewritten` (the same circuit after Qiskit's optimiser) the gates come from it,
    otherwise from the generated gates, so the drawing does not need Qiskit.
    """
    labels = wire_labels(circuit)
    if rewritten is not None:
        return qiskit_diagram(rewritten, labels, register_width=circuit.register_width)
    gates = [_controlled_x(gate.target, gate.controls, "X") for gate in circuit.gates]
    gates += [_meter(qubit) for qubit in circuit.measured]
    return {"wires": labels, "register_width": circuit.register_width, **_layered(gates, circuit.qubits)}
