"""
Running Qrisp programs written as source text, to compare their circuits with ours. A program defines `main()` that returns
the register to measure; it is run, compiled by Qrisp, and read into a Qiskit circuit with that register measured. Qrisp is an
optional dependency (the `compare` extra).
"""
import contextlib
import io
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class QrispCircuit:
    circuit: object  # a Qiskit QuantumCircuit, with the returned register measured
    result: Tuple[int, ...]  # the qubit of each bit of the returned register (its mantissa), least significant first
    exponent: int  # a QuantumFloat holds mantissa * 2 ** exponent
    labels: Tuple[str, ...]  # a label per wire, from the Qrisp variable it belongs to
    groups: Tuple[int, ...]  # which variable each wire belongs to, to colour labels


def build(source: str) -> QrispCircuit:
    """Run a Qrisp program's `main()` and compile what it returns to a Qiskit circuit with that register measured."""
    from qiskit import ClassicalRegister

    namespace: Dict[str, object] = {}
    exec(source, namespace)  # trusted: the programs the apps show, exactly as shown
    with contextlib.redirect_stdout(io.StringIO()):  # Qrisp prints progress while it simulates
        result = namespace["main"]()
        compiled = result.qs.compile(intended_measurements=list(result.reg))
        indices = tuple(compiled.qubits.index(q) for q in result.reg)
        circuit = compiled.to_qiskit()
        exponent = int(getattr(result, "exponent", 0))  # a boolean has none
    labels, groups, seen = [], [], {}
    for qubit in circuit.qubits:
        registers = circuit.find_bit(qubit).registers
        label = registers[0][0].name if registers else str(len(labels))
        labels.append(label)
        groups.append(seen.setdefault(label.split(".")[0], len(seen)))
    circuit.add_register(ClassicalRegister(len(indices), "result"))
    for position, qubit in enumerate(indices):
        circuit.measure(qubit, position)
    return QrispCircuit(circuit, indices, exponent, tuple(labels), tuple(groups))
