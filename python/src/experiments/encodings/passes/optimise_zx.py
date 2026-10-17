"""
A ZX-calculus optimiser, as a second way to optimise the circuit a lowering builds, for comparison with Qiskit's.

It uses PyZX's standard routines and nothing of its own: the circuit is expanded into Clifford+T gates, read into a ZX
graph, simplified with `full_reduce`, turned back into a circuit by `extract_circuit`, and tidied by `basic_optimization`.
The result stays in the Clifford+T gate set (h, s, t, rz, x, z and cx) PyZX works in, so a T gate is still one gate.

PyZX's extraction works on unitary circuits only, so the measurements at the end are taken off before the circuit goes in
and put back, on the same qubits and with the same readout recipe, afterwards. Everything else about the circuit, its
registers included, is kept.

PyZX is an optional dependency (the `zx` extra).
"""
from typing import Optional

from qiskit import ClassicalRegister, QuantumCircuit, qasm2, transpile

from compiler.compiler import CIRCUIT, compile_pass
from compiler.passes.qasm import measured_qubits
from compiler.system import System

CLIFFORD_T = ["cx", "h", "s", "sdg", "t", "tdg", "x", "z", "rz"]


def zx_available() -> bool:
    try:
        import pyzx  # noqa: F401
    except ImportError:
        return False
    return True


@compile_pass(
    CIRCUIT, CIRCUIT,
    title="Optimise with a ZX optimiser (PyZX)",
    summary=(
        "The circuit in Clifford+T gates, simplified as a ZX graph with PyZX's full_reduce and extracted back to gates, "
        "with the final measurements taken off for the optimiser and put back afterwards."
    ),
)
def optimise_zx(circuit: QuantumCircuit, system: Optional[System] = None) -> QuantumCircuit:
    import pyzx as zx

    measured = measured_qubits(circuit)
    body = circuit.remove_final_measurements(inplace=False)
    expanded = transpile(body, basis_gates=CLIFFORD_T, optimization_level=0)
    graph = zx.Circuit.from_qasm(qasm2.dumps(expanded)).to_graph()
    zx.simplify.full_reduce(graph)
    extracted = zx.basic_optimization(zx.extract_circuit(graph.copy()).to_basic_gates())
    optimised = QuantumCircuit(*circuit.qregs)
    optimised.compose(qasm2.loads(extracted.to_qasm()), qubits=list(range(circuit.num_qubits)), inplace=True)
    if measured:
        optimised.add_register(ClassicalRegister(len(measured), "result"))
        for bit, qubit in enumerate(measured):
            optimised.measure(optimised.qubits[qubit], optimised.clbits[bit])
    optimised.metadata = circuit.metadata
    return optimised
