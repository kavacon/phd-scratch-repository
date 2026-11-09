"""
Gate-level optimisers applied to the circuit a pipeline produces, so that the same circuit can be compared under each and with or
without the source-level rewrites.

A circuit is taken before the qubit pool (`assign-qubits`), where it has a wire for every register: the pool reuses qubits, which
puts unrelated gates next to each other on one wire and changes what an optimiser can cancel. Every optimiser's output is brought to
the same basis (Clifford, T and phases) so that T counts are comparable, and is a Qiskit circuit that `metrics.of_clifford_t` measures.

    none      the circuit written in the basis, nothing else
    mqt       MQT's default QCO pipeline (merge rotations, combine controlled gates, canonicalize), run in MLIR
    qiskit1   Qiskit's transpiler at optimisation level 1
    qiskit3   Qiskit's transpiler at optimisation level 3
    pyzx      ZX-calculus simplification (`full_reduce`), run in a child process that is stopped after a time limit

A Toffoli becomes the standard seven T gates; a gate with more controls becomes whatever Qiskit's synthesis gives, and the phases of
that have no exact T cost (they are counted as rotations).
"""
import subprocess
import sys
from typing import Callable, Dict

from checking import qiskit_tools
from experiment_existing_ir import pipeline

BASIS = ["cx", "h", "t", "tdg", "s", "sdg", "x", "y", "z", "rz", "ry", "rx", "p"]
TIMEOUT = 60  # seconds an optimiser may take on one circuit


class TimedOut(Exception):
    pass


def unpooled(stages):
    """A pipeline without the qubit pool."""
    return tuple(stage for stage in stages if stage != "--assign-qubits")


def read(qc: str):
    """The QC text as a Qiskit circuit with the named adders written out."""
    circuit = qiskit_tools.read(pipeline.export(qc, "openqasm3"))
    adders = lambda: [name for name in circuit.count_ops() if name.startswith("add")]
    while adders():
        circuit = circuit.decompose(gates_to_decompose=adders())
    return circuit


def clifford_t(circuit, level: int = 0):
    from qiskit import transpile
    return transpile(circuit, basis_gates=BASIS, optimization_level=level, seed_transpiler=0)


def none(qc: str):
    return clifford_t(read(qc))


def mqt(qc: str):
    return clifford_t(read(pipeline.optimise(qc, "--qc-to-qco", "--mqt-qco-default", "--qco-to-qc")))


def qiskit(level: int) -> Callable[[str], object]:
    return lambda qc: clifford_t(read(qc), level)


def pyzx(qc: str):
    """ZX simplification of the circuit without its measurements (which the extraction does not take), which are put back after."""
    from qiskit import QuantumCircuit, qasm2
    circuit = clifford_t(read(qc))
    body = QuantumCircuit(*circuit.qregs)
    measures = []
    for instruction in circuit.data:
        if instruction.operation.name == "measure":
            measures.append((circuit.find_bit(instruction.qubits[0]).index, circuit.find_bit(instruction.clbits[0]).index))
        else:
            body.append(instruction.operation, instruction.qubits)
    try:
        done = subprocess.run([sys.executable, "-m", "experiment_existing_ir.optimisers"], input=qasm2.dumps(body), capture_output=True,
                              text=True, timeout=TIMEOUT, check=True)
    except subprocess.TimeoutExpired as expired:
        raise TimedOut(f"pyzx stopped after {TIMEOUT} s") from expired
    optimised = QuantumCircuit.from_qasm_str(done.stdout)
    result = QuantumCircuit(optimised.num_qubits, circuit.num_clbits)
    result.compose(optimised, inplace=True)
    for qubit, bit in measures:
        result.measure(qubit, bit)
    return result


OPTIMISERS: Dict[str, Callable[[str], object]] = {"none": none, "mqt": mqt, "qiskit1": qiskit(1), "qiskit3": qiskit(3), "pyzx": pyzx}


def main() -> None:
    """The child process of `pyzx`: OpenQASM 2 in, the simplified circuit out."""
    import pyzx as zx
    graph = zx.Circuit.from_qasm(sys.stdin.read()).to_graph()
    zx.full_reduce(graph, quiet=True)
    print(zx.basic_optimization(zx.extract_circuit(graph.copy()).to_basic_gates()).to_qasm())


if __name__ == "__main__":
    main()
