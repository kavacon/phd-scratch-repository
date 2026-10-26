"""
Thin helpers over Qiskit and Qiskit Aer (Aer and the OpenQASM reader are optional extras): reading and writing OpenQASM 3, counting gates and
wires, and running a circuit on a simulator. The optimiser is a compiler pass (`compiler.passes.optimise`).
"""
import importlib.util
from typing import Any, Dict, List, Tuple

from compiler.passes.optimise import BASIS, optimise, qiskit_available  # noqa: F401  (re-exported)

MAX_SHOTS = 4096


def aer_available() -> bool:
    return importlib.util.find_spec("qiskit_aer") is not None


def read(text: str):
    """OpenQASM 3 text as a Qiskit circuit."""
    from qiskit import qasm3
    return qasm3.loads(text)


def write(circuit) -> str:
    """A Qiskit circuit as OpenQASM 3 text."""
    from qiskit import qasm3
    return qasm3.dumps(circuit)


def measure(circuit) -> Dict[str, Any]:
    """Gate and wire counts for a Qiskit circuit, from Qiskit's own counters."""
    from qiskit.converters import circuit_to_dag

    ops = dict(circuit.count_ops())
    measurements = ops.get("measure", 0)
    return {
        "gates": circuit.size() - measurements,  # size() already leaves out barriers
        "measurements": measurements,
        "ops": ops,
        "two_qubit": circuit.num_nonlocal_gates(),
        "depth": circuit.depth(),
        "wires": circuit.num_qubits,
        "touched": circuit.num_qubits - len(list(circuit_to_dag(circuit).idle_wires())),
    }


def _with_rest_measured(circuit):
    """A copy of `circuit` that also measures every qubit it does not already measure, and which qubits those are."""
    from qiskit import ClassicalRegister

    run = circuit.copy()
    already = {run.find_bit(i.qubits[0]).index for i in run.data if i.operation.name == "measure"}
    rest = [q for q in range(run.num_qubits) if q not in already]
    if rest:
        run.add_register(ClassicalRegister(len(rest), "rest"))
        for i, q in enumerate(rest):
            run.measure(q, run.num_clbits - len(rest) + i)
    return run, rest


def execute(circuit, result_bits: int, shots: int = 8) -> Dict[str, Any]:
    """
    Run a Qiskit circuit from all zeros on Qiskit Aer and read what it returns. The circuits only ever hold one
    computational basis state, so the matrix product state method simulates them at any width; the other qubits are
    measured too (in a copy), so a qubit left set shows up.

    Returns the counts of the returned value, the qubits that were ever set outside it, and whether every shot agreed.
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    if circuit.num_clbits != result_bits:
        raise ValueError("the circuit must measure exactly the returned register")
    run, rest = _with_rest_measured(circuit)
    simulator = AerSimulator(method="matrix_product_state")
    runnable = transpile(run, basis_gates=BASIS, optimization_level=0)  # the simulator runs u and cx
    counts = simulator.run(runnable, shots=shots).result().get_counts()
    values: Dict[int, int] = {}
    leftover = set()
    for key, count in counts.items():
        *rest_bits, value = key.split()  # registers are listed last first: "rest result"
        values[int(value, 2)] = values.get(int(value, 2), 0) + count
        for position, bit in enumerate(reversed(rest_bits[0]) if rest_bits else ()):
            if bit == "1":
                leftover.add(rest[position])
    return {"counts": values, "value": next(iter(values)) if len(values) == 1 else None,
            "leaked": sorted(leftover), "deterministic": len(values) == 1}


def measured_qubits(circuit) -> List[int]:
    """The qubit measured into each classical bit, in the order of the bits (so least significant first for a register)."""
    by_bit = {circuit.find_bit(i.clbits[0]).index: circuit.find_bit(i.qubits[0]).index
              for i in circuit.data if i.operation.name == "measure"}
    return [by_bit[bit] for bit in sorted(by_bit)]


def final_probabilities(circuit) -> Tuple[List[int], Dict[int, float]]:
    """
    The probability of each state of every qubit a circuit uses (or measures) at its end, with the final measurements left
    out: the qubits, and a dictionary from the state (bit k of the number is the k-th of those qubits) to its probability.
    The matrix product state method of Qiskit Aer gives these exactly for the circuits here, which are too wide for a
    state vector: it finds the qubits that are ever set and lists the states of just those. Phases are not part of it.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    run = circuit.remove_final_measurements(inplace=False)
    used = {run.find_bit(q).index for i in run.data for q in i.qubits} | set(measured_qubits(circuit))
    kept = sorted(used)
    position = {q: k for k, q in enumerate(kept)}
    reduced = QuantumCircuit(len(kept))
    for instruction in run.data:
        if instruction.operation.name != "barrier":
            reduced.append(instruction.operation, [position[run.find_bit(q).index] for q in instruction.qubits])
    runnable = transpile(reduced, basis_gates=BASIS, optimization_level=0)
    simulator = AerSimulator(method="matrix_product_state")
    probe = runnable.copy()  # first, which qubits are ever set, so the full list of states only needs those
    for k in range(len(kept)):
        probe.save_probabilities(qubits=[k], label=f"q{k}")
    saved = simulator.run(probe).result().data(0)
    active = [k for k in range(len(kept)) if saved[f"q{k}"][1] > 1e-9]
    if not active:
        return kept, {0: 1.0}
    runnable.save_probabilities_dict(qubits=active)
    states = simulator.run(runnable).result().data(0)["probabilities"]
    return kept, {sum(((state >> j) & 1) << k for j, k in enumerate(active)): p for state, p in states.items() if p > 1e-9}
