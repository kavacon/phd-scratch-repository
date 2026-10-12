"""
Thin helpers over Qiskit and Qiskit Aer, optional dependencies: reading and writing OpenQASM 3, optimising a circuit as
a black box, counting gates and wires, and running a circuit on a simulator. Nothing here knows about the compiler.
"""
from typing import Any, Dict

BASIS = ["u", "cx"]  # the gates every circuit is rewritten into
MAX_SHOTS = 4096


def _importable(*modules: str) -> bool:
    import importlib.util
    return all(importlib.util.find_spec(module) is not None for module in modules)


def qiskit_available() -> bool:
    return _importable("qiskit", "qiskit_qasm3_import")


def aer_available() -> bool:
    return _importable("qiskit_aer")


def read(text: str):
    """OpenQASM 3 text as a Qiskit circuit."""
    from qiskit import qasm3
    return qasm3.loads(text)


def write(circuit) -> str:
    """A Qiskit circuit as OpenQASM 3 text."""
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
