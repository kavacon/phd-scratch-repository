"""
Optimising a circuit with Qiskit's own optimiser at its highest level, into `u` and `cx`. It is treated as a black
box: it also rewrites every gate into those two, since a multi-controlled X cannot be counted as one gate on a device,
so there is no separate decomposition step.
"""
import importlib.util
from typing import Optional

from qiskit import QuantumCircuit, transpile

from compiler.compiler import CIRCUIT, compile_pass
from compiler.system import System

BASIS = ["u", "cx"]  # the gates every optimised circuit is rewritten into


def qiskit_available() -> bool:
    """Whether Qiskit can also read OpenQASM text (qiskit-qasm3-import), which only checking needs."""
    return importlib.util.find_spec("qiskit_qasm3_import") is not None


@compile_pass(
    CIRCUIT, CIRCUIT,
    title="Optimise (Qiskit)",
    summary="Qiskit's optimiser as a black box, rewriting every gate into u and cx.",
)
def optimise(circuit: QuantumCircuit, system: Optional[System] = None) -> QuantumCircuit:
    """The circuit rewritten into `u` and `cx` and optimised by Qiskit."""
    return transpile(circuit, basis_gates=BASIS, optimization_level=3, seed_transpiler=0)


optimise_circuit = optimise  # for callers with a Qiskit circuit of their own and no system
