"""Fourier (phase) encoding, with register addition done by the binary adder with the two roles swapped."""
from typing import Optional

from qiskit import QuantumCircuit

from compiler.compiler import CIRCUIT, ISA, compile_pass
from compiler.parser.instructions import IsaProgram
from compiler.system import System


@compile_pass(
    ISA, CIRCUIT,
    title="Lower to a Fourier-encoded circuit (adder with roles swapped)",
    summary=(
        "Integers held as phases; adding two registers runs the ordinary binary subtractor the other way round, "
        "which needs no conversion but uses the same gates as binary."
    ),
)
def to_circuit_fourier_dual(program: IsaProgram, system: Optional[System] = None) -> QuantumCircuit:
    raise NotImplementedError("fourier-dual: the binary adder circuit applied to Fourier-encoded registers, roles swapped")
