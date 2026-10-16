"""Angle encoding: one qubit's phase holds a value, read out statistically over many shots."""
from typing import Optional

from qiskit import QuantumCircuit

from compiler.compiler import CIRCUIT, ISA, compile_pass
from compiler.parser.instructions import IsaProgram
from compiler.system import System


@compile_pass(
    ISA, CIRCUIT,
    title="Lower to an angle-encoded circuit",
    summary=(
        "One qubit per value, the integer stored as a phase. Adding a constant is one rotation; the value is read "
        "from the statistics of many shots, so the lowering also chooses how many."
    ),
)
def to_circuit_angle(program: IsaProgram, system: Optional[System] = None) -> QuantumCircuit:
    raise NotImplementedError("angle: constant add and subtract as rotations, and a many-shot readout recipe")
