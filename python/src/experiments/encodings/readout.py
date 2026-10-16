"""
How to read a circuit's answer. A lowering chooses it, because an encoding that cannot be read exactly needs more than
one shot, and it travels on the circuit (its metadata) so the experiment runs each circuit its own way. A circuit with
no recipe is read as plain two's complement bits in one shot, which is what the shared binary lowering produces.
"""
from dataclasses import dataclass
from typing import Dict

from qiskit import QuantumCircuit

from checking.circuits import to_signed


@dataclass(frozen=True)
class Readout:
    shots: int  # 1 for binary and Fourier
    decode: str  # how counts become a value, by name: "signed-bits"

    def attach(self, circuit: QuantumCircuit) -> QuantumCircuit:
        circuit.metadata = {**(circuit.metadata or {}), "readout": {"shots": self.shots, "decode": self.decode}}
        return circuit


def readout_of(circuit: QuantumCircuit) -> Readout:
    recipe = (circuit.metadata or {}).get("readout")
    return Readout(recipe["shots"], recipe["decode"]) if recipe else Readout(1, "signed-bits")


def decode(readout: Readout, counts: Dict[int, int], width: int) -> int:
    """The value the counts of a run stand for."""
    if readout.decode != "signed-bits":
        raise ValueError(f"unknown decoder {readout.decode}")
    if len(counts) != 1:
        raise ValueError(f"expected one outcome from a {readout.shots}-shot read of plain bits, got {sorted(counts)}")
    (value,) = counts
    return to_signed(value, width)
