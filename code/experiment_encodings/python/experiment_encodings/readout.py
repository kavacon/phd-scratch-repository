"""
How to read a circuit's answer. A lowering chooses it, because an encoding that cannot be read exactly needs more than
one shot, and it travels on the circuit (its metadata) so the experiment runs each circuit its own way. A circuit with
no recipe is read as plain two's complement bits in one shot, which is what the shared binary lowering produces.
"""
import math
from dataclasses import dataclass
from typing import Dict, Sequence

from qiskit import QuantumCircuit

from checking.circuits import to_signed


@dataclass(frozen=True)
class Readout:
    shots: int  # 1 for binary and Fourier
    decode: str  # how counts become a value, by name: "signed-bits", or "rns:" and the moduli, such as "rns:3,4,5"

    def attach(self, circuit: QuantumCircuit) -> QuantumCircuit:
        circuit.metadata = {**(circuit.metadata or {}), "readout": {"shots": self.shots, "decode": self.decode}}
        return circuit


def readout_of(circuit: QuantumCircuit) -> Readout:
    recipe = (circuit.metadata or {}).get("readout")
    return Readout(recipe["shots"], recipe["decode"]) if recipe else Readout(1, "signed-bits")


def remainders_to_value(remainders: Sequence[int], moduli: Sequence[int]) -> int:
    """The number below the product of the moduli that has these remainders (the Chinese remainder theorem)."""
    value, step = 0, 1
    for remainder, modulus in zip(remainders, moduli, strict=True):
        while value % modulus != remainder:
            value += step
        step *= modulus
    return value


def decode(readout: Readout, counts: Dict[int, int], width: int) -> int:
    """The value the counts of a run stand for."""
    if readout.decode != "signed-bits" and not readout.decode.startswith("rns:"):
        raise ValueError(f"unknown decoder {readout.decode}")
    if len(counts) != 1:
        raise ValueError(f"expected one outcome from a {readout.shots}-shot read of plain bits, got {sorted(counts)}")
    (value,) = counts
    if readout.decode == "signed-bits":
        return to_signed(value, width)
    moduli = [int(m) for m in readout.decode[len("rns:"):].split(",")]
    remainders = []
    for modulus in moduli:  # each remainder is held in as many bits as its largest value needs, the first modulus in the lowest bits
        size = (modulus - 1).bit_length()
        remainders.append(value & ((1 << size) - 1))
        value >>= size
    total = math.prod(moduli)
    number = remainders_to_value(remainders, moduli)
    return number - total if number >= total // 2 else number
