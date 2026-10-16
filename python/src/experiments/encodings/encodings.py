"""
The encodings the experiment compares. Each is a name for the command line, a title, and the lowering pass that turns
instructions into a circuit in that encoding. Everything before the lowering (parse, normalise, instructions,
allocation) is the same for every encoding.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from compiler.passes.qasm import to_openqasm
from experiments.encodings.passes.angle import to_circuit_angle
from experiments.encodings.passes.binary_ttk import to_circuit_binary_ttk
from experiments.encodings.passes.fourier import to_circuit_fourier
from experiments.encodings.passes.fourier_dual import to_circuit_fourier_dual


@dataclass(frozen=True)
class Encoding:
    name: str  # the value of the --encoding flag
    title: str
    summary: str
    lowering: Callable  # the ISA -> circuit pass


ENCODINGS: Dict[str, Encoding] = {e.name: e for e in (
    Encoding("binary", "Binary, Cuccaro adder", "Computational-basis binary with a ripple-carry adder: the baseline.",
             to_openqasm),
    Encoding("binary-ttk", "Binary, no-ancilla adder", "Binary with the Takahashi, Tani and Kunihiro adder.",
             to_circuit_binary_ttk),
    Encoding("fourier", "Fourier, rotation adder", "Integers as phases; register add by controlled rotations.",
             to_circuit_fourier),
    Encoding("fourier-dual", "Fourier, roles-swapped adder", "Integers as phases; register add by the binary adder, roles swapped.",
             to_circuit_fourier_dual),
    Encoding("angle", "Angle, one qubit per value", "A value as one qubit's phase; many-shot readout.",
             to_circuit_angle),
)}

DEFAULT: Tuple[str, ...] = ("binary", "binary-ttk", "fourier")
