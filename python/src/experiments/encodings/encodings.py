"""
The encodings the experiment compares. Each is a name for the command line, a title, and the lowering pass that turns
instructions into a circuit in that encoding. Everything before the lowering (parse, normalise, instructions,
allocation) is the same for every encoding.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from compiler.passes.qasm import to_openqasm
from experiments.encodings.passes.binary_ttk import to_circuit_binary_ttk
from experiments.encodings.passes.fourier import to_circuit_fourier
from experiments.encodings.passes.fourier_dual import to_circuit_fourier_dual
from experiments.encodings.passes.rns import modulus_of, to_circuit_rns


@dataclass(frozen=True)
class Encoding:
    name: str  # the value of the --encoding flag
    title: str
    summary: str
    lowering: Callable  # the ISA -> circuit pass
    modulus: Optional[Callable] = None  # for an encoding whose range is not 2^width: its system -> how many values it holds


ENCODINGS: Dict[str, Encoding] = {e.name: e for e in (
    Encoding("binary", "Binary, Cuccaro adder", "Computational-basis binary with a ripple-carry adder: the baseline.",
             to_openqasm),
    Encoding("binary-ttk", "Binary, no-ancilla adder", "Binary with the Takahashi, Tani and Kunihiro adder.",
             to_circuit_binary_ttk),
    Encoding("fourier", "Fourier, rotation adder", "Integers as phases; register add by controlled rotations.",
             to_circuit_fourier),
    Encoding("fourier-dual", "Fourier, roles-swapped adder", "Integers as phases; register add by the binary adder, roles swapped.",
             to_circuit_fourier_dual),
    Encoding("rns", "Residue number system, binary remainders",
             "A number as its remainders by 2^n-1, 2^n and 2^n+1; add and multiply with no carry between them.",
             to_circuit_rns, modulus_of),
)}

DEFAULT: Tuple[str, ...] = ("binary", "binary-ttk", "fourier", "fourier-dual")  # rns is slower to optimise, so it is a switch on, not on
