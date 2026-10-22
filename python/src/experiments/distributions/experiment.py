"""
Working with distributions: compile a program and produce an OpenQASM 3 circuit.

    PYTHONPATH=src .venv/bin/python -m experiments.distributions.experiment --file my_program.txt [--width 4]

The source is read with grammar v2, which adds `x = 0..4;` (an equal superposition of 0 to 3). The pipeline is the shared
front end, then constants folded, unread values dropped, constant steps combined and updates made in place on the
program before it is renamed, the guard of each conditional raised into a boolean variable, a rename that keeps the name of an in-place update, instructions with each register released as soon as it can be undone, multiplication by a power of two as a shift, slots, and circuit
generation.
"""
import argparse
from pathlib import Path
from typing import Optional, Sequence

from qiskit import QuantumCircuit

from compiler.compiler import Compiler
from compiler.passes.allocate import allocate
from compiler.passes.balance import balance
from compiler.passes.flatten import flatten
from compiler.passes.inline import inline
from compiler.passes.isa import lower_to_isa
from compiler.passes.optimise import optimise
from compiler.passes.qasm import openqasm_text, to_openqasm
from compiler.passes.unroll import unroll
from compiler.system import System
from experiments.distributions.passes.alias_identities import alias_identities
from experiments.distributions.passes.collapse_updates import collapse_updates
from experiments.distributions.passes.condition_bits import condition_bits
from experiments.distributions.passes.drop_noops import drop_noops
from experiments.distributions.passes.drop_unread import drop_unread, drop_unread_versions
from experiments.distributions.passes.flatten_conditionals import flatten_conditionals
from experiments.distributions.passes.flatten_controls import flatten_controls
from experiments.distributions.passes.fold_constants import fold_constants
from experiments.distributions.passes.fuse_steps import fuse_steps
from experiments.distributions.passes.release import release_values
from experiments.distributions.passes.rename import rename
from experiments.distributions.passes.shifts import shift_multiplications

FRONT_END = (inline, unroll, flatten, flatten_conditionals, balance)
PIPELINE = (*FRONT_END, fold_constants, drop_unread, fuse_steps, collapse_updates, condition_bits, rename, drop_unread_versions,
            lower_to_isa, alias_identities, drop_noops, flatten_controls,
            release_values, shift_multiplications, allocate, to_openqasm, optimise)
SYSTEM = System(register_width=4, qubits=100)


def compiler(system: Optional[System] = None) -> Compiler:
    return Compiler(system or SYSTEM, grammar="v2").load(*PIPELINE)


def compile_circuit(source: str, system: Optional[System] = None) -> QuantumCircuit:
    return compiler(system).run(source, until="to_openqasm").circuit


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--file", type=Path, required=True, help="a source file to compile")
    parser.add_argument("--width", type=int, default=SYSTEM.register_width, help="qubits per register")
    parser.add_argument("--qubits", type=int, default=SYSTEM.qubits, help="total qubits")
    args = parser.parse_args(argv)
    print(openqasm_text(compile_circuit(args.file.read_text(encoding="utf-8"), System(args.width, args.qubits))))


if __name__ == "__main__":
    main()
