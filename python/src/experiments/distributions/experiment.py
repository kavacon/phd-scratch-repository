"""
Working with distributions: compile a program and produce an OpenQASM 3 circuit.

    PYTHONPATH=src .venv/bin/python -m experiments.distributions.experiment --file my_program.txt [--width 4]

The source is read with grammar v2, which adds `x = 0..4;` (an equal superposition of 0 to 3). The pipeline is the shared front end, instructions, slots and circuit generation with no experiment-specific pass yet.
The passes that decide what becomes superposed, and that uncompute what it entangles, go in `passes/` and are loaded here.
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
from compiler.passes.rename import rename
from compiler.passes.unroll import unroll
from compiler.system import System

FRONT_END = (inline, unroll, flatten, balance, rename)
PIPELINE = (*FRONT_END, lower_to_isa, allocate, to_openqasm, optimise)
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
