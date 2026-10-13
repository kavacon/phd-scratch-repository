"""
Allocation automation: compile a program that only allocates, let the compiler insert the releases, and produce an
OpenQASM 3 circuit.

    PYTHONPATH=src .venv/bin/python -m experiments.allocation_poc.experiment --program area
    PYTHONPATH=src .venv/bin/python -m experiments.allocation_poc.experiment --program area --stages
    PYTHONPATH=src .venv/bin/python -m experiments.allocation_poc.experiment --file my_program.txt --width 8
"""
import argparse
from pathlib import Path
from typing import Any, List, Optional, Sequence

from qiskit import QuantumCircuit

from checking.simulate import execute
from compiler.compiler import Compiler, PassError
from compiler.parser.grammar import Node
from compiler.passes.allocate import allocate
from compiler.passes.flatten import flatten
from compiler.passes.inline import inline
from compiler.passes.isa import lower_to_isa
from compiler.passes.optimise import optimise
from compiler.passes.qasm import openqasm_text, to_openqasm
from compiler.passes.unroll import unroll
from compiler.printer import unparse, unparse_isa
from compiler.system import System
from experiments.allocation_poc.passes.allocation import lower_allocation
from experiments.allocation_poc.passes.balance import balance
from experiments.allocation_poc.passes.rename import rename
from experiments.allocation_poc.programs import PROGRAMS

# The AST passes in the order they run (see the pipeline notes for why this order).
FRONT_END = (inline, unroll, flatten, balance, rename, lower_allocation)
# The whole pipeline: the AST passes, instructions, physical slots, a circuit, and Qiskit's optimiser (a black box).
PIPELINE = (*FRONT_END, lower_to_isa, allocate, to_openqasm, optimise)


def compiler(system: Optional[System] = None) -> Compiler:
    return Compiler(system).load(*PIPELINE)


def compile_source(source: str, passes=FRONT_END, system: Optional[System] = None) -> List[Node]:
    """Parse source and apply the AST passes in order. A failing pass raises its own exception."""
    try:
        return Compiler(system).load(*passes).run(source).output
    except PassError as error:
        raise error.cause from None


def compile_circuit(source: str, system: Optional[System] = None) -> QuantumCircuit:
    """Source to OpenQASM, before the optimiser. A failing pass raises its own exception."""
    try:
        return compiler(system).run(source, until="to_openqasm").circuit
    except PassError as error:
        raise error.cause from None


def text_of(output: Any) -> str:
    """A stage's output printed: the program, the instructions, or the OpenQASM."""
    if isinstance(output, list):
        return unparse(output)
    if hasattr(output, "body"):  # instructions
        return unparse_isa(output)
    return openqasm_text(output) if isinstance(output, QuantumCircuit) else str(output)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--program", choices=[p["id"] for p in PROGRAMS], default=PROGRAMS[0]["id"], help="a preloaded program")
    which.add_argument("--file", type=Path, help="a source file to compile instead")
    parser.add_argument("--width", type=int, default=6, help="qubits per register")
    parser.add_argument("--qubits", type=int, default=100, help="total qubits")
    parser.add_argument("--stages", action="store_true", help="report what every stage returned instead of printing the circuit")
    args = parser.parse_args(argv)
    source = args.file.read_text(encoding="utf-8") if args.file else next(p for p in PROGRAMS if p["id"] == args.program)["source"]
    system = System(args.width, args.qubits)
    if not args.stages:
        print(openqasm_text(compile_circuit(source, system)))
        return
    for stage in compiler(system).run(source, until="to_openqasm").stages:
        result = execute(stage.output)
        detail = "circuit" if result is None else f"value {result.value}, peak {max(result.peak.values(), default=0)}, leftover {dict(result.leaks) or 'none'}"
        print(f"{stage.name:<17}{detail}")


if __name__ == "__main__":
    main()
