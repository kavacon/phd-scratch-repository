"""
Same program, different encodings: compile one source program once per chosen encoding and compare what each returns
with an independent reference and with each other.

    PYTHONPATH=src .venv/bin/python -m experiments.encodings.experiment                       # every program, default encodings
    PYTHONPATH=src .venv/bin/python -m experiments.encodings.experiment --program wrap --encoding binary binary-ttk
    PYTHONPATH=src .venv/bin/python -m experiments.encodings.experiment --list
    PYTHONPATH=src .venv/bin/python -m experiments.encodings.experiment --plan

Each encoding is a different lowering pass in an otherwise identical pipeline. A row is printed for each stage whose
output is a circuit (the lowering's, then Qiskit's optimised one), because nothing is simulated until a circuit exists.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit

from checking import qiskit_tools
from checking.simulate import simulate
from compiler.compiler import Compilation, Compiler, PassError
from compiler.parser.parser import parse
from compiler.passes.allocate import allocate
from compiler.passes.balance import balance
from compiler.passes.flatten import flatten
from compiler.passes.inline import inline
from compiler.passes.isa import lower_to_isa
from compiler.passes.optimise import optimise
from compiler.passes.qasm import measured_qubits
from compiler.passes.rename import rename
from compiler.passes.unroll import unroll
from compiler.system import System
from experiments.encodings.encodings import DEFAULT, ENCODINGS
from experiments.encodings.programs import PROGRAMS
from experiments.encodings.readout import decode, readout_of

# Normalisation only: this experiment has no release strategy, so nothing is uncomputed beyond what Qiskit's
# optimiser removes.
FRONT_END = (inline, unroll, flatten, balance, rename)
SYSTEM = System(register_width=4, qubits=100)


def compiler(encoding: str, system: Optional[System] = None) -> Compiler:
    """The whole pipeline for one encoding: the shared front end, instructions, slots, that encoding's lowering, Qiskit."""
    return Compiler(system or SYSTEM).load(*FRONT_END, lower_to_isa, allocate, ENCODINGS[encoding].lowering, optimise)


@dataclass(frozen=True)
class Outcome:
    """
    One row of the results: what one encoding's circuit returned for one program, measured at one stage of the
    compilation. Only stages whose output is a circuit give a row (the lowering's, then Qiskit's optimised one), because
    nothing is simulated until a circuit exists. A compilation keeps every stage's output, so the rows are compared by
    `stage`.
    """
    encoding: str
    stage: str  # the pass whose output was measured, e.g. "to_openqasm" or "optimise"
    value: Optional[int]  # the decoded value; None only if the circuit could not be run
    reference: int  # what the reference machine returns under the same signed, wrapping rules
    shots: int  # from the lowering's readout recipe
    qubits_touched: int
    gates: int
    depth: int
    ops: Dict[str, int]  # the circuit's gates by name, to show what the gate count is made of

    @property
    def agrees(self) -> bool:
        return self.value == self.reference


def compile_all(source: str, encodings: Sequence[str], system: Optional[System] = None) -> Dict[str, Compilation]:
    """The source compiled once per encoding. A failing pass raises its own exception."""
    compilations = {}
    for name in encodings:
        try:
            compilations[name] = compiler(name, system).run(source)
        except PassError as error:
            raise error.cause from None
    return compilations


def reference(source: str, system: Optional[System] = None) -> int:
    """What the program returns on the reference machine, wrapping as the circuits do."""
    return int(simulate(parse(source), system or SYSTEM).value)


def measure(encoding: str, stage: str, circuit: QuantumCircuit, expected: int, system: System) -> Outcome:
    """Run one circuit with its own readout recipe, and count what it is made of."""
    readout = readout_of(circuit)
    ran = qiskit_tools.execute(circuit, len(measured_qubits(circuit)), readout.shots)
    counted = qiskit_tools.measure(circuit)
    ops: Dict[str, int] = {}
    for name, n in counted["ops"].items():  # "mcx_o5" is an mcx with some controls open: still an mcx
        if name != "measure":
            ops[name.split("_o")[0]] = ops.get(name.split("_o")[0], 0) + n
    return Outcome(encoding, stage, decode(readout, ran["counts"], system.register_width), expected, readout.shots,
                   counted["touched"], counted["gates"], counted["depth"], ops)


def outcomes(source: str, compilations: Dict[str, Compilation], system: Optional[System] = None) -> List[Outcome]:
    """A row for every circuit stage of every compilation."""
    system = system or SYSTEM
    expected = reference(source, system)
    return [measure(name, stage.name, stage.output, expected, system)
            for name, compilation in compilations.items() for stage in compilation.stages
            if isinstance(stage.output, QuantumCircuit)]


def run(source: str, encodings: Sequence[str], system: Optional[System] = None) -> List[Outcome]:
    """Compile `source` once per encoding, run each circuit with its own readout recipe, and return the rows."""
    return outcomes(source, compile_all(source, encodings, system), system)


def plan(encodings: Sequence[str], system: Optional[System] = None) -> List[Tuple[str, List[str]]]:
    """For each encoding, the passes it runs in order."""
    return [(name, compiler(name, system).names) for name in encodings]


def table(rows: Sequence[Tuple[str, Outcome]]) -> str:
    header = ("program", "encoding", "stage", "value", "reference", "ok", "shots", "qubits", "gates", "depth")
    body = [(program, o.encoding, o.stage, str(o.value), str(o.reference), "yes" if o.agrees else "NO",
             str(o.shots), str(o.qubits_touched), str(o.gates), str(o.depth)) for program, o in rows]
    widths = [max(len(line[i]) for line in [header, *body]) for i in range(len(header))]
    return "\n".join("  ".join(cell.ljust(width) for cell, width in zip(line, widths)) for line in [header, *body])


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--encoding", nargs="+", choices=sorted(ENCODINGS), default=list(DEFAULT),
                        help="the encodings to compile with (default: %(default)s)")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--program", nargs="+", choices=[p["id"] for p in PROGRAMS], help="preloaded programs (default: all)")
    which.add_argument("--file", type=Path, help="a source file to compile instead")
    parser.add_argument("--list", action="store_true", help="list the encodings")
    parser.add_argument("--plan", action="store_true", help="show the passes each chosen encoding runs")
    parser.add_argument("--width", type=int, default=SYSTEM.register_width, help="qubits per register")
    args = parser.parse_args(argv)
    system = System(args.width, SYSTEM.qubits)
    if args.list:
        for encoding in ENCODINGS.values():
            print(f"{encoding.name:<13}{encoding.title}")
        return
    if args.plan:
        for name, passes in plan(args.encoding, system):
            print(f"{name}: " + " -> ".join(["source", *passes]))
        return
    sources = ([(args.file.name, args.file.read_text(encoding="utf-8"))] if args.file else
               [(p["id"], p["source"]) for p in PROGRAMS if args.program is None or p["id"] in args.program])
    print(table([(name, row) for name, source in sources for row in run(source, args.encoding, system)]))


if __name__ == "__main__":
    main()
