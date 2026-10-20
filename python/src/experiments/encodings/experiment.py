"""
Same program, different encodings: compile one source program once per chosen encoding and compare what each returns
with an independent reference and with each other.

    PYTHONPATH=src .venv/bin/python -m experiments.encodings.experiment                       # every program, default encodings
    PYTHONPATH=src .venv/bin/python -m experiments.encodings.experiment --program wrap --encoding binary binary-ttk
    PYTHONPATH=src .venv/bin/python -m experiments.encodings.experiment --list
    PYTHONPATH=src .venv/bin/python -m experiments.encodings.experiment --plan

Each encoding is a different lowering pass in an otherwise identical pipeline. A row is printed for each stage whose
output is a circuit (the lowering's, Qiskit's optimised one, and the same circuit optimised by PyZX when it is
installed), because nothing is simulated until a circuit exists.
"""
import argparse
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit, transpile

from checking import qiskit_tools
from checking.simulate import simulate
from compiler.compiler import CIRCUIT, Compilation, Compiler, PassError, Stage
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
from experiments.encodings.passes.optimise_zx import optimise_zx, zx_available
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
    two_qubit: int = 0  # gates on more than one qubit: CNOTs after optimising, and a Toffoli is one gate in a circuit as built
    two_qubit_depth: int = 0  # the depth counting only those gates, which does not depend on how single-qubit gates are merged
    t_gates: Optional[int] = None  # the T gates it has exactly, written out in Clifford+T
    rotations: Optional[int] = None  # rotations that are not a multiple of a quarter turn, which can only be approximated
    t_cost: Optional[int] = None  # T gates in all, charging each of those rotations the T gates that approximate it

    @property
    def agrees(self) -> bool:
        return self.value == self.reference


def with_zx(compilation: Compilation) -> Compilation:
    """
    The compilation with a second optimised circuit: the lowering's circuit, optimised by PyZX instead of Qiskit. If PyZX
    fails the compilation is returned as it was, with the reason as a warning on its last stage.
    """
    lowered, spec = compilation.stages[-2], optimise_zx.compile_pass  # the last stage is Qiskit's optimise
    try:
        stage = Stage(spec.name, CIRCUIT, optimise_zx(lowered.output, compilation.system), (), spec.title)
    except Exception as error:
        last = compilation.stages[-1]
        return Compilation(compilation.system, (*compilation.stages[:-1], replace(last, warnings=(*last.warnings, f"{spec.name} failed: {type(error).__name__}: {error}"))))
    return Compilation(compilation.system, (*compilation.stages, stage))


def optimisers() -> Tuple[str, ...]:
    """The stages that optimise the lowering's circuit, in the order they appear."""
    return ("optimise", "optimise_zx") if zx_available() else ("optimise",)


def compile_all(source: str, encodings: Sequence[str], system: Optional[System] = None, zx: bool = True) -> Dict[str, Compilation]:
    """The source compiled once per encoding, then optimised a second way when `zx` is on and PyZX is installed. A failing pass raises its own exception."""
    compilations = {}
    for name in encodings:
        try:
            compilations[name] = compiler(name, system).run(source)
        except PassError as error:
            raise error.cause from None
        if zx and zx_available():
            compilations[name] = with_zx(compilations[name])
    return compilations


def reference(source: str, system: Optional[System] = None, modulus: Optional[int] = None) -> int:
    """
    What the program returns on the reference machine, wrapping as the circuits do: at the register width, or, for an
    encoding whose range is not 2^width, into `modulus` values (signed, like two's complement). That wrap is taken on the
    value the program returns without wrapping, which is right for programs made of add, subtract, multiply and equality.
    """
    if modulus is None:
        return int(simulate(parse(source), system or SYSTEM).value)
    value = int(simulate(parse(source)).value)
    return (value + modulus // 2) % modulus - modulus // 2


CLIFFORD_T = ["cx", "h", "s", "sdg", "t", "tdg", "x", "z", "rz"]
PRECISION = 1e-10  # how closely a rotation that is not exactly Clifford+T is approximated by T gates (Qiskit's default)
UNREADABLE = ("optimise",)  # Qiskit's circuit has its gates merged into u gates, so its T gates cannot be counted


@lru_cache(maxsize=None)
def _synthesised(angle: float) -> int:
    """The T gates it takes to approximate a rotation about Z by `angle` to PRECISION (gridsynth, Qiskit 2.5 or later)."""
    from qiskit.synthesis import gridsynth_rz
    counted = gridsynth_rz(angle, PRECISION).count_ops()
    return counted.get("t", 0) + counted.get("tdg", 0)


def t_cost(circuit: QuantumCircuit) -> Tuple[int, int, int]:
    """
    What the circuit costs in T gates once it is written out in Clifford+T: the T gates it has exactly (a Toffoli is
    seven, and a rotation by an odd multiple of a quarter turn is one), the rotations that are not a multiple of a
    quarter turn and so can only be approximated, and the total with each of those charged its T gates at PRECISION.
    """
    expanded = transpile(circuit.remove_final_measurements(inplace=False), basis_gates=CLIFFORD_T, optimization_level=0)
    exact = rotations = synthesised = 0
    for item in expanded.data:
        if item.operation.name in ("t", "tdg"):
            exact += 1
        elif item.operation.name == "rz":
            angle = float(item.operation.params[0]) % (2 * math.pi)
            quarters = angle / (math.pi / 4)
            if abs(quarters - round(quarters)) < 1e-9:
                exact += int(round(quarters)) % 2  # odd multiples are one T gate, even ones are Clifford
            else:
                rotations += 1
                synthesised += _synthesised(round(angle, 12))
    return exact, rotations, exact + synthesised


def count(circuit: QuantumCircuit, t: bool = True) -> Dict[str, Any]:
    """What a circuit is made of, as the fields of an `Outcome`: qubits touched, gates, depth, the gates by name (measurements left out), and the rest."""
    counted = qiskit_tools.measure(circuit)
    ops: Dict[str, int] = {}
    for name, n in counted["ops"].items():  # "mcx_o5" is an mcx with some controls open: still an mcx
        if name != "measure":
            ops[name.split("_o")[0]] = ops.get(name.split("_o")[0], 0) + n
    return {
        "qubits_touched": counted["touched"], "gates": counted["gates"], "depth": counted["depth"], "ops": ops,
        "two_qubit": counted["two_qubit"],
        "two_qubit_depth": circuit.depth(filter_function=lambda item: item.operation.num_qubits > 1),
        **(dict(zip(("t_gates", "rotations", "t_cost"), t_cost(circuit))) if t else {}),
    }


def measure(encoding: str, stage: str, circuit: QuantumCircuit, expected: int, system: System) -> Outcome:
    """Run one circuit with its own readout recipe, and count what it is made of."""
    readout = readout_of(circuit)
    ran = qiskit_tools.execute(circuit, len(measured_qubits(circuit)), readout.shots)
    return Outcome(encoding, stage, decode(readout, ran["counts"], system.register_width), expected, readout.shots,
                   **count(circuit, t=stage not in UNREADABLE))


def outcomes(source: str, compilations: Dict[str, Compilation], system: Optional[System] = None) -> List[Outcome]:
    """A row for every circuit stage of every compilation."""
    system = system or SYSTEM
    rows = []
    for name, compilation in compilations.items():
        modulus = ENCODINGS[name].modulus
        expected = reference(source, system, modulus(system) if modulus else None)
        rows += [measure(name, stage.name, stage.output, expected, system)
                 for stage in compilation.stages if isinstance(stage.output, QuantumCircuit)]
    return rows


def run(source: str, encodings: Sequence[str], system: Optional[System] = None, zx: bool = True) -> List[Outcome]:
    """Compile `source` once per encoding, run each circuit with its own readout recipe, and return the rows."""
    return outcomes(source, compile_all(source, encodings, system, zx), system)


def plan(encodings: Sequence[str], system: Optional[System] = None) -> List[Tuple[str, List[str]]]:
    """For each encoding, the passes it runs in order."""
    return [(name, compiler(name, system).names) for name in encodings]


def table(rows: Sequence[Tuple[str, Outcome]]) -> str:
    header = ("program", "encoding", "stage", "value", "reference", "ok", "shots", "qubits", "gates", "2q gates", "depth", "2q depth", "T cost")
    body = [(program, o.encoding, o.stage, str(o.value), str(o.reference), "yes" if o.agrees else "NO",
             str(o.shots), str(o.qubits_touched), str(o.gates), str(o.two_qubit), str(o.depth), str(o.two_qubit_depth),
             "-" if o.t_cost is None else str(o.t_cost)) for program, o in rows]
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
    parser.add_argument("--no-zx", action="store_true", help="leave out the ZX optimised circuit, which is slow on large programs")
    args = parser.parse_args(argv)
    system = System(args.width, SYSTEM.qubits)
    if args.list:
        for encoding in ENCODINGS.values():
            print(f"{encoding.name:<13}{encoding.title}")
        return
    if args.plan:
        for name, passes in plan(args.encoding, system):
            print(f"{name}: " + " -> ".join(["source", *passes]))
        if zx_available():
            print("and optimise_zx, which optimises the lowering's circuit with PyZX instead of Qiskit")
        return
    sources = ([(args.file.name, args.file.read_text(encoding="utf-8"))] if args.file else
               [(p["id"], p["source"]) for p in PROGRAMS if args.program is None or p["id"] in args.program])
    rows, refused = [], []
    for name, source in sources:
        for encoding in args.encoding:
            try:
                rows += [(name, row) for row in run(source, [encoding], system, zx=not args.no_zx)]
            except Exception as error:  # an encoding that cannot express a program says why, and the run goes on
                refused.append(f"{name}  {encoding}  refused: {error}")
    print(table(rows))
    for line in refused:
        print(line)


if __name__ == "__main__":
    main()
