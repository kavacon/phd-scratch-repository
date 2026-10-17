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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

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
    non_clifford: int = 0  # single-qubit gates that are not Clifford, with the circuit written out as single-qubit gates and CNOTs

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


def reference(source: str, system: Optional[System] = None) -> int:
    """What the program returns on the reference machine, wrapping as the circuits do."""
    return int(simulate(parse(source), system or SYSTEM).value)


_PAULIS = [np.array(m, dtype=complex) for m in ([[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]])]


def _is_clifford(matrix: np.ndarray) -> bool:
    """A single-qubit gate is Clifford when it turns X and Z into signed Paulis."""
    for pauli in (_PAULIS[0], _PAULIS[2]):
        conjugated = matrix @ pauli @ matrix.conj().T
        if not any(np.allclose(conjugated, sign * other, atol=1e-8) for other in _PAULIS for sign in (1, -1)):
            return False
    return True


def non_clifford(circuit: QuantumCircuit) -> int:
    """
    The single-qubit gates that are not Clifford, once the circuit is written out as single-qubit gates and CNOTs with no
    merging, so a T gate is one gate and a Toffoli is seven. A circuit whose single-qubit gates have already been merged
    (Qiskit's optimised one) counts each merged gate once, which can be fewer than the T gates it came from.
    """
    expanded = transpile(circuit.remove_final_measurements(inplace=False), basis_gates=["u", "cx"], optimization_level=0)
    return sum(1 for item in expanded.data if item.operation.num_qubits == 1 and not _is_clifford(Operator(item.operation).data))


def count(circuit: QuantumCircuit) -> Dict[str, Any]:
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
        "non_clifford": non_clifford(circuit),
    }


def measure(encoding: str, stage: str, circuit: QuantumCircuit, expected: int, system: System) -> Outcome:
    """Run one circuit with its own readout recipe, and count what it is made of."""
    readout = readout_of(circuit)
    ran = qiskit_tools.execute(circuit, len(measured_qubits(circuit)), readout.shots)
    return Outcome(encoding, stage, decode(readout, ran["counts"], system.register_width), expected, readout.shots,
                   **count(circuit))


def outcomes(source: str, compilations: Dict[str, Compilation], system: Optional[System] = None) -> List[Outcome]:
    """A row for every circuit stage of every compilation."""
    system = system or SYSTEM
    expected = reference(source, system)
    return [measure(name, stage.name, stage.output, expected, system)
            for name, compilation in compilations.items() for stage in compilation.stages
            if isinstance(stage.output, QuantumCircuit)]


def run(source: str, encodings: Sequence[str], system: Optional[System] = None, zx: bool = True) -> List[Outcome]:
    """Compile `source` once per encoding, run each circuit with its own readout recipe, and return the rows."""
    return outcomes(source, compile_all(source, encodings, system, zx), system)


def plan(encodings: Sequence[str], system: Optional[System] = None) -> List[Tuple[str, List[str]]]:
    """For each encoding, the passes it runs in order."""
    return [(name, compiler(name, system).names) for name in encodings]


def table(rows: Sequence[Tuple[str, Outcome]]) -> str:
    header = ("program", "encoding", "stage", "value", "reference", "ok", "shots", "qubits", "gates", "2q gates", "depth", "2q depth", "non-Clifford")
    body = [(program, o.encoding, o.stage, str(o.value), str(o.reference), "yes" if o.agrees else "NO",
             str(o.shots), str(o.qubits_touched), str(o.gates), str(o.two_qubit), str(o.depth), str(o.two_qubit_depth),
             str(o.non_clifford)) for program, o in rows]
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
        if zx_available():
            print("and optimise_zx, which optimises the lowering's circuit with PyZX instead of Qiskit")
        return
    sources = ([(args.file.name, args.file.read_text(encoding="utf-8"))] if args.file else
               [(p["id"], p["source"]) for p in PROGRAMS if args.program is None or p["id"] in args.program])
    print(table([(name, row) for name, source in sources for row in run(source, args.encoding, system)]))


if __name__ == "__main__":
    main()
