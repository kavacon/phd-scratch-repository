"""
Measurements of what a pipeline or a simulation produced. Nothing here compiles or runs a program: each function takes an output
(QC text, an OpenQASM 3 artifact, the stages of a build, or the counts of a run) and describes it, so the same measurements apply to
the output of any pipeline.

    . ./env.sh && .venv/bin/python -m experiment_existing_ir.metrics "a = 0..4; return a * 3;" [qubits in the system]

The T cost counts the T gates a Clifford+T circuit needs for what is exact in it: a Toffoli is seven T gates (the standard
construction, without ancillas) and a T or its adjoint is one. A rotation (the preparation of an interval that is not a power of two)
has no exact T cost and a gate with three or more controls has several, so both are counted separately, not folded in.
"""
import re
import sys
from collections import Counter
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from checking import qiskit_tools
from compiler.system import System
from experiment_existing_ir import pipeline
from experiment_existing_ir.pipeline import Export
from experiment_existing_ir.translate import translate
from compiler.parser.parser import parse

TOFFOLI_T_COUNT = 7
ROTATIONS = {"rx", "ry", "rz", "p", "u", "u1", "u2", "u3", "crx", "cry", "crz", "cp", "cu"}


class Circuit(NamedTuple):
    qubits: int  # wires the circuit touches
    gates: int  # every gate once the named subroutines (the adders) are written out
    two_qubit: int
    depth: int
    toffolis: int
    t_count: int  # exact T gates: T and its adjoint, and seven per Toffoli
    rotations: int  # gates with an angle, which have no exact T cost
    wide: int  # gates with three or more controls, which have no T cost counted here


class Fit(NamedTuple):
    """Whether a circuit fits a system's qubits, and when it does not, by how much and which registers make it up."""
    fits: bool
    needed: int  # the most qubits live at once
    available: int  # the system's qubits
    over: int  # qubits beyond the budget (0 when it fits)
    registers: List[int]  # sizes of the QC program's registers, largest first


class Run(NamedTuple):
    shots: int
    outcomes: int  # distinct values measured
    law: Dict[int, float]  # the chance of each value, as measured


def of_circuit(circuit) -> Circuit:
    """A Qiskit circuit. Named subroutines (`add5`, ...) are written out first, so the counts are of gates."""
    named = lambda: [name for name in circuit.count_ops() if name.startswith("add")]
    while named():
        circuit = circuit.decompose(gates_to_decompose=named())
    counts = circuit.count_ops()
    measured = qiskit_tools.measure(circuit)
    wide = sum(n for name, n in counts.items() if re.fullmatch(r"c[3-9]\d*[a-z]+|mc[a-z]+", name))
    return Circuit(qubits=measured["touched"], gates=measured["gates"], two_qubit=measured["two_qubit"], depth=measured["depth"],
                   toffolis=counts.get("ccx", 0), t_count=counts.get("t", 0) + counts.get("tdg", 0)
                   + TOFFOLI_T_COUNT * counts.get("ccx", 0),
                   rotations=sum(n for name, n in counts.items() if name in ROTATIONS), wide=wide)


def of_openqasm(text: str) -> Circuit:
    """An OpenQASM 3 artifact, as `pipeline.build(..., Export("openqasm3"))` gives it."""
    return of_circuit(qiskit_tools.read(text))


def of_qc(qc: str) -> Circuit:
    """The QC text a pipeline ends in."""
    return of_openqasm(pipeline.export(qc, "openqasm3"))


def of_ir(text: str) -> Dict[str, int]:
    """How often each operation occurs in an MLIR program (any stage of a pipeline)."""
    return dict(Counter(re.findall(r"^\s*(?:%[\w#:, ]+ = )?\"?([a-z_]+\.[a-z_0-9.]+)", text, re.M)))


def of_stages(program: str, stages: Sequence[str]) -> List[Tuple[str, Dict[str, int], Optional[Circuit]]]:
    """Each stage's output measured: the operations of the program after it and, once the program is QC, its circuit."""
    rows = []
    for k, stage in enumerate(stages):
        text = pipeline.optimise(program, *stages[:k + 1])
        rows.append((stage, of_ir(text), of_qc(text) if "qc." in text and "func.func @main" in text else None))
    return rows


def fit(qc: str, system: System) -> Fit:
    """The QC program a pipeline ends in against a system: the most qubits it holds at once, which `lower-to-qc` records on `main`,
    at most the system's qubits. (Given the system up front, the tool refuses instead; this assesses a program that was built
    without one.)"""
    needed = int(re.search(r"existing_ir\.peak_qubits = (\d+)", qc).group(1))
    sizes = sorted((int(n) for n in re.findall(r"memref\.alloc\(\)(?: \{[^}]*\})? : memref<(\d+)x!qc\.qubit>", qc)), reverse=True)
    return Fit(needed <= system.qubits, needed, system.qubits, max(needed - system.qubits, 0), sizes)


def of_run(counts: Dict[int, int]) -> Run:
    """The counts of a simulation, as `pipeline.execute` gives them."""
    shots = sum(counts.values())
    return Run(shots, len(counts), {value: n / shots for value, n in sorted(counts.items())})


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "a = 0..4; b = 0..3; return a * b;"
    qubits = int(sys.argv[2]) if len(sys.argv) > 2 else None  # the system's qubits, when a budget is to be assessed
    program = translate(parse(source, "v2"))
    print(f"{'stage':<46}{'ops':>6}{'qubits':>8}{'gates':>7}{'depth':>7}{'T':>6}{'rot':>5}{'wide':>6}")
    for stage, ops, circuit in of_stages(program, pipeline.DEFAULT):
        cells = (circuit.qubits, circuit.gates, circuit.depth, circuit.t_count, circuit.rotations, circuit.wide) if circuit else ("",) * 6
        print(f"{stage[:45]:<46}{sum(ops.values()):>6}" + "".join(f"{c:>{w}}" for c, w in zip(cells, (8, 7, 7, 6, 5, 6))))
    if qubits is not None:
        print(f"\n{fit(pipeline.build(program, *pipeline.DEFAULT), System(qubits=qubits))}")
    print(f"\nrun: {of_run(pipeline.run(program))}")


if __name__ == "__main__":
    main()
