"""
Our circuits next to Qrisp's for the same task.

Each demo program goes through our compiler as usual (inline, allocate, generate OpenQASM). The same task, written by
hand as an ordinary Qrisp program using only Qrisp's own operators (qrisp_programs.py), goes through Qrisp's own compiler.
Both are measured the same way, as built and after the same Qiskit optimisation into `u` and `cx`, and either can be
run on Qiskit Aer on request. `demos/compare.py` shows the result.

It needs the optional dependencies: pip install '.[circuits,compare]'
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict

from checking import circuits, qiskit_tools
from checking.simulate import simulate_isa
from compiler.passes.allocate import allocate
from compiler.passes.isa import lower_to_isa
from compiler.passes.optimise import optimise
from compiler.passes.qasm import measured_qubits, openqasm_text, to_openqasm
from compiler.system import System
from experiments.allocation_poc import qrisp_programs
from experiments.allocation_poc.experiment import compile_source
from experiments.allocation_poc.programs import PROGRAMS
from demo.drawing import diagram, qiskit_diagram

MAX_WIDTH = 16
DEFAULT_WIDTH = 8
SIDES = ("ours", "qrisp")
VARIANTS = ("built", "optimised")
_PROGRAMS = {program["id"]: program for program in PROGRAMS}


@dataclass(frozen=True)
class Side:
    """One compiler's circuit for a program, as built and as optimised, with what is needed to draw and run it."""
    built: Any
    optimised: Any
    result_bits: int
    exponent: int  # a Qrisp QuantumFloat holds mantissa * 2 ** exponent
    text: Dict[str, str]  # the OpenQASM of each variant
    drawings: Dict[str, Dict[str, Any]]

    def variant(self, name: str):
        return self.built if name == "built" else self.optimised


@dataclass(frozen=True)
class Built:
    expected: int
    sources: Dict[str, str]
    sides: Dict[str, Side]


def _text(circuit) -> str:
    try:
        return openqasm_text(circuit)
    except Exception as error:  # a circuit with gates OpenQASM cannot name
        return f"// this circuit cannot be written as OpenQASM: {error}"


def _ours(source: str, width: int):
    isa = lower_to_isa(compile_source(source))
    system = System(width, 100_000)
    generated = to_openqasm(allocate(isa, system), system)
    optimised = optimise(generated)
    side = Side(
        generated, optimised, len(measured_qubits(generated)), 0,
        {"built": openqasm_text(generated), "optimised": _text(optimised)},
        {"built": diagram(generated), "optimised": diagram(optimised)},
    )
    return side, circuits.wrap(simulate_isa(isa).value, width)


def _qrisp(program_id: str, width: int) -> Side:
    qrisp = qrisp_programs.run(program_id, width)
    optimised = qiskit_tools.optimise(qrisp.circuit)
    return Side(
        qrisp.circuit, optimised, len(qrisp.result), qrisp.exponent,
        {"built": _text(qrisp.circuit), "optimised": _text(optimised)},
        {"built": qiskit_diagram(qrisp.circuit, qrisp.labels, qrisp.groups),
         "optimised": qiskit_diagram(optimised, qrisp.labels, qrisp.groups)},
    )


@lru_cache(maxsize=6)  # so running one circuit does not rebuild everything
def build(program_id: str, width: int) -> Built:
    """Both circuits for a program, as built and optimised."""
    if program_id not in _PROGRAMS:
        raise KeyError(f"unknown program {program_id}")
    ours, expected = _ours(_PROGRAMS[program_id]["source"], width)
    return Built(
        expected,
        {"ours": _PROGRAMS[program_id]["source"], "qrisp": qrisp_programs.source_text(program_id, width)},
        {"ours": ours, "qrisp": _qrisp(program_id, width)},
    )


def metrics(built: Built, side: str, variant: str) -> Dict[str, Any]:
    """Gate and wire counts of one of the four circuits, counted the same way."""
    return qiskit_tools.measure(built.sides[side].variant(variant))


def run_circuit(program_id: str, width: int, side: str, variant: str, shots: int) -> Dict[str, Any]:
    """Run one of the four circuits on Qiskit Aer and report the measurement against the expected value."""
    if side not in SIDES or variant not in VARIANTS:
        return {"error": "unknown circuit"}
    if not qiskit_tools.aer_available():
        return {"error": "Running a circuit needs Qiskit Aer: pip install qiskit-aer"}
    try:
        built = build(program_id, width)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}
    shots = min(max(shots, 1), qiskit_tools.MAX_SHOTS)
    entry = built.sides[side]
    outcome = qiskit_tools.execute(entry.variant(variant), entry.result_bits, shots)
    return {"side": side, "variant": variant,
            **circuits.report(outcome, shots=shots, expected=built.expected, scale=2 ** entry.exponent)}
