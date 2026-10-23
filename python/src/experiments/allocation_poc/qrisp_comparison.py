"""
Our circuits next to Qrisp's for the same task.

Each demo program goes through our compiler as usual (inline, allocate, generate OpenQASM). The same task, written by
hand as an ordinary Qrisp program using only Qrisp's own operators (qrisp_programs.py), goes through Qrisp's own compiler.
Both are measured the same way, as built and after the same Qiskit optimisation into `u` and `cx`, and either can be
run on Qiskit Aer on request. `demos/compare.py` shows the result.

It needs the optional dependencies: pip install '.[circuits,compare]'
"""
from functools import lru_cache
from typing import Any, Dict

from checking import circuits
from checking.simulate import simulate_isa
from compiler.passes.allocate import allocate
from compiler.passes.isa import lower_to_isa
from compiler.passes.optimise import optimise
from compiler.passes.qasm import measured_qubits, to_openqasm
from compiler.system import System
from demo import comparison
from demo.comparison import SIDES, VARIANTS, Built, Side, metrics  # noqa: F401  (the shared pieces, re-exported)
from experiments.allocation_poc import qrisp_programs
from experiments.allocation_poc.experiment import compile_source
from experiments.allocation_poc.programs import PROGRAMS

MAX_WIDTH = 16
DEFAULT_WIDTH = 8
_PROGRAMS = {program["id"]: program for program in PROGRAMS}


def _ours(source: str, width: int):
    isa = lower_to_isa(compile_source(source))
    system = System(width, 100_000)
    generated = to_openqasm(allocate(isa, system), system)
    side = comparison.ours_side(generated, optimise(generated), len(measured_qubits(generated)))
    return side, circuits.wrap(simulate_isa(isa).value, width)


@lru_cache(maxsize=6)  # so running one circuit does not rebuild everything
def build(program_id: str, width: int) -> Built:
    """Both circuits for a program, as built and optimised."""
    if program_id not in _PROGRAMS:
        raise KeyError(f"unknown program {program_id}")
    ours, expected = _ours(_PROGRAMS[program_id]["source"], width)
    return Built(
        expected,
        {"ours": _PROGRAMS[program_id]["source"], "qrisp": qrisp_programs.source_text(program_id, width)},
        {"ours": ours, "qrisp": comparison.qrisp_side(qrisp_programs.run(program_id, width))},
    )


def run_circuit(program_id: str, width: int, side: str, variant: str, shots: int) -> Dict[str, Any]:
    """Run one of the four circuits on Qiskit Aer and report the measurement against the expected value."""
    try:
        built = build(program_id, width)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}
    return comparison.run_circuit(built, side, variant, shots)
