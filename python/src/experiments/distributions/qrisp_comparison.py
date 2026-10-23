"""
Our circuits next to Qrisp's for the same task, at 8-bit registers.

Each program goes through our whole pipeline to a circuit as usual. The same task, written by hand as an ordinary Qrisp
program using only Qrisp's own operators (qrisp_programs.py), goes through Qrisp's own compiler. Both are measured the same way,
as built and after the same Qiskit optimisation into `u` and `cx`, and either can be run on Qiskit Aer on request. The programs
return a distribution, so a run is compared with the probability the instruction machine gives each value, not with one value.
The machine runs the program as lowered with none of the simplifying passes, so it does not share their changes with the circuit
it checks. `demos/compare.py` shows the result.

It needs the optional dependencies: pip install '.[circuits,compare]'
"""
from functools import lru_cache
from typing import Any, Dict

from checking import circuits
from checking.simulate import simulate_isa
from compiler.passes.qasm import measured_qubits
from compiler.system import System
from demo import comparison
from demo.comparison import SIDES, VARIANTS, Built, metrics  # noqa: F401  (the shared pieces, re-exported)
from experiments.distributions import experiment, qrisp_programs
from experiments.distributions.programs import PROGRAMS

MAX_WIDTH = 10  # a distribution over a count that is not a power of two is prepared from a vector of 2 to the width values
DEFAULT_WIDTH = 8
REFERENCE_SKIPS = ["flatten_conditionals", "fold_constants", "drop_unread", "fuse_steps", "collapse_updates", "condition_bits"]
_PROGRAMS = {program["id"]: program for program in PROGRAMS}


def _expected(source: str, system: System) -> Any:
    """What the program returns by the instruction machine, as registers hold it: unsigned, by value or by probability."""
    program = experiment.compiler(system).run(source, until="lower_to_isa", skip=REFERENCE_SKIPS).output
    value = simulate_isa(program, system).value
    if isinstance(value, dict):
        return {circuits.wrap(v, system.register_width): p for v, p in value.items()}
    return circuits.wrap(value, system.register_width)


@lru_cache(maxsize=12)  # so running one circuit does not rebuild everything
def build(program_id: str, width: int) -> Built:
    """Both circuits for a program, as built and optimised, and what the program should return."""
    if program_id not in _PROGRAMS:
        raise KeyError(f"unknown program {program_id}")
    source = _PROGRAMS[program_id]["source"]
    system = System(width, 100_000)
    compilation = experiment.compiler(system).run(source, until="optimise")
    generated = compilation["to_openqasm"]
    ours = comparison.ours_side(generated, compilation["optimise"], len(measured_qubits(generated)))
    return Built(
        _expected(source, system),
        {"ours": source, "qrisp": qrisp_programs.source_text(program_id, width)},
        {"ours": ours, "qrisp": comparison.qrisp_side(qrisp_programs.run(program_id, width))},
    )


def _machine_state(program_id: str, width: int) -> Dict[tuple, float]:
    """The states the instruction machine gives the allocated program, by slot: the non-zero (slot, value) pairs and probability."""
    system = System(width, 100_000)
    program = experiment.compiler(system).run(_PROGRAMS[program_id]["source"], until="allocate").output
    names, state = simulate_isa(program, system).state
    states: Dict[tuple, float] = {}
    for values, amplitude in state.items():
        key = tuple(sorted((name, circuits.wrap(v, width)) for name, v in zip(names, values) if circuits.wrap(v, width)))
        states[key] = states.get(key, 0.0) + abs(amplitude) ** 2
    return states


@lru_cache(maxsize=24)  # it can take a while
def final_state(program_id: str, width: int, side: str) -> Dict[str, Any]:
    """
    The state one side's circuit (as built) ends in, by register, with which registers are still held and whether the result is
    entangled with them. For ours it is also checked against the states of the instruction machine for the allocated program.
    """
    entry = build(program_id, width).sides[side]
    report = circuits.state_report(entry.built, entry.names, reference=_machine_state(program_id, width) if side == "ours" else None)
    return {**report, "scale": 2 ** entry.exponent}


def run_circuit(program_id: str, width: int, side: str, variant: str, shots: int) -> Dict[str, Any]:
    """Run one of the four circuits on Qiskit Aer and report the measurement against the expected distribution."""
    try:
        built = build(program_id, width)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}
    return comparison.run_circuit(built, side, variant, shots)
