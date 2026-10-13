"""
Pipeline explorer: pick a program or write your own, choose the passes, and step through what each one changes.

Every stage shows the program before and after the pass with additions and changes marked, runs it on a register
machine to show the result is unchanged and how many registers are left unreleased, and a circuit stage can be run
on a simulator on request. The passes are the experiment's (`experiment.pipeline`); this chooses which to apply,
keeps going past a failed step so the stages up to it are still shown, and describes the result for the shared page.

    PYTHONPATH=src .venv/bin/python -m experiments.allocation_poc.demos.explorer
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lark.exceptions import UnexpectedInput

from checking import circuits, qiskit_tools
from checking.simulate import SimulationError, simulate
from compiler.compiler import PassError
from compiler.parser.errors import SourceProblem, describe_parse_error
from compiler.parser.parser import parse
from compiler.passes.allocate import WidthExceeded
from compiler.passes.qasm import measured_qubits
from compiler.system import System
from demo import app
from demo.blocks import measurement
from demo.model import Case, Number, Result, SourceError, Step, Text
from experiments.allocation_poc.demos.blocks import Info, Ran, is_circuit, stage_for
from experiments.allocation_poc import experiment
from experiments.allocation_poc.experiment import text_of
from experiments.allocation_poc.programs import PROGRAMS

NEEDS_AER = "Running a circuit needs Qiskit and Qiskit Aer: pip install qiskit qiskit-aer qiskit-qasm3-import"

# The passes name and explain themselves (`@compile_pass(title=, summary=, requires=)`), so the demo reads them.
STEPS = [Info(spec.name, spec.title, spec.summary, spec.requires) for spec in (fn.compile_pass for fn in experiment.compiler().passes)]
INFO = {step.id: step for step in STEPS}
INFO["source"] = Info("source", "Source", "Your program, parsed and printed back out.")


def _system(inputs: Dict[str, Any]) -> System:
    return System(inputs["register_width"], inputs["qubits"])


def run_steps(source: str, enabled: Sequence[str], system: System) -> Tuple[List[Ran], Optional[SourceProblem]]:
    """
    Parse `source` and apply the enabled passes in compiler order. A pass that fails stops the run: its stage shows its
    input again (or, for an allocation that does not fit, the packing it would have needed) and says why.
    """
    compiler = experiment.compiler(system)
    skip = [name for name in compiler.names if name not in enabled]
    try:
        stages, failed = compiler.run(source, skip=skip).stages, None
    except UnexpectedInput as error:
        return [], describe_parse_error(error)
    except PassError as error:
        stages, failed = error.stages, error
    except Exception as error:  # the grammar's transformer can reject a program in ways the parser does not
        return [], SourceProblem(str(error))
    ran = [Ran(st.name, st.output, text_of(st.output), None, st.warnings) for st in stages]
    if failed is not None:
        if isinstance(failed.cause, WidthExceeded):
            message = f"{failed.cause} ({system.qubits} qubits ÷ {system.register_width} per register)"
        else:
            message = f"{type(failed.cause).__name__}: {failed.cause}"
        shown = failed.partial if failed.partial is not None else ran[-1].output
        text = text_of(shown) if failed.partial is not None else ran[-1].text
        expects = INFO[failed.name].requires if failed.partial is None and failed.name in INFO else ()
        ran.append(Ran(failed.name, shown, text, message, (), expects))
    return ran, None


def run(inputs: Dict[str, Any], enabled: List[str]) -> Result:
    ran, problem = run_steps(inputs["source"], enabled, _system(inputs))
    if problem:
        return Result(error=SourceError(problem.message, problem.line, problem.column))
    system = _system(inputs)
    return Result(tuple(stage_for(ran, i, inputs["source"], INFO, system) for i in range(len(ran))))


def simulate_stage(inputs: Dict[str, Any], enabled: List[str], stage_id: str, shots: int) -> Dict[str, Any]:
    """Run one circuit stage on Qiskit Aer and report the measurement next to what the program should return."""
    system = _system(inputs)
    ran, _ = run_steps(inputs["source"], enabled, system)
    stage = next((r for r in ran if r.id == stage_id), None)
    if stage is None or not is_circuit(stage.output):
        return {"error": f"{stage_id} is not a circuit stage that was produced"}
    if not qiskit_tools.aer_available():
        return {"error": NEEDS_AER}
    shots = min(max(shots, 1), qiskit_tools.MAX_SHOTS)
    outcome = qiskit_tools.execute(stage.output, len(measured_qubits(stage.output)), shots)
    try:
        expected = circuits.wrap(simulate(parse(inputs["source"])).value, system.register_width)
    except SimulationError:
        expected = None
    return {"stage": stage_id, **circuits.report(outcome, shots=shots, expected=expected)}


def simulate_action(inputs: Dict[str, Any], enabled: List[str], payload: Dict[str, Any]) -> Sequence[Any]:
    try:
        shots = int(payload.get("shots", 100))
    except (TypeError, ValueError):
        shots = 100
    return measurement(simulate_stage(inputs, enabled, str(payload.get("stage", "")), shots))


DEMO = app.Demo(
    title="Pipeline Explorer",
    description="Pick a program or write your own, choose the passes, and step through what each one changes.",
    run=run,
    inputs=(
        Text("source", "Source", 16),
        Number("register_width", "Qubits per register", 6, 1, 64),
        Number("qubits", "Total qubits", 100, 0, 4096, "One register is kept for scratch; the rest are slots."),
    ),
    cases=tuple(Case(p["id"], p["name"], p["description"], {"source": p["source"]}) for p in PROGRAMS),
    steps=tuple(Step(s.id, s.name, s.summary, s.requires) for s in STEPS),
    actions={"simulate": simulate_action},
    cases_label="Programs",
    port=8000,
)


def make_server(port: int):
    return app.make_server(DEMO, port)


def main() -> None:
    app.serve(DEMO)


if __name__ == "__main__":
    main()
