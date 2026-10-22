"""
Pipeline explorer: pick a program or write your own, choose the passes, and step through what each one changes.

Every stage shows the program before and after the pass with the changes marked. Once the program is instructions, the
instruction machine runs it and the stage shows what it returns, the state vector it ends in, and which registers are
entangled: by the state itself, and by the rule that any operation acting on several variables entangles them. The
circuit stages are not built yet, so the run stops at the physical allocation.

    PYTHONPATH=src .venv/bin/python -m experiments.distributions.demos.explorer
"""
from typing import Any, Dict, List, Sequence, Tuple

from lark.exceptions import UnexpectedInput

from checking.simulate import SimulationError, simulate_isa
from compiler.compiler import PassError
from compiler.parser.errors import SourceProblem, describe_parse_error
from compiler.parser.instructions import IsaProgram
from compiler.printer import unparse, unparse_isa
from compiler.system import System
from demo import app
from demo.model import Bar, Card, Case, Cell, Columns, Diff, Metrics, Note, Number, Result, Row, SourceError, Stage, Stats, Step, Text
from experiments.distributions import experiment
from experiments.distributions.passes.entanglement import analyse
from experiments.distributions.programs import PROGRAMS
from experiments.distributions.views import entangled_groups, groups_by_joint_operation

SPECS = [fn.compile_pass for fn in experiment.compiler().passes]
LAST = next(n for n, spec in enumerate(SPECS) if spec.name == "allocate")  # the circuit stages are not built yet
STEPS = SPECS[:LAST + 1]
SUMMARY = {spec.name: spec.summary for spec in SPECS} | {"source": "Your program, parsed and printed back out."}
MOST_ENTRIES = 24  # of the state vector, in the listing


def text_of(output: Any) -> str:
    return unparse_isa(output) if isinstance(output, IsaProgram) else unparse(output)


def _groups(groups: List[List[str]]) -> Tuple[str, ...]:
    return tuple(" · ".join(group) + ("" if len(group) > 1 else "   (alone)") for group in groups)


def _amplitude(amplitude: complex) -> str:
    return f"{amplitude.real:.4f}" if not complex(amplitude).imag else f"{complex(amplitude):.4f}"


def _state_blocks(program: IsaProgram, system: System) -> List[Any]:
    """What the instruction machine makes of the program."""
    try:
        result = simulate_isa(program, system)
    except SimulationError as error:
        return [Note(f"The instruction machine refused this program: {error}", "bad")]
    names, state = result.state
    value = result.value
    blocks: List[Any] = [Metrics((
        Card("Returns", str(value) if not isinstance(value, dict) else f"{len(value)} values", "each with the probability below" if isinstance(value, dict) else "a definite value"),
        Card("Registers at most", str(result.peak["<program>"]), "held at once"),
        Card("Held at the end", str(len(names)), ", ".join(names) or "none"),
    ))]
    if isinstance(value, dict):
        most = max(value.values())  # bars are scaled to the likeliest value, here and in the state vector below
        blocks.append(Stats(("value", "probability", ""), tuple(
            Row(str(v), (Cell(f"{p:.0%}"), Cell(bars=(Bar(p / most, "series-1"),)))) for v, p in value.items())))
    blocks.append(Note(f"State vector: one bar per basis state, its length the probability of that combination of register "
                       f"values ({len(state)} entries{f', first {MOST_ENTRIES}' if len(state) > MOST_ENTRIES else ''}). "
                       f"Registers whose values move together are entangled.", "muted"))
    top = max((abs(a) ** 2 for a in state.values()), default=1)
    shown = sorted(state.items())[:MOST_ENTRIES]
    blocks.append(Stats(("basis state", "probability", ""), tuple(
        Row(", ".join(f"{n}={v}" for n, v in zip(names, key, strict=True)) or "(no registers)",
            (Cell(f"{abs(a) ** 2:.0%}"), Cell(bars=(Bar(abs(a) ** 2 / top, "series-1" if complex(a).real >= 0 else "series-2"),))),
            f"amplitude {_amplitude(a)}")
        for key, a in shown)))
    try:
        actual = Note("Entangled, from the state: registers in one group are", "", _groups(entangled_groups(names, state)))
    except ValueError as error:
        actual = Note(str(error), "muted")
    reported = Note("By the analysis (never fewer than the state):", "", _groups(sorted(sorted(g) for g in analyse(program)[-1].groups)))
    rule = Note("By the existing rule (any joint operation):", "", _groups(groups_by_joint_operation(program)))
    blocks.append(Columns(((actual,), (reported,), (rule,))))
    return blocks


def _stage(stage, previous: str, system: System) -> Tuple[Stage, str]:
    text = text_of(stage.output)
    blocks: List[Any] = [Diff.of(previous if previous != "" else None, text, "Before", "After")]
    blocks += [Note(w, "warn") for w in stage.warnings]
    if isinstance(stage.output, IsaProgram):
        blocks += _state_blocks(stage.output, system)
    name = stage.title or stage.name
    return Stage(stage.name, name, SUMMARY.get(stage.name, ""), tuple(blocks)), text


def run(inputs: Dict[str, Any], enabled: List[str]) -> Result:
    system = System(inputs["register_width"], experiment.SYSTEM.qubits)
    compiler = experiment.compiler(system)
    skip = [name for name in compiler.names if name not in enabled]  # the circuit passes are never steps, so they are skipped
    failure = None
    try:
        stages = compiler.run(inputs["source"], skip=skip).stages
    except UnexpectedInput as error:
        problem = describe_parse_error(error)
        return Result(error=SourceError(problem.message, problem.line, problem.column))
    except PassError as error:
        stages, failure = error.stages, error
    except Exception as error:  # the grammar's transformer can reject a program in ways the parser does not
        return Result(error=SourceError(str(SourceProblem(str(error)).message)))
    described, previous = [], ""
    for stage in stages:
        built, previous = _stage(stage, previous, system)
        described.append(built)
    if failure is not None:
        described.append(Stage(failure.name, failure.name, SUMMARY.get(failure.name, ""),
                               (Note(f"{type(failure.cause).__name__}: {failure.cause}", "bad"),), failed=True))
    return Result(tuple(described))


DEMO = app.Demo(
    title="Distributions Explorer",
    description="Pick a program or write your own, choose the passes, and step through what each one changes.",
    run=run,
    inputs=(
        Text("source", "Source", 12),
        Number("register_width", "Qubits per register", experiment.SYSTEM.register_width, 2, 8,
               "Integers are two's complement and wrap at this width."),
    ),
    cases=tuple(Case(p["id"], p["name"], p["description"], {"source": p["source"]}) for p in PROGRAMS),
    steps=tuple(Step(s.name, s.title, s.summary, s.requires) for s in STEPS),
    cases_label="Programs",
    port=8004,
)


def make_server(port: int):
    return app.make_server(DEMO, port)


def main() -> None:
    app.serve(DEMO)


if __name__ == "__main__":
    main()
