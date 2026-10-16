"""
Encodings side by side: pick a program, choose the encodings (the steps), and see what each returned against the
reference, what it cost, and the circuits. A row is shown for each circuit stage: the lowering's, then Qiskit's optimised.

    PYTHONPATH=src .venv/bin/python -m experiments.encodings.demos.comparison
"""
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from lark.exceptions import UnexpectedInput

from compiler.parser.errors import describe_parse_error
from compiler.parser.parser import parse
from compiler.passes.qasm import openqasm_text
from compiler.system import System
from demo import app
from demo.drawing import diagram
from demo.model import (
    Bar, Case, Cell, Code, Diagram, Note, Number, Option, Result, Row, SourceError, Stage, Stats, Step, Switcher, Text,
)
from demo.text import shorten
from experiments.encodings import experiment
from experiments.encodings.encodings import DEFAULT, ENCODINGS
from experiments.encodings.programs import PROGRAMS


def _mix(r: experiment.Outcome) -> str:
    return " · ".join(f"{n} {name}" for name, n in sorted(r.ops.items(), key=lambda item: -item[1]))


def _bar(value: int, top: int) -> Cell:
    return Cell(str(value), bars=(Bar(value / top, "series-1"),))


def _results(rows: List[experiment.Outcome], reference: int, enabled: List[str], failures: Dict[str, Note]) -> List[Any]:
    """A heading and a table for each encoding, its stages as the rows. Bars share one scale across the encodings."""
    top = {f: max((getattr(r, f) for r in rows), default=1) or 1 for f in ("qubits_touched", "gates", "depth")}
    blocks: List[Any] = [Note(f"The reference machine returns {reference}, wrapping as the circuits do.", "muted")]
    for name in enabled:
        blocks.append(Note(ENCODINGS[name].title, "heading", (ENCODINGS[name].summary,)))
        if name in failures:
            blocks.append(failures[name])
            continue
        blocks.append(Stats(
            ("stage", "value", "agrees", "shots", "qubits touched", "gates", "depth"),
            tuple(
                Row(r.stage, note=_mix(r), cells=(
                    Cell(str(r.value), tone="good" if r.agrees else "bad"),
                    Cell("yes" if r.agrees else "no", tone="good" if r.agrees else "bad"),
                    Cell(str(r.shots)),
                    _bar(r.qubits_touched, top["qubits_touched"]),
                    _bar(r.gates, top["gates"]),
                    _bar(r.depth, top["depth"]),
                ))
                for r in rows if r.encoding == name
            ),
        ))
    return blocks


def _circuits(compilations: Dict[str, Any]) -> List[Any]:
    """A heading and a row of stage buttons for each encoding, one button per circuit its pipeline produced."""
    blocks: List[Any] = []
    for name, compilation in compilations.items():
        options = []
        for stage in compilation.stages:
            if not hasattr(stage.output, "qregs"):
                continue
            picture = diagram(stage.output)
            cut = (f"first {picture['layers']} layers, {picture['shown']:,} of {picture['total']:,} gates" if picture["shown"] < picture["total"]
                   else f"{picture['total']:,} gates, {picture['layers']} layers")
            options.append(Option(stage.name, stage.name, (Switcher("circuit-view", (
                Option("diagram", "Circuit diagram", (Diagram(picture, stage.name, cut),)),
                Option("text", "OpenQASM text", (Code.of(shorten(openqasm_text(stage.output)), stage.name),)),
            )),)))
        blocks += [Note(ENCODINGS[name].title, "heading"), Switcher(f"circuit-{name}", tuple(options))]
    return blocks


ALL = "all"  # the case that shows every program at once instead of one

MEASURES = (
    ("Qubits after optimising", "the qubits the optimised circuit touches", lambda built, optimised: optimised.qubits_touched),
    ("Toffoli gates", "ccx gates in the circuit as built", lambda built, optimised: built.ops.get("ccx", 0)),
    ("Multi-controlled gates", "mcx gates in the circuit as built, which Qiskit has to break up", lambda built, optimised: built.ops.get("mcx", 0)),
    ("Two-qubit gates after optimising", "cx gates once Qiskit has rewritten everything into u and cx", lambda built, optimised: optimised.ops.get("cx", 0)),
    ("Depth after optimising", "layers once Qiskit has rewritten everything into u and cx", lambda built, optimised: optimised.depth),
)


@lru_cache(maxsize=16)
def _suite(width: int, enabled: Tuple[str, ...]):
    """
    For each program and encoding, the circuit as built and the optimised one, and why an encoding gave nothing for any
    program (a placeholder). A program one encoding cannot express is left out of that encoding's column.
    """
    system = System(width, experiment.SYSTEM.qubits)
    suite: Dict[str, Dict[str, Tuple[experiment.Outcome, experiment.Outcome]]] = {p["name"]: {} for p in PROGRAMS}
    errors: Dict[str, Exception] = {}
    for program in PROGRAMS:
        for name in enabled:
            try:
                suite[program["name"]][name] = tuple(experiment.run(program["source"], [name], system))
            except Exception as error:
                errors[name] = error
    return suite, {n: e for n, e in errors.items() if not any(n in per for per in suite.values())}


def _suite_tables(width: int, enabled: List[str]) -> List[Any]:
    suite, failed = _suite(width, tuple(enabled))
    names = [n for n in enabled if n not in failed]
    blocks: List[Any] = [Note(f"{ENCODINGS[n].title}: {type(error).__name__}: {error}", "bad") for n, error in failed.items()]
    if not names:
        return blocks
    for title, meaning, measure in MEASURES:
        values = {program: {n: measure(*pair) for n, pair in per.items()} for program, per in suite.items()}
        top = max((v for per in values.values() for v in per.values()), default=1) or 1
        rows = []
        for program, per in values.items():
            best, worst = min(per.values(), default=0), max(per.values(), default=0)
            cells = tuple(
                Cell(str(per[n]), tone="good" if best < worst and per[n] == best else "", bars=(Bar(per[n] / top, "series-1"),)) if n in per else Cell("–")
                for n in names
            )
            rows.append(Row(program, cells))
        blocks += [Note(title, "heading", (meaning,)), Stats(("program", *(ENCODINGS[n].title for n in names)), tuple(rows))]
    return blocks


def _single(inputs: Dict[str, Any], enabled: List[str], system: System) -> Result:
    source = inputs["source"]
    try:
        parse(source)
    except UnexpectedInput as error:
        problem = describe_parse_error(error)
        return Result(error=SourceError(problem.message, problem.line, problem.column))
    compilations: Dict[str, Any] = {}
    failures: Dict[str, Note] = {}
    for name in enabled:
        try:
            compilations.update(experiment.compile_all(source, [name], system))
        except Exception as error:  # a placeholder, or a program the lowering cannot express: say so beside the others
            failures[name] = Note(f"{type(error).__name__}: {error}", "bad")
    stages = [Stage("source", "Source", "", (Code.of(source, "Source"),))]
    if compilations:
        reference = experiment.reference(source, system)
        rows = experiment.outcomes(source, compilations, system)
        stages += [
            Stage("results", "Results", "What each encoding returned, and what it cost.", tuple(_results(rows, reference, enabled, failures))),
            Stage("circuits", "Circuits", "", tuple(_circuits(compilations))),
        ]
    else:
        stages.append(Stage("results", "Results", "", tuple(failures.values())))
    return Result(tuple(stages))


def run(inputs: Dict[str, Any], enabled: List[str]) -> Result:
    system = System(inputs["register_width"], experiment.SYSTEM.qubits)
    if not enabled:
        return Result((Stage("none", "Nothing to compare", "Switch an encoding on in the steps below."),))
    if inputs["case"] == ALL:
        return Result((Stage("suite", "All programs", "Every program at this register width. Green marks the encoding that needs least.",
                             tuple(_suite_tables(system.register_width, enabled))),))
    return _single(inputs, enabled, system)


DEMO = app.Demo(
    title="Same program, different encodings",
    description="Compile one program once per encoding and compare what each returns and what it costs.",
    run=run,
    inputs=(
        Text("source", "Source", 14),
        Number("register_width", "Qubits per register", experiment.SYSTEM.register_width, 2, 8,
               "Integers are two's complement and wrap at this width."),
    ),
    cases=(Case(ALL, "All programs", "Every program side by side, one table per measure."),
           *(Case(p["id"], p["name"], p["description"], {"source": p["source"]}) for p in PROGRAMS)),
    steps=tuple(Step(e.name, e.title, e.summary, enabled=e.name in DEFAULT) for e in ENCODINGS.values()),
    workflow=False,
    cases_label="Programs",
    port=8002,
)


def make_server(port: int):
    return app.make_server(DEMO, port)


def main() -> None:
    app.serve(DEMO)


if __name__ == "__main__":
    main()
