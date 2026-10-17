"""
Encodings side by side: pick a program, choose the encodings (the steps), and see what each returned against the
reference, what it cost, and the circuits. A row is shown for each circuit stage: the lowering's, then Qiskit's optimised.

    PYTHONPATH=src .venv/bin/python -m experiments.encodings.demos.comparison
"""
import threading
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from lark.exceptions import UnexpectedInput

from compiler.parser.errors import describe_parse_error
from compiler.parser.parser import parse
from qiskit import QuantumCircuit

from compiler.system import System
from demo import app
from demo.model import Bar, Case, Cell, Code, Note, Number, Result, Row, SourceError, Stage, Stats, Step, Text
from experiments.encodings import experiment
from experiments.encodings.demos.blocks import bar, circuits, mix
from experiments.encodings.encodings import DEFAULT, ENCODINGS
from experiments.encodings.programs import PROGRAMS


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
            ("stage", "value", "agrees", "shots", "qubits touched", "gates", "two-qubit gates", "depth", "two-qubit depth", "T cost"),
            tuple(
                Row(r.stage, note=mix(r.ops), cells=(
                    Cell(str(r.value), tone="good" if r.agrees else "bad"),
                    Cell("yes" if r.agrees else "no", tone="good" if r.agrees else "bad"),
                    Cell(str(r.shots)),
                    bar(r.qubits_touched, top["qubits_touched"]),
                    bar(r.gates, top["gates"]),
                    Cell(str(r.two_qubit)),
                    bar(r.depth, top["depth"]),
                    Cell(str(r.two_qubit_depth)),
                    Cell("–") if r.t_cost is None else Cell(str(r.t_cost), note=f"{r.t_gates} exact + {r.rotations} rotations"),
                ))
                for r in rows if r.encoding == name
            ),
        ))
    return blocks


ALL = "all"  # the case that shows every program at once instead of one

OPTIMISED = ("optimise", "optimise_zx")
LABEL = {"built": " · as built", "optimise": " · Qiskit", "optimise_zx": " · ZX"}

# (title, meaning, the stages to read it at, what to read)
MEASURES = (
    ("Qubits after optimising", "the qubits the optimised circuit touches", OPTIMISED, lambda o: o.qubits_touched),
    ("Toffoli gates", "ccx gates in the circuit as built", ("built",), lambda o: o.ops.get("ccx", 0)),
    ("Multi-controlled gates", "mcx gates in the circuit as built, which the optimisers have to break up", ("built",), lambda o: o.ops.get("mcx", 0)),
    ("Two-qubit gates after optimising", "gates on two qubits in the optimised circuit", OPTIMISED, lambda o: o.two_qubit),
    ("Two-qubit depth after optimising", "layers counting only the two-qubit gates, which single-qubit merging does not change", OPTIMISED, lambda o: o.two_qubit_depth),
    ("Depth after optimising", "layers of all gates; Qiskit's circuit is in u and cx, ZX's in Clifford+T gates", OPTIMISED, lambda o: o.depth),
    ("T cost", f"T gates in Clifford+T, with each rotation that is not an exact quarter turn charged the T gates that approximate it to {experiment.PRECISION:g}; Qiskit's circuit merges its T gates, so it has none", ("built", "optimise_zx"), lambda o: o.t_cost),
)


def _by_stage(rows: List[experiment.Outcome]) -> Dict[str, experiment.Outcome]:
    """One encoding's rows by what they are: the circuit as built (the first row), then each optimiser."""
    return {"built": rows[0], **{r.stage: r for r in rows[1:]}}


@lru_cache(maxsize=16)
def _suite(width: int, enabled: Tuple[str, ...]):
    """
    For each program and encoding, its outcomes by stage, and why an encoding gave nothing for any program (a
    placeholder). A program one encoding cannot express is left out of that encoding's columns.
    """
    system = System(width, experiment.SYSTEM.qubits)
    suite: Dict[str, Dict[str, Dict[str, experiment.Outcome]]] = {p["name"]: {} for p in PROGRAMS}
    errors: Dict[str, Exception] = {}
    for program in PROGRAMS:
        for name in enabled:
            try:
                suite[program["name"]][name] = _by_stage(experiment.run(program["source"], [name], system))
            except Exception as error:
                errors[name] = error
    return suite, {n: e for n, e in errors.items() if not any(n in per for per in suite.values())}


def _suite_tables(width: int, enabled: List[str]) -> List[Any]:
    suite, failed = _suite(width, tuple(enabled))
    names = [n for n in enabled if n not in failed]
    blocks: List[Any] = [Note(f"{ENCODINGS[n].title}: {type(error).__name__}: {error}", "bad") for n, error in failed.items()]
    if not names:
        return blocks
    for title, meaning, stages, read in MEASURES:
        columns = [(n, stage) for n in names for stage in stages if any(stage in per.get(n, {}) for per in suite.values())]
        values = {program: {(n, st): read(per[n][st]) for n, st in columns if n in per and st in per[n]} for program, per in suite.items()}
        top = max((v for per in values.values() for v in per.values()), default=1) or 1
        rows = []
        for program, per in values.items():
            best, worst = min(per.values(), default=0), max(per.values(), default=0)
            rows.append(Row(program, tuple(
                Cell(str(per[c]), tone="good" if best < worst and per[c] == best else "", bars=(Bar(per[c] / top, "series-1"),)) if c in per else Cell("–")
                for c in columns)))
        blocks += [Note(title, "heading", (meaning,)), Stats(("program", *(f"{n}{LABEL[st]}" for n, st in columns)), tuple(rows))]
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
            Stage("circuits", "Circuits", "", tuple(circuits({name: [(st.name, st.output) for st in c.stages if isinstance(st.output, QuantumCircuit)] for name, c in compilations.items()}))),
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
    cases=(*(Case(p["id"], p["name"], p["description"], {"source": p["source"]}) for p in PROGRAMS),
           Case(ALL, "All programs", "Every program side by side, one table per measure. The first time takes about a minute.")),
    steps=tuple(Step(e.name, e.title, e.summary, enabled=e.name in DEFAULT) for e in ENCODINGS.values()),
    workflow=False,
    cases_label="Programs",
    port=8002,
)


def make_server(port: int):
    return app.make_server(DEMO, port)


def main() -> None:
    # The all-programs tables take a minute to build and are kept, so build them for the usual settings while the page loads.
    threading.Thread(target=_suite, args=(experiment.SYSTEM.register_width, tuple(DEFAULT)), daemon=True).start()
    app.serve(DEMO)


if __name__ == "__main__":
    main()
