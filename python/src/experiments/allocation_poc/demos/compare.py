"""
Qrisp comparison: our circuits next to Qrisp's for the same task, built, counted and optionally run the same way.

    PYTHONPATH=src .venv/bin/python -m experiments.allocation_poc.demos.compare

It needs the optional dependencies: pip install '.[circuits,compare]'
"""
from typing import Any, Dict, List, Sequence

from experiments.allocation_poc import qrisp_comparison as comparison
from experiments.allocation_poc.programs import PROGRAMS
from demo import app
from demo.blocks import measurement
from demo.model import (
    Action, Bar, Case, Cell, Code, Columns, Diagram, Field, LegendItem, Note, Number, Option, Result, Row, Stage, Stats, Switcher,
)
from demo.text import shorten

SIDES = (("ours", "Ours"), ("qrisp", "Qrisp"))
VARIANT_NAMES = {"built": "as built", "optimised": "optimised"}
ROWS = (
    ("Wires used", "touched", lambda m: f"{m['wires']} declared"),
    ("Gates", "gates", lambda m: "multi-controlled X counts as one as built"),
    ("Two-qubit gates", "two_qubit", lambda m: "gates on two or more qubits"),
    ("Depth", "depth", lambda m: "layers that cannot overlap"),
)
EXPLANATION = (
    "Ours: inline the functions, insert the releases, pack registers into fixed slots with one shared scratch register, and "
    "write the gates (a Cuccaro adder, shift-and-add multiply, multi-controlled X comparisons).",
    "Qrisp: the same task written by hand as ordinary Qrisp, using only what Qrisp provides: QuantumFloat registers, its own "
    "+, - , * and comparison operators, `with condition:` for an if, and @auto_uncompute for the temporaries. Nothing is "
    "wrapped or annotated, so the circuit is what Qrisp's compiler makes of the code, with its own adders, qubit allocation "
    "and ancillae.",
    "So the gate constructions are not the same. This compares the two generation processes as they come, not matched adders.",
    "Where the Qrisp text had to adapt to Qrisp: a QuantumFloat is a mantissa times a power of two, so multiplying by 2 only "
    "shifts the exponent and Qrisp then refuses to add an odd constant, so doubling is written a + a where that happens. "
    "Where our source updates a name inside an if, the Qrisp updates in place, but copies first where updating would stop "
    "Qrisp uncomputing the condition (one-armed). The measured value is the mantissa times two to the exponent.",
    "What Qrisp leaves undone: @auto_uncompute does not clean a register loaded with a constant, so it is still set when the "
    "circuit ends. Run a circuit to see the qubits left set. In our compiler a constant is just a variable and is released "
    "like any other.",
    "Same measuring: every circuit is read into Qiskit and counted the same way, as built (a multi-controlled X, or a Qrisp "
    "QFT, counts as one gate) and after Qiskit's optimiser into u and cx.",
)


def _ratio(ours: int, theirs: int) -> str:
    if ours == theirs:
        return "same"
    return "—" if not ours else f"×{theirs / ours:.2f}"


def _statistics(built: comparison.Built) -> List[Any]:
    measured = {(side, variant): comparison.metrics(built, side, variant) for side, _ in SIDES for variant in comparison.VARIANTS}
    rows = []
    for label, key, note in ROWS:
        cells = []
        for variant in comparison.VARIANTS:
            ours, theirs = measured["ours", variant][key], measured["qrisp", variant][key]
            top = max(ours, theirs, 1)
            cells.append(Cell(f"ours {ours:,} · Qrisp {theirs:,}", bars=(Bar(ours / top, "series-1"), Bar(theirs / top, "series-2"))))
            cells.append(Cell(_ratio(ours, theirs), "Qrisp ÷ ours"))
        rows.append(Row(label, tuple(cells), note(measured["ours", "built"])))
    mix = [
        f"{name} {VARIANT_NAMES[variant]}: " + " · ".join(
            f"{op} {n:,}" for op, n in sorted(measured[side, variant]["ops"].items(), key=lambda item: -item[1]))
        for variant in comparison.VARIANTS for side, name in SIDES
    ]
    table = Stats(("", "As built", "", "Optimised", ""), tuple(rows), (LegendItem("ours", "series-1"), LegendItem("Qrisp", "series-2")))
    return [table, Note("Gate mix", "muted", tuple(mix))]


def _circuit_pane(built: comparison.Built, side: str, title: str, variant: str, view: str, program: str, width: int):
    entry = built.sides[side]
    metrics = comparison.metrics(built, side, variant)
    name = f"{title} · {VARIANT_NAMES[variant]}"
    if view == "diagram":
        picture = entry.drawings[variant]
        meta = (f"first {picture['layers']} layers, {picture['shown']:,} of {picture['total']:,} gates"
                if picture["shown"] < picture["total"] else f"{picture['layers']} layers")
        shown = Diagram(picture, name, meta)
    else:
        shown = Code.of(shorten(entry.text[variant]), name, f"{metrics['gates']:,} gates")
    run = Action("Run on simulator", "simulate", {"side": side, "variant": variant}, (Field("shots", "shots", 100, 1, 4096),), "Qiskit Aer")
    return (shown, run)


def _circuits(built: comparison.Built, program: str, width: int) -> Switcher:
    def view(variant: str) -> Switcher:
        return Switcher("circuit-view", tuple(
            Option(name, label, (Columns(tuple(
                _circuit_pane(built, side, title, variant, name, program, width) for side, title in SIDES)),))
            for name, label in (("diagram", "Diagram"), ("text", "OpenQASM text"))))
    return Switcher("circuit-variant", tuple(Option(v, VARIANT_NAMES[v].capitalize(), (view(v),)) for v in comparison.VARIANTS))


def _case(inputs: Dict[str, Any]) -> str:
    return inputs["case"] or PROGRAMS[0]["id"]


def run(inputs: Dict[str, Any], enabled: List[str]) -> Result:
    program, width = _case(inputs), inputs["width"]
    try:
        built = comparison.build(program, width)
    except Exception as error:  # Qrisp or Qiskit missing, or a program Qrisp cannot build
        return Result((Stage("error", "Could not build the circuits", blocks=(Note(f"{type(error).__name__}: {error}", "bad"),), failed=True),))
    return Result((
        Stage("programs", "The two programs",
              "The same task, built two ways: our compiler turns the allocation-only program on the left into a circuit, and "
              "Qrisp's own compiler turns the Qrisp program on the right into one. Both are measured the same way below.",
              (Columns((
                  (Code.of(built.sources["ours"], "Ours · source language", "compiler inserts the releases"),),
                  (Code.of(built.sources["qrisp"], "Qrisp · Python", "@auto_uncompute"),),
              )), Note("What is and is not equivalent", "muted", EXPLANATION))),
        Stage("statistics", "Statistics", "", tuple(_statistics(built))),
        Stage("circuits", "Circuits", "", (_circuits(built, program, width),)),
    ))


def simulate(inputs: Dict[str, Any], enabled: List[str], payload: Dict[str, Any]) -> Sequence[Any]:
    try:
        shots = int(payload.get("shots", 100))
    except (TypeError, ValueError):
        shots = 100
    report = comparison.run_circuit(_case(inputs), inputs["width"], str(payload.get("side", "")), str(payload.get("variant", "")), shots)
    return measurement(report)


DEMO = app.Demo(
    title="Compiler comparison: ours and Qrisp",
    description="The same task built by our compiler and by Qrisp's, with the circuits and counts side by side.",
    run=run,
    inputs=(Number("width", "Register width (qubits)", comparison.DEFAULT_WIDTH, 1, comparison.MAX_WIDTH,
                   "Both circuits use registers this wide, unsigned, modulo 2 to this width."),),
    cases=tuple(Case(p["id"], p["name"], p["description"]) for p in PROGRAMS),
    actions={"simulate": simulate},
    workflow=False,
    cases_label="Programs",
    port=8001,
)


def make_server(port: int):
    return app.make_server(DEMO, port)


def main() -> None:
    app.serve(DEMO)


if __name__ == "__main__":
    main()
