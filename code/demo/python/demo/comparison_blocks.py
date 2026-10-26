"""
The blocks for a comparison of two compilers' circuits (demo.comparison): the counts side by side, and the circuits as
diagrams or OpenQASM text with a button to run each on the simulator.
"""
from typing import Any, List

from demo import comparison
from demo.model import Action, Bar, Cell, Code, Columns, Diagram, Field, LegendItem, Note, Option, Row, Stats, Switcher
from demo.text import shorten

SIDES = (("ours", "Ours"), ("qrisp", "Qrisp"))
VARIANT_NAMES = {"built": "as built", "optimised": "optimised"}
ROWS = (
    ("Wires used", "touched", lambda m: f"{m['wires']} declared"),
    ("Gates", "gates", lambda m: "multi-controlled X counts as one as built"),
    ("Two-qubit gates", "two_qubit", lambda m: "gates on two or more qubits"),
    ("Depth", "depth", lambda m: "layers that cannot overlap"),
)
def _ratio(ours: int, theirs: int) -> str:
    if ours == theirs:
        return "same"
    return "—" if not ours else f"×{theirs / ours:.2f}"


def statistics(built: comparison.Built) -> List[Any]:
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


def _circuit_pane(built: comparison.Built, side: str, title: str, variant: str, view: str):
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


def circuits(built: comparison.Built) -> Switcher:
    def view(variant: str) -> Switcher:
        return Switcher("circuit-view", tuple(
            Option(name, label, (Columns(tuple(
                _circuit_pane(built, side, title, variant, name) for side, title in SIDES)),))
            for name, label in (("diagram", "Diagram"), ("text", "OpenQASM text"))))
    return Switcher("circuit-variant", tuple(Option(v, VARIANT_NAMES[v].capitalize(), (view(v),)) for v in comparison.VARIANTS))
