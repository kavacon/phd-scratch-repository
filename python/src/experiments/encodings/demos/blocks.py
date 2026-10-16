"""Blocks both demo pages draw the same way: a bar cell, a gate mix, and a heading with the circuits of each encoding."""
from typing import Any, Dict, List, Sequence, Tuple

from qiskit import QuantumCircuit

from compiler.passes.qasm import openqasm_text
from demo.drawing import diagram
from demo.model import Bar, Cell, Code, Diagram, Note, Option, Switcher
from demo.text import shorten
from experiments.encodings.encodings import ENCODINGS


def bar(value: int, top: int) -> Cell:
    return Cell(str(value), bars=(Bar(value / top, "series-1"),))


def mix(ops: Dict[str, int]) -> str:
    """The gates a circuit is made of, most common first, as one line."""
    return " · ".join(f"{n} {name}" for name, n in sorted(ops.items(), key=lambda item: -item[1]))


def circuits(stages: Dict[str, Sequence[Tuple[str, QuantumCircuit]]]) -> List[Any]:
    """
    A heading and a row of stage buttons for each encoding, one button per circuit its pipeline produced, each showing
    the circuit as a diagram or as OpenQASM text. `stages` maps an encoding to its (stage name, circuit) pairs.
    """
    blocks: List[Any] = []
    for name, pairs in stages.items():
        options = []
        for stage, circuit in pairs:
            picture = diagram(circuit)
            cut = (f"first {picture['layers']} layers, {picture['shown']:,} of {picture['total']:,} gates" if picture["shown"] < picture["total"]
                   else f"{picture['total']:,} gates, {picture['layers']} layers")
            options.append(Option(stage, stage, (Switcher("circuit-view", (
                Option("diagram", "Circuit diagram", (Diagram(picture, stage, cut),)),
                Option("text", "OpenQASM text", (Code.of(shorten(openqasm_text(circuit)), stage),)),
            )),)))
        blocks += [Note(ENCODINGS[name].title, "heading"), Switcher(f"circuit-{name}", tuple(options))]
    return blocks
