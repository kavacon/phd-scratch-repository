"""
Instruction circuits in isolation: pick an instruction, choose up to two encodings, and see the circuit each builds for
that instruction alone, as its lowering writes it.

    PYTHONPATH=src .venv/bin/python -m experiments.encodings.demos.instructions
"""
from typing import Any, Dict, List

from compiler.printer import unparse_instruction
from compiler.system import System
from demo import app
from demo.drawing import diagram
from demo.model import Case, Diagram, Note, Number, Result, Stage, Step
from experiments.encodings import experiment
from experiments.encodings.encodings import ENCODINGS
from experiments.encodings.isolated import VARIANTS

SHOWN = ("binary", "fourier")  # on when the page opens
MOST = 2


def _circuit(name: str, program, system: System) -> List[Any]:
    """A heading and the circuit one encoding builds for the program, or why it cannot."""
    heading = Note(ENCODINGS[name].title, "heading", (ENCODINGS[name].summary,))
    try:
        picture = diagram(ENCODINGS[name].lowering(program, system))
    except Exception as error:  # a placeholder, or an instruction the lowering cannot express
        return [heading, Note(f"{type(error).__name__}: {error}", "bad")]
    cut = f"first {picture['layers']} layers, {picture['shown']:,} of {picture['total']:,} gates" if picture["shown"] < picture["total"] else ""
    return [heading, Diagram(picture, "", cut)]


def run(inputs: Dict[str, Any], enabled: List[str]) -> Result:
    variant = next((v for v in VARIANTS if v.id == inputs["case"]), VARIANTS[0])
    system = System(inputs["register_width"], experiment.SYSTEM.qubits)
    program = variant.program(inputs["constant"])
    summary = f"{variant.description}  ({unparse_instruction(program.body[-1]).strip()})"
    blocks = [block for name in enabled for block in _circuit(name, program, system)]
    if not blocks:
        blocks = [Note("Choose an encoding above.", "muted")]
    return Result((Stage("circuits", variant.name, summary, tuple(blocks)),))


DEMO = app.Demo(
    title="Instruction circuits",
    description="One instruction at a time: the circuit each encoding builds for it.",
    run=run,
    inputs=(
        Number("register_width", "Qubits per register", experiment.SYSTEM.register_width, 2, 8,
               "Integers are two's complement and wrap at this width."),
        Number("constant", "Constant operand", 5, -128, 127, "Used by the instructions that take a constant."),
    ),
    cases=tuple(Case(v.id, v.name, v.description) for v in VARIANTS),
    steps=tuple(Step(e.name, e.title, e.summary, enabled=e.name in SHOWN) for e in ENCODINGS.values()),
    workflow=False,
    cases_label="Instructions",
    steps_label=f"Encodings (up to {MOST})",
    steps_first=True,
    max_steps=MOST,
    port=8003,
)


def make_server(port: int):
    return app.make_server(DEMO, port)


def main() -> None:
    app.serve(DEMO)


if __name__ == "__main__":
    main()
