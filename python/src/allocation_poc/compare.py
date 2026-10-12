"""
Comparison app: our circuits next to Qrisp's for the same task.

Each demo program goes through our compiler as usual (inline, allocate, generate OpenQASM). The same task, written by
hand as an ordinary Qrisp program using only Qrisp's own operators (qrisp_programs.py), goes through Qrisp's own compiler.
Both are measured the same way, as built and after the same Qiskit optimisation into `u` and `cx`, with gate and wire
counts and a drawing, and either can be run on Qiskit Aer on request (nothing runs on its own).

Run it from the python directory:

    PYTHONPATH=src .venv/bin/python -m allocation_poc.compare

It needs the optional dependencies: pip install '.[circuits,compare]'
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from allocation_poc import circuits, qrisp_programs
from allocation_poc.drawing import diagram, qiskit_diagram
from allocation_poc.programs import PROGRAMS
from allocation_poc.simulate import simulate_isa
from parser.parser import parse
from passes.allocate import allocate
from passes.isa import lower_to_isa
from passes.qasm import to_openqasm
from utils import http_server, qiskit_tools
from utils.text import shorten

STATIC = Path(__file__).parent / "static"
MAX_WIDTH = 16
DEFAULT_WIDTH = 8
SIDES = ("ours", "qrisp")
VARIANTS = ("built", "optimised")
_PROGRAMS = {program["id"]: program for program in PROGRAMS}


def clean_width(raw) -> int:
    try:
        return min(max(int(raw), 1), MAX_WIDTH)
    except (TypeError, ValueError):
        return DEFAULT_WIDTH


@dataclass(frozen=True)
class Side:
    """One compiler's circuit for a program, as built and as optimised, with what is needed to draw and run it."""
    built: Any
    optimised: Any
    result_bits: int
    exponent: int  # a Qrisp QuantumFloat holds mantissa * 2 ** exponent
    text: Dict[str, str]  # the OpenQASM of each variant
    drawings: Dict[str, Dict[str, Any]]

    def variant(self, name: str):
        return self.built if name == "built" else self.optimised


@dataclass(frozen=True)
class Built:
    expected: int
    sources: Dict[str, str]
    sides: Dict[str, Side]


def _text(circuit) -> str:
    try:
        return qiskit_tools.write(circuit)
    except Exception as error:  # a circuit with gates OpenQASM cannot name
        return f"// this circuit cannot be written as OpenQASM: {error}"


def _ours(source: str, width: int):
    isa = lower_to_isa(parse(source))
    generated = to_openqasm(allocate(isa, width, 100_000))
    built = qiskit_tools.read(generated.text)
    optimised = qiskit_tools.optimise(built)
    side = Side(
        built, optimised, len(generated.measured), 0,
        {"built": generated.text, "optimised": _text(optimised)},
        {"built": diagram(generated), "optimised": diagram(generated, optimised)},
    )
    return side, circuits.wrap(simulate_isa(isa).value, width)


def _qrisp(program_id: str, width: int) -> Side:
    qrisp = qrisp_programs.run(program_id, width)
    optimised = qiskit_tools.optimise(qrisp.circuit)
    return Side(
        qrisp.circuit, optimised, len(qrisp.result), qrisp.exponent,
        {"built": _text(qrisp.circuit), "optimised": _text(optimised)},
        {"built": qiskit_diagram(qrisp.circuit, qrisp.labels, qrisp.groups),
         "optimised": qiskit_diagram(optimised, qrisp.labels, qrisp.groups)},
    )


@lru_cache(maxsize=6)  # so running one circuit does not rebuild everything
def build(program_id: str, width: int) -> Built:
    """Both circuits for a program, as built and optimised."""
    if program_id not in _PROGRAMS:
        raise KeyError(f"unknown program {program_id}")
    ours, expected = _ours(_PROGRAMS[program_id]["source"], width)
    return Built(
        expected,
        {"ours": _PROGRAMS[program_id]["source"], "qrisp": qrisp_programs.source_text(program_id, width)},
        {"ours": ours, "qrisp": _qrisp(program_id, width)},
    )


def compare(program_id: str, width: int) -> Dict[str, Any]:
    """Everything the page shows for one program: both sources, and counts, text and drawing for each of the four circuits."""
    try:
        built = build(program_id, width)
    except Exception as error:
        return {"error": {"message": f"{type(error).__name__}: {error}"}}
    return {
        "width": width,
        "expected": built.expected,
        "sources": built.sources,
        "circuits": {
            name: {
                variant: {
                    "metrics": qiskit_tools.measure(side.variant(variant)),
                    "text": shorten(side.text[variant]),
                    "diagram": side.drawings[variant],
                }
                for variant in VARIANTS
            }
            for name, side in built.sides.items()
        },
    }


def simulate(program_id: str, width: int, side: str, variant: str, shots: int) -> Dict[str, Any]:
    """Run one of the four circuits on Qiskit Aer and report the measurement against the expected value."""
    if side not in SIDES or variant not in VARIANTS:
        return {"error": "unknown circuit"}
    if not qiskit_tools.aer_available():
        return {"error": "Running a circuit needs Qiskit Aer: pip install qiskit-aer"}
    try:
        built = build(program_id, width)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}
    shots = min(max(shots, 1), qiskit_tools.MAX_SHOTS)
    entry = built.sides[side]
    outcome = qiskit_tools.execute(entry.variant(variant), entry.result_bits, shots)
    return {"side": side, "variant": variant,
            **circuits.report(outcome, shots=shots, expected=built.expected, scale=2 ** entry.exponent)}


def _compare_route(request: Dict[str, Any]) -> Dict[str, Any]:
    return compare(http_server.text_field(request, "program"), clean_width(request.get("width")))


def _simulate_route(request: Dict[str, Any]) -> Dict[str, Any]:
    return simulate(http_server.text_field(request, "program"), clean_width(request.get("width")),
                    http_server.text_field(request, "side"), http_server.text_field(request, "variant"),
                    http_server.int_field(request, "shots", 100))


class Handler(http_server.JsonHandler):
    static_dir = STATIC
    pages = {"/": "compare.html", "/index.html": "compare.html", "/circuit.js": "circuit.js"}
    get_routes = {
        "/api/programs": lambda: [{"id": p["id"], "name": p["name"], "description": p["description"]} for p in PROGRAMS],
    }
    post_routes = {"/api/compare": _compare_route, "/api/simulate": _simulate_route}


def make_server(port: int):
    return http_server.make_server(Handler, port)


def main() -> None:
    http_server.serve(Handler, name="Qrisp comparison", description=__doc__.split("\n\n")[0], default_port=8001)


if __name__ == "__main__":
    main()
