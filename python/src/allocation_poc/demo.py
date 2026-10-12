"""
Interactive demo of the compiler pipeline, for walking someone through what each pass does.

Pick a preloaded program or type your own, choose which passes to apply, and step through the stages. Each stage
shows the program before and after the pass with additions and changes highlighted, and runs it on a register
machine to show that the result is unchanged and how many registers are left unreleased. A circuit stage can be run
on a simulator on request.

Run it from the python directory:

    PYTHONPATH=src .venv/bin/python -m allocation_poc.demo

It serves one page on localhost and uses the real passes, so it always shows what the compiler does now.
"""
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedInput, UnexpectedToken

from allocation_poc import circuits
from allocation_poc.drawing import diagram
from allocation_poc.printer import unparse, unparse_isa
from allocation_poc.programs import PROGRAMS
from allocation_poc.simulate import Result, SimulationError, simulate, simulate_isa
from allocation_poc.views import allocation_view, liveness_view
from parser.parser import parse
from passes.allocate import Allocation, WidthExceeded, allocate
from passes.allocation import lower_allocation
from passes.balance import balance
from passes.flatten import flatten
from passes.inline import inline
from passes.isa import lower_to_isa
from passes.qasm import Circuit, to_openqasm
from passes.rename import rename
from passes.unroll import unroll
from utils import http_server, qiskit_tools
from utils.text import shorten
from utils.text_diff import diff_lines, line

STATIC = Path(__file__).parent / "static"


@dataclass(frozen=True)
class PassInfo:
    id: str
    name: str
    summary: str
    requires: Tuple[str, ...]
    apply: Callable
    render: Callable = unparse  # how the pass's output is printed
    run: Optional[Callable] = simulate  # how the pass's output is executed; None if it is only run on request
    takes_options: bool = False  # whether the pass is also given the machine settings


# --- circuit stages ---------------------------------------------------------------------------------------------

NEEDS_QISKIT = "This stage uses Qiskit, which is not installed: pip install qiskit qiskit-qasm3-import 'openqasm3[parser]' qiskit-aer"
NEEDS_AER = "Running a circuit needs Qiskit and Qiskit Aer: pip install qiskit qiskit-aer qiskit-qasm3-import"


@dataclass(frozen=True)
class CircuitStage:
    """The circuit as generated, and the same circuit as Qiskit currently has it."""
    circuit: Circuit
    qiskit: Any
    text: str


def _generate(allocation: Allocation) -> CircuitStage:
    circuit = to_openqasm(allocation)
    return CircuitStage(circuit, qiskit_tools.read(circuit.text) if qiskit_tools.qiskit_available() else None, circuit.text)


def _rewrite(rewrite: Callable) -> Callable[[CircuitStage], CircuitStage]:
    def apply(stage: CircuitStage) -> CircuitStage:
        if stage.qiskit is None:
            raise RuntimeError(NEEDS_QISKIT)
        rewritten = rewrite(stage.qiskit)
        return CircuitStage(stage.circuit, rewritten, qiskit_tools.write(rewritten))
    return apply


def circuit_metrics(stage: CircuitStage) -> Optional[Dict]:
    """Gate and wire counts, from Qiskit's reading of the circuit, so every circuit stage is counted the same way."""
    return qiskit_tools.measure(stage.qiskit) if stage.qiskit is not None else None


# In the order the compiler runs them.
PASSES = [
    PassInfo(
        "inline", "Inline functions",
        "A circuit has no calls, so every call is replaced by a copy of the function body and the definitions are "
        "dropped. The copy's locals get their own names (area1.p, area2.p) and a plain variable or literal argument "
        "stands in for its parameter with no copy. The release pass then treats the callee's locals like any other.",
        (), inline,
    ),
    PassInfo(
        "unroll", "Unroll loops",
        "Loop bounds are fixed numbers, so each for loop is replaced by one copy of its body per iteration, with "
        "the loop variable replaced by the iteration number. Later passes only see straight-line code and ifs.",
        (), unroll,
    ),
    PassInfo(
        "flatten", "Flatten expressions",
        "One operation per step. Anything nested, like the 2 * a in (2 * a) + 1, goes into a temporary of its own "
        "just before it is needed, so every intermediate value can be released like any other.",
        ("unroll",), flatten,
    ),
    PassInfo(
        "balance", "Balance branches",
        "If only one branch of an if changes a variable, the other branch gets an identity assignment (x = x;), "
        "so afterwards the variable has a new value on every path.",
        (), balance,
    ),
    PassInfo(
        "rename", "Rename",
        "Gives every assignment its own version (x becomes x#1, x#2, ...), so each value has exactly one "
        "producer. Both branches of an if share the same final version.",
        ("balance",), rename,
    ),
    PassInfo(
        "allocation", "Insert releases",
        "Works out where each temporary last matters, following Faro, Marino and Messina's liveness, and inserts "
        "a release there. x ~= e; means x currently equals e, so it can be uncomputed and freed.",
        ("rename",), lower_allocation,
    ),
    PassInfo(
        "isa", "Lower to instructions",
        "Each assignment becomes ALLOC plus one instruction that computes the value into the clean register, and each "
        "release is its explicit inverse (IMUL undoes MUL) plus FREE. An if becomes a CTRL region and an ICTRL region "
        "(controlled on the condition and on its negation), each closed by ENDCTRL, with registers allocated before "
        "them and freed after. The program becomes one flat list of instructions. Functions were inlined earlier, so "
        "there are no calls.",
        ("allocation", "inline"), lower_to_isa, unparse_isa, simulate_isa,
    ),
    PassInfo(
        "allocate", "Assign physical qubits",
        "Maps every logical register onto a numbered slot of the machine, reusing a slot as soon as its register is "
        "freed. The machine is set by two numbers: qubits per register and total qubits. One register is set aside as "
        "scratch for the operations' ancillae, and the rest are the slots available. Step through to see which register "
        "holds each slot, and where it came from in the source.",
        ("isa",), allocate, lambda allocation: unparse_isa(allocation.program),
        lambda allocation: simulate_isa(allocation.program), True,
    ),
    PassInfo(
        "qasm", "Generate OpenQASM",
        "Writes each slot as w qubits of one register (least significant bit first) with the scratch register last, "
        "and each instruction as gates: X, CNOT, Toffoli and multi-controlled X. An inverse instruction is the reversed "
        "gates, and a CTRL region puts its control on every gate inside it. The returned register is measured. The "
        "counts are Qiskit's reading of the text, and the circuit can be run on Qiskit Aer on request.",
        ("allocate",), _generate, lambda stage: stage.text, None,
    ),
    PassInfo(
        "optimise", "Optimise (Qiskit)",
        "Qiskit's own optimiser at its highest level. It also rewrites every gate into one- and two-qubit gates (u and cx), "
        "since a multi-controlled X cannot be counted as one gate on a device. It is treated as a black box here: the "
        "point is what it does to the gate and wire counts.",
        ("qasm",), _rewrite(qiskit_tools.optimise), lambda stage: stage.text, None,
    ),
]


DEFAULT_OPTIONS = {"register_width": 6, "qubits": 100}  # room for every preloaded program, and values up to 63
_OPTION_LIMITS = {"register_width": (1, 64), "qubits": (0, 4096)}


def clean_options(raw) -> Dict[str, int]:
    """The machine settings from a request, with defaults for anything missing or out of range."""
    options = dict(DEFAULT_OPTIONS)
    for key, (low, high) in _OPTION_LIMITS.items():
        try:
            value = int((raw or {}).get(key, options[key]))
        except (TypeError, ValueError):
            continue
        options[key] = min(max(value, low), high)
    return options


# --- building the stages ----------------------------------------------------------------------------------------

def _measure(nodes, run: Optional[Callable]) -> Dict:
    """Run a stage's output on its machine, if it has one, and report the value, leftover registers and peak."""
    if run is None:  # a circuit, which is only run on request
        return {"value": "not run", "notrun": True, "leaked": [], "peak": None}
    try:
        result: Result = run(nodes)
    except RecursionError:
        return {"error": "recursion went too deep"}
    except Exception as error:  # a user program can fail in many ways; show it instead of crashing the demo
        return {"error": str(error) or type(error).__name__}
    leaked = sorted(
        name if scope == "<program>" else f"{scope}: {name}"
        for scope, names in result.leaks.items() for name in names
    )
    value = result.value
    return {
        "value": "true" if value is True else "false" if value is False else str(value),
        "leaked": leaked,
        "peak": max(result.peak.values(), default=0),
    }


def _count(lines: List[Dict], kind: str) -> int:
    return sum(1 for entry in lines if entry["kind"] == kind)


def _stage(info: Dict, nodes, previous: str, text: str, notes: Sequence[str] = (), liveness=None,
           run: Optional[Callable] = simulate) -> Dict:
    if previous:
        before, after = diff_lines(previous.splitlines(), text.splitlines())
    else:  # the first stage has nothing to compare against
        before, after = [], [line(t, "same") for t in text.splitlines()]
    return {
        **info,
        "text": text,
        "before": before,
        "after": after,
        "stats": {
            "lines": len(after),
            "added": _count(after, "added"),
            "changed": _count(after, "changed"),
            "removed": _count(before, "removed"),
            "releases": sum(1 for entry in after if "~=" in entry["text"] or entry["text"].strip().startswith("FREE")),
        },
        "notes": list(notes),
        "liveness": liveness,
        "run": _measure(nodes, run),
    }


def _circuit_stage(details: Dict, stage: CircuitStage, notes: Sequence[str]) -> Dict:
    """A circuit stage: only the start of the text is shown, there is no comparison with the last stage, and it carries counts and a drawing."""
    entry = _stage(details, stage, "", shorten(stage.text), notes, run=None)
    entry["single"] = True
    entry["circuit"] = circuit_metrics(stage)
    entry["diagram"] = diagram(stage.circuit, None if stage.text == stage.circuit.text else stage.qiskit)
    return entry


def describe_parse_error(error: Exception) -> Dict:
    """A syntax error as a message for the viewer, with where it happened."""
    if isinstance(error, UnexpectedEOF) or (isinstance(error, UnexpectedToken) and error.token.type == "$END"):
        return {"message": "The program ended early. Every program, and every function, ends with a return."}
    if isinstance(error, UnexpectedToken):
        if "INT" in set(error.expected or ()) and "NUMBER" not in set(error.expected or ()):
            message = f"Unexpected '{error.token}': loop bounds must be whole-number literals, not variables or expressions."
        else:
            message = f"Unexpected '{error.token}'."
        return {"message": message, "line": error.line, "column": error.column}
    if isinstance(error, UnexpectedCharacters):
        return {"message": f"Unexpected character '{error.char}'.", "line": error.line, "column": error.column}
    return {"message": str(error)}


# --- running the pipeline ---------------------------------------------------------------------------------------

def _apply(info: PassInfo, nodes, options: Dict[str, int]) -> Tuple[Any, Optional[str], List[str]]:
    """Run one pass: its output, the failure message if it failed, and any warnings. A failed pass leaves its input as the output."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            output = info.apply(nodes, options["register_width"], options["qubits"]) if info.takes_options else info.apply(nodes)
            failure = None
        except WidthExceeded as error:  # still show the packing it would have needed
            output, failure = error.allocation, f"{error} ({options['qubits']} qubits ÷ {options['register_width']} per register)"
        except Exception as error:  # a pass can fail on a user's program in many ways; show it instead of crashing the demo
            output, failure = nodes, f"{type(error).__name__}: {error}"
    return output, failure, [str(w.message) for w in caught]


def _liveness_for(info: PassInfo, ran: Sequence[str], nodes):
    """The liveness view, for the release pass once every name has been made a unique version."""
    if info.id != "allocation" or "rename" not in ran:
        return None
    try:
        return liveness_view(nodes)
    except Exception:  # the analysis is only for display; without it the stage just has no chart
        return None


def _pipeline(source: str, pass_ids: Sequence[str], options=None) -> Tuple[Dict, Dict]:
    """Every stage as data for the viewer, and what each pass produced (by pass id), for anything that needs the objects."""
    options = clean_options(options)
    outputs: Dict[str, object] = {}
    try:
        nodes = parse(source, passes=())
    except UnexpectedInput as error:
        return {"error": describe_parse_error(error)}, outputs
    except Exception as error:  # the grammar's transformer can reject a program in ways the parser does not
        return {"error": {"message": str(error)}}, outputs

    text, run = unparse(nodes), simulate
    stages = [_stage({"id": "source", "name": "Source", "summary": "Your program, parsed and printed back out."}, nodes, "", text)]
    for info in PASSES:
        if info.id not in pass_ids:
            continue
        details = {"id": info.id, "name": info.name, "summary": info.summary}
        liveness = _liveness_for(info, [s["id"] for s in stages], nodes)  # of what the pass is given
        lowered, failure, notes = _apply(info, nodes, options)
        if failure and lowered is nodes:
            hint = [f"This pass expects {', '.join(info.requires)} to have run first."] if info.requires else []
            stage = _stage(details, nodes, text, text, hint, run=run)
        else:
            previous, nodes, text, run = text, lowered, info.render(lowered), info.run
            outputs[info.id] = lowered
            if isinstance(lowered, CircuitStage):
                stage = _circuit_stage(details, lowered, notes)
            else:
                stage = _stage(details, nodes, previous, text, notes, liveness, run)
            if isinstance(lowered, Allocation):
                stage["allocation"] = allocation_view(lowered, source)
        if failure:
            stage["error"] = failure
        stages.append(stage)
        if failure:
            break
    return {"stages": stages, "options": options}, outputs


def run_pipeline(source: str, pass_ids: Sequence[str], options=None) -> Dict:
    """Parse `source` and apply the chosen passes in compiler order, returning every stage."""
    return _pipeline(source, pass_ids, options)[0]


def _expected_value(source: str, width: int) -> Optional[int]:
    """What the program returns when run directly, as a register of `width` bits holds it."""
    try:
        return circuits.wrap(simulate(parse(source, passes=())).value, width)
    except SimulationError:
        return None


def simulate_stage(source: str, pass_ids: Sequence[str], options, stage_id: str, shots: int = 100) -> Dict:
    """
    Run one circuit stage on Qiskit Aer, on request, and report what was measured next to what the program should
    return. The circuit is rebuilt from the source, so the answer is for exactly what the viewer shows.
    """
    _, outputs = _pipeline(source, pass_ids, options)
    stage = outputs.get(stage_id)
    if not isinstance(stage, CircuitStage):
        return {"error": f"{stage_id} is not a circuit stage that was produced"}
    if stage.qiskit is None or not qiskit_tools.aer_available():
        return {"error": NEEDS_AER}
    shots = min(max(shots, 1), qiskit_tools.MAX_SHOTS)
    outcome = qiskit_tools.execute(stage.qiskit, len(stage.circuit.measured), shots)
    expected = _expected_value(source, stage.circuit.register_width)
    return {"stage": stage_id, **circuits.report(outcome, shots=shots, expected=expected)}


# --- server -----------------------------------------------------------------------------------------------------

def _run_route(request: Dict) -> Dict:
    passes = [p for p in request.get("passes", []) if isinstance(p, str)]
    options = request.get("options")
    return run_pipeline(http_server.text_field(request, "source"), passes, options if isinstance(options, dict) else None)


def _simulate_route(request: Dict) -> Dict:
    passes = [p for p in request.get("passes", []) if isinstance(p, str)]
    options = request.get("options")
    return simulate_stage(http_server.text_field(request, "source"), passes, options if isinstance(options, dict) else None,
                          http_server.text_field(request, "stage"), http_server.int_field(request, "shots", 100))


class Handler(http_server.JsonHandler):
    static_dir = STATIC
    pages = {"/": "index.html", "/index.html": "index.html", "/circuit.js": "circuit.js"}
    get_routes = {
        "/api/programs": lambda: PROGRAMS,
        "/api/passes": lambda: [{"id": p.id, "name": p.name, "summary": p.summary, "requires": list(p.requires)} for p in PASSES],
    }
    post_routes = {"/api/run": _run_route, "/api/simulate": _simulate_route}


def make_server(port: int):
    return http_server.make_server(Handler, port)


def main() -> None:
    http_server.serve(Handler, name="Pipeline demo", description=__doc__.split("\n\n")[0], default_port=8000)


if __name__ == "__main__":
    main()
