"""
Interactive demo of the compiler pipeline, for walking someone through what each pass does.

Pick a preloaded program or type your own, choose which passes to apply, and step through the stages. Each
stage shows the program before and after the pass with additions and changes highlighted, and runs it on a
register machine to show that the result is unchanged and how many registers are left unreleased.

Run it from the python directory:

    PYTHONPATH=src .venv/bin/python -m allocation_poc.demo

It serves one page on localhost and uses the real passes, so it always shows what the compiler does now.
"""
import argparse
import difflib
import json
import re
import warnings
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedInput, UnexpectedToken

from allocation_poc import circuits
from allocation_poc.printer import unparse, unparse_instruction, unparse_isa, unparse_statement
from allocation_poc.programs import PROGRAMS
from allocation_poc.simulate import Result, simulate, simulate_isa
from parser.grammar import Assign, FunctionDef, If, Node
from parser.instructions import Alloc, Free, IOp, Op
from parser.parser import parse
from passes.allocate import Allocation, WidthExceeded, allocate
from passes.allocation import lower_allocation, prepare
from passes.balance import balance
from passes.flatten import flatten
from passes.inline import inline
from passes.isa import lower_to_isa
from passes.liveness import Liveness
from passes.names import results
from passes.qasm import Circuit, to_openqasm
from passes.rename import rename
from passes.unroll import unroll

STATIC = Path(__file__).parent / "static"


@dataclass(frozen=True)
class PassInfo:
    id: str
    name: str
    summary: str
    requires: Tuple[str, ...]
    apply: Callable
    render: Callable = unparse  # how the pass's output is printed
    run: Callable = simulate  # how the pass's output is executed
    takes_options: bool = False  # whether the pass is also given the machine settings


# --- circuit stages ---------------------------------------------------------------------------------------------

MAX_CIRCUIT_LINES = 600  # a circuit can run to thousands of lines, so only the start is shown
MAX_CHECK_QUBITS = 18  # a circuit Qiskit has rewritten is checked on a state vector, which needs 2^n numbers

NEEDS_QISKIT = "This stage uses Qiskit, which is not installed: pip install qiskit qiskit-qasm3-import 'openqasm3[parser]'"


@dataclass(frozen=True)
class CircuitStage:
    """The circuit as generated, and the same circuit as Qiskit currently has it."""
    circuit: Circuit
    qiskit: object
    text: str


def _generate(allocation: Allocation) -> CircuitStage:
    circuit = to_openqasm(allocation)
    return CircuitStage(circuit, circuits.read(circuit.text) if circuits.qiskit_available() else None, circuit.text)


def _rewrite(rewrite: Callable) -> Callable:
    def apply(stage: CircuitStage) -> CircuitStage:
        if stage.qiskit is None:
            raise RuntimeError(NEEDS_QISKIT)
        rewritten = rewrite(stage.qiskit)
        return CircuitStage(stage.circuit, rewritten, circuits.write(rewritten))
    return apply


def _run_circuit(stage: CircuitStage) -> Result:
    """The value and leftover qubits of a circuit. The generated gates are run exactly; a Qiskit circuit on a state vector."""
    names = lambda qubits: {f"q[{q}]" for q in qubits}
    if stage.qiskit is None or stage.text == stage.circuit.text:
        state = circuits.run_gates(stage.circuit.gates, stage.circuit.qubits)
        return Result(circuits.value_of(state, stage.circuit.result),
                      {"<program>": names(circuits.leftover_qubits(state, stage.circuit.result))}, {"<program>": 0})
    outcome = circuits.run_state(stage.qiskit, stage.circuit.result, MAX_CHECK_QUBITS)
    if outcome is None:
        return Result(None, {"<program>": set()}, {"<program>": 0})
    qubits = outcome["leaked"]
    return Result(outcome["value"], {"<program>": {q if isinstance(q, str) else f"q[{q}]" for q in qubits}}, {"<program>": 0})


def circuit_metrics(stage: CircuitStage) -> Optional[Dict]:
    """Gate and wire counts, from Qiskit's reading of the circuit, so every circuit stage is counted the same way."""
    if stage.qiskit is None:
        return None
    return circuits.measure(stage.qiskit)


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
        "them and freed after. The "
        "program becomes one flat list of instructions. Functions were inlined earlier, so there are no calls.",
        ("allocation", "inline"), lower_to_isa, unparse_isa, simulate_isa,
    ),
    PassInfo(
        "allocate", "Assign physical qubits",
        "Maps every logical register onto a numbered slot of the machine, reusing a slot as soon as its register is "
        "freed. The machine is set by two numbers: qubits per register and total qubits. One register is set aside as "
        "scratch for the operations' ancillae, and the rest are the slots available. Step through to see which register holds each slot, and where it came from in the source.",
        ("isa",), allocate, lambda allocation: unparse_isa(allocation.program),
        lambda allocation: simulate_isa(allocation.program), True,
    ),
    PassInfo(
        "qasm", "Generate OpenQASM",
        "Writes each slot as w qubits of one register (least significant bit first) with the scratch register last, "
        "and each instruction as gates: X, CNOT, Toffoli and multi-controlled X. An inverse instruction is the reversed "
        "gates, and a CTRL region puts its control on every gate inside it. The counts are Qiskit's reading of the text.",
        ("allocate",), _generate, lambda stage: stage.text, _run_circuit,
    ),
    PassInfo(
        "optimise", "Optimise (Qiskit)",
        "Qiskit's own optimiser at its highest level. It also rewrites every gate into one- and two-qubit gates (u and cx), "
        "since a multi-controlled X cannot be counted as one gate on a device. It is treated as a black box here: the "
        "point is what it does to the gate and wire counts.",
        ("qasm",), _rewrite(circuits.optimise), lambda stage: stage.text, _run_circuit,
    ),
]

DEFAULT_OPTIONS = {"register_width": 6, "qubits": 100}  # room for every preloaded program, and values up to 63


def clean_options(raw) -> Dict[str, int]:
    """The machine settings from a request, with defaults for anything missing or out of range."""
    options = dict(DEFAULT_OPTIONS)
    limits = {"register_width": (1, 64), "qubits": (0, 4096)}
    for key, (low, high) in limits.items():
        try:
            value = int((raw or {}).get(key, options[key]))
        except (TypeError, ValueError):
            continue
        options[key] = min(max(value, low), high)
    return options


# --- highlighting ---------------------------------------------------------------------------------------------

_TOKEN = re.compile(r"[A-Za-z_][\w#]*|\d+(?:\.\d+)?|\s+|.")


def _line(text: str, kind: str, spans=()) -> Dict:
    return {"text": text, "kind": kind, "spans": [list(s) for s in spans]}


def _changed_spans(old: str, new: str):
    """Character ranges that differ between two lines, for the old and the new text."""
    a, b = _TOKEN.findall(old), _TOKEN.findall(new)
    starts_a = [0]
    for t in a:
        starts_a.append(starts_a[-1] + len(t))
    starts_b = [0]
    for t in b:
        starts_b.append(starts_b[-1] + len(t))
    old_spans, new_spans = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:
            old_spans.append((starts_a[i1], starts_a[i2]))
        if j2 > j1:
            new_spans.append((starts_b[j1], starts_b[j2]))
    return old_spans, new_spans


def diff_lines(before: List[str], after: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """Mark each line of `before` and `after` as same, added, removed or changed (with the changed ranges)."""
    old: List[Dict] = [{}] * len(before)
    new: List[Dict] = [{}] * len(after)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                old[i], new[j] = _line(before[i], "same"), _line(after[j], "same")
            continue
        paired = 0
        if tag == "replace":
            for k in range(min(i2 - i1, j2 - j1)):
                i, j = i1 + k, j1 + k
                if difflib.SequenceMatcher(None, before[i], after[j]).ratio() < 0.5:
                    break  # too different to call a modification
                old_spans, new_spans = _changed_spans(before[i], after[j])
                old[i], new[j] = _line(before[i], "changed", old_spans), _line(after[j], "changed", new_spans)
                paired += 1
        for i in range(i1 + paired, i2):
            old[i] = _line(before[i], "removed")
        for j in range(j1 + paired, j2):
            new[j] = _line(after[j], "added")
    return old, new


# --- running the pipeline ---------------------------------------------------------------------------------------

def _measure(nodes, run: Callable = simulate) -> Dict:
    """Run the program on the register machine."""
    try:
        result = run(nodes)
    except RecursionError:
        return {"error": "recursion went too deep"}
    except Exception as error:  # a user program can fail in many ways; show it instead of crashing the demo
        return {"error": str(error) or type(error).__name__}
    leaked = sorted(
        name if scope == "<program>" else f"{scope}: {name}"
        for scope, names in result.leaks.items() for name in names
    )
    value = result.value
    if value is None:  # a circuit too big to run on a state vector
        return {"value": "unchecked", "unchecked": True, "leaked": [], "peak": None}
    return {
        "value": "true" if value is True else "false" if value is False else str(value),
        "leaked": leaked,
        "peak": max(result.peak.values(), default=0),
    }


# --- liveness view ----------------------------------------------------------------------------------------------

def _first_line(statement: Node) -> str:
    lines = [l.strip() for l in unparse_statement(statement).splitlines() if not l.strip().startswith("//")]
    return lines[0] + (" …" if len(lines) > 1 else "")


def _scope(title: str, body: Sequence[Node], analysis: Liveness, params: Sequence[str], depth: int, out: List[Dict]) -> None:
    """One block as rows (statements) and columns (versions), with where each version is live and released."""
    n = len(body)
    defined_row: Dict[str, int] = {}
    for k, statement in enumerate(body):
        if isinstance(statement, Assign):
            defined_row[statement.name] = k
        elif isinstance(statement, If):
            defined_row.update({name: k for name in sorted(results(statement))})

    order = {}  # position among the versions released at the same point
    for name in analysis.release_order:
        point = analysis.reclaim_at[name]
        order[name] = 1 + sum(1 for other in order if analysis.reclaim_at[other] == point)

    columns = []
    names = list(params) + sorted((n_ for n_ in defined_row if n_ not in params), key=lambda v: (defined_row[v], v))
    for name in names:
        is_param = name in params
        start = 0 if is_param else defined_row[name]
        reclaim = analysis.reclaim_at.get(name)
        died = analysis.dies_at.get(name)
        if is_param:
            kind = "param"
        elif name in analysis.kept:
            kind = "kept"  # in a branch: needed to release what the if produced, so it outlives the branch
        elif name in analysis.outputs:
            kind = "output"
        else:
            kind = "temp"
        conservative_end = n - 1 if died is None else died - 1
        live_end = reclaim - 1 if kind == "temp" else n - 1 if kind in ("output", "kept") else conservative_end
        columns.append({
            "name": name,
            "kind": kind,
            "defined": None if is_param else defined_row[name],
            "start": start,
            "live_end": live_end,
            "conservative_end": max(conservative_end, live_end),
            "release_row": reclaim - 1 if kind == "temp" else None,
            "order": order.get(name) if kind == "temp" else None,
        })
    out.append({"title": title, "depth": depth, "rows": [_first_line(s) for s in body], "columns": columns})

    for k, statement in enumerate(body):
        label = f"row {k + 1}"
        if isinstance(statement, FunctionDef):
            continue  # a function is its own scope, added by the caller
        if isinstance(statement, If):
            for branch, block, child in zip(("then", "else"), (statement.body, statement.orelse), analysis.nested[k]):
                _scope(f"{branch} branch of {label}", block, child, (), depth + 1, out)


def liveness_view(nodes: Sequence[Node]) -> Dict:
    """The liveness analysis behind the release pass, as scopes of rows and columns for display."""
    named, analyses = prepare(nodes)
    scopes: List[Dict] = []
    _scope("Program", named, analyses["<program>"], (), 0, scopes)
    for statement in named:
        if isinstance(statement, FunctionDef):
            _scope(f"function {statement.name}", statement.body, analyses[statement.name], [p.name for p in statement.params], 0, scopes)
    return {"scopes": scopes}


def _stage(info: Dict, nodes, previous: str, text: str, notes: Sequence[str] = (), liveness=None, run: Callable = simulate) -> Dict:
    if previous:
        before, after = diff_lines(previous.splitlines(), text.splitlines())
    else:  # the first stage has nothing to compare against
        before, after = [], [_line(t, "same") for t in text.splitlines()]
    count = lambda lines, kind: sum(1 for line in lines if line["kind"] == kind)
    return {
        **info,
        "text": text,
        "before": before,
        "after": after,
        "stats": {
            "lines": len(after),
            "added": count(after, "added"),
            "changed": count(after, "changed"),
            "removed": count(before, "removed"),
            "releases": sum(1 for line in after if "~=" in line["text"] or line["text"].strip().startswith("FREE")),
        },
        "notes": list(notes),
        "liveness": liveness,
        "run": _measure(nodes, run),
    }


def _check_circuit(stage: Dict, stages: List[Dict], circuit: CircuitStage) -> None:
    """Say whether a circuit stage returns what the source returns. The value is the register's, modulo 2 to its width."""
    run = stage["run"]
    if "error" in run or run.get("unchecked"):
        if run.get("unchecked"):
            run["note"] = f"more than {MAX_CHECK_QUBITS} qubits, too many to run on a state vector"
        return
    source = stages[0]["run"].get("value")
    expected = {"true": 1, "false": 0}.get(source)
    if expected is None:
        try:
            expected = int(float(source))
        except (TypeError, ValueError):
            return
    run["matches"] = int(run["value"]) == expected & ((1 << circuit.circuit.register_width) - 1)


def _describe(error: Exception) -> Dict:
    if isinstance(error, UnexpectedEOF):
        return {"message": "The program ended early. Every program, and every function, ends with a return."}
    if isinstance(error, UnexpectedToken):
        if error.token.type == "$END":
            return {"message": "The program ended early. Every program, and every function, ends with a return."}
        expected = set(error.expected or ())
        if "INT" in expected and "NUMBER" not in expected:
            message = f"Unexpected '{error.token}': loop bounds must be whole-number literals, not variables or expressions."
        else:
            message = f"Unexpected '{error.token}'."
        return {"message": message, "line": error.line, "column": error.column}
    if isinstance(error, UnexpectedCharacters):
        return {"message": f"Unexpected character '{error.char}'.", "line": error.line, "column": error.column}
    return {"message": str(error)}


# --- allocation view --------------------------------------------------------------------------------------------

class _SourceIndex:
    """Where each variable is assigned in the source, by function, so a register can point back at its lines."""

    _FUNCTION = re.compile(r"^\s*function\s+(\w+)\s*\(")

    def __init__(self, source: str) -> None:
        self.lines = source.splitlines()
        self.scope_of: List[str] = []
        scope, depth = "main", 0
        for line in self.lines:
            code = line.split("//")[0]
            match = self._FUNCTION.match(code)
            if match and depth == 0:
                scope = match.group(1)
            self.scope_of.append(scope)
            depth += code.count("{") - code.count("}")
            if depth <= 0:
                depth, scope = 0, "main"
        self.cache: Dict[Tuple[str, str], List[int]] = {}

    def lines_for(self, scope: str, variable: str) -> List[int]:
        key = (scope.split("@")[0], variable)
        if key not in self.cache:
            name = re.escape(variable)
            pattern = re.compile(rf"\b{name}\s*~?=(?!=)|\b{name}\s*:|\bfor\s+{name}\b")
            self.cache[key] = [
                number for number, line in enumerate(self.lines, 1)
                if self.scope_of[number - 1] == key[0] and pattern.search(line.split("//")[0])
            ]
        return self.cache[key]


def source_origin(register: str) -> Tuple[Optional[str], str]:
    """
    Where a logical register came from in the source, as (variable, function): x#2 is (x, main), area1.p#1 is
    (p, area) because it was inlined from area, and h1.g1.m#1 is (m, g). Loop copies (t@1) are the variable t.
    Compiler temporaries ($t1#1) have no variable.
    """
    base = register.split("#")[0]
    if base.startswith("$"):
        return None, "main"
    parts = base.split(".")
    function = re.sub(r"\d+$", "", parts[-2]) if len(parts) > 1 else "main"
    return parts[-1].split("@")[0], function


def _acted_on(instruction) -> str:
    if isinstance(instruction, (Alloc, Free)):
        return instruction.register
    if isinstance(instruction, (Op, IOp)):
        return instruction.dst
    return ""


def allocation_view(allocation: Allocation, source: str) -> Dict:
    """The allocation as data for the viewer: the machine, and for every step what each slot held."""
    index = _SourceIndex(source)

    def label(register: str) -> Dict:
        variable, function = source_origin(register) if register else (None, "main")
        return {
            "register": register,
            "function": function,
            "variable": variable,
            "lines": index.lines_for(function, variable) if variable else [],
        }

    steps = []
    for step in allocation.trace:
        steps.append({
            "logical": unparse_instruction(step.logical),
            "physical": unparse_instruction(step.physical),
            "held": [{"slot": h.slot, **label(h.register)} for h in step.held],
            "changed": step.changed,
            "focus": label(_acted_on(step.logical)),
        })
    return {
        "register_width": allocation.register_width,
        "qubits": allocation.qubits,
        "slots": allocation.slots,
        "scratch": allocation.scratch_qubits,
        "qubits_used": allocation.qubits_used,
        "needed": allocation.needed,
        "fits": allocation.fits,
        "steps": steps,
    }


def run_pipeline(source: str, pass_ids: Sequence[str], options=None) -> Dict:
    """Parse `source` and apply the chosen passes in compiler order, returning every stage."""
    options = clean_options(options)
    try:
        nodes = parse(source, passes=())
    except UnexpectedInput as error:
        return {"error": _describe(error)}
    except Exception as error:
        return {"error": {"message": str(error)}}

    text = unparse(nodes)
    run = simulate
    stages = [_stage({"id": "source", "name": "Source", "summary": "Your program, parsed and printed back out."}, nodes, "", text)]
    for info in PASSES:
        if info.id not in pass_ids:
            continue
        details = {"id": info.id, "name": info.name, "summary": info.summary}
        liveness = None
        if info.id == "allocation" and any(stage["id"] == "rename" for stage in stages):
            try:  # the analysis only means something once every name is a unique version
                liveness = liveness_view(nodes)
            except Exception:
                liveness = None
        failure = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                lowered = info.apply(nodes, options["register_width"], options["qubits"]) if info.takes_options else info.apply(nodes)
            except WidthExceeded as error:  # still show the packing it would have needed
                lowered, failure = error.allocation, f"{error} ({options['qubits']} qubits ÷ {options['register_width']} per register)"
            except Exception as error:
                lowered, failure = nodes, f"{type(error).__name__}: {error}"
        if failure and lowered is nodes:
            hint = [f"This pass expects {', '.join(info.requires)} to have run first."] if info.requires else []
            stage = _stage(details, nodes, text, text, hint, run=run)
            stage["error"] = failure
            stages.append(stage)
            break
        nodes, previous, text = lowered, text, info.render(lowered)
        run = info.run
        is_circuit = isinstance(lowered, CircuitStage)
        if is_circuit:  # thousands of lines are not worth a line by line comparison, or showing in full
            lines = text.splitlines()
            shown = lines[:MAX_CIRCUIT_LINES] + ([f"// ... {len(lines) - MAX_CIRCUIT_LINES} more lines"] if len(lines) > MAX_CIRCUIT_LINES else [])
            stage = _stage(details, nodes, "", "\n".join(shown), [str(w.message) for w in caught], liveness, run)
            stage["single"] = True
            stage["run"]["peak"] = None  # registers do not exist at this level; the wire counts replace it
            stage["circuit"] = circuit_metrics(lowered)
            stage["diagram"] = circuits.diagram(lowered.circuit, None if lowered.text == lowered.circuit.text else lowered.qiskit)
            _check_circuit(stage, stages, lowered)
        else:
            stage = _stage(details, nodes, previous, text, [str(w.message) for w in caught], liveness, run)
        if isinstance(lowered, Allocation):
            stage["allocation"] = allocation_view(lowered, source)
        if failure:
            stage["error"] = failure
        stages.append(stage)
        if failure:
            break
    return {"stages": stages, "options": options}


# --- server -----------------------------------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json(self, payload, status: int = 200) -> None:
        self.reply(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.reply(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/programs":
            self.json(PROGRAMS)
        elif self.path == "/api/passes":
            self.json([{"id": p.id, "name": p.name, "summary": p.summary, "requires": list(p.requires)} for p in PASSES])
        else:
            self.json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/run":
            return self.json({"error": "not found"}, 404)
        length = int(self.headers.get("Content-Length") or 0)
        if length > 200_000:
            return self.json({"error": "program too large"}, 413)
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
            source = str(request.get("source", ""))
            passes = [p for p in request.get("passes", []) if isinstance(p, str)]
            options = request.get("options")
        except (ValueError, AttributeError):
            return self.json({"error": "bad request"}, 400)
        self.json(run_pipeline(source, passes, options if isinstance(options, dict) else None))


def make_server(port: int) -> HTTPServer:
    """
    One request at a time, all on the main thread. A server thread per request crashes the whole process: the
    Qiskit wheel's native allocator segfaults when a thread starts after an earlier thread has used Qiskit and
    finished (reproduced on Python 3.14 with qiskit 2.5.2), which is exactly one request after another.
    """
    return HTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = parser.parse_args()
    server = make_server(args.port)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Pipeline demo running at {url} (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
