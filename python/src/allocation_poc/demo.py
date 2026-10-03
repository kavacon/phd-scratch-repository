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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedInput, UnexpectedToken

from allocation_poc.printer import unparse, unparse_statement
from allocation_poc.programs import PROGRAMS
from allocation_poc.simulate import simulate
from parser.grammar import Assign, FunctionDef, If, Node
from parser.parser import parse
from passes.allocation import lower_allocation, prepare
from passes.balance import balance
from passes.liveness import Liveness
from passes.names import results
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


# In the order the compiler runs them.
PASSES = [
    PassInfo(
        "unroll", "Unroll loops",
        "Loop bounds are fixed numbers, so each for loop is replaced by one copy of its body per iteration, with "
        "the loop variable replaced by the iteration number. Later passes only see straight-line code and ifs.",
        (), unroll,
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
]


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

def _measure(nodes) -> Dict:
    """Run the program on the register machine."""
    try:
        result = simulate(nodes)
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
        elif name in analysis.outputs:
            kind = "output"
        else:
            kind = "temp"
        conservative_end = n - 1 if died is None else died - 1
        live_end = reclaim - 1 if kind == "temp" else n - 1 if kind == "output" else conservative_end
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


def _stage(info: Dict, nodes, previous: str, text: str, notes: Sequence[str] = (), liveness=None) -> Dict:
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
            "releases": sum(1 for line in after if "~=" in line["text"]),
        },
        "notes": list(notes),
        "liveness": liveness,
        "run": _measure(nodes),
    }


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


def run_pipeline(source: str, pass_ids: Sequence[str]) -> Dict:
    """Parse `source` and apply the chosen passes in compiler order, returning every stage."""
    try:
        nodes = parse(source, passes=())
    except UnexpectedInput as error:
        return {"error": _describe(error)}
    except Exception as error:
        return {"error": {"message": str(error)}}

    text = unparse(nodes)
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                lowered = info.apply(nodes)
            except Exception as error:
                lowered, failure = nodes, f"{type(error).__name__}: {error}"
            else:
                failure = None
        if failure:
            hint = [f"This pass expects {', '.join(info.requires)} to have run first."] if info.requires else []
            stage = _stage(details, nodes, text, text, hint)
            stage["error"] = failure
            stages.append(stage)
            break
        nodes, previous, text = lowered, text, unparse(lowered)
        stages.append(_stage(details, nodes, previous, text, [str(w.message) for w in caught], liveness))
    return {"stages": stages}


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
        except (ValueError, AttributeError):
            return self.json({"error": "bad request"}, 400)
        self.json(run_pipeline(source, passes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
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
