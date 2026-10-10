"""
Comparison app: our circuits next to Qrisp's for the same task.

Each demo program goes through our compiler as usual (inline, allocate, generate OpenQASM). The same task, written by
hand as an ordinary Qrisp program using only Qrisp's own operators (qrisp_programs.py), goes through Qrisp's own compiler. Both are measured the same way, as built
and after the same Qiskit optimisation into `u` and `cx`, with gate and wire counts and a drawing, and either can be run on
Qiskit Aer on request (nothing runs on its own).

Run it from the python directory:

    PYTHONPATH=src .venv/bin/python -m allocation_poc.compare

It needs the optional dependencies: pip install '.[circuits,compare]'
"""
import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict

from allocation_poc import circuits
from allocation_poc.demo import MAX_CIRCUIT_LINES
from allocation_poc.programs import PROGRAMS
from allocation_poc.simulate import simulate_isa
from parser.parser import parse
from passes.allocate import allocate
from passes.isa import lower_to_isa
from passes.qasm import to_openqasm

STATIC = Path(__file__).parent / "static"
MAX_WIDTH = 16
_cache: Dict[Any, Dict[str, Any]] = {}  # built circuits by (program, width), so running one does not rebuild everything


def clean_width(raw) -> int:
    try:
        return min(max(int(raw), 1), MAX_WIDTH)
    except (TypeError, ValueError):
        return 6


def _text(circuit, limit: int = MAX_CIRCUIT_LINES) -> str:
    try:
        lines = circuits.write(circuit).splitlines()
    except Exception as error:  # a circuit with gates OpenQASM cannot name
        return f"// this circuit cannot be written as OpenQASM: {error}"
    return "\n".join(lines[:limit] + ([f"// ... {len(lines) - limit} more lines"] if len(lines) > limit else []))


def _program(program_id: str) -> Dict[str, Any]:
    for program in PROGRAMS:
        if program["id"] == program_id:
            return program
    raise KeyError(f"unknown program {program_id}")


def build(program_id: str, width: int) -> Dict[str, Any]:
    """Both circuits for a program, as built and optimised."""
    key = (program_id, width)
    if key in _cache:
        return _cache[key]
    from allocation_poc import qrisp_programs

    program = _program(program_id)
    isa = lower_to_isa(parse(program["source"]))
    expected = int(simulate_isa(isa).value) & ((1 << width) - 1)
    ours = to_openqasm(allocate(isa, width, 100_000))
    ours_built = circuits.read(ours.text)
    qrisp = qrisp_programs.run(program_id, width)
    built = {
        "ours": {"built": ours_built, "optimised": circuits.optimise(ours_built), "result_bits": len(ours.measured), "exponent": 0},
        "qrisp": {"built": qrisp.circuit, "optimised": circuits.optimise(qrisp.circuit), "result_bits": len(qrisp.result), "exponent": qrisp.exponent},
        "expected": expected,
        "ours_circuit": ours,
        "qrisp_circuit": qrisp,
    }
    if len(_cache) >= 6:
        _cache.pop(next(iter(_cache)))
    _cache[key] = built
    return built


def compare(program_id: str, width: int) -> Dict[str, Any]:
    """Everything the page shows for one program: both sources, and counts, text and drawing for each of the four circuits."""
    from allocation_poc import qrisp_programs
    try:
        built = build(program_id, width)
    except Exception as error:
        return {"error": {"message": f"{type(error).__name__}: {error}"}}
    ours, qrisp = built["ours_circuit"], built["qrisp_circuit"]
    drawings = {
        ("ours", "built"): circuits.diagram(ours),
        ("ours", "optimised"): circuits.diagram(ours, built["ours"]["optimised"]),
        ("qrisp", "built"): circuits.qiskit_diagram(qrisp.circuit, qrisp.labels, qrisp.groups),
        ("qrisp", "optimised"): circuits.qiskit_diagram(built["qrisp"]["optimised"], qrisp.labels, qrisp.groups),
    }
    result: Dict[str, Any] = {
        "width": width,
        "expected": built["expected"],
        "sources": {"ours": _program(program_id)["source"], "qrisp": qrisp_programs.source_text(program_id, width)},
        "circuits": {},
    }
    for side in ("ours", "qrisp"):
        result["circuits"][side] = {}
        for variant in ("built", "optimised"):
            circuit = built[side][variant]
            text = ours.text if (side, variant) == ("ours", "built") else _text(circuit)
            lines = text.splitlines()
            if len(lines) > MAX_CIRCUIT_LINES:
                text = "\n".join(lines[:MAX_CIRCUIT_LINES] + [f"// ... {len(lines) - MAX_CIRCUIT_LINES} more lines"])
            result["circuits"][side][variant] = {
                "metrics": circuits.measure(circuit),
                "text": text,
                "diagram": drawings[(side, variant)],
            }
    return result


def simulate(program_id: str, width: int, side: str, variant: str, shots: int) -> Dict[str, Any]:
    """Run one of the four circuits on Qiskit Aer and report the measurement against the expected value."""
    if side not in ("ours", "qrisp") or variant not in ("built", "optimised"):
        return {"error": "unknown circuit"}
    if not circuits.aer_available():
        return {"error": "Running a circuit needs Qiskit Aer: pip install qiskit-aer"}
    try:
        built = build(program_id, width)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}
    shots = min(max(int(shots), 1), 4096)
    entry = built[side]
    outcome = circuits.execute(entry[variant], entry["result_bits"], shots)
    scale = 2 ** entry["exponent"]  # a Qrisp QuantumFloat is a mantissa times a power of two
    counts = {value * scale: count for value, count in outcome["counts"].items()}
    return {
        "side": side, "variant": variant, "shots": shots,
        "counts": sorted(([value, count] for value, count in counts.items()), key=lambda item: -item[1]),
        "leaked": [f"q[{q}]" for q in outcome["leaked"]],
        "expected": built["expected"],
        "matches": len(counts) == 1 and next(iter(counts)) == built["expected"],
    }


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
            self.reply(200, (STATIC / "compare.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/circuit.js":
            self.reply(200, (STATIC / "circuit.js").read_bytes(), "application/javascript; charset=utf-8")
        elif self.path == "/api/programs":
            self.json([{"id": p["id"], "name": p["name"], "description": p["description"]} for p in PROGRAMS])
        else:
            self.json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path not in ("/api/compare", "/api/simulate"):
            return self.json({"error": "not found"}, 404)
        length = int(self.headers.get("Content-Length") or 0)
        if length > 200_000:
            return self.json({"error": "program too large"}, 413)
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
            program_id = str(request.get("program", ""))
            width = clean_width(request.get("width"))
            shots = int(request.get("shots", 100))
            side, variant = str(request.get("side", "")), str(request.get("variant", ""))
        except (ValueError, AttributeError, TypeError):
            return self.json({"error": "bad request"}, 400)
        if self.path == "/api/simulate":
            return self.json(simulate(program_id, width, side, variant, shots))
        self.json(compare(program_id, width))


def make_server(port: int) -> HTTPServer:
    """One request at a time on the main thread (a thread per request crashes the Qiskit native allocator)."""
    return HTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = parser.parse_args()
    server = make_server(args.port)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Qrisp comparison running at {url} (Ctrl+C to stop)")
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
