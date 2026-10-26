"""
A tiny JSON and static-file HTTP server for small local tools.

Each tool subclasses `JsonHandler`, sets `static_dir` and fills in three tables: the pages and scripts it serves, the JSON it returns for a
GET, and what it does with a JSON POST. A POST handler takes the request as a dict, returns a JSON-able value, and
raises `BadRequest` for a field it cannot read.

Requests are handled one at a time on the main thread. A server thread per request crashes the whole process: the
Qiskit wheel's native allocator segfaults when a thread starts after an earlier thread has used Qiskit and finished
(reproduced on Python 3.14 with Qiskit 2.5.2), which is exactly one request after another.
"""
import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Type

MAX_BODY_BYTES = 200_000
_CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}
SHARED_STATIC = Path(__file__).parent / "static"  # pages and scripts every experiment's app can use, like circuit.js


class BadRequest(Exception):
    """A request field is missing or the wrong type."""


def text_field(request: Dict[str, Any], key: str, default: str = "") -> str:
    return str(request.get(key, default))


def int_field(request: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(request.get(key, default))
    except (TypeError, ValueError):
        raise BadRequest(f"{key} must be a whole number") from None


class JsonHandler(BaseHTTPRequestHandler):
    static_dir: Path = Path(".")  # where the experiment's own pages are served from; anything missing falls back to SHARED_STATIC
    pages: Dict[str, str] = {}  # URL path -> file in static_dir or SHARED_STATIC
    get_routes: Dict[str, Callable[[], Any]] = {}  # URL path -> the JSON to return
    post_routes: Dict[str, Callable[[Dict[str, Any]], Any]] = {}  # URL path -> what to do with the request

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (the name is the base class's)
        """Stay quiet: the apps are run in a terminal the user is also using."""

    def reply(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json(self, payload: Any, status: int = 200) -> None:
        self.reply(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:
        if self.path in self.pages:
            filename = self.pages[self.path]
            path = self.static_dir / filename
            if not path.is_file():
                path = SHARED_STATIC / filename
            self.reply(200, path.read_bytes(), _CONTENT_TYPES[Path(filename).suffix])
        elif self.path in self.get_routes:
            self.json(self.get_routes[self.path]())
        else:
            self.json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        route = self.post_routes.get(self.path)
        if route is None:
            return self.json({"error": "not found"}, 404)
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            return self.json({"error": "request too large"}, 413)
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(request, dict):
                raise BadRequest("the request must be a JSON object")
            self.json(route(request))
        except (ValueError, BadRequest) as error:
            self.json({"error": f"bad request: {error}"}, 400)


def make_server(handler: Type[JsonHandler], port: int) -> HTTPServer:
    return HTTPServer(("127.0.0.1", port), handler)


def serve(handler: Type[JsonHandler], *, name: str, description: str, default_port: int, argv: Optional[Sequence[str]] = None) -> None:
    """Parse the command line, start the server, open a browser, and run until interrupted."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = parser.parse_args(argv)
    server = make_server(handler, args.port)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{name} running at {url} (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
