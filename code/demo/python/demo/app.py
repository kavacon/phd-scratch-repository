"""
The shared demo app. A demo describes itself with a `Demo` (inputs, cases, steps, a `run` function and optional
actions) and the demo package serves one generic page for it.

The page loads `/api/bootstrap` once: the title, the inputs and their defaults, the cases, the steps, and the url of
every action. It then posts the current inputs to `/api/run` and draws the `Result` it gets back (see
`demo.model` for what it can draw). An action block posts to its url and the page draws the blocks that come back.

    demo = Demo("My demo", "What it shows", run=run, inputs=[Number("width", "Width", 8, 1, 16)])
    if __name__ == "__main__":
        serve(demo)
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from demo import http_server
from demo.model import Case, Note, Number, Result, SourceError, Step, Theme, to_data

Run = Callable[[Dict[str, Any], List[str]], Result]
ActionFn = Callable[[Dict[str, Any], List[str], Dict[str, Any]], Sequence[Any]]


@dataclass
class Demo:
    title: str
    description: str
    run: Run  # (inputs, enabled step ids) -> Result
    inputs: Sequence[Any] = ()  # Text and Number
    cases: Sequence[Case] = ()
    steps: Sequence[Step] = ()
    workflow: bool = True  # stages are stepped through; False shows them all, one under another
    actions: Mapping[str, ActionFn] = field(default_factory=dict)  # (inputs, steps, payload) -> blocks
    cases_label: str = "Cases"
    steps_label: str = "Steps"
    steps_first: bool = False  # the steps panel goes above the results instead of below them
    max_steps: int = 0  # at most this many steps on at once (0 for any number); turning on another turns off the oldest
    theme: Optional[Theme] = None  # the demo's colours; the shared look when None
    port: int = 8000

    def clean_inputs(self, raw: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """The inputs from a request: every declared input, with defaults for anything missing or out of range."""
        raw = raw or {}
        case = next((c for c in self.cases if c.id == raw.get("case")), None)
        cleaned: Dict[str, Any] = {}
        for spec in self.inputs:
            value = raw.get(spec.id, case.values.get(spec.id, spec.default) if case else spec.default)
            if isinstance(spec, Number):
                try:
                    value = min(max(int(value), spec.low), spec.high)
                except (TypeError, ValueError):
                    value = spec.default
            else:
                value = value if isinstance(value, str) else spec.default
            cleaned[spec.id] = value
        cleaned["case"] = case.id if case else None
        return cleaned

    def clean_steps(self, raw: Any) -> List[str]:
        known = [step.id for step in self.steps]
        chosen = set(raw) if isinstance(raw, list) else set(known)
        steps = [step for step in known if step in chosen]  # in the demo's order
        return steps[:self.max_steps] if self.max_steps else steps

    def bootstrap(self) -> Dict[str, Any]:
        """What the page needs before anything runs."""
        return {
            "title": self.title,
            "description": self.description,
            "workflow": self.workflow,
            "cases_label": self.cases_label,
            "steps_label": self.steps_label,
            "steps_first": self.steps_first,
            "max_steps": self.max_steps,
            "theme": self.theme.css() if self.theme is not None else None,
            "inputs": [{"kind": "number" if isinstance(i, Number) else "text", **to_data(i)} for i in self.inputs],
            "cases": [to_data(c) for c in self.cases],
            "steps": [to_data(s) for s in self.steps],
            "urls": {"run": "/api/run", "actions": {name: f"/api/action/{name}" for name in self.actions}},
        }

    def run_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        inputs, steps = self.clean_inputs(request.get("inputs")), self.clean_steps(request.get("steps"))
        try:
            result = self.run(inputs, steps)
        except Exception as error:  # a demo can fail on a user's input in many ways; show it instead of crashing the app
            result = Result(error=SourceError(f"{type(error).__name__}: {error}"))
        return to_data(result)

    def action_request(self, name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        payload = request.get("payload")
        try:
            blocks = self.actions[name](
                self.clean_inputs(request.get("inputs")), self.clean_steps(request.get("steps")),
                payload if isinstance(payload, dict) else {},
            )
        except Exception as error:
            blocks = [Note(f"{type(error).__name__}: {error}", "bad")]
        return {"blocks": to_data(list(blocks))}


def make_handler(demo: Demo):
    class Handler(http_server.JsonHandler):
        static_dir = http_server.SHARED_STATIC
        pages = {"/": "app.html", "/index.html": "app.html", "/app.js": "app.js", "/app.css": "app.css", "/circuit.js": "circuit.js"}
        get_routes = {"/api/bootstrap": demo.bootstrap}
        post_routes = {
            "/api/run": demo.run_request,
            **{f"/api/action/{name}": (lambda request, name=name: demo.action_request(name, request)) for name in demo.actions},
        }
    return Handler


def make_server(demo: Demo, port: int):
    return http_server.make_server(make_handler(demo), port)


def serve(demo: Demo, argv: Optional[Sequence[str]] = None) -> None:
    http_server.serve(make_handler(demo), name=demo.title, description=demo.description, default_port=demo.port, argv=argv)
