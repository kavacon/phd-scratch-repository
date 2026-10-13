"""
The explorer's stages as blocks for the shared page: the diff or the liveness chart, the result cards, the allocation
timeline, and the circuit's counts and drawing. It describes what `explorer.run_steps` produced; nothing here runs the
compiler.
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from checking import qiskit_tools
from checking.simulate import execute
from qiskit import QuantumCircuit

from demo.blocks import change
from demo.drawing import diagram
from demo.model import (
    Action, Card, Code, Columns, Diagram, Diff, Field, Frame, Grid, GridCell, GridColumn, GridRow, LegendItem, Line,
    Metrics, Note, Option, Stage, Switcher, Timeline,
)
from demo.text import shorten
from compiler.system import System
from experiments.allocation_poc.views import allocation_view, liveness_view


@dataclass(frozen=True)
class Ran:
    """One stage of a run: what the step produced, printed, and whether (and why) it failed."""
    id: str
    output: Any
    text: str
    failure: Optional[str] = None
    warnings: Tuple[str, ...] = ()
    expects: Tuple[str, ...] = ()  # the steps it expects first, when it failed with nothing to show


@dataclass(frozen=True)
class Info:
    id: str
    name: str
    summary: str = ""
    requires: Tuple[str, ...] = ()


# --- liveness --------------------------------------------------------------------------------------------------------

_LIVENESS_LEGEND = (
    LegendItem("defined", mark="●"), LegendItem("live", "temp"), LegendItem("released here, no longer live", mark="×"),
    LegendItem("returned", "output"), LegendItem("caller-owned parameter", "param"), LegendItem("kept for the release of an if", "kept"),
)


def _describe(column: Dict) -> str:
    rows = lambda n: f"row {n + 1}"  # noqa: E731
    name = column["name"]
    if column["kind"] == "param":
        return f"{name}: a parameter, owned by the caller, so it is never released here."
    if column["kind"] == "output":
        return f"{name}: returned (or passed out of the block), so it is never released here."
    if column["kind"] == "kept":
        return f"{name}: kept until the result of the if is released, because that release needs it."
    return f"{name}: defined at {rows(column['defined'])}, released after {rows(column['release_row'])}."


def _liveness_grids(view: Dict) -> List[Grid]:
    grids = []
    for scope in (s for s in view["scopes"] if s["columns"]):
        columns = scope["columns"]
        rows = tuple(
            GridRow(text, tuple(
                GridCell(
                    "live" if c["start"] <= r <= c["live_end"] else "",
                    ("●" if c["defined"] == r else "") + ("×" if c["release_row"] == r else ""),
                    _describe(c),
                ) for c in columns))
            for r, text in enumerate(scope["rows"])
        )
        grids.append(Grid(
            scope["title"], tuple(GridColumn(c["name"], c["kind"], c["name"], _describe(c)) for c in columns), rows,
            _LIVENESS_LEGEND if not grids else (), scope["depth"],
        ))
    return grids


_RELEASE = re.compile(r"(\S+)\s+~=")


def _release_key(text: str) -> Optional[str]:
    match = _RELEASE.search(text)
    return match.group(1) if match else None


# --- allocation ------------------------------------------------------------------------------------------------------

def _timeline(view: Dict, source: str) -> Timeline:
    steps = view["steps"]
    cells = max(view["slots"], view["needed"])
    panes = (
        Code(tuple(Line(text or " ") for text in source.split("\n")), "Source · where the registers came from",
             "tinted: a register for that variable is held"),
        Code(tuple(
            Line(step["physical"].replace("\n", " ") + ("" if step["logical"] == step["physical"] else "   ← " + step["logical"].replace("\n", " ")))
            for step in steps), "Instructions", "physical ← logical"),
    )
    frames = []
    for k, step in enumerate(steps):
        by_slot = {entry["slot"]: entry for entry in step["held"]}
        cards = []
        for i in range(cells):
            entry = by_slot.get(i)
            over = i >= view["slots"]
            origin = ""
            if entry:
                origin = ("" if entry["function"] == "main" else f"from {entry['function']} · ") + (
                    f"variable {entry['variable']}" if entry["variable"] else "compiler temporary")
            tone = "bad" if over else "warn" if step["changed"] == i else "accent" if entry else "muted"
            cards.append(Card(f"R{i}" + (" · over budget" if over else ""), entry["register"] if entry else "free", origin, tone))
        cards.append(Card("scratch · reserved", f"{view['scratch']} qubit{'' if view['scratch'] == 1 else 's'}", "shared by every operation", "muted"))
        marks = (
            tuple([(n - 1, "has") for entry in step["held"] for n in entry["lines"]] + [(n - 1, "now") for n in step["focus"]["lines"]]),
            ((k, "now"),),
        )
        now = Code((Line(step["physical"].replace("\n", " ")), Line("logical: " + step["logical"].replace("\n", " "))))
        frames.append(Frame(f"step {k + 1} of {len(steps)}", (Metrics(tuple(cards)), now), marks))
    return Timeline("Step through the program", tuple(frames), panes, "This program has no instructions.")


def _allocation_blocks(logical, physical, system: System, source: str) -> List[Any]:
    view = allocation_view(logical, physical, system, source)
    plural = "" if view["slots"] == 1 else "s"
    facts = Metrics((
        Card("Slots available", str(view["slots"]),
             f"{view['qubits']} qubits ÷ {view['register_width']}, less a scratch register"),
        Card("Slots needed", str(view["needed"]), "fits" if view["fits"] else "too many", "good" if view["fits"] else "bad"),
        Card("Qubits used", f"{view['qubits_used']} of {view['qubits']}", f"including {view['scratch']} for scratch"),
    ))
    return [facts, _timeline(view, source)]


# --- circuits --------------------------------------------------------------------------------------------------------

def is_circuit(output: Any) -> bool:
    return isinstance(output, QuantumCircuit)


def circuit_metrics(output: Any) -> Optional[Dict]:
    """Gate and wire counts, from Qiskit's own counters, so every circuit stage is counted the same way."""
    return qiskit_tools.measure(output) if is_circuit(output) else None


def _circuit_blocks(ran: Sequence[Ran], index: int, info: Dict[str, Info]) -> List[Any]:
    stage = ran[index]
    name = info[stage.id].name
    shown = shorten(stage.text)
    picture = diagram(stage.output)
    cut = (f"first {picture['layers']} layers, {picture['shown']:,} of {picture['total']:,} gates" if picture["shown"] < picture["total"]
           else f"{picture['total']:,} gates, {picture['layers']} layers")
    blocks: List[Any] = [Switcher("circuit-view", (
        Option("text", "OpenQASM text", (Code.of(shown, name, f"{len(shown.splitlines()):,} lines shown"),)),
        Option("diagram", "Circuit diagram", (Diagram(picture, name, cut),)),
    ))]
    metrics = circuit_metrics(stage.output)
    if metrics:
        earlier = next((r for r in reversed(ran[:index]) if is_circuit(r.output)), None)
        before = circuit_metrics(earlier.output) if earlier else None
        since = info[earlier.id].name if earlier else ""

        def card(label, key, unit, default):
            return Card(label, f"{metrics[key]:,}", change(metrics[key], before and before[key], unit, since) or default)

        blocks.append(Metrics((
            card("Gates", "gates", "gates", "every gate counts as one, however many controls"),
            card("Two-qubit gates", "two_qubit", "gates", "gates on two or more qubits"),
            card("Depth", "depth", "layers", "layers of gates that cannot overlap"),
            Card("Wires used", str(metrics["touched"]), f"{metrics['wires']} declared"
                 + (f", {change(metrics['touched'], before['touched'], 'wires', since)}" if before else "")),
        ), 4))
        mix = " · ".join(f"{op} {n:,}" for op, n in sorted(metrics["ops"].items(), key=lambda item: -item[1]))
        blocks.append(Note(f"Gate mix: {mix}"))
    blocks.append(Action("Run on simulator", "simulate", {"stage": stage.id}, (Field("shots", "shots", 100, 1, 4096),),
                         "Qiskit Aer, matrix product state. Runs the circuit as shown and measures the returned register."))
    return blocks


# --- stages ----------------------------------------------------------------------------------------------------------

def _outcome(output: Any) -> Dict[str, Any]:
    """How a stage's output runs on its reference machine, for the result cards."""
    try:
        result = execute(output)
    except RecursionError:
        return {"error": "recursion went too deep"}
    except Exception as error:  # a user program can fail in many ways; report it instead of crashing
        return {"error": str(error) or type(error).__name__}
    if result is None:
        return {"not_run": True}
    leaked = sorted(name if scope == "<program>" else f"{scope}: {name}" for scope, names in result.leaks.items() for name in names)
    value = result.value
    return {"value": "true" if value is True else "false" if value is False else str(value), "leaked": leaked,
            "peak": max(result.peak.values(), default=0)}


def _result_cards(ran: Sequence[Ran], index: int) -> Any:
    outcome, first = _outcome(ran[index].output), _outcome(ran[0].output)
    if "error" in outcome:
        return Note(f"Could not run this program here: {outcome['error']}", "bad")
    previous = _outcome(ran[index - 1].output) if index else {}
    before = previous.get("peak")
    same = "error" in first or outcome.get("value") == first.get("value")
    cards = []
    if outcome.get("not_run"):
        cards.append(Card("Result", "not run", "circuits are only run on request: use the simulator below"))
    else:
        cards.append(Card("Result", outcome["value"], "what the program returns" if index == 0 else "same as the source" if same
                          else "differs from the source!", "" if same else "bad"))
        left = len(outcome["leaked"])
        cards.append(Card("Registers not released", str(left), ", ".join(outcome["leaked"]) if left else "nothing left over when it returns",
                          "" if left else "good"))
    if outcome.get("peak") is not None:
        cards.append(Card("Peak registers", str(outcome["peak"]),
                          change(outcome["peak"], before, "register(s)") or "most held at once (a stand-in for circuit width)"))
    return Metrics(tuple(cards), 3)


def stage_for(ran: Sequence[Ran], index: int, source: str, info: Dict[str, Info], system: System) -> Stage:
    stage = ran[index]
    here = info[stage.id]
    previous = ran[index - 1] if index else None
    blocks: List[Any] = []
    if index == 0:
        blocks.append(Code.of(stage.text, "Source"))
    elif is_circuit(stage.output):
        blocks += _circuit_blocks(ran, index, info)
    else:
        comparison = Diff.of(previous.text, stage.text, f"Before · {info[previous.id].name}", f"After · {here.name}", _release_key)
        grids: List[Grid] = []
        if stage.id == "lower_allocation" and "rename" in [r.id for r in ran[:index]]:
            try:
                grids = _liveness_grids(liveness_view(previous.output))
            except Exception:  # the analysis is only for display; without it the stage just has no chart
                grids = []
        if grids:
            blocks.append(Switcher("compare", (
                Option("diff", "Before · After", (comparison,)),
                Option("liveness", "Liveness · After", (Columns((tuple(grids), (comparison.after,))),)),
            )))
        else:
            blocks.append(comparison)
        if stage.id == "allocate":
            blocks += _allocation_blocks(previous.output, stage.output, system, source)
    if stage.failure:
        blocks.append(Note(stage.failure, "bad"))
    if stage.expects:
        blocks.append(Note(f"This pass expects {', '.join(stage.expects)} to have run first."))
    blocks += [Note(text) for text in stage.warnings]
    blocks.append(_result_cards(ran, index))
    return Stage(stage.id, here.name, here.summary, tuple(blocks), failed=bool(stage.failure))
