"""
The data model every demo page is built from.

A demo's `run` returns a `Result`: stages, each holding a list of blocks. The page (`static/app.js`) knows how to
draw each block type and nothing else, so a demo never ships HTML or JavaScript: it describes what to show in these
terms, and a new kind of display is added here and in app.js for every demo to use.

    code      numbered lines, each with a kind (same, added, removed, changed, or a mark a timeline sets)
    diff      two code panes, before and after, with the changed lines and the changed words marked
    metrics   cards: a label, a value, a sub-line and a tone
    stats     a table whose cells can carry text and bars
    switcher  one of several panels, chosen by the viewer; the choice is remembered by name
    grid      rows by columns of cells with a state, for things like liveness
    timeline  frames to step through, with fixed panes whose lines are marked per frame
    note      a message, with an optional list
    action    a button that asks the server for more blocks (a named action declared by the demo)
    diagram   a circuit drawing (data from demo.drawing)
    columns   blocks side by side

An element with a `key` highlights every other element with the same key while it is hovered.
"""
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, ClassVar, Dict, Mapping, Optional, Sequence, Tuple, Union

from demo.text_diff import diff_lines

# tones a block may use: "", "good", "bad", "muted", "accent", "warn", "series-1", "series-2"


@dataclass(frozen=True)
class Line:
    text: str
    kind: str = "same"
    spans: Tuple[Tuple[int, int], ...] = ()  # character ranges to emphasise within the line
    key: Optional[str] = None


@dataclass(frozen=True)
class Code:
    type: ClassVar[str] = "code"
    lines: Tuple[Line, ...]
    title: str = ""
    meta: str = ""
    empty: str = ""  # shown when there are no lines

    @classmethod
    def of(cls, text: str, title: str = "", meta: str = "", empty: str = "") -> "Code":
        return cls(tuple(Line(t) for t in text.splitlines()), title, meta, empty)


@dataclass(frozen=True)
class Diff:
    type: ClassVar[str] = "diff"
    before: Optional[Code]
    after: Code

    @classmethod
    def of(cls, old: Optional[str], new: str, before_title: str = "Before", after_title: str = "After",
           key_of: Optional[Callable[[str], Optional[str]]] = None) -> "Diff":
        """
        The comparison of two texts. With no `old` there is nothing to compare, and only the new text is shown.
        `key_of` gives a line a key (see the module notes) from its text.
        """
        def keyed(entries, title, meta="") -> Code:
            lines = tuple(
                Line(e["text"], e["kind"], tuple(tuple(span) for span in e["spans"]), key_of(e["text"]) if key_of else None)
                for e in entries
            )
            return Code(lines, title, meta)

        if old is None:
            return cls(None, keyed([{"text": t, "kind": "same", "spans": []} for t in new.splitlines()], after_title))
        before, after = diff_lines(old.splitlines(), new.splitlines())
        counts = [(sum(1 for e in after if e["kind"] == kind), kind) for kind in ("added", "changed")]
        removed = sum(1 for e in before if e["kind"] == "removed")
        parts = [f"{n} {kind}" for n, kind in counts if n] + ([f"{removed} removed"] if removed else [])
        return cls(keyed(before, before_title), keyed(after, after_title, " · ".join(parts) or "no changes"))


@dataclass(frozen=True)
class Card:
    label: str
    value: str
    sub: str = ""
    tone: str = ""
    key: Optional[str] = None


@dataclass(frozen=True)
class Metrics:
    type: ClassVar[str] = "metrics"
    cards: Tuple[Card, ...]
    columns: int = 0  # 0 lets the page choose


@dataclass(frozen=True)
class Bar:
    fraction: float  # 0 to 1
    tone: str = ""


@dataclass(frozen=True)
class Cell:
    text: str = ""
    note: str = ""
    tone: str = ""
    bars: Tuple[Bar, ...] = ()


@dataclass(frozen=True)
class Row:
    label: str
    cells: Tuple[Cell, ...]
    note: str = ""


@dataclass(frozen=True)
class LegendItem:
    label: str
    tone: str = ""  # a swatch in this tone
    mark: str = ""  # or a literal mark


@dataclass(frozen=True)
class Stats:
    type: ClassVar[str] = "stats"
    headers: Tuple[str, ...]  # one per column, the first for the row labels
    rows: Tuple[Row, ...]
    legend: Tuple[LegendItem, ...] = ()


@dataclass(frozen=True)
class Option:
    id: str
    label: str
    blocks: Tuple["Block", ...]
    disabled: str = ""  # why it cannot be chosen, if it cannot


@dataclass(frozen=True)
class Switcher:
    type: ClassVar[str] = "switcher"
    name: str  # switchers with the same name share the viewer's choice
    options: Tuple[Option, ...]


@dataclass(frozen=True)
class GridColumn:
    label: str
    kind: str = ""  # colours the column's live cells
    key: Optional[str] = None
    tip: str = ""


@dataclass(frozen=True)
class GridCell:
    state: str = ""  # "live" fills the cell in its column's colour
    mark: str = ""
    tip: str = ""


@dataclass(frozen=True)
class GridRow:
    label: str
    cells: Tuple[GridCell, ...]


@dataclass(frozen=True)
class Grid:
    type: ClassVar[str] = "grid"
    title: str
    columns: Tuple[GridColumn, ...]
    rows: Tuple[GridRow, ...]
    legend: Tuple[LegendItem, ...] = ()
    indent: int = 0
    empty: str = ""


@dataclass(frozen=True)
class Frame:
    label: str
    blocks: Tuple["Block", ...] = ()
    marks: Tuple[Tuple[Tuple[int, str], ...], ...] = ()  # per pane: (line index, kind) pairs


@dataclass(frozen=True)
class Timeline:
    type: ClassVar[str] = "timeline"
    title: str
    frames: Tuple[Frame, ...]
    panes: Tuple[Code, ...] = ()  # drawn once; each frame marks lines in them
    empty: str = ""


@dataclass(frozen=True)
class Note:
    type: ClassVar[str] = "note"
    text: str
    tone: str = ""
    items: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Field:
    name: str
    label: str
    default: int
    low: int = 1
    high: int = 4096


@dataclass(frozen=True)
class Action:
    type: ClassVar[str] = "action"
    label: str
    action: str  # a name in the demo's actions; the page gets its url from the bootstrap
    payload: Mapping[str, Any]
    fields: Tuple[Field, ...] = ()  # numbers the viewer sets, sent with the payload
    hint: str = ""


@dataclass(frozen=True)
class Diagram:
    type: ClassVar[str] = "diagram"
    picture: Mapping[str, Any]
    title: str = ""
    meta: str = ""


@dataclass(frozen=True)
class Columns:
    type: ClassVar[str] = "columns"
    children: Tuple[Tuple["Block", ...], ...]


Block = Union[Code, Diff, Metrics, Stats, Switcher, Grid, Timeline, Note, Action, Diagram, Columns]


@dataclass(frozen=True)
class Stage:
    id: str
    name: str
    summary: str = ""
    blocks: Tuple[Block, ...] = ()
    failed: bool = False


@dataclass(frozen=True)
class SourceError:
    message: str
    line: Optional[int] = None
    column: Optional[int] = None


@dataclass(frozen=True)
class Result:
    stages: Tuple[Stage, ...] = ()
    error: Optional[SourceError] = None  # the input could not be read at all


# --- inputs and the demo description ----------------------------------------------------------------------------

_THEME_VARIABLES = {
    "background": "--bg", "panel": "--panel", "text": "--text", "muted": "--muted", "border": "--border",
    "accent": "--accent", "accent_soft": "--accent-soft",
    "added": "--add", "added_strong": "--add-strong", "changed": "--chg", "changed_strong": "--chg-strong",
    "removed": "--rem", "removed_strong": "--rem-strong", "good": "--ok", "bad": "--bad",
    "series_1": "--s1", "series_2": "--s2",
}
_SURFACES = ("background", "panel", "text", "muted", "border")
_FORBIDDEN = set(";{}<>\n\\")


@dataclass(frozen=True)
class Theme:
    """
    The colours of one demo's page; anything left out keeps the shared look. Values are CSS colours ("#0f766e",
    "rgb(15 118 110)", "teal").

    The brand colours (accent, the diff colours, good and bad, the two series) apply in both the light and the dark
    scheme. The surface colours (background, panel, text, muted, border) apply in the light scheme only, because a
    light background needs dark text; give `dark=Theme(...)` to set them for the dark scheme, or to use other
    brand colours there.
    """
    background: Optional[str] = None
    panel: Optional[str] = None
    text: Optional[str] = None
    muted: Optional[str] = None
    border: Optional[str] = None
    accent: Optional[str] = None  # the selected stage, buttons, links between elements
    accent_soft: Optional[str] = None  # the tint behind selected things
    added: Optional[str] = None  # a diff's added lines, and its stronger mark within a line
    added_strong: Optional[str] = None
    changed: Optional[str] = None
    changed_strong: Optional[str] = None
    removed: Optional[str] = None
    removed_strong: Optional[str] = None
    good: Optional[str] = None
    bad: Optional[str] = None
    series_1: Optional[str] = None  # the first and second thing a chart or table compares
    series_2: Optional[str] = None
    dark: Optional["Theme"] = None

    def __post_init__(self) -> None:
        for name in _THEME_VARIABLES:
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip() or _FORBIDDEN & set(value)):
                raise ValueError(f"{name} is not a CSS colour: {value!r}")
        if self.dark is not None and self.dark.dark is not None:
            raise ValueError("a dark theme has no dark theme of its own")

    def css(self) -> Dict[str, Dict[str, str]]:
        """The CSS variables to set in each scheme, for the page."""
        own = {var: getattr(self, name) for name, var in _THEME_VARIABLES.items() if getattr(self, name) is not None}
        brand = {var: getattr(self, name) for name, var in _THEME_VARIABLES.items()
                 if name not in _SURFACES and getattr(self, name) is not None}
        dark = dict(brand)
        if self.dark is not None:
            dark.update(self.dark.css()["light"])
        return {"light": own, "dark": dark}


@dataclass(frozen=True)
class Text:
    id: str
    label: str
    rows: int = 14
    hint: str = ""
    default: str = ""


@dataclass(frozen=True)
class Number:
    id: str
    label: str
    default: int
    low: int = 0
    high: int = 4096
    hint: str = ""


@dataclass(frozen=True)
class Case:
    """A preloaded input set. Choosing it sets the demo's inputs to `values`; its id is sent as the input `case`."""
    id: str
    name: str
    description: str = ""
    values: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", dict(self.values or {}))


@dataclass(frozen=True)
class Step:
    """A stage of the workflow the viewer can switch on and off."""
    id: str
    name: str
    summary: str = ""
    requires: Tuple[str, ...] = ()
    enabled: bool = True


def to_data(value: Any) -> Any:
    """A model value as JSON-able data. Blocks carry their `type`."""
    if is_dataclass(value) and not isinstance(value, type):
        data: Dict[str, Any] = {f.name: to_data(getattr(value, f.name)) for f in fields(value)}
        kind = getattr(type(value), "type", None)
        if kind:
            data["type"] = kind
        return data
    if isinstance(value, Mapping):
        return {str(k): to_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_data(v) for v in value]
    return value


def walk(blocks: Sequence[Any]):
    """Every block in a tree, for checks and tests."""
    for block in blocks:
        yield block
        if isinstance(block, Switcher):
            for option in block.options:
                yield from walk(option.blocks)
        elif isinstance(block, Columns):
            for child in block.children:
                yield from walk(child)
        elif isinstance(block, Timeline):
            yield from walk(block.panes)
            for frame in block.frames:
                yield from walk(frame.blocks)
