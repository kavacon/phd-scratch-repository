"""
Data for the explorer's two inspection views, built from what the passes produce: the liveness analysis behind the
release pass, and which logical register held each physical slot at every step of an allocation, with the source lines
each register came from.
"""
import re
from typing import Dict, List, Optional, Sequence, Tuple

from allocation_poc.printer import unparse_instruction, unparse_statement
from parser.grammar import Assign, FunctionDef, If, Node
from parser.instructions import Alloc, Free, IOp, Op
from passes.allocate import Allocation
from passes.allocation import prepare
from passes.liveness import Liveness
from passes.names import results

PROGRAM_SCOPE = "main"  # the name given to code that is not inside a function


# --- liveness view --------------------------------------------------------------------------------------------

def _first_line(statement: Node) -> str:
    code = [text.strip() for text in unparse_statement(statement).splitlines() if not text.strip().startswith("//")]
    return code[0] + (" …" if len(code) > 1 else "")


def _defined_rows(body: Sequence[Node]) -> Dict[str, int]:
    """The row each version is defined in; an `if` defines the versions it produces."""
    rows: Dict[str, int] = {}
    for row, statement in enumerate(body):
        if isinstance(statement, Assign):
            rows[statement.name] = row
        elif isinstance(statement, If):
            rows.update(dict.fromkeys(sorted(results(statement)), row))
    return rows


def _release_positions(analysis: Liveness) -> Dict[str, int]:
    """Each released version's position among the versions released at the same point."""
    positions: Dict[str, int] = {}
    for name in analysis.release_order:
        point = analysis.reclaim_at[name]
        positions[name] = 1 + sum(1 for other in positions if analysis.reclaim_at[other] == point)
    return positions


def _column_kind(name: str, is_param: bool, analysis: Liveness) -> str:
    if is_param:
        return "param"
    if name in analysis.kept:
        return "kept"  # in a branch: needed to release what the if produced, so it outlives the branch
    return "output" if name in analysis.outputs else "temp"


def _column(name: str, rows: int, defined: Dict[str, int], params: Sequence[str], analysis: Liveness, positions: Dict[str, int]) -> Dict:
    is_param = name in params
    kind = _column_kind(name, is_param, analysis)
    reclaim = analysis.reclaim_at.get(name)
    died = analysis.dies_at.get(name)
    conservative_end = rows - 1 if died is None else died - 1
    if kind == "temp":
        live_end = reclaim - 1
    elif kind in ("output", "kept"):
        live_end = rows - 1
    else:
        live_end = conservative_end
    return {
        "name": name,
        "kind": kind,
        "defined": None if is_param else defined[name],
        "start": 0 if is_param else defined[name],
        "live_end": live_end,
        "conservative_end": max(conservative_end, live_end),
        "release_row": reclaim - 1 if kind == "temp" else None,
        "order": positions.get(name) if kind == "temp" else None,
    }


def _scope(title: str, body: Sequence[Node], analysis: Liveness, params: Sequence[str], depth: int, out: List[Dict]) -> None:
    """One block as rows (statements) and columns (versions), with where each version is live and released."""
    defined = _defined_rows(body)
    positions = _release_positions(analysis)
    names = [*params, *sorted((name for name in defined if name not in params), key=lambda name: (defined[name], name))]
    out.append({
        "title": title,
        "depth": depth,
        "rows": [_first_line(statement) for statement in body],
        "columns": [_column(name, len(body), defined, params, analysis, positions) for name in names],
    })
    for row, statement in enumerate(body):
        if isinstance(statement, If):  # a function is its own scope, added by the caller
            branches = zip(("then", "else"), (statement.body, statement.orelse), analysis.nested[row], strict=True)
            for branch, block, child in branches:
                _scope(f"{branch} branch of row {row + 1}", block, child, (), depth + 1, out)


def liveness_view(nodes: Sequence[Node]) -> Dict:
    """The liveness analysis behind the release pass, as scopes of rows and columns for display."""
    named, analyses = prepare(nodes)
    scopes: List[Dict] = []
    _scope("Program", named, analyses["<program>"], (), 0, scopes)
    for statement in named:
        if isinstance(statement, FunctionDef):
            _scope(f"function {statement.name}", statement.body, analyses[statement.name], [p.name for p in statement.params], 0, scopes)
    return {"scopes": scopes}


# --- allocation view ------------------------------------------------------------------------------------------

class _SourceIndex:
    """Where each variable is assigned in the source, by function, so a register can point back at its lines."""

    _FUNCTION = re.compile(r"^\s*function\s+(\w+)\s*\(")

    def __init__(self, source: str) -> None:
        self.lines = source.splitlines()
        self.scope_of = self._scopes()
        self._cache: Dict[Tuple[str, str], List[int]] = {}

    def _scopes(self) -> List[str]:
        """The function each source line is in, or the program's own scope."""
        scopes: List[str] = []
        scope, depth = PROGRAM_SCOPE, 0
        for text in self.lines:
            code = text.split("//")[0]
            match = self._FUNCTION.match(code)
            if match and depth == 0:
                scope = match.group(1)
            scopes.append(scope)
            depth += code.count("{") - code.count("}")
            if depth <= 0:
                depth, scope = 0, PROGRAM_SCOPE
        return scopes

    def lines_for(self, scope: str, variable: str) -> List[int]:
        key = (scope.split("@")[0], variable)
        if key not in self._cache:
            name = re.escape(variable)
            pattern = re.compile(rf"\b{name}\s*~?=(?!=)|\b{name}\s*:|\bfor\s+{name}\b")
            self._cache[key] = [
                number for number, text in enumerate(self.lines, 1)
                if self.scope_of[number - 1] == key[0] and pattern.search(text.split("//")[0])
            ]
        return self._cache[key]


def source_origin(register: str) -> Tuple[Optional[str], str]:
    """
    Where a logical register came from in the source, as (variable, function): x#2 is (x, main), area1.p#1 is
    (p, area) because it was inlined from area, and h1.g1.m#1 is (m, g). Loop copies (t@1) are the variable t.
    Compiler temporaries ($t1#1) have no variable.
    """
    base = register.split("#")[0]
    if base.startswith("$"):
        return None, PROGRAM_SCOPE
    parts = base.split(".")
    function = re.sub(r"\d+$", "", parts[-2]) if len(parts) > 1 else PROGRAM_SCOPE
    return parts[-1].split("@")[0], function


def _acted_on(instruction) -> str:
    """The register an instruction allocates, frees or writes, or an empty string for a control marker."""
    if isinstance(instruction, (Alloc, Free)):
        return instruction.register
    if isinstance(instruction, (Op, IOp)):
        return instruction.dst
    return ""


def allocation_view(allocation: Allocation, source: str) -> Dict:
    """The allocation as data for the viewer: the machine, and for every step what each slot held."""
    index = _SourceIndex(source)

    def label(register: str) -> Dict:
        variable, function = source_origin(register) if register else (None, PROGRAM_SCOPE)
        return {
            "register": register,
            "function": function,
            "variable": variable,
            "lines": index.lines_for(function, variable) if variable else [],
        }

    steps = [{
        "logical": unparse_instruction(step.logical),
        "physical": unparse_instruction(step.physical),
        "held": [{"slot": held.slot, **label(held.register)} for held in step.held],
        "changed": step.changed,
        "focus": label(_acted_on(step.logical)),
    } for step in allocation.trace]
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
