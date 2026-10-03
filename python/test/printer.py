"""Renders an AST back to readable source-like text, for inspecting what passes produce."""
from typing import List, Sequence

from parser.grammar import (
    Assign, BinOp, Boolean, Call, Equals, ExprStmt, ForLoop, FunctionDef, If, Node, Not,
    Number, Phi, Return, Unassign, Variable, WhileLoop,
)


def unparse(program: Sequence[Node]) -> str:
    return "\n".join(_block(program, 0))


def _expr(node: Node, nested: bool = False) -> str:
    if isinstance(node, (Number, Variable)):
        return str(node.value if isinstance(node, Number) else node.name)
    if isinstance(node, Boolean):
        return "true" if node.value else "false"
    if isinstance(node, Call):
        return f"{node.name}({', '.join(_expr(a) for a in node.args)})"
    if isinstance(node, Phi):
        return f"phi({', '.join(_expr(o) for o in node.operands)})"
    if isinstance(node, Not):
        return f"!{_expr(node.operand, True)}"
    text = f"{_expr(node.left, True)} {node.op} {_expr(node.right, True)}" if isinstance(node, BinOp) \
        else " == ".join(_expr(o, True) for o in node.operands)
    return f"({text})" if nested else text


def _block(body: Sequence[Node], depth: int) -> List[str]:
    return [line for statement in body for line in _stmt(statement, depth)]


def _phis(label: str, assigns: Sequence[Node], pad: str) -> List[str]:
    return [f"{pad}// {label}: {a.name} = {_expr(a.value)}" for a in assigns]


def _stmt(node: Node, depth: int) -> List[str]:
    pad = "    " * depth
    if isinstance(node, Assign):
        return [f"{pad}{node.name} = {_expr(node.value)};"]
    if isinstance(node, ExprStmt):
        return [f"{pad}{_expr(node.expr)};"]
    if isinstance(node, Return):
        return [f"{pad}return {_expr(node.value)};"]
    if isinstance(node, Unassign):
        return [f"{pad}{node.name} ~= {_expr(node.witness)};"]
    if isinstance(node, If):
        lines = [f"{pad}if {_expr(node.condition)} {{"] + _block(node.body, depth + 1)
        if node.orelse:
            lines += [f"{pad}}} else {{"] + _block(node.orelse, depth + 1)
        return lines + [f"{pad}}}"] + _phis("join", node.join, pad)
    if isinstance(node, WhileLoop):
        return _phis("header", node.header, pad) + [f"{pad}while {_expr(node.condition)} {{"] \
            + _block(node.body, depth + 1) + [f"{pad}}}"]
    if isinstance(node, ForLoop):
        return _phis("header", node.header, pad) \
            + [f"{pad}for {node.var} in {_expr(node.start)} .. {_expr(node.stop)} {{"] \
            + _block(node.body, depth + 1) + [f"{pad}}}"]
    if isinstance(node, FunctionDef):
        params = ", ".join(f"{p.name}: {p.type}" for p in node.params)
        returns = f": {node.return_type}" if node.return_type else ""
        return [f"{pad}function {node.name}({params}){returns} {{"] + _block(node.body, depth + 1) + [f"{pad}}}"]
    raise TypeError(f"Cannot print {type(node).__name__}")
