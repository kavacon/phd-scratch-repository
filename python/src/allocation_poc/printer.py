"""Renders an AST, or an instruction program, as readable text, for inspecting what passes produce."""
from typing import List, Sequence

from parser.grammar import (
    Assign, BinOp, Boolean, Call, Equals, ExprStmt, ForLoop, FunctionDef, If, Node, Not,
    Number, Return, Unassign, Variable,
)


def unparse(program: Sequence[Node]) -> str:
    return "\n".join(_block(program, 0))


def unparse_statement(statement: Node) -> str:
    return "\n".join(_stmt(statement, 0))


def _expr(node: Node, nested: bool = False) -> str:
    if isinstance(node, (Number, Variable)):
        return str(node.value if isinstance(node, Number) else node.name)
    if isinstance(node, Boolean):
        return "true" if node.value else "false"
    if isinstance(node, Call):
        return f"{node.name}({', '.join(_expr(a) for a in node.args)})"
    if isinstance(node, Not):
        return f"!{_expr(node.operand, True)}"
    text = f"{_expr(node.left, True)} {node.op} {_expr(node.right, True)}" if isinstance(node, BinOp) \
        else " == ".join(_expr(o, True) for o in node.operands)
    return f"({text})" if nested else text


def _block(body: Sequence[Node], depth: int) -> List[str]:
    return [line for statement in body for line in _stmt(statement, depth)]


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
        return lines + [f"{pad}}}"]
    if isinstance(node, ForLoop):
        return [f"{pad}for {node.var} in {_expr(node.start)} .. {_expr(node.stop)} {{"] \
            + _block(node.body, depth + 1) + [f"{pad}}}"]
    if isinstance(node, FunctionDef):
        params = ", ".join(f"{p.name}: {p.type}" for p in node.params)
        returns = f": {node.return_type}" if node.return_type else ""
        return [f"{pad}function {node.name}({params}){returns} {{"] + _block(node.body, depth + 1) + [f"{pad}}}"]
    raise TypeError(f"Cannot print {type(node).__name__}")


def _operand_text(operand) -> str:
    from parser.instructions import Reg
    if isinstance(operand, Reg):
        return operand.name
    value = operand.value
    return "true" if value is True else "false" if value is False else str(value)


def _instruction_lines(instruction, depth: int) -> List[str]:
    from parser.instructions import Alloc, Ctrl, EndCtrl, Free, ICtrl, IOp, Op
    pad = "    " * depth
    if isinstance(instruction, Alloc):
        return [f"{pad}ALLOC {instruction.register}"]
    if isinstance(instruction, Free):
        return [f"{pad}FREE  {instruction.register}"]
    if isinstance(instruction, (Op, IOp)):
        sources = ", ".join(_operand_text(s) for s in instruction.sources)
        mnemonic = ("I" if isinstance(instruction, IOp) else "") + instruction.opcode.upper()
        return [f"{pad}{mnemonic:<5} {instruction.dst}, {sources}"]
    if isinstance(instruction, Ctrl):
        return [f"{pad}CTRL    {_operand_text(instruction.control)}"]
    if isinstance(instruction, ICtrl):
        return [f"{pad}ICTRL   {_operand_text(instruction.control)}"]
    if isinstance(instruction, EndCtrl):
        return [f"{pad}ENDCTRL"]
    raise TypeError(f"Cannot print {type(instruction).__name__}")


def unparse_isa(program) -> str:
    """The instructions in order, then the returned value. Instructions inside a control region are indented, which
    is only for reading: the list itself is flat."""
    from parser.instructions import Ctrl, EndCtrl, ICtrl
    lines: List[str] = []
    depth = 0
    for instruction in program.body:
        if isinstance(instruction, EndCtrl):
            depth -= 1
        lines += _instruction_lines(instruction, depth)
        if isinstance(instruction, (Ctrl, ICtrl)):
            depth += 1
    return "\n".join(lines + [f"RETURN {_operand_text(program.result)}"])
