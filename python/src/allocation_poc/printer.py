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
    from parser.instructions import Alloc, Call as CallInstruction, Ctrl, EndCtrl, Free, InvCtrl, Op
    pad = "    " * depth
    if isinstance(instruction, Alloc):
        return [f"{pad}ALLOC {instruction.register}"]
    if isinstance(instruction, Free):
        return [f"{pad}FREE  {instruction.register}"]
    if isinstance(instruction, Op):
        sources = ", ".join(_operand_text(s) for s in instruction.sources)
        return [f"{pad}{instruction.opcode.upper():<5} {instruction.dst}, {sources}"]
    if isinstance(instruction, CallInstruction):
        args = ", ".join(_operand_text(a) for a in instruction.args)
        target = f"{instruction.dst}, " if instruction.dst else ""
        return [f"{pad}CALL  {target}{instruction.function}({args})"]
    if isinstance(instruction, Ctrl):
        return [f"{pad}CTRL    {_operand_text(instruction.control)}"]
    if isinstance(instruction, InvCtrl):
        return [f"{pad}INVCTRL {_operand_text(instruction.control)}"]
    if isinstance(instruction, EndCtrl):
        return [f"{pad}ENDCTRL"]
    raise TypeError(f"Cannot print {type(instruction).__name__}")


def unparse_isa(program) -> str:
    """Each instruction set under a label, `main` last. A call names the label of the set it runs.
    Instructions inside a control region are indented, which is only for reading: the list itself is flat."""
    from parser.instructions import Ctrl, EndCtrl, InvCtrl
    lines: List[str] = []
    for instruction_set in list(program.functions) + [program.main]:
        lines.append(f"{instruction_set.name}({', '.join(instruction_set.params)}):" if instruction_set.params
                     else f"{instruction_set.name}:")
        depth = 1
        for instruction in instruction_set.body:
            if isinstance(instruction, EndCtrl):
                depth -= 1
            lines += _instruction_lines(instruction, depth)
            if isinstance(instruction, (Ctrl, InvCtrl)):
                depth += 1
        lines += [f"    RETURN {_operand_text(instruction_set.result)}", ""]
    return "\n".join(lines).rstrip()
