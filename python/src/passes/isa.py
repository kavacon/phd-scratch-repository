"""
Lowering to the instruction set (parser.instructions), after the passes up to and including release insertion.

The result is one flat list of instructions with no blocks, and the value it returns. Calls were descoped for this
version: the inline pass must have run, so there are no functions and no calls to lower.

Each assignment is `ALLOC r` followed by one instruction that computes the value into it, and each `~=` is the
inverse instruction (`IMUL` undoes `MUL`) followed by `FREE r`. An `if` becomes a `CTRL c` region and an `ICTRL c` region, each closed by
`ENDCTRL`, and both are always in the circuit (the second is controlled on the negation, so no negation is
computed). A controlled region may not allocate or free, so every register an `if` touches is allocated before
the outermost `if` that contains it and freed after. A register shared by both branches is allocated once. An `if`
on a constant is resolved here and only the branch that runs is emitted.

The program must be flat: each assignment a plain value or one operation on plain values (see passes.flatten).
"""
from typing import List, Optional, Tuple

from parser.grammar import (
    Assign, BinOp, Boolean, Call as CallNode, Equals, ExprStmt, FunctionDef, If, Node, Not, Number, Return,
    Unassign, Variable,
)
from parser.instructions import (
    Alloc, Const, Ctrl, EndCtrl, Free, ICtrl, Instruction, IsaProgram, Op, Operand, Reg, inverse,
)

_BINARY = {"+": "add", "-": "sub", "*": "mul", "/": "div", "<": "lt", ">": "gt", "!=": "ne", "&&": "and", "||": "or"}


class IsaError(Exception):
    pass


def _operand(node: Node) -> Operand:
    if isinstance(node, Variable):
        return Reg(node.name)
    if isinstance(node, (Number, Boolean)):
        return Const(node.value)
    raise IsaError(f"expected a plain value, got {type(node).__name__}; run flatten first")


def _compute(dst: str, value: Node) -> Instruction:
    """The instruction that XORs `value` into `dst`."""
    if isinstance(value, BinOp):
        return Op(_BINARY[value.op], dst, (_operand(value.left), _operand(value.right)))
    if isinstance(value, Not):
        return Op("not", dst, (_operand(value.operand),))
    if isinstance(value, Equals):
        if len(value.operands) != 2:
            raise IsaError("an equality chain must be flattened first")
        return Op("eq", dst, tuple(_operand(o) for o in value.operands))
    if isinstance(value, CallNode):
        raise IsaError("calls must be inlined first")
    return Op("copy", dst, (_operand(value),))


class _Hoisted:
    """Registers an outermost `if` must allocate before its controlled blocks and free after them."""

    def __init__(self) -> None:
        self.allocs: List[str] = []
        self.frees: List[str] = []


def _unique(names: List[str]) -> List[str]:
    return list(dict.fromkeys(names))


class _Lowerer:
    def block(self, body: Tuple[Node, ...], out: List[Instruction], hoisted: Optional[_Hoisted]) -> None:
        for statement in body:
            self.statement(statement, out, hoisted)

    def statement(self, node: Node, out: List[Instruction], hoisted: Optional[_Hoisted]) -> None:
        if isinstance(node, Assign):
            if hoisted is None:
                out.append(Alloc(node.name))
            else:
                hoisted.allocs.append(node.name)
            out.append(_compute(node.name, node.value))
        elif isinstance(node, Unassign):
            out.append(inverse(_compute(node.name, node.witness)))
            if hoisted is None:
                out.append(Free(node.name))
            else:
                hoisted.frees.append(node.name)
        elif isinstance(node, ExprStmt):
            pass  # a value that is not used leaves nothing behind
        elif isinstance(node, If):
            self.if_statement(node, out, hoisted)
        else:
            raise IsaError(f"cannot lower {type(node).__name__}")

    def if_statement(self, node: If, out: List[Instruction], hoisted: Optional[_Hoisted]) -> None:
        control = _operand(node.condition)
        if isinstance(control, Const):  # decided now, so only the branch that runs is emitted
            self.block(node.body if control.value else node.orelse, out, hoisted)
            return
        inner = hoisted if hoisted is not None else _Hoisted()
        blocks: List[Instruction] = []
        for opener, branch in ((Ctrl, node.body), (ICtrl, node.orelse)):
            body: List[Instruction] = []
            self.block(branch, body, inner)
            if body:
                blocks += [opener(control), *body, EndCtrl()]
        if hoisted is None:
            out.extend(Alloc(r) for r in _unique(inner.allocs))
            out.extend(blocks)
            out.extend(Free(r) for r in _unique(inner.frees))
        else:
            out.extend(blocks)

    def program(self, body: Tuple[Node, ...]) -> IsaProgram:
        *statements, last = body
        if not isinstance(last, Return):
            raise IsaError("the program does not end with a return")
        out: List[Instruction] = []
        self.block(tuple(statements), out, None)
        return IsaProgram(tuple(out), _operand(last.value))


def lower_to_isa(program: List[Node]) -> IsaProgram:
    if any(isinstance(s, FunctionDef) for s in program):
        raise IsaError("functions must be inlined first")
    return _Lowerer().program(tuple(program))
