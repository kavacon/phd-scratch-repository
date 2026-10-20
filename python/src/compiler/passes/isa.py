"""
Lowering to the instruction set (parser.instructions), after the passes up to and including release insertion.

The result is one flat list of instructions with no blocks, and the value it returns. Calls were descoped for this
version: the inline pass must have run, so there are no functions and no calls to lower.

Each assignment is `ALLOC r` followed by one instruction that computes the value into it (a distribution, `x = 0..4`, is
`DIST r, 0, 4`), and each `~=` is the
inverse instruction (`IMUL` undoes `MUL`) followed by `FREE r`. An `if` becomes a `CTRL c` region and an `ICTRL c` region, each closed by
`ENDCTRL`, and both are always in the circuit (the second is controlled on the negation, so no negation is
computed). A controlled region may not allocate or free, so every register an `if` touches is allocated before
the outermost `if` that contains it and freed after. A register shared by both branches is allocated once. An `if`
on a constant is resolved here and only the branch that runs is emitted.

The program must be flat: each assignment a plain value or one operation on plain values, and it returns a
variable (see passes.flatten).
"""
from typing import Dict, List, Optional, Tuple

from compiler.compiler import ISA, NODE, compile_pass
from compiler.system import System
from compiler.parser.grammar import (
    Assign,
    BinOp,
    Boolean,
    Distribution,
    Equals,
    ExprStmt,
    FunctionDef,
    If,
    Node,
    Not,
    Number,
    Return,
    Unassign,
    Variable,
)
from compiler.parser.grammar import (
    Call as CallNode,
)
from compiler.parser.instructions import (
    Alloc,
    Const,
    Ctrl,
    Dist,
    EndCtrl,
    Free,
    ICtrl,
    Instruction,
    IsaProgram,
    Op,
    Operand,
    Reg,
    Update,
    inverse,
)

_BINARY = {"+": "add", "-": "sub", "*": "mul", "/": "div", "<": "lt", ">": "gt", "!=": "ne", "&&": "and", "||": "or"}


class IsaError(Exception):
    pass


def _operand(node: Node, reg=lambda name: name) -> Operand:
    """The operand for a plain value; `reg` says which register a version lives in."""
    if isinstance(node, Variable):
        return Reg(reg(node.name))
    if isinstance(node, (Number, Boolean)):
        return Const(node.value)
    raise IsaError(f"expected a plain value, got {type(node).__name__}; run flatten first")


def _compute(dst: str, value: Node, reg=lambda name: name) -> Instruction:
    """The instruction that XORs `value` into `dst`, or, for a distribution, prepares it there."""
    if isinstance(value, Distribution):
        return Dist(dst, value.start.value, value.stop.value)
    if isinstance(value, BinOp):
        return Op(_BINARY[value.op], dst, (_operand(value.left, reg), _operand(value.right, reg)))
    if isinstance(value, Not):
        return Op("not", dst, (_operand(value.operand, reg),))
    if isinstance(value, Equals):
        if len(value.operands) != 2:
            raise IsaError("an equality chain must be flattened first")
        return Op("eq", dst, tuple(_operand(o, reg) for o in value.operands))
    if isinstance(value, CallNode):
        raise IsaError("calls must be inlined first")
    return Op("copy", dst, (_operand(value, reg),))


class _Hoisted:
    """Registers an outermost `if` must allocate before its controlled blocks and free after them."""

    def __init__(self) -> None:
        self.allocs: List[str] = []
        self.frees: List[str] = []


def _unique(names: List[str]) -> List[str]:
    return list(dict.fromkeys(names))


class _Lowerer:
    """
    Lowers a flat, renamed program. An in-place update (`x#2 = x#1 + e`, marked) has no register of its own: its new
    version is another name for the register of the version it replaces, recorded in `alias`.
    """

    def __init__(self) -> None:
        self.alias: Dict[str, str] = {}

    def reg(self, name: str) -> str:
        return self.alias.get(name, name)

    def operand(self, node: Node) -> Operand:
        return _operand(node, self.reg)

    def block(self, body: Tuple[Node, ...], out: List[Instruction], hoisted: Optional[_Hoisted]) -> None:
        for statement in body:
            self.statement(statement, out, hoisted)

    def update_target(self, node: Assign) -> Tuple[str, Node]:
        """The register an in-place update changes and the operand it adds or subtracts."""
        value = node.value
        if not (isinstance(value, BinOp) and value.op in ("+", "-") and isinstance(value.left, Variable)):
            raise IsaError("an in-place update must add or subtract a plain value")
        return self.reg(value.left.name), value.right

    def in_place(self, node: Assign, out: List[Instruction]) -> None:
        target, rhs = self.update_target(node)
        source = self.operand(rhs)
        if source == Reg(target):
            raise IsaError(f"'{node.name}' is updated with itself, which cannot be undone, so it has no circuit")
        out.append(Update("add" if node.value.op == "+" else "sub", target, source))
        self.alias[node.name] = target

    def claim_updates(self, block: Tuple[Node, ...]) -> None:
        """Before the branches of an `if` are lowered, give every version an update makes its register, in order."""
        for statement in block:
            if isinstance(statement, Assign) and statement.inplace:
                self.alias[statement.name] = self.update_target(statement)[0]
            elif isinstance(statement, If):
                self.claim_updates(statement.body)
                self.claim_updates(statement.orelse)

    def statement(self, node: Node, out: List[Instruction], hoisted: Optional[_Hoisted]) -> None:
        if isinstance(node, Assign):
            if node.inplace:
                self.in_place(node, out)
                return
            if node.name in self.alias:
                # the other branch of an `if` updated this variable in place, so this branch either leaves it alone
                # (an identity copy from balance, which is nothing to do) or cannot be expressed
                if isinstance(node.value, Variable) and self.reg(node.value.name) == self.alias[node.name]:
                    return
                raise IsaError(f"'{node.name}' is updated in place in one branch of an if and reassigned in the other; "
                               f"use += or -= in both")
            if hoisted is None:
                out.append(Alloc(node.name))
            else:
                hoisted.allocs.append(node.name)
            out.append(_compute(node.name, node.value, self.reg))
        elif isinstance(node, Unassign):
            out.append(inverse(_compute(self.reg(node.name), node.witness, self.reg)))
            if hoisted is None:
                out.append(Free(self.reg(node.name)))
            else:
                hoisted.frees.append(self.reg(node.name))
        elif isinstance(node, ExprStmt):
            pass  # a value that is not used leaves nothing behind
        elif isinstance(node, If):
            self.if_statement(node, out, hoisted)
        else:
            raise IsaError(f"cannot lower {type(node).__name__}")

    def if_statement(self, node: If, out: List[Instruction], hoisted: Optional[_Hoisted]) -> None:
        control = self.operand(node.condition)
        if isinstance(control, Const):  # decided now, so only the branch that runs is emitted
            self.block(node.body if control.value else node.orelse, out, hoisted)
            return
        self.claim_updates(node.body)
        self.claim_updates(node.orelse)
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
        if not isinstance(last.value, Variable):
            raise IsaError("the program must return a variable; run flatten first")
        out: List[Instruction] = []
        self.block(tuple(statements), out, None)
        return IsaProgram(tuple(out), self.operand(last.value))


@compile_pass(
    NODE, ISA,
    title="Lower to instructions",
    requires=("inline", "flatten", "rename"),
    summary=(
        "Turns the program into one flat list of instructions: ALLOC and an operation per assignment, the "
        "inverse and FREE per release, and CTRL regions for ifs."
    ),
)
def lower_to_isa(program: List[Node], system: Optional[System] = None) -> IsaProgram:
    if any(isinstance(s, FunctionDef) for s in program):
        raise IsaError("functions must be inlined first")
    return _Lowerer().program(tuple(program))
