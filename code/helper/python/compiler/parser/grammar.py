"""Typed AST nodes mirroring the grammars in resources/."""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Node:
    pass


@dataclass(frozen=True)
class Number(Node):
    value: int | float


@dataclass(frozen=True)
class Boolean(Node):
    value: bool


@dataclass(frozen=True)
class Variable(Node):
    name: str


@dataclass(frozen=True)
class BinOp(Node):
    op: str  # "+", "-", "*", "/", "<", ">", "!=", "&&" or "||"
    left: Node
    right: Node


@dataclass(frozen=True)
class Not(Node):
    operand: Node


@dataclass(frozen=True)
class Equals(Node):
    operands: Tuple[Node, ...]  # a == b == c


@dataclass(frozen=True)
class Call(Node):
    name: str
    args: Tuple[Node, ...] = ()


@dataclass(frozen=True)
class Assign(Node):
    """
    `name = value;`. An in-place update (`x += e;` or `x -= e;`) is the same assignment marked `inplace`: its value is
    `x + e` or `x - e` with the target's previous version on the left, and the new version reuses that version's
    register instead of getting one of its own.
    """
    name: str
    value: Node
    inplace: bool = False


@dataclass(frozen=True)
class ExprStmt(Node):
    expr: Node


@dataclass(frozen=True)
class Return(Node):
    value: Node


@dataclass(frozen=True)
class Unassign(Node):
    """Inverse of Assign: `name ~= witness;` asserts name currently equals witness, and releases it."""
    name: str
    witness: Node


@dataclass(frozen=True)
class ForLoop(Node):
    var: str
    start: Node  # a Number: loop bounds are constants for now (see the TODO in grammar_v1.lark)
    stop: Node  # a Number, exclusive
    body: Tuple[Node, ...]  # the unroll pass removes every loop, so later passes never see one


@dataclass(frozen=True)
class Distribution(Node):
    """
    `x = start..stop;` (grammar v2): an equal superposition of the integers from start up to, not including, stop, read
    in the language as a distribution over them. Bounds are constants for now, as a loop's are; a range one value wide
    is an ordinary definite value.
    """
    start: Node  # a Number
    stop: Node  # a Number, exclusive


@dataclass(frozen=True)
class If(Node):
    condition: Node
    body: Tuple[Node, ...]
    orelse: Tuple[Node, ...] = ()


@dataclass(frozen=True)
class Param(Node):
    name: str
    type: str


@dataclass(frozen=True)
class FunctionDef(Node):
    name: str
    params: Tuple[Param, ...]
    return_type: Optional[str]
    body: Tuple[Node, ...]
