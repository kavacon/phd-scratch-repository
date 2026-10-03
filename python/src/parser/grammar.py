"""Typed AST nodes mirroring grammar_v1.lark."""
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
    name: str
    value: Node


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
    start: Node
    stop: Node  # exclusive
    body: Tuple[Node, ...]


@dataclass(frozen=True)
class WhileLoop(Node):
    condition: Node
    body: Tuple[Node, ...]


@dataclass(frozen=True)
class If(Node):
    condition: Node
    body: Tuple[Node, ...]
    orelse: Tuple[Node, ...] = ()  # an `else if` is a single If in here


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
