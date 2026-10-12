"""
Function inlining, the first pass: every call is replaced by the callee's body, and the definitions are dropped.

A circuit has no call, and which registers a callee may use depends on what its caller holds at that point, so
instead of keeping calls the compiler copies the callee into each call site. The callee's locals then belong to
the caller, and the release pass handles them like any other temporary, at their last use, instead of having a
call undo its whole body. Functions cannot recurse (a circuit cannot either), so this always terminates.

For a call `f(a, b)`:
- the callee's statements are placed just before the statement that holds the call, in the same block, so a call
  inside an `if` or a loop body stays inside it, and a call in an `if` condition runs before the `if`;
- the call itself is replaced by the callee's returned expression;
- each parameter is replaced by its argument when that is a variable or a literal, so passing `x` costs no copy.
  An argument that is an expression is computed once into a local, and so is an argument whose parameter the
  callee assigns to (a parameter is by value, so the caller's variable must not change);
- every other name the callee uses is prefixed with the callee and the call number (`area1.p`, `area2.p`), so two
  calls never share a name and none clashes with the caller's names. Calls inside a callee are inlined first, so
  names nest (`h1.g1.m`).

A call whose value is discarded still runs its body, and its locals are released like any other.
"""
from dataclasses import fields, replace
from typing import Dict, List, Sequence, Set, Tuple

from parser.grammar import (
    Assign,
    BinOp,
    Boolean,
    Call,
    Equals,
    ExprStmt,
    ForLoop,
    FunctionDef,
    If,
    Node,
    Not,
    Number,
    Return,
    Unassign,
    Variable,
)

_ATOMS = (Variable, Number, Boolean)


class InlineError(Exception):
    pass


def _written(block: Sequence[Node]) -> Set[str]:
    """Names a block assigns, releases or uses as a loop variable, including inside nested blocks."""
    names: Set[str] = set()
    for statement in block:
        if isinstance(statement, (Assign, Unassign)):
            names.add(statement.name)
        elif isinstance(statement, ForLoop):
            names.add(statement.var)
            names |= _written(statement.body)
        elif isinstance(statement, If):
            names |= _written(statement.body) | _written(statement.orelse)
    return names


def _rename(node, mapping: Dict[str, Node], prefix: str):
    """A copy of `node` with parameters replaced by their mapping and every other name prefixed."""
    if isinstance(node, Variable):
        return mapping.get(node.name) or Variable(prefix + node.name)
    if isinstance(node, tuple):
        return tuple(_rename(child, mapping, prefix) for child in node)
    if not isinstance(node, Node):
        return node
    changes = {}
    for field in fields(node):
        value = getattr(node, field.name)
        if field.name in ("name", "var") and isinstance(value, str):
            mapped = mapping.get(value)
            changes[field.name] = mapped.name if isinstance(mapped, Variable) else prefix + value
        else:
            changes[field.name] = _rename(value, mapping, prefix)
    return replace(node, **changes)


class _Inliner:
    def __init__(self, definitions: Dict[str, FunctionDef]) -> None:
        self.definitions = definitions
        self.counts: Dict[str, int] = {}
        self.bodies: Dict[str, Tuple[Tuple[str, ...], Tuple[Node, ...], Node]] = {}
        self.active: List[str] = []

    def callee(self, name: str) -> Tuple[Tuple[str, ...], Tuple[Node, ...], Node]:
        """A function as (parameter names, statements, returned expression), with its own calls already inlined."""
        if name in self.bodies:
            return self.bodies[name]
        if name not in self.definitions:
            raise InlineError(f"function {name} is not defined")
        if name in self.active:
            raise InlineError(f"{name} calls itself; a circuit cannot recurse")
        self.active.append(name)
        definition = self.definitions[name]
        *statements, last = definition.body
        if not isinstance(last, Return):
            raise InlineError(f"{name} does not end with a return")
        out: List[Node] = []
        for statement in statements:
            self.statement(statement, out)
        result = self.expr(last.value, out)
        self.active.pop()
        self.bodies[name] = (tuple(p.name for p in definition.params), tuple(out), result)
        return self.bodies[name]

    def block(self, body: Sequence[Node]) -> Tuple[Node, ...]:
        out: List[Node] = []
        for statement in body:
            self.statement(statement, out)
        return tuple(out)

    def statement(self, node: Node, out: List[Node]) -> None:
        if isinstance(node, Assign):
            out.append(Assign(node.name, self.expr(node.value, out)))
        elif isinstance(node, Unassign):
            out.append(Unassign(node.name, self.expr(node.witness, out)))
        elif isinstance(node, ExprStmt):
            value = self.expr(node.expr, out)
            if not isinstance(node.expr, Call):  # a call's value is discarded, but its body had to run
                out.append(ExprStmt(value))
        elif isinstance(node, Return):
            out.append(Return(self.expr(node.value, out)))
        elif isinstance(node, If):
            condition = self.expr(node.condition, out)
            out.append(If(condition, self.block(node.body), self.block(node.orelse)))
        elif isinstance(node, ForLoop):
            out.append(replace(node, body=self.block(node.body)))
        elif isinstance(node, FunctionDef):
            raise InlineError("functions cannot be nested")
        else:
            out.append(node)

    def expr(self, node: Node, out: List[Node]) -> Node:
        """`node` with every call replaced by the callee's result. The callee's statements are appended to `out`."""
        if isinstance(node, Call):
            return self.call(node, out)
        if isinstance(node, BinOp):
            left = self.expr(node.left, out)
            return BinOp(node.op, left, self.expr(node.right, out))
        if isinstance(node, Not):
            return Not(self.expr(node.operand, out))
        if isinstance(node, Equals):
            return Equals(tuple(self.expr(o, out) for o in node.operands))
        return node

    def call(self, node: Call, out: List[Node]) -> Node:
        params, statements, result = self.callee(node.name)
        if len(node.args) != len(params):
            raise InlineError(f"{node.name} takes {len(params)} argument(s) but is called with {len(node.args)}")
        arguments = [self.expr(a, out) for a in node.args]
        self.counts[node.name] = self.counts.get(node.name, 0) + 1
        prefix = f"{node.name}{self.counts[node.name]}."
        written = _written(statements)
        mapping: Dict[str, Node] = {}
        for param, argument in zip(params, arguments, strict=True):
            if isinstance(argument, _ATOMS) and param not in written:
                mapping[param] = argument
            else:  # computed once into a local, or copied because the callee changes it
                out.append(Assign(prefix + param, argument))
                mapping[param] = Variable(prefix + param)
        out.extend(_rename(statement, mapping, prefix) for statement in statements)
        return _rename(result, mapping, prefix)


def inline(program: List[Node]) -> List[Node]:
    """The program with every call replaced by the callee's body and the function definitions removed."""
    definitions = {s.name: s for s in program if isinstance(s, FunctionDef)}
    inliner = _Inliner(definitions)
    out: List[Node] = []
    for statement in program:
        if not isinstance(statement, FunctionDef):
            inliner.statement(statement, out)
    return out
