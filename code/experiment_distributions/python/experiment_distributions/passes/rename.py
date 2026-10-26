"""
Rename, for this experiment: every assignment gets a fresh version, written `name#n`, except an in-place update outside a
branch, which keeps the name of the value it updates (`a#1 += 3`, not `a#2 = a#1 + 3`).

A version names a value, and the lowering gives each one a register, so after renaming a name is assigned once along any
path through the program, apart from those updates. An update changes the register in place, so it does not make a new
value to name: the register holds the earlier value and then the new one under the same name, and nothing needs the
two told apart (the passes here that run after this one work on the value a register holds when it is read, and release
registers, after lowering). Inside a branch an update still gets a fresh version, because both branches of an `if` must
give the variables they assign the same version afterwards, and that is how they are told apart.

Scoping is lexical, as in the shared pass: a name first assigned inside a block is local to it, and assigning to a name that
exists in an enclosing scope makes a new version of that outer variable. Branches are exclusive, so an outer variable
assigned in both gets the then-branch's version in both, which needs the balance pass to have run. Loops and calls are
expanded before this.
"""
from dataclasses import fields, replace
from typing import Dict, List, Optional, Sequence

from compiler.compiler import NODE, compile_pass
from compiler.parser.grammar import Assign, BinOp, Call, Equals, ExprStmt, ForLoop, FunctionDef, If, Node, Not, Return, Unassign, Variable
from compiler.passes.rename import RenameError
from compiler.system import System


class _Renamer:
    def __init__(self) -> None:
        self.counts: Dict[str, int] = {}
        self.depth = 0  # how many branches deep

    def fresh(self, name: str) -> str:
        self.counts[name] = self.counts.get(name, 0) + 1
        return f"{name}#{self.counts[name]}"

    def lookup(self, name: str, env: Dict[str, str]) -> str:
        if name not in env:
            raise RenameError(f"'{name}' is not defined here")
        return env[name]

    def expr(self, node: Node, env: Dict[str, str]) -> Node:
        if isinstance(node, Variable):
            return Variable(self.lookup(node.name, env))
        if isinstance(node, BinOp):
            return BinOp(node.op, self.expr(node.left, env), self.expr(node.right, env))
        if isinstance(node, Not):
            return Not(self.expr(node.operand, env))
        if isinstance(node, Equals):
            return Equals(tuple(self.expr(o, env) for o in node.operands))
        if isinstance(node, Call):
            return Call(node.name, tuple(self.expr(a, env) for a in node.args))
        return node  # literals and distributions

    def block(self, body: Sequence[Node], env: Dict[str, str]) -> tuple:
        return tuple(self.stmt(s, env) for s in body)

    def stmt(self, node: Node, env: Dict[str, str]) -> Node:
        if isinstance(node, Assign):
            value = self.expr(node.value, env)
            if node.inplace and self.depth == 0:
                return Assign(self.lookup(node.name, env), value, True)  # the same value's register, updated
            env[node.name] = name = self.fresh(node.name)
            return Assign(name, value, node.inplace)
        if isinstance(node, ExprStmt):
            return ExprStmt(self.expr(node.expr, env))
        if isinstance(node, Return):
            return Return(self.expr(node.value, env))
        if isinstance(node, Unassign):
            witness = self.expr(node.witness, env)
            name = self.lookup(node.name, env)
            del env[node.name]
            return Unassign(name, witness)
        if isinstance(node, If):
            return self.if_stmt(node, env)
        if isinstance(node, (ForLoop, FunctionDef)):
            raise RenameError("loops must be unrolled and functions inlined before renaming")
        raise RenameError(f"Cannot rename {type(node).__name__}")

    def if_stmt(self, node: If, env: Dict[str, str]) -> If:
        condition = self.expr(node.condition, env)
        then_env, else_env = dict(env), dict(env)
        self.depth += 1
        body = self.block(node.body, then_env)
        orelse = self.block(node.orelse, else_env)
        self.depth -= 1
        shared = {}
        for name, before in env.items():
            if name not in then_env or name not in else_env:
                raise RenameError(f"'{name}' cannot be unassigned inside a block it was not defined in")
            then_version, else_version = then_env[name], else_env[name]
            if (then_version == before) != (else_version == before):
                raise RenameError(f"'{name}' is assigned in only one branch of an if; run balance first")
            if then_version != else_version:  # assigned in both: use the then-branch's version in both
                orelse = tuple(_substitute(s, else_version, then_version) for s in orelse)
                shared[name] = then_version
        env.update(shared)
        return If(condition, body, orelse)


def _substitute(node: Node, old: str, new: str) -> Node:
    """Replace every occurrence of the name `old` (a version, so it cannot clash with anything else) with `new`."""
    changes = {}
    for f in fields(node):
        value = getattr(node, f.name)
        if value == old:
            changes[f.name] = new
        elif isinstance(value, Node):
            changes[f.name] = _substitute(value, old, new)
        elif isinstance(value, tuple):
            changes[f.name] = tuple(
                _substitute(c, old, new) if isinstance(c, Node) else (new if c == old else c) for c in value
            )
    return replace(node, **changes)


@compile_pass(
    NODE, NODE,
    title="Rename",
    requires=("balance",),
    summary="Gives every assignment its own version (x#1, x#2, ...), except an in-place update outside a branch, which "
            "keeps the name of the value it updates.",
)
def rename(program: List[Node], system: Optional[System] = None) -> List[Node]:
    renamer, env = _Renamer(), {}
    return [renamer.stmt(s, env) for s in program]
