"""
Rename pass: gives every binding a unique version, written `name#n` (parameters are `name#0`,
assignments start at `name#1`). After renaming each name is assigned exactly once, so a value's
producer and its uses can be tracked without worrying about overwrites.

Scoping is lexical: a name first assigned inside a block is local to that block, while assigning to
a name that already exists in an enclosing scope creates a new version of that outer variable.
Functions cannot see variables outside themselves.

Unassign (`x ~= e;`) releases the current version of x, so x is no longer in scope afterwards.

Where control flow merges, `Phi` assignments select between versions:
- `If.join`: after the branches, for each outer variable the branches left at different versions.
- `WhileLoop.header` / `ForLoop.header`: at the loop header, for each outer variable assigned in the
  body, choosing between its entry version and the version at the end of the body. Code after the loop
  sees the header version.
"""
from typing import Dict, Iterator, List, Sequence

from parser.grammar import (
    Assign, BinOp, Call, Equals, ExprStmt, ForLoop, FunctionDef, If, Node, Not, Param, Phi,
    Return, Unassign, Variable, WhileLoop,
)


class RenameError(Exception):
    pass


def _assigned(body: Sequence[Node]) -> Iterator[str]:
    """Names assigned anywhere in `body`, including nested blocks."""
    for statement in body:
        if isinstance(statement, Assign):
            yield statement.name
        elif isinstance(statement, If):
            yield from _assigned(statement.body)
            yield from _assigned(statement.orelse)
        elif isinstance(statement, (WhileLoop, ForLoop)):
            yield from _assigned(statement.body)


class _Renamer:
    def __init__(self):
        self.counts: Dict[str, int] = {}

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
        return node  # literals

    def block(self, body: Sequence[Node], env: Dict[str, str]) -> tuple:
        return tuple(self.stmt(s, env) for s in body)

    def stmt(self, node: Node, env: Dict[str, str]) -> Node:
        if isinstance(node, Assign):
            value = self.expr(node.value, env)
            env[node.name] = name = self.fresh(node.name)
            return Assign(name, value)
        if isinstance(node, ExprStmt):
            return ExprStmt(self.expr(node.expr, env))
        if isinstance(node, Return):
            return Return(self.expr(node.value, env))
        if isinstance(node, Unassign):
            witness = self.expr(node.witness, env)
            name = self.lookup(node.name, env)
            del env[node.name]  # released: later use is an error
            return Unassign(name, witness)
        if isinstance(node, If):
            return self.if_stmt(node, env)
        if isinstance(node, WhileLoop):
            return self.loop(node, env)
        if isinstance(node, ForLoop):
            return self.loop(node, env)
        if isinstance(node, FunctionDef):
            return self.function(node)
        raise RenameError(f"Cannot rename {type(node).__name__}")

    def check_still_defined(self, name: str, *envs: Dict[str, str]) -> None:
        if any(name not in e for e in envs):
            raise RenameError(f"'{name}' cannot be unassigned inside a block it was not defined in")

    def if_stmt(self, node: If, env: Dict[str, str]) -> If:
        condition = self.expr(node.condition, env)
        then_env, else_env = dict(env), dict(env)
        body = self.block(node.body, then_env)
        orelse = self.block(node.orelse, else_env)
        join, merged = [], {}
        for name in env:
            self.check_still_defined(name, then_env, else_env)
            then_version, else_version = then_env[name], else_env[name]
            if then_version != else_version:
                merged[name] = self.fresh(name)
                join.append(Assign(merged[name], Phi((Variable(then_version), Variable(else_version)))))
        env.update(merged)
        return If(condition, body, orelse, tuple(join))

    def loop(self, node, env: Dict[str, str]):
        is_for = isinstance(node, ForLoop)
        if is_for:  # bounds are evaluated once, on entry
            start, stop = self.expr(node.start, env), self.expr(node.stop, env)
        carried = [n for n in dict.fromkeys(_assigned(node.body)) if n in env and not (is_for and n == node.var)]
        entry = {n: env[n] for n in carried}
        headers = {n: self.fresh(n) for n in carried}
        body_env = {**env, **headers}
        if is_for:
            body_env[node.var] = var = self.fresh(node.var)
        else:
            condition = self.expr(node.condition, body_env)
        body = self.block(node.body, body_env)
        for name in env:
            self.check_still_defined(name, body_env)
        header = tuple(
            Assign(headers[n], Phi((Variable(entry[n]), Variable(body_env[n])))) for n in carried
        )
        env.update(headers)
        if is_for:
            return ForLoop(var, start, stop, body, header)
        return WhileLoop(condition, body, header)

    def function(self, node: FunctionDef) -> FunctionDef:
        local = _Renamer()
        params = tuple(Param(f"{p.name}#0", p.type) for p in node.params)
        env = {p.name: f"{p.name}#0" for p in node.params}
        return FunctionDef(node.name, params, node.return_type, local.block(node.body, env))


def rename(program: List[Node]) -> List[Node]:
    renamer, env = _Renamer(), {}
    return [renamer.stmt(s, env) for s in program]
