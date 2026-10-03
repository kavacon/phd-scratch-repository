from dataclasses import fields, replace
from importlib.resources import files
from typing import Callable, Iterator, List, Sequence

from lark import Lark, Transformer, v_args

from parser.grammar import (
    Assign, BinOp, Boolean, Call, Equals, ExprStmt, ForLoop, FunctionDef, If, Node, Not,
    Number, Param, Return, Unassign, Variable,
)
from passes.allocation import lower_allocation
from passes.balance import balance
from passes.rename import rename
from passes.unroll import unroll

# Define lark based AST parsing for grammar file.
_grammar = (
    files("parser")
    .joinpath("resources/grammar_v1.lark")
    .read_text(encoding="utf-8")
)

_parser = Lark(_grammar, parser="lalr", start="start")

Pass = Callable[[List[Node]], List[Node]]

class _Params(tuple):
    pass


class _ReturnType(str):
    pass


@v_args(inline=True)
class _ToGrammar(Transformer):
    """Lark parse tree -> typed AST nodes."""

    start = lambda self, *nodes: list(nodes)
    block = args = function_body = lambda self, *items: tuple(items)
    params = lambda self, *items: _Params(items)
    return_type = lambda self, t: _ReturnType(t)
    type = lambda self, name: str(name)
    param = lambda self, name, t: Param(str(name), t)
    assignment = lambda self, name, value: Assign(str(name), value)
    return_stmt = lambda self, value: Return(value)
    unassign_stmt = lambda self, name, witness: Unassign(str(name), witness)
    expr_statement = lambda self, expr: ExprStmt(expr)
    for_loop = lambda self, var, start, stop, body: ForLoop(str(var), Number(int(start)), Number(int(stop)), body)
    if_stmt = lambda self, cond, body, orelse=(): If(cond, body, orelse)
    else_clause = lambda self, block: block
    function_call = lambda self, name, args=(): Call(str(name), args)
    equality = lambda self, *operands: Equals(operands)
    add = lambda self, l, r: BinOp("+", l, r)
    sub = lambda self, l, r: BinOp("-", l, r)
    mul = lambda self, l, r: BinOp("*", l, r)
    div = lambda self, l, r: BinOp("/", l, r)
    and_ = lambda self, l, r: BinOp("&&", l, r)
    or_ = lambda self, l, r: BinOp("||", l, r)
    not_ = lambda self, x: Not(x)
    ne = lambda self, l, r: BinOp("!=", l, r)
    lt = lambda self, l, r: BinOp("<", l, r)
    gt = lambda self, l, r: BinOp(">", l, r)
    true = lambda self: Boolean(True)
    false = lambda self: Boolean(False)
    variable = lambda self, name: Variable(str(name))

    def number(self, n):
        return Number(int(n)) if str(n).isdigit() else Number(float(n))

    def function_def(self, name, *rest):
        params = next((p for p in rest if isinstance(p, _Params)), ())
        return_type = next((str(r) for r in rest if isinstance(r, _ReturnType)), None)
        return FunctionDef(str(name), tuple(params), return_type, rest[-1])


def map_children(node: Node, fn: Callable[[Node], Node]) -> Node:
    """Rebuild `node` with `fn` applied to each direct child."""
    changes = {}
    for f in fields(node):
        value = getattr(node, f.name)
        if isinstance(value, Node):
            changes[f.name] = fn(value)
        elif isinstance(value, tuple):
            changes[f.name] = tuple(fn(v) if isinstance(v, Node) else v for v in value)
    return replace(node, **changes)


def rewrite(node: Node, fn: Callable[[Node], Node]) -> Node:
    """Bottom-up rewrite: children first, then `fn` on the rebuilt node."""
    return fn(map_children(node, lambda child: rewrite(child, fn)))


def walk(node: Node) -> Iterator[Node]:
    """Pre-order traversal."""
    yield node
    children: List[Node] = []
    map_children(node, lambda child: children.append(child) or child)
    for child in children:
        yield from walk(child)


def parse(program: str, passes: Sequence[Pass] = (unroll, balance, rename, lower_allocation)) -> List[Node]:
    """Parse source into the typed AST, then apply each pass in order. Use `passes=()` for the raw AST."""
    nodes = _ToGrammar().transform(_parser.parse(program))
    for p in passes:
        nodes = p(nodes)
    return nodes
