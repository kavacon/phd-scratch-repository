"""Parsing source text into the typed AST. Passes over the AST are composed in `compiler.pipeline`."""
import re
from importlib.resources import files
from typing import Dict, List

from lark import Lark, Transformer, v_args

from compiler.parser.grammar import (
    Assign,
    BinOp,
    Boolean,
    Call,
    Distribution,
    Equals,
    ExprStmt,
    ForLoop,
    FunctionDef,
    If,
    Node,
    Not,
    Number,
    Param,
    Return,
    Unassign,
    Variable,
)

GRAMMARS = ("v1", "v2")  # resources/grammar_<name>.lark: each is a complete syntax, kept apart so an experiment picks its own
_parsers: Dict[str, Lark] = {}


def _parser(grammar: str) -> Lark:
    if grammar not in GRAMMARS:
        raise ValueError(f"no grammar named {grammar!r}: choose from {', '.join(GRAMMARS)}")
    if grammar not in _parsers:
        text = files("compiler.parser").joinpath(f"resources/grammar_{grammar}.lark").read_text(encoding="utf-8")
        _parsers[grammar] = Lark(text, parser="lalr", start="start")
    return _parsers[grammar]


class _Params(tuple):
    """Marks the parameter list among a function definition's optional children."""


class _ReturnType(str):
    """Marks the return type among a function definition's optional children."""


def _binary(op: str):
    def build(_transformer, left, right):
        return BinOp(op, left, right)
    return build


@v_args(inline=True)
class _ToGrammar(Transformer):
    """Lark parse tree -> typed AST nodes."""

    add, sub, mul, div = _binary("+"), _binary("-"), _binary("*"), _binary("/")
    and_, or_, ne, lt, gt = _binary("&&"), _binary("||"), _binary("!="), _binary("<"), _binary(">")

    def start(self, *nodes):
        return list(nodes)

    def block(self, *items):
        return tuple(items)

    args = function_body = block

    def params(self, *items):
        return _Params(items)

    def return_type(self, name):
        return _ReturnType(name)

    def type(self, name):
        return str(name)

    def param(self, name, type_):
        return Param(str(name), type_)

    def assignment(self, name, value):
        return Assign(str(name), value)

    def add_update(self, name, value):
        return Assign(str(name), BinOp("+", Variable(str(name)), value), inplace=True)

    def sub_update(self, name, value):
        return Assign(str(name), BinOp("-", Variable(str(name)), value), inplace=True)

    def distribution(self, text):
        start, stop = (int(bound) for bound in re.fullmatch(r"(-?\d+)\s*\.\.\s*(-?\d+)", str(text)).groups())
        if start >= stop:
            raise ValueError(f"a distribution needs at least one value, but {start}..{stop} has none")
        return Distribution(Number(start), Number(stop))

    def neg(self, operand):
        if isinstance(operand, Number):  # a negative literal
            return Number(-operand.value)
        return BinOp("-", Number(0), operand)

    def return_stmt(self, value):
        return Return(value)

    def unassign_stmt(self, name, witness):
        return Unassign(str(name), witness)

    def expr_statement(self, expr):
        return ExprStmt(expr)

    def for_loop(self, var, start, stop, body):
        return ForLoop(str(var), Number(int(start)), Number(int(stop)), body)

    def if_stmt(self, condition, body, orelse=()):
        return If(condition, body, orelse)

    def else_clause(self, block):
        return block

    def function_call(self, name, args=()):
        return Call(str(name), args)

    def equality(self, *operands):
        return Equals(operands)

    def not_(self, operand):
        return Not(operand)

    def true(self):
        return Boolean(True)

    def false(self):
        return Boolean(False)

    def variable(self, name):
        return Variable(str(name))

    def number(self, text):
        return Number(int(text)) if str(text).isdigit() else Number(float(text))

    def function_def(self, name, *rest):
        params = next((p for p in rest if isinstance(p, _Params)), ())
        return_type = next((str(r) for r in rest if isinstance(r, _ReturnType)), None)
        return FunctionDef(str(name), tuple(params), return_type, rest[-1])


def parse(program: str, grammar: str = "v1") -> List[Node]:
    """Parse source into the typed AST, with the named grammar."""
    return _ToGrammar().transform(_parser(grammar).parse(program))
