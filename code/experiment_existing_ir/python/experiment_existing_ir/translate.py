"""
Translate a source program (the typed AST from `compiler.parser`) into the standard `func` and `arith` dialects.

Every integer is given one fixed width, chosen by the front-end, the way a classical front-end gives every `int` its
declared width. That width is an upper bound: a later pass narrows values to the width they need. Results that do not
fit wrap, as they do in C. Change `WIDTH` (or pass `width`) to try another bound.

This covers what the first milestone needs: constants, variables, `+`, `-` and a final return. Anything else is
refused by name, so a later milestone adds it deliberately.
"""
from typing import Dict, Sequence

from xdsl.dialects import func
from xdsl.dialects.arith import AddiOp, ConstantOp, SubiOp
from xdsl.dialects.builtin import IntegerType, ModuleOp
from xdsl.ir import Block, Operation, Region, SSAValue

from compiler.parser.grammar import Assign, BinOp, Node, Number, Return, Variable

WIDTH = 32


class TranslationError(ValueError):
    pass


class _Translator:
    def __init__(self, width: int) -> None:
        self.width = width
        self.ops: list[Operation] = []
        self.env: Dict[str, SSAValue] = {}

    def emit(self, op: Operation) -> Operation:
        self.ops.append(op)
        return op

    def constant(self, value: int) -> SSAValue:
        low, high = -(1 << (self.width - 1)), (1 << (self.width - 1)) - 1
        if not low <= value <= high:
            raise TranslationError(f"{value} does not fit in i{self.width}")
        return self.emit(ConstantOp.from_int_and_width(value, self.width)).result

    def binary(self, node: BinOp) -> SSAValue:
        op_type = {"+": AddiOp, "-": SubiOp}.get(node.op)
        if op_type is None:
            raise TranslationError(f"operator {node.op!r} is not supported yet")
        return self.emit(op_type(self.expression(node.left), self.expression(node.right))).result

    def expression(self, node: Node) -> SSAValue:
        if isinstance(node, Number):
            if not isinstance(node.value, int):
                raise TranslationError("only integer constants are supported yet")
            return self.constant(node.value)
        if isinstance(node, Variable):
            if node.name not in self.env:
                raise TranslationError(f"{node.name} is used before it is assigned")
            return self.env[node.name]
        if isinstance(node, BinOp):
            return self.binary(node)
        raise TranslationError(f"{type(node).__name__} is not supported yet")

    def program(self, program: Sequence[Node]) -> ModuleOp:
        returned = None
        for statement in program:
            if isinstance(statement, Assign):
                self.env[statement.name] = self.expression(statement.value)
            elif isinstance(statement, Return):
                returned = self.expression(statement.value)
            else:
                raise TranslationError(f"{type(statement).__name__} is not supported yet")
        if returned is None:
            raise TranslationError("a program must end with a return")
        self.emit(func.ReturnOp(returned))
        main = func.FuncOp("main", ((), (IntegerType(self.width),)), Region(Block(self.ops)))
        return ModuleOp([main])


def translate(program: Sequence[Node], width: int = WIDTH) -> ModuleOp:
    return _Translator(width).program(program)
