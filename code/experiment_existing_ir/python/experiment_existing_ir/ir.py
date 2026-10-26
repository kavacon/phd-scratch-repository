"""
Thin helpers around xDSL: a context with the dialects we use, parsing and printing MLIR text, and an interpreter for
the result. Passes are run with xDSL's own `PassPipeline`. Nothing here is specific to a program.
"""
from io import StringIO

from xdsl.context import Context
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.parser import Parser
from xdsl.printer import Printer


def context() -> Context:
    ctx = Context()
    for dialect in (Builtin, Func, Arith):
        ctx.load_dialect(dialect)
    return ctx


def parse(text: str, ctx: Context | None = None) -> ModuleOp:
    module = Parser(ctx or context(), text).parse_module()
    module.verify()
    return module


def show(module: ModuleOp, *, generic: bool = False) -> str:
    out = StringIO()
    Printer(out, print_generic_format=generic).print_op(module)
    return out.getvalue().strip()


def evaluate(module: ModuleOp) -> int:
    """Run `main` with xDSL's interpreter. Integers come back as signed values."""
    interpreter = Interpreter(module)
    interpreter.register_implementations(ArithFunctions())
    interpreter.register_implementations(FuncFunctions())
    (result,) = interpreter.call_op("main", ())
    return result
