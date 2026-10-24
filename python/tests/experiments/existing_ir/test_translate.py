import unittest

from xdsl.passes import PassPipeline
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.dead_code_elimination import DeadCodeElimination

from checking.simulate import simulate
from compiler.parser.parser import parse as parse_source
from experiments.existing_ir.ir import context, evaluate, parse, show
from experiments.existing_ir.translate import TranslationError, translate


def translated(source: str, **options) -> str:
    return show(translate(parse_source(source), **options))


def expected(mlir: str) -> str:
    """The expected program as xDSL prints it, so the comparison ignores how the text was laid out."""
    return show(parse(mlir))


class Translate(unittest.TestCase):
    def test_translation_is_the_expected_program(self):
        self.assertEqual(translated("x = 5; y = x + 3; y = y - 1; return y;"), expected("""
            func.func @main() -> i32 {
              %0 = arith.constant 5 : i32
              %1 = arith.constant 3 : i32
              %2 = arith.addi %0, %1 : i32
              %3 = arith.constant 1 : i32
              %4 = arith.subi %2, %3 : i32
              func.return %4 : i32
            }"""))

    def test_a_reassigned_variable_reads_its_latest_value(self):
        self.assertEqual(translated("x = 2; x = x - 9; return x;"), expected("""
            func.func @main() -> i32 {
              %0 = arith.constant 2 : i32
              %1 = arith.constant 9 : i32
              %2 = arith.subi %0, %1 : i32
              func.return %2 : i32
            }"""))

    def test_the_width_is_chosen_by_the_caller(self):
        self.assertEqual(translated("x = 1; return x;", width=16), expected("""
            func.func @main() -> i16 {
              %0 = arith.constant 1 : i16
              func.return %0 : i16
            }"""))

    def test_translated_program_gives_the_value_of_the_source_interpreter(self):
        for source in ("x = 5; y = x + 3; y = y - 1; return y;", "x = 2; y = x - 9; return y;"):
            nodes = parse_source(source)
            self.assertEqual(evaluate(translate(nodes)), simulate(nodes).value, source)

    def test_constructs_not_yet_supported_are_refused(self):
        with self.assertRaisesRegex(TranslationError, "'\\*'"):
            translate(parse_source("x = 2 * 3; return x;"))
        with self.assertRaisesRegex(TranslationError, "does not fit in i8"):
            translate(parse_source("x = 300; return x;"), width=8)


class StandardPasses(unittest.TestCase):
    def test_canonicalize_evaluates_a_translated_program(self):
        module = translate(parse_source("x = 5; y = x + 3; y = y - 1; return y;"))
        PassPipeline((CanonicalizePass(), DeadCodeElimination())).apply(context(), module)
        self.assertEqual(show(module), expected("""
            func.func @main() -> i32 {
              %0 = arith.constant 7 : i32
              func.return %0 : i32
            }"""))


if __name__ == "__main__":
    unittest.main()
