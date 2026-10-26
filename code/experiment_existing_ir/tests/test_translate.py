import unittest

from compiler.parser.parser import parse
from experiment_existing_ir.translate import TranslationError, translate


def translated(source: str, grammar: str = "v1", **options) -> str:
    return translate(parse(source, grammar), **options)


class Translate(unittest.TestCase):
    def test_translation_is_the_expected_program(self):
        self.assertEqual(translated("x = 5; y = x + 3; y = y - 1; return y;"), """func.func @main() -> i32 {
  %0 = arith.constant 5 : i32
  %1 = arith.constant 3 : i32
  %2 = arith.addi %0, %1 : i32
  %3 = arith.constant 1 : i32
  %4 = arith.subi %2, %3 : i32
  func.return %4 : i32
}""")

    def test_the_width_is_chosen_by_the_caller(self):
        self.assertEqual(translated("x = 1; return x;", width=16), """func.func @main() -> i16 {
  %0 = arith.constant 1 : i16
  func.return %0 : i16
}""")

    def test_a_conditional_on_a_distribution_yields_the_variables_either_arm_changed(self):
        self.assertEqual(translated("a = 0..4; x = 2; if a < 2 { x += 3; } else { x -= 1; } return x;", "v2"),
                         """func.func @main() -> i32 {
  %0 = "dist.uniform"() <{start = 0 : i32, stop = 4 : i32}> : () -> i32
  %1 = arith.constant 2 : i32
  %2 = arith.constant 2 : i32
  %3 = arith.cmpi slt, %0, %2 : i32
  %8 = scf.if %3 -> (i32) {
    %4 = arith.constant 3 : i32
    %5 = arith.addi %1, %4 : i32
    scf.yield %5 : i32
  } else {
    %6 = arith.constant 1 : i32
    %7 = arith.subi %1, %6 : i32
    scf.yield %7 : i32
  }
  func.return %8 : i32
}""")

    def test_constructs_not_yet_supported_are_refused(self):
        with self.assertRaisesRegex(TranslationError, "'\\*'"):
            translated("x = 2 * 3; return x;")
        with self.assertRaisesRegex(TranslationError, "does not fit in i8"):
            translated("x = 300; return x;", width=8)


if __name__ == "__main__":
    unittest.main()
