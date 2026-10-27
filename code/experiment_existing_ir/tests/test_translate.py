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

    def test_a_distribution_is_changed_by_an_in_place_update_and_reading_it_measures_it(self):
        self.assertEqual(translated("a = 0..4; a += 3; return a;", "v2"), """func.func @main() -> i32 {
  %0 = dist.uniform 0, 4 : !dist.dist<i32>
  %1 = arith.constant 3 : i32
  %2 = dist.add_assign %0, %1 : !dist.dist<i32>, i32 -> !dist.dist<i32>
  %3 = dist.measure %2 : !dist.dist<i32> -> i32
  func.return %3 : i32
}""")

    def test_a_distribution_can_be_read_by_several_conditions(self):
        text = translated("a = 0..4; x = 0; y = 0; if a < 2 { x += 1; } if a < 3 { y += 2; } return x;", "v2")
        self.assertEqual(text.count("dist.slt %0,"), 2)  # both read the one distribution

    def test_a_conditional_on_a_distribution_computes_both_arms_and_selects_once(self):
        self.assertEqual(translated("a = 0..4; x = 2; if a < 2 { x += 3; } else { x -= 1; } return x;", "v2"),
                         """func.func @main() -> i32 {
  %0 = dist.uniform 0, 4 : !dist.dist<i32>
  %1 = arith.constant 2 : i32
  %2 = arith.constant 2 : i32
  %3 = dist.slt %0, %2 : !dist.dist<i32>, i32 -> !dist.dist<i1>
  %4 = arith.constant 3 : i32
  %5 = arith.addi %1, %4 : i32
  %6 = arith.constant 1 : i32
  %7 = arith.subi %1, %6 : i32
  %8 = dist.select %3, (%5), (%7) : !dist.dist<i1>, (i32), (i32) -> (!dist.dist<i32>)
  %9 = dist.measure %8 : !dist.dist<i32> -> i32
  func.return %9 : i32
}""")

    def test_a_conditional_on_a_definite_value_yields_the_variables_either_arm_changed(self):
        self.assertEqual(translated("x = 2; if x < 3 { x += 5; } else { x -= 1; } return x;"), """func.func @main() -> i32 {
  %0 = arith.constant 2 : i32
  %1 = arith.constant 3 : i32
  %2 = arith.cmpi slt, %0, %1 : i32
  %7 = scf.if %2 -> (i32) {
    %3 = arith.constant 5 : i32
    %4 = arith.addi %0, %3 : i32
    scf.yield %4 : i32
  } else {
    %5 = arith.constant 1 : i32
    %6 = arith.subi %0, %5 : i32
    scf.yield %6 : i32
  }
  func.return %7 : i32
}""")

    def test_constructs_not_yet_supported_are_refused(self):
        with self.assertRaisesRegex(TranslationError, "'\\*'"):
            translated("x = 2 * 3; return x;")
        with self.assertRaisesRegex(TranslationError, "does not fit in i8"):
            translated("x = 300; return x;", width=8)

    def test_a_plain_name_is_an_alias_so_an_update_through_one_name_is_seen_through_the_other(self):
        self.assertEqual(translated("a = 0..4; b = a; b += 1; return a;", "v2"), """func.func @main() -> i32 {
  %0 = dist.uniform 0, 4 : !dist.dist<i32>
  %1 = arith.constant 1 : i32
  %2 = dist.add_assign %0, %1 : !dist.dist<i32>, i32 -> !dist.dist<i32>
  %3 = dist.measure %2 : !dist.dist<i32> -> i32
  func.return %3 : i32
}""")

    def test_a_computed_value_assigned_to_a_new_variable_is_marked_as_an_instance(self):
        self.assertEqual(translated("a = 0..4; c = a + 1; return c - a;", "v2"), """func.func @main() -> i32 {
  %0 = dist.uniform 0, 4 : !dist.dist<i32>
  %1 = arith.constant 1 : i32
  %2 = dist.addi %0, %1 : !dist.dist<i32>, i32 -> !dist.dist<i32>
  %3 = dist.instance %2 : !dist.dist<i32> -> !dist.dist<i32>
  %4 = dist.subi %3, %0 : !dist.dist<i32>, !dist.dist<i32> -> !dist.dist<i32>
  %5 = dist.measure %4 : !dist.dist<i32> -> i32
  func.return %5 : i32
}""")

    def test_dropping_a_distribution_or_adding_it_to_itself_is_refused(self):
        for source, message in (("a = 0..4; a = 5; return a;", "would drop"),
                                ("a = 0..4; a += a; return a;", "to itself"),
                                ("a = 0..4; b = a; b += a; return b;", "to itself")):
            with self.assertRaisesRegex(TranslationError, message):
                translated(source, "v2")

if __name__ == "__main__":
    unittest.main()
