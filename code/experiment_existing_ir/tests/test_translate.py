import unittest

from compiler.parser.parser import parse
from experiment_existing_ir import simulate
from experiment_existing_ir.translate import TranslationError, translate


def translated(source: str, grammar: str = "v1", **options) -> str:
    return translate(parse(source, grammar), **options)


def promoted(source: str, grammar: str = "v1") -> str:
    """The translation with its loops unrolled and its slots promoted to values, without inlining (which also tidies)."""
    return simulate.optimise(translated(source, grammar), simulate.UNROLL, simulate.PROMOTE).rstrip("\n")


def resolved(source: str, grammar: str = "v2") -> str:
    """The program once its references are resolved and its sums are done in place where they can be."""
    return simulate.optimise(translated(source, grammar), *simulate.PREPARE, *simulate.RESOLVE, "--canonicalize",
                             "--in-place-updates").rstrip("\n")


class Translate(unittest.TestCase):
    def test_translation_keeps_the_sources_variables_as_slots_that_are_loaded_and_stored(self):
        self.assertEqual(translated("x = 5; y = x + 3; y = y - 1; return y;"), """func.func @main() -> i32 {
  %0 = arith.constant 5 : i32
  %1 = memref.alloca() : memref<i32>
  memref.store %0, %1[] : memref<i32>
  %2 = memref.load %1[] : memref<i32>
  %3 = arith.constant 3 : i32
  %4 = arith.addi %2, %3 : i32
  %5 = memref.alloca() : memref<i32>
  memref.store %4, %5[] : memref<i32>
  %6 = memref.load %5[] : memref<i32>
  %7 = arith.constant 1 : i32
  %8 = arith.subi %6, %7 : i32
  memref.store %8, %5[] : memref<i32>
  %9 = memref.load %5[] : memref<i32>
  func.return %9 : i32
}""")

    def test_the_width_is_chosen_by_the_caller(self):
        self.assertIn("arith.constant 1 : i16", translated("x = 1; return x;", width=16))

    def test_a_loop_is_an_affine_for_that_reads_and_writes_the_outer_slots(self):
        self.assertEqual(translated("s = 0; for i in 0..4 { s += i; } return s;"), """func.func @main() -> i32 {
  %0 = arith.constant 0 : i32
  %1 = memref.alloca() : memref<i32>
  memref.store %0, %1[] : memref<i32>
  affine.for %2 = 0 to 4 {
    %3 = arith.index_cast %2 : index to i32
    %4 = memref.alloca() : memref<i32>
    memref.store %3, %4[] : memref<i32>
    %5 = memref.load %1[] : memref<i32>
    %6 = memref.load %4[] : memref<i32>
    %7 = arith.addi %5, %6 : i32
    memref.store %7, %1[] : memref<i32>
  }
  %8 = memref.load %1[] : memref<i32>
  func.return %8 : i32
}""")

    def test_constructs_not_yet_supported_are_refused(self):
        with self.assertRaisesRegex(TranslationError, "'\\*'"):
            translated("x = 2 * 3; return x;")
        with self.assertRaisesRegex(TranslationError, "does not fit in i8"):
            translated("x = 300; return x;", width=8)

    def test_a_loop_may_not_assign_its_variable_or_be_too_large_to_unroll(self):
        with self.assertRaisesRegex(TranslationError, "loop variable i"):
            translated("for i in 0..3 { i += 1; } return 0;")
        with self.assertRaisesRegex(TranslationError, "cannot be unrolled"):
            translated("s = 0; for i in 0..200 { for j in 0..100 { s += 1; } } return s;")

    def test_a_call_must_name_a_function_with_that_many_parameters_and_not_recurse(self):
        for source, message in (("return g(1);", "is not a function"),
                                ("function f(p) { return p; }\nreturn f(1, 2);", "takes 1 argument"),
                                ("function f(p) { return f(p); }\nreturn f(1);", "calls itself")):
            with self.assertRaisesRegex(TranslationError, message):
                translated(source, "v2")

    def test_a_variable_that_may_hold_a_distribution_holds_a_reference_and_a_copy_is_a_marker(self):
        # x may become a distribution, so it holds a reference to an object; `y = x` is a copy or an alias, which only
        # folding can tell, so the translation selects between the two on `dist.is_single` and decides nothing. A variable that never touches a
        # distribution stays a plain integer.
        text = translated("a = 0..4; x = 2; for i in 0..3 { x += a; } y = x; return y;", "v2")
        self.assertIn("memref<memref<!dist.dist<i32>>>", text)  # the slot of a name holds a reference to a cell
        self.assertEqual(text.count("dist.is_single"), 1)  # the copy-or-alias question of `y = x`
        self.assertNotIn("dist.", translated("x = 2; y = x; y += 1; return y;", "v2"))


@unittest.skipUnless(simulate.built(), "existing-ir-opt is not built: run helper/cpp/setup.sh and cpp/build.sh")
class Promoted(unittest.TestCase):
    """What the translation means, read after the slots have become values."""

    def test_definite_variables_become_the_values_they_hold(self):
        self.assertEqual(promoted("x = 5; y = x + 3; y = y - 1; return y;"), """module {
  func.func @main() -> i32 {
    %c5_i32 = arith.constant 5 : i32
    %c3_i32 = arith.constant 3 : i32
    %0 = arith.addi %c5_i32, %c3_i32 : i32
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.subi %0, %c1_i32 : i32
    return %1 : i32
  }
}""")

    def test_a_distribution_is_changed_by_an_in_place_update_and_reading_it_measures_it(self):
        self.assertEqual(resolved("a = 0..4; a += 3; return a;"), """module {
  func.func @main() -> i32 {
    %c3_i32 = arith.constant 3 : i32
    %0 = dist.uniform 0, 4 : !dist.dist<i32>
    %1 = dist.add_assign %0, %c3_i32 : !dist.dist<i32>, i32 -> !dist.dist<i32>
    %2 = dist.measure %1 : !dist.dist<i32> -> i32
    return %2 : i32
  }
}""")

    def test_a_distribution_can_be_read_by_several_conditions(self):
        text = resolved("a = 0..4; x = 0; y = 0; if a < 2 { x += 1; } if a < 3 { y += 2; } return x + y;")
        self.assertEqual(text.count("dist.slt %3,"), 2)  # both read the one distribution

    def test_a_conditional_on_a_distribution_computes_both_arms_and_selects_once(self):
        # x is 2 before the conditional, so its two arms are the integers 5 and 1, and one select ties x to a < 2
        self.assertEqual(resolved("a = 0..4; x = 2; if a < 2 { x += 3; } else { x -= 1; } return x;"), """module {
  func.func @main() -> i32 {
    %0 = dist.constant <1 : i32> : !dist.dist<i32>
    %1 = dist.constant <5 : i32> : !dist.dist<i32>
    %c2_i32 = arith.constant 2 : i32
    %2 = dist.uniform 0, 4 : !dist.dist<i32>
    %3 = dist.slt %2, %c2_i32 : !dist.dist<i32>, i32 -> !dist.dist<i1>
    %4 = dist.select %3, (%1), (%0) : !dist.dist<i1>, (!dist.dist<i32>), (!dist.dist<i32>) -> (!dist.dist<i32>)
    %5 = dist.measure %4 : !dist.dist<i32> -> i32
    return %5 : i32
  }
}""")

    def test_a_conditional_on_a_definite_value_yields_the_variables_either_arm_changed(self):
        self.assertEqual(promoted("x = 2; if x < 3 { x += 5; } else { x -= 1; } return x;"), """module {
  func.func @main() -> i32 {
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %0 = arith.cmpi slt, %c2_i32, %c3_i32 : i32
    %1 = scf.if %0 -> (i32) {
      %c5_i32 = arith.constant 5 : i32
      %2 = arith.addi %c2_i32, %c5_i32 : i32
      scf.yield %2 : i32
    } else {
      %c1_i32 = arith.constant 1 : i32
      %2 = arith.subi %c2_i32, %c1_i32 : i32
      scf.yield %2 : i32
    }
    return %1 : i32
  }
}""")

    def test_a_plain_name_is_an_alias_so_an_update_through_one_name_is_seen_through_the_other(self):
        self.assertEqual(resolved("a = 0..4; b = a; b += 1; return a;"), """module {
  func.func @main() -> i32 {
    %c1_i32 = arith.constant 1 : i32
    %0 = dist.uniform 0, 4 : !dist.dist<i32>
    %1 = dist.add_assign %0, %c1_i32 : !dist.dist<i32>, i32 -> !dist.dist<i32>
    %2 = dist.measure %1 : !dist.dist<i32> -> i32
    return %2 : i32
  }
}""")

    def test_a_computed_value_assigned_to_a_new_variable_is_marked_as_an_instance(self):
        self.assertEqual(resolved("a = 0..4; c = a + 1; return c;"), """module {
  func.func @main() -> i32 {
    %c1_i32 = arith.constant 1 : i32
    %0 = dist.uniform 0, 4 : !dist.dist<i32>
    %1 = dist.add_assign %0, %c1_i32 : !dist.dist<i32>, i32 -> !dist.dist<i32>
    %2 = dist.instance %1 : !dist.dist<i32> -> !dist.dist<i32>
    %3 = dist.measure %2 : !dist.dist<i32> -> i32
    return %3 : i32
  }
}""")

    def test_a_loop_that_runs_no_times_disappears_and_the_others_are_unrolled(self):
        self.assertNotIn("affine", promoted("s = 5; for i in 2..2 { s += 1; } return s;"))
        self.assertNotIn("affine.for", promoted("s = 0; for i in 0..4 { s += i; } return s;"))


if __name__ == "__main__":
    unittest.main()
