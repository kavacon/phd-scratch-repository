import subprocess
import unittest

from compiler.parser.parser import parse
from experiment_existing_ir import pipeline
from experiment_existing_ir.translate import translate

HEADER = """  %0 = dist.uniform 0, 4 : !dist.dist<i32>
  %1 = arith.constant 1 : i32
"""
FOOTER = """  %9 = dist.measure %3 : !dist.dist<i32> -> i32
  func.return %9 : i32
}"""
# a distribution read twice, each time leaving it as it was: allowed
READ_TWICE = "func.func @main() -> i32 {\n" + HEADER + """  %2 = dist.slt %0, %1 : !dist.dist<i32>, i32 -> !dist.dist<i1>
  %4 = dist.slt %0, %1 : !dist.dist<i32>, i32 -> !dist.dist<i1>
  %3 = dist.add_assign %0, %1 : !dist.dist<i32>, i32 -> !dist.dist<i32>
""" + FOOTER
# a distribution read after it was updated in place: the update consumed it
READ_AFTER_UPDATE = "func.func @main() -> i32 {\n" + HEADER + """  %3 = dist.add_assign %0, %1 : !dist.dist<i32>, i32 -> !dist.dist<i32>
  %2 = dist.slt %0, %1 : !dist.dist<i32>, i32 -> !dist.dist<i1>
""" + FOOTER


def status(source: str) -> str:
    return pipeline.optimise(translate(parse(source, "v2")), *pipeline.PREPARE, *pipeline.RESOLVE, "--dist-status")


@unittest.skipUnless(pipeline.built(), "existing-ir-opt is not built: run helper/cpp/setup.sh and cpp/build.sh")
class Distributions(unittest.TestCase):
    def test_a_distribution_may_be_read_again_but_not_after_it_was_updated_in_place(self):
        pipeline.optimise(READ_TWICE)
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            pipeline.optimise(READ_AFTER_UPDATE)
        self.assertIn("uses a distribution after it was updated in place", caught.exception.stderr)

    def test_independent_distributions_form_separate_groups_until_an_update_joins_them(self):
        separate = status("a = 0..4; p = 0..2; a += 1; p += 1; return a;")
        self.assertIn("dist.groups = 1 : i64", separate)  # p is never read, so its updates are removed with it
        self.assertIn("dist.largest_group = 2 : i64", separate)  # a and its update
        two = status("a = 0..4; p = 0..2; a += 1; p += 1; return a - a;")
        self.assertIn("dist.groups = 1 : i64", two)
        joined = status("a = 0..4; p = 0..2; a += p; return a;")
        self.assertIn("dist.groups = 1 : i64", joined)
        self.assertIn("dist.largest_group = 3 : i64", joined)  # a, p and the updated a

    def test_a_definite_amount_does_not_join_a_group(self):
        self.assertIn("dist.groups = 1 : i64", status("a = 0..4; k = 7; a += k; return a;"))

    def test_a_comparison_the_range_decides_is_folded_with_its_conditional(self):
        # a is in 0..3, so a < 10 holds whatever the roll: the program is the constant 1, with no distribution left
        ir = pipeline.optimise(translate(parse("a = 0..4; x = 0; if a < 10 { x = 1; } return x;", "v2")), *pipeline.BEFORE_LOWERING)
        self.assertNotIn("dist.", ir)

    def test_a_chain_of_constant_steps_on_one_value_is_one_step(self):
        ir = pipeline.optimise(translate(parse("a = 0..4; y = a + 1; z = y + 1; w = z - 3; return w;", "v2")), *pipeline.BEFORE_LOWERING)
        self.assertEqual(ir.count("dist.sub_assign"), 1)  # a + 1 + 1 - 3 is a - 1
        self.assertNotIn("dist.add_assign", ir)

    def test_work_common_to_both_arms_of_a_conditional_is_done_once(self):
        ir = pipeline.optimise(translate(parse("a = 0..4; b = 0..3; x = 0; if a < 2 { x = b * b + 1; } else { x = b * b + 2; } return x;", "v2")),
                               *pipeline.BEFORE_LOWERING)
        self.assertEqual(ir.count("dist.muli"), 1)

    def test_conditionals_nested_around_the_same_value_become_one_condition(self):
        ir = pipeline.optimise(translate(parse("a = 0..4; b = 0..4; c = 0..4; x = 0; if a < 2 { if b < 2 { if c < 2 { x += 1; } } } return x;", "v2")),
                               *pipeline.BEFORE_LOWERING)
        self.assertEqual(ir.count("dist.select"), 1)
        self.assertEqual(ir.count("dist.andi"), 2)


if __name__ == "__main__":
    unittest.main()
