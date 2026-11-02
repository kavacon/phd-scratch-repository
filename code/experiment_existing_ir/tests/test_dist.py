import subprocess
import unittest

from compiler.parser.parser import parse
from experiment_existing_ir import simulate
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
    return simulate.optimise(translate(parse(source, "v2")), *simulate.PREPARE, *simulate.RESOLVE, "--dist-status")


@unittest.skipUnless(simulate.built(), "existing-ir-opt is not built: run helper/cpp/setup.sh and cpp/build.sh")
class Distributions(unittest.TestCase):
    def test_a_distribution_may_be_read_again_but_not_after_it_was_updated_in_place(self):
        simulate.optimise(READ_TWICE)
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            simulate.optimise(READ_AFTER_UPDATE)
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


if __name__ == "__main__":
    unittest.main()
