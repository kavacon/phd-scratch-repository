import unittest

from checking.simulate import SimulationError, simulate_isa
from compiler.parser.grammar import If
from compiler.parser.instructions import Alloc, Const, Dist, Free, IOp, IsaProgram, Op, Reg
from compiler.passes.isa import lower_to_isa
from compiler.printer import unparse, unparse_isa
from experiments.distributions import experiment
from experiments.distributions.programs import PROGRAMS


def stage(source, name):
    return experiment.compiler().run(source, until=name).output


class FoldConstantsTestCase(unittest.TestCase):
    def test_a_chain_of_constants_is_folded_into_its_use_and_the_unread_values_are_dropped(self):
        source = next(p["source"] for p in PROGRAMS if p["id"] == "mixed")
        self.assertEqual("a#1 = 0..3;\na#2 = a#1 + 1;  // in place\na#3 = a#2 + 3;  // in place\nreturn a#3;",
                         unparse(stage(source, "drop_unread")))

    def test_arithmetic_wraps_and_a_known_condition_picks_its_branch(self):
        self.assertIn("+ -8", unparse(stage("x = 7;\ny = x + 1;\nz = 0..2;\nw = z + y;\nreturn w;\n", "fold_constants")))
        folded = stage("c = 1;\nx = 5;\nif c {\n    x = 6;\n}\nreturn x;\n", "drop_unread")
        self.assertFalse(any(isinstance(s, If) for s in folded))
        self.assertEqual(6, simulate_isa(lower_to_isa(folded), experiment.SYSTEM).value)

    def test_a_value_nothing_reads_is_dropped_even_when_it_is_not_a_constant(self):
        self.assertEqual("p#1 = 0..4;\nreturn p#1;", unparse(stage("p = 0..4;\nq = p * 3;\nreturn p;\n", "drop_unread")))


class CollapseUpdatesTestCase(unittest.TestCase):
    def test_an_add_whose_source_is_not_read_again_updates_it_in_place_and_renames_the_later_uses(self):
        self.assertEqual("a = 0..4;\na += 3;\nc = a * 2;\nreturn c;",
                         unparse(stage("a = 0..4;\nb = a + 3;\nc = b * 2;\nreturn c;\n", "collapse_updates")))
        self.assertEqual(1, unparse_isa(stage("a = 0..4;\nb = a + 3;\nreturn b;\n", "lower_to_isa")).count("ALLOC"))

    def test_it_is_left_alone_while_the_source_is_read_or_either_name_is_assigned_again(self):
        read_again = unparse(stage("a = 0..4;\nb = a + 3;\nc = a + b;\nreturn c;\n", "collapse_updates"))
        self.assertEqual("a = 0..4;\nb = a + 3;\na += b;\nreturn a;", read_again)  # b stays; c = a + b is the last read of a
        assigned_again = unparse(stage("a = 0..4;\nb = a + 3;\na = 7;\nc = a + b;\nreturn c;\n", "collapse_updates"))
        self.assertIn("b = a + 3;", assigned_again)
        for source in ("a = 0..4;\nb = a + 3;\nc = b * 2;\nreturn c;\n", "a = 0..4;\nb = a + 3;\na = 7;\nc = a + b;\nreturn c;\n"):
            with_pass = simulate_isa(experiment.compiler().run(source, until="allocate").output, experiment.SYSTEM).value
            without = simulate_isa(experiment.compiler().run(
                source, until="allocate", skip=["collapse_updates"]).output, experiment.SYSTEM).value
            self.assertEqual(without, with_pass)


class ShiftTestCase(unittest.TestCase):
    def test_a_multiplication_by_two_becomes_a_shift_with_the_same_result(self):
        source = next(p["source"] for p in PROGRAMS if p["id"] == "arithmetic")
        shifted = stage(source, "shift_multiplications")
        self.assertTrue(any(isinstance(i, Op) and i.opcode == "shl" for i in shifted.body))
        plain = experiment.compiler().run(source, until="allocate", skip=["shift_multiplications"]).output
        self.assertEqual(simulate_isa(plain, experiment.SYSTEM).value, simulate_isa(
            experiment.compiler().run(source, until="allocate").output, experiment.SYSTEM).value)

    def test_a_right_shift_releases_the_source_only_when_the_dropped_bit_was_zero(self):
        def release(start, stop):
            return IsaProgram((Alloc("b"), Dist("b", start, stop), Alloc("c"), Op("shl", "c", (Reg("b"), Const(1))),
                               IOp("shr", "b", (Reg("c"), Const(1))), Free("b")))
        simulate_isa(release(0, 8), experiment.SYSTEM)
        with self.assertRaises(SimulationError):
            simulate_isa(release(-2, 2), experiment.SYSTEM)


if __name__ == '__main__':
    unittest.main()
