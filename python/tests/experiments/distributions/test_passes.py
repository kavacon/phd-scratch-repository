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
        self.assertEqual("a = 0..3;\nb = a + 1;\nd = b + 3;\nreturn d;", unparse(stage(source, "drop_unread")))

    def test_arithmetic_is_kept_exact_a_comparison_reads_the_register_width_and_a_known_condition_picks_its_branch(self):
        self.assertIn("+ 8", unparse(stage("x = 7;\ny = x + 1;\nz = 0..2;\nw = z + y;\nreturn w;\n", "fold_constants")))
        self.assertIn("c = true;", unparse(stage("x = 7;\ny = x + 1;\nc = y < 0;\nreturn c;\n", "fold_constants")))  # 8 is -8 at 4 bits
        source = "c = 1;\nx = 5;\nif c {\n    x = 6;\n}\nreturn x;\n"
        self.assertFalse(any(isinstance(s, If) for s in stage(source, "drop_unread")))
        self.assertEqual(6, simulate_isa(stage(source, "lower_to_isa"), experiment.SYSTEM).value)

    def test_a_value_is_forgotten_when_it_is_assigned_something_unknown_or_in_a_branch_that_may_not_run(self):
        self.assertIn("y = x + 1;", unparse(stage("x = 3;\nx = 0..2;\ny = x + 1;\nreturn y;\n", "fold_constants")))
        branch = unparse(stage("a = 0..2;\nx = 3;\nif a {\n    x = 4;\n}\ny = x + 1;\nreturn y;\n", "fold_constants"))
        self.assertIn("y = x + 1;", branch)  # x is 3 or 4 afterwards, so it is not substituted

    def test_a_value_nothing_reads_is_dropped_even_when_it_is_not_a_constant_or_is_overwritten_first(self):
        self.assertEqual("p = 0..4;\nreturn p;", unparse(stage("p = 0..4;\nq = p * 3;\nreturn p;\n", "drop_unread")))
        self.assertEqual("x = 6;\nreturn x;", unparse(stage("x = 5;\nx = 6;\nreturn x;\n", "drop_unread")))


class IdentityTestCase(unittest.TestCase):
    SOURCE = "a = 0..3;\nx = 1;\nif a {\n    x = a + 5;\n}\nreturn x;\n"

    def test_an_identity_is_not_folded_and_lowers_to_no_operation_with_the_other_arm_clearing_the_old_value(self):
        self.assertIn("x = x;", unparse(stage(self.SOURCE, "fold_constants")))
        text = unparse_isa(stage(self.SOURCE, "drop_noops"))
        self.assertNotIn("COPY  x#2", text)  # no copy of the old version into the new one, and one register for x
        self.assertIn("ICOPY x#1, 1", text)  # the arm that assigns clears the 1 first, under its control
        with_alias = simulate_isa(stage(self.SOURCE, "drop_noops"), experiment.SYSTEM)
        without = simulate_isa(experiment.compiler().run(self.SOURCE, until="release_values", skip=["alias_identities"]).output,
                               experiment.SYSTEM)
        self.assertEqual(without.value, with_alias.value)

    def test_two_identity_arms_of_a_nested_conditional_leave_the_value_unchanged_on_both_paths(self):
        source = "a = 0..2;\nb = 0..2;\nx = 1;\nif a {\n    if b {\n        x = 3;\n    }\n}\nreturn x;\n"
        skip = ["flatten_conditionals"]
        program = experiment.compiler().run(source, until="drop_noops", skip=skip).output
        self.assertEqual({1: 0.75, 3: 0.25}, simulate_isa(program, experiment.SYSTEM).value)  # x is 1 unless a and b are 1
        self.assertEqual(1, unparse_isa(program).count("ALLOC x"))

    def test_a_load_of_zero_and_the_region_it_leaves_empty_are_dropped(self):
        source = "a = 0..2;\nb = 0;\nif a {\n    b = 1;\n}\nreturn b;\n"
        self.assertNotIn("ICTRL", unparse_isa(stage(source, "drop_noops")))
        self.assertNotIn("COPY  b#1, 0", unparse_isa(stage(source, "drop_noops")))


class ConditionBitsTestCase(unittest.TestCase):
    def test_a_guard_that_is_not_a_boolean_is_raised_into_one_and_a_boolean_guard_is_left_unless_the_conditional_reads_it(self):
        self.assertIn("$b1 = a != 0;\nif $b1 {", unparse(stage("a = 0..4;\nx = 1;\nif a {\n    x = 5;\n}\nreturn x;\n", "condition_bits")))
        comparison = "a = 0..4;\nc = a < 2;\nx = 1;\nif c {\n    x = 5;\n}\nreturn x;\n"
        self.assertNotIn("$b", unparse(stage(comparison, "condition_bits")))
        reads_it = unparse(stage("a = 0..4;\nc = a < 2;\nx = 1;\nif c {\n    x = c + 5;\n}\nreturn x;\n", "condition_bits"))
        self.assertIn("$b1 = c && true;", reads_it)

    def test_the_variable_is_shared_by_conditionals_on_the_same_guard_until_the_guard_is_assigned_again(self):
        source = "a = 0..4;\nx = 1;\ny = 1;\nif a {\n    x = 5;\n}\nif a {\n    y = 5;\n}\na = 0..2;\nif a {\n    x = 7;\n}\nreturn x + y;\n"
        text = unparse(stage(source, "condition_bits"))
        self.assertEqual(2, text.count("!= 0;"))  # one for the two conditionals before a changes, one after


class FuseStepsTestCase(unittest.TestCase):
    def test_two_steps_of_the_same_kind_through_an_intermediate_nothing_else_reads_become_one(self):
        self.assertEqual("a = 0..4;\nu = a * 15;\nreturn u;",  # exact: 15 wraps when the program is lowered, not here
                         unparse(stage("a = 0..4;\nt = a * 3;\nu = t * 5;\nreturn u;\n", "fuse_steps")))
        self.assertEqual("a = 0..3;\nd = a + 11;\nreturn d;",
                         unparse(stage("a = 0..3;\nb = a + 6;\nd = b + 5;\nreturn d;\n", "fuse_steps")))
        self.assertEqual("a = 0..3;\nd = a + 4;\nreturn d;",
                         unparse(stage("a = 0..3;\nb = a + 1;\nd = b - -3;\nreturn d;\n", "fuse_steps")))

    def test_the_intermediate_is_kept_when_it_is_read_again_and_a_multiply_and_an_add_do_not_combine(self):
        read_twice = "a = 0..4;\nt = a * 3;\nu = t * 5;\nv = t + u;\nreturn v;\n"
        self.assertEqual(unparse(stage(read_twice, "drop_unread")), unparse(stage(read_twice, "fuse_steps")))
        mixed_kinds = "a = 0..4;\nt = a * 3;\nu = t + 5;\nreturn u;\n"
        self.assertEqual(unparse(stage(mixed_kinds, "drop_unread")), unparse(stage(mixed_kinds, "fuse_steps")))


class CollapseUpdatesTestCase(unittest.TestCase):
    def test_an_add_whose_source_is_not_read_again_updates_it_in_place_and_renames_the_later_uses(self):
        self.assertEqual("a = 0..4;\na += 3;\nc = a * 2;\nreturn c;",
                         unparse(stage("a = 0..4;\nb = a + 3;\nc = b * 2;\nreturn c;\n", "collapse_updates")))
        self.assertEqual(1, unparse_isa(stage("a = 0..4;\nb = a + 3;\nreturn b;\n", "lower_to_isa")).count("ALLOC"))

    def test_it_is_left_alone_while_the_source_is_read_or_either_name_is_assigned_again(self):
        read_again = unparse(stage("a = 0..4;\nb = a + 3;\nc = a + b;\nreturn c;\n", "collapse_updates"))
        self.assertEqual("a = 0..4;\nb = a + 3;\na += b;\nreturn a;", read_again)  # b stays; c = a + b is the last read of a
        assigned_again = unparse(stage("a = 0..4;\nb = a + 3;\na = 0..2;\nc = a + b;\nreturn c;\n", "collapse_updates"))
        self.assertIn("b = a + 3;", assigned_again)
        for source in ("a = 0..4;\nb = a + 3;\nc = b * 2;\nreturn c;\n", "a = 0..4;\nb = a + 3;\na = 0..2;\nc = a + b;\nreturn c;\n"):
            with_pass = simulate_isa(experiment.compiler().run(source, until="allocate").output, experiment.SYSTEM).value
            without = simulate_isa(experiment.compiler().run(
                source, until="allocate", skip=["collapse_updates"]).output, experiment.SYSTEM).value
            self.assertEqual(without, with_pass)


class RenameTestCase(unittest.TestCase):
    def test_an_in_place_update_keeps_its_name_outside_a_branch_and_gets_a_version_inside_one(self):
        self.assertEqual("a#1 = 0..4;\na#1 += 3;\nreturn a#1;", unparse(stage("a = 0..4;\na += 3;\nreturn a;\n", "rename")))
        branch = unparse(experiment.compiler().run(  # without folding, which would settle x
            "a = 0..2;\nx = 1;\nif a {\n    x += 1;\n}\nreturn x;\n", until="rename", skip=["fold_constants", "drop_unread"]).output)
        self.assertIn("x#2 = x#1 + 1;  // in place", branch)  # both branches give x the version x#2
        self.assertIn("x#2 = x#1;", branch)


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
