import unittest

from checking.simulate import simulate_isa
from compiler.parser.instructions import Const, IDist, IOp, Reg
from experiments.distributions import experiment
from experiments.distributions.programs import PROGRAMS

UNOPTIMISED = ["collapse_updates", "fold_constants", "drop_unread", "fuse_steps", "drop_unread_versions"]
SOURCES = {p["id"]: p["source"] for p in PROGRAMS}
SHARED = "a = 0..3;\np = 0..2;\nk = 2;\nx = a + k;\ny = p + k;\nreturn x;\n"


def run(source, skip=(), pass_on=True):
    program = experiment.compiler().run(source, until="release_values" if pass_on else "lower_to_isa", skip=list(skip)).output
    held, state = simulate_isa(program, experiment.SYSTEM).state
    return program, set(held), simulate_isa(program, experiment.SYSTEM).value


class ReleaseTestCase(unittest.TestCase):
    def test_a_register_nothing_is_entangled_with_is_released_by_undoing_it_and_the_value_is_unchanged(self):
        program, held, value = run(SHARED, UNOPTIMISED)
        self.assertEqual({"a#1", "x#1"}, held)  # y by reversal, k reset, p by running its preparation backwards
        self.assertTrue(any(isinstance(i, IDist) and i.dst == "p#1" for i in program.body))
        self.assertEqual(value, run(SHARED, UNOPTIMISED, pass_on=False)[2])

    def test_what_the_result_needs_stays_held_and_the_rest_of_a_chain_goes(self):
        _, held, value = run(SOURCES["arithmetic"], ["collapse_updates"])
        self.assertEqual({"a#1", "c#1"}, held)  # c = b * 2 cannot be run backwards, so a stays; b is reversed
        _, held, _ = run(SOURCES["mixed"], UNOPTIMISED)
        self.assertEqual({"a#1", "d#1"}, held)  # the constants k, m, c and the temporary b are all released
        self.assertEqual(value, run(SOURCES["arithmetic"], ["collapse_updates"], pass_on=False)[2])

    def test_an_input_is_computed_back_from_the_result_when_that_step_can_be_run_backwards(self):
        program, held, value = run("a = 0..4;\nc = a * 3;\nreturn c;\n")
        self.assertEqual({"c#1"}, held)
        self.assertIn(IOp("mul", "a#1", (Reg("c#1"), Const(-5))), program.body)  # 3 * 11 = 1 modulo 16, and 11 is -5
        self.assertEqual(value, run("a = 0..4;\nc = a * 3;\nreturn c;\n", pass_on=False)[2])

    def test_the_default_pipeline_leaves_one_register_where_steps_combine_and_two_where_they_do_not(self):
        self.assertEqual({"u#1"}, run(SOURCES["fused"])[1])  # 3 * 5 combine, 15 is odd, so a is computed back from u
        self.assertEqual({"a#1", "u#1"}, run(SOURCES["temporary"])[1])  # t is reversed; a stays

    def test_a_register_computed_before_a_region_is_released_after_it_and_one_written_inside_is_undone_under_its_control(self):
        arms = SOURCES["branch-arms"]  # $t1 = a < 2 is computed before the branch and only read as its control
        _, held, value = run(arms)
        self.assertEqual({"a#1", "x#2"}, held)
        self.assertEqual(value, run(arms, pass_on=False)[2])
        program, held, _ = run(SOURCES["branch-temporary"])
        self.assertTrue(any(isinstance(i, IOp) and i.opcode == "mul" and i.dst == "t#1" for i in program.body))
        self.assertEqual({"a#1", "x#1"}, held)  # t is written under a control and undone under the same control


    def test_a_guard_bit_shared_by_two_conditionals_is_undone_after_the_last_of_them(self):
        source = "a = 0..4;\nx = 1;\ny = 1;\nif a {\n    x = 5;\n}\nif a {\n    y = 5;\n}\nreturn x + y;\n"
        _, held, _ = run(source)
        self.assertEqual([], [name for name in held if name.startswith("$")])  # the bit made for the guard is released

if __name__ == '__main__':
    unittest.main()
