import unittest

from checking.simulate import SimulationError, simulate_isa
from compiler.parser.instructions import Alloc, Const, Dist, Free, IDist, IsaProgram, Update, inverse
from experiments.distributions import experiment


def allocated(source):
    return experiment.compiler().run(source, until="allocate").output


class DistributionInstructionTestCase(unittest.TestCase):
    def test_a_distribution_lowers_to_dist_and_runs_as_equal_weights(self):
        program = allocated("x = 0..4;\nreturn x;\n")
        self.assertTrue(any(isinstance(i, Dist) and (i.start, i.stop) == (0, 4) for i in program.body))
        self.assertEqual({0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}, simulate_isa(program, experiment.SYSTEM).value)

    def test_a_conditional_on_a_distribution_correlates_what_it_writes_with_it(self):
        program = allocated("a = 0..2;\nb = 3;\nif a {\n    b = 5;\n}\nreturn b;\n")
        result = simulate_isa(program, experiment.SYSTEM)
        self.assertEqual({3: 0.5, 5: 0.5}, result.value)
        self.assertEqual(2, len(result.state[1]))  # a and b are entangled: independent registers would give four entries

    def test_idist_undoes_dist_only_on_the_state_it_made(self):
        self.assertEqual(IDist("x", 0, 4), inverse(Dist("x", 0, 4)))
        simulate_isa(IsaProgram((Alloc("x"), Dist("x", 0, 4), IDist("x", 0, 4), Free("x"))))
        with self.assertRaises(SimulationError):
            simulate_isa(IsaProgram((Alloc("x"), Dist("x", 0, 4), Update("add", "x", Const(1)), IDist("x", 0, 4))))


if __name__ == '__main__':
    unittest.main()
