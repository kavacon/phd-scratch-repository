import subprocess
import unittest

from compiler.parser.parser import parse
from compiler.system import System
from experiment_existing_ir import metrics, pipeline
from experiment_existing_ir.translate import translate
from tests.golden import TOLERANCE, distance

SHOTS = 4000
EXAMPLE = "a = 0..4; a += 3; return a;"


def law(source, stages):
    return pipeline.execute(pipeline.export(pipeline.optimise(translate(parse(source, "v2")), *stages), "openqasm3"), shots=SHOTS)


@unittest.skipUnless(pipeline.built(), "existing-ir-opt is not built: run helper/cpp/setup.sh and cpp/build.sh")
class Increment(unittest.TestCase):
    def test_a_constant_can_be_added_in_four_qubits_by_carrying_through_the_register(self):
        program = translate(parse(EXAMPLE, "v2"))
        increment = pipeline.with_increment(pipeline.DEFAULT)
        with self.assertRaises(subprocess.CalledProcessError):  # the adder needs a copy of the constant and a carry: 9 qubits
            pipeline.optimise(pipeline.with_system(program, System(qubits=4)), *pipeline.DEFAULT)
        fitted = pipeline.optimise(pipeline.with_system(program, System(qubits=4)), *increment)
        self.assertEqual(metrics.of_qc(fitted).qubits, 4)
        self.assertLess(distance(law(EXAMPLE, increment), {3: 1/4, 4: 1/4, 5: 1/4, 6: 1/4}), TOLERANCE)

    def test_the_network_gives_the_adder_s_outcomes_for_additions_and_subtractions(self):
        for source in ("a = 0..4; a -= 2; return a;", "a = 0..8; a += 5; a -= 1; return a;", "a = 0..3; a += -4; return a;",
                       "a = 0..6; a += 13; return a;"):
            with self.subTest(source):
                expected = {k: v / SHOTS for k, v in law(source, pipeline.DEFAULT).items()}
                self.assertLess(distance(law(source, pipeline.with_increment(pipeline.DEFAULT)), expected), TOLERANCE)


if __name__ == "__main__":
    unittest.main()
