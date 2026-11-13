import re
import subprocess
import unittest
from pathlib import Path

from compiler.parser.parser import parse
from compiler.system import System
from experiment_existing_ir import constant_add, metrics, pipeline
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
        # (7, 15 and 31 are written in signed digits: 8 - 1, 16 - 1, 32 - 1, and 13 is 16 - 4 + 1)
        for source in ("a = 0..4; a -= 2; return a;", "a = 0..8; a += 5; a -= 1; return a;", "a = 0..3; a += -4; return a;",
                       "a = 0..6; a += 13; return a;", "a = 0..2; a += 7; return a;", "a = 0..2; a += 15; return a;", "a = 0..3; a -= 7; return a;"):
            with self.subTest(source):
                expected = {k: v / SHOTS for k, v in law(source, pipeline.DEFAULT).items()}
                self.assertLess(distance(law(source, pipeline.with_increment(pipeline.DEFAULT)), expected), TOLERANCE)

    def test_the_automatic_choice_follows_the_machine_and_the_cost(self):
        auto = pipeline.with_auto_constant_add(pipeline.DEFAULT)
        small, dense = translate(parse(EXAMPLE, "v2")), translate(parse("a = 0..2; a += 63; return a;", "v2"))
        # a small constant with room to spare: the network, which is no worse on two-qubit gates and on depth
        self.assertLess(metrics.of_qc(pipeline.optimise(small, *auto)).qubits, metrics.of_qc(pipeline.optimise(small, *pipeline.DEFAULT)).qubits)
        # a dense constant on a wide register: the adder is cheaper, so the circuit is the adder's
        self.assertEqual(pipeline.optimise(dense, *auto), pipeline.optimise(dense, *pipeline.DEFAULT))
        # the same program on a machine that holds only the network: the network, though the adder is cheaper
        with self.assertRaises(subprocess.CalledProcessError):
            pipeline.optimise(pipeline.with_system(dense, System(qubits=9)), *pipeline.DEFAULT)
        self.assertEqual(metrics.of_qc(pipeline.optimise(pipeline.with_system(dense, System(qubits=9)), *auto)).qubits, 8)

    def test_the_costs_the_compiler_uses_are_the_ones_the_study_measures(self):
        header = (Path(__file__).resolve().parents[1] / "cpp" / "ConstantAdd.h").read_text()
        two = [int(x) for x in re.search(r"TWO_QUBIT\[\] = \{([^}]*)\}", header).group(1).split(",")]
        depth = [int(x) for x in re.search(r"DEPTH\[\] = \{([^}]*)\}", header).group(1).split(",")]
        for controls, (t, d) in enumerate(zip(two, depth)):
            self.assertEqual((t, d), (constant_add.table(controls).two_qubit, constant_add.table(controls).depth), controls)


if __name__ == "__main__":
    unittest.main()
