import unittest

from experiments.encodings import experiment
from experiments.encodings.programs import PROGRAMS
from experiments.encodings.readout import Readout, readout_of
from compiler.system import System

BINARY = ("binary", "binary-ttk")


def rows(source, width=4, encodings=BINARY):
    return experiment.run(source, encodings, System(width, 100), zx=False)  # PyZX has its own test


class BinaryEncodingsTestCase(unittest.TestCase):
    def test_both_return_what_the_reference_returns_for_every_program(self):
        for program in PROGRAMS:
            with self.subTest(program=program["id"]):
                result = rows(program["source"])
                self.assertEqual(4, len(result))  # two encodings, each with the lowering's circuit and the optimised one
                self.assertTrue(all(row.agrees for row in result), result)

    def test_the_ancilla_free_adder_saves_a_qubit_when_registers_are_added_and_not_compared(self):
        touched = {r.encoding: r.qubits_touched for r in rows("a = 3; b = 2; c = a + b; return c;") if r.stage != "optimise"}
        self.assertEqual(touched["binary"] - 1, touched["binary-ttk"])

    def test_the_readout_recipe_travels_with_the_circuit_through_the_optimiser(self):
        compilation = experiment.compile_all("a = 3; return a;", ["binary-ttk"], System(4, 100), zx=False)["binary-ttk"]
        self.assertEqual(Readout(1, "signed-bits"), readout_of(compilation["optimise"]))


if __name__ == '__main__':
    unittest.main()
