import unittest

from experiments.encodings import experiment
from experiments.encodings.programs import PROGRAMS
from experiments.encodings.readout import Readout, decode, readout_of
from compiler.passes.qasm import openqasm_text
from compiler.system import System

BINARY = ("binary", "binary-ttk")


def rows(source, width=4, encodings=BINARY):
    return experiment.run(source, encodings, System(width, 100))


class AgreementTestCase(unittest.TestCase):
    def test_both_binary_encodings_return_what_the_reference_returns_for_every_program(self):
        for program in PROGRAMS:
            with self.subTest(program=program["id"]):
                result = rows(program["source"])
                self.assertEqual(4, len(result))  # two encodings, each with the lowering's circuit and the optimised one
                for row in result:
                    self.assertTrue(row.agrees, row)

    def test_a_narrower_register_wraps_the_same_way_in_the_reference_and_the_circuits(self):
        for program in PROGRAMS[:4]:
            with self.subTest(program=program["id"]):
                for row in rows(program["source"], width=3):
                    self.assertTrue(row.agrees, row)

    def test_a_row_is_made_for_each_circuit_stage_only(self):
        result = rows("a = 3; return a;", encodings=["binary-ttk"])
        self.assertEqual(["to_circuit_binary_ttk", "optimise"], [r.stage for r in result])


class CostTestCase(unittest.TestCase):
    """The ancilla-free adder saves a qubit exactly when the program adds registers and never compares."""

    def touched(self, source, name):
        return {r.encoding: r.qubits_touched for r in rows(source) if r.stage != "optimise"}[name]

    def test_it_saves_a_qubit_in_a_program_that_adds_registers_and_does_not_compare(self):
        source = "a = 3; b = 2; c = a + b; return c;"
        self.assertEqual(self.touched(source, "binary") - 1, self.touched(source, "binary-ttk"))

    def test_it_saves_nothing_when_a_comparison_borrows_the_scratch_qubit(self):
        source = "a = 3; b = 2; c = a < b; return c;"
        self.assertEqual(self.touched(source, "binary"), self.touched(source, "binary-ttk"))

    def test_the_two_agree_exactly_when_there_is_no_register_addition(self):
        source = "a = 7; b = a + 1; return b;"  # constants only
        by = {(r.encoding, r.stage): (r.gates, r.depth, r.qubits_touched) for r in rows(source)}
        self.assertEqual(by["binary", "optimise"], by["binary-ttk", "optimise"])


class GateCountsTestCase(unittest.TestCase):
    def test_the_gate_names_are_counted_without_their_open_control_variants(self):
        for row in rows(PROGRAMS[5]["source"]):
            self.assertFalse([name for name in row.ops if "_o" in name], row.ops)


class ReadoutTestCase(unittest.TestCase):
    def test_the_shared_binary_lowering_has_no_recipe_and_is_read_as_signed_bits_in_one_shot(self):
        compilation = experiment.compile_all("a = 3; return a;", ["binary"], System(4, 100))["binary"]
        self.assertEqual(Readout(1, "signed-bits"), readout_of(compilation["to_openqasm"]))

    def test_the_local_lowering_attaches_its_recipe_and_it_survives_the_optimiser(self):
        compilation = experiment.compile_all("a = 3; return a;", ["binary-ttk"], System(4, 100))["binary-ttk"]
        self.assertEqual({"shots": 1, "decode": "signed-bits"}, compilation["to_circuit_binary_ttk"].metadata["readout"])
        self.assertEqual({"shots": 1, "decode": "signed-bits"}, compilation["optimise"].metadata["readout"])

    def test_decoding_reads_two_s_complement_and_insists_on_one_outcome_for_a_one_shot_read(self):
        self.assertEqual(-2, decode(Readout(1, "signed-bits"), {14: 1}, 4))
        self.assertEqual(5, decode(Readout(1, "signed-bits"), {5: 1}, 4))
        with self.assertRaises(ValueError):
            decode(Readout(1, "signed-bits"), {1: 3, 2: 2}, 4)
        with self.assertRaises(ValueError):
            decode(Readout(1, "nonsense"), {1: 1}, 4)


class CircuitsTestCase(unittest.TestCase):
    def test_the_two_circuits_differ_only_where_a_register_is_added(self):
        compare = "a = 3; b = 2; c = a < b; return c;"
        shared, local = (experiment.compile_all(compare, [name], System(4, 100))[name].stages for name in BINARY)
        self.assertEqual(openqasm_text(shared[-2].output), openqasm_text(local[-2].output))
        add = "a = 3; b = 2; c = a + b; return c;"
        shared, local = (experiment.compile_all(add, [name], System(4, 100))[name].stages for name in BINARY)
        self.assertNotEqual(openqasm_text(shared[-2].output), openqasm_text(local[-2].output))


if __name__ == '__main__':
    unittest.main()
