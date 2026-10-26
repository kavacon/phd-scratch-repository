import subprocess
import unittest

from compiler.parser.parser import parse
from experiment_existing_ir import simulate
from experiment_existing_ir.translate import translate

ARITHMETIC = "x = 5; y = x + 3; y = y - 1; return y;"
CONDITIONAL = "a = 0..4; x = 2; if a < 2 { x += 3; } else { x -= 1; } return x;"


def program(source: str, grammar: str = "v1", **options) -> str:
    return translate(parse(source, grammar), **options)


def registers(qc: str) -> int:
    return qc.count("memref.alloc")


@unittest.skipUnless(simulate.built(), "existing-ir-opt is not built: run helper/cpp/setup.sh and cpp/build.sh")
class Pipeline(unittest.TestCase):
    def test_a_definite_program_returns_its_value_with_and_without_folding(self):
        for passes in (("--lower-to-qc",), ("--canonicalize", "--lower-to-qc")):
            self.assertEqual(simulate.run(program(ARITHMETIC), passes), {7: 2000}, passes)

    def test_folding_the_source_first_removes_the_arithmetic_from_the_circuit(self):
        plain = simulate.optimise(program(ARITHMETIC), "--lower-to-qc")
        folded = simulate.optimise(program(ARITHMETIC), "--canonicalize", "--lower-to-qc")
        self.assertEqual((registers(plain), "qc.call" in plain), (9, True))
        self.assertEqual((registers(folded), "qc.call" in folded), (1, False))  # 7 is 0111, in one register

    def test_arithmetic_wraps_modulo_the_type_width(self):
        # 7 + 7 is 14, which is -2 in four bits of two's complement
        self.assertEqual(simulate.run(program("a = 7; b = a + a; return b;", width=4), shots=100), {-2: 100})

    def test_a_conditional_on_a_distribution_gives_each_outcome_with_its_probability(self):
        # a is 0, 1, 2 or 3 with equal probability, so x is 5 when a < 2 and 1 otherwise: each half the time
        counts = simulate.run(program(CONDITIONAL, "v2"))
        self.assertEqual(set(counts), {1, 5})
        self.assertTrue(all(800 < count < 1200 for count in counts.values()), counts)

    def test_an_in_place_update_shifts_every_outcome(self):
        # a is 0, 1, 2 or 3, each a quarter of the time; the first result fits a's register, the second needs a wider one
        for source, outcomes in (("a = 0..4; a -= 1; return a;", {-1, 0, 1, 2}), ("a = 0..4; a += 3; return a;", {3, 4, 5, 6})):
            counts = simulate.run(program(source, "v2"))
            self.assertEqual(set(counts), outcomes, source)
            self.assertTrue(all(400 < count < 600 for count in counts.values()), (source, counts))

    def test_updates_reuse_the_distribution_s_register_however_far_they_widen_it(self):
        # three constants, the distribution, and for each update only a temporary and a carry: no fresh result register
        source = program("a = 0..4; a += 3; a += 5; a -= 20; return a;", "v2")
        lowered = simulate.optimise(source, "--canonicalize", "--lower-to-qc")
        self.assertEqual(registers(lowered), 3 + 1 + 3 * 2)
        self.assertEqual(set(simulate.run(source)), {-12, -11, -10, -9})  # 0..3 + 3 + 5 - 20

    def test_adding_a_distribution_in_place_gives_the_distribution_of_the_sum(self):
        # a is 0..3 and p is 0..1 independently, so a + p is 0 to 4 with weights 1, 2, 2, 2, 1 out of 8
        counts = simulate.run(program("a = 0..4; p = 0..2; a += p; return a;", "v2"), shots=4000)
        self.assertEqual(set(counts), {0, 1, 2, 3, 4})
        for value, weight in zip(range(5), (1, 2, 2, 2, 1)):
            self.assertAlmostEqual(counts[value] / 4000, weight / 8, delta=0.04, msg=value)

    def test_a_conditional_on_a_definite_value_lowers_only_once_it_has_become_a_select(self):
        source = "x = 2; if x < 3 { x += 5; } else { x -= 1; } return x;"
        self.assertEqual(simulate.run(program(source)), {7: 2000})  # canonicalize folds the arms and the select
        with self.assertRaises(subprocess.CalledProcessError):  # a condition held in qubits cannot be branched on
            simulate.optimise(program(source), "--lower-to-qc")

if __name__ == "__main__":
    unittest.main()
