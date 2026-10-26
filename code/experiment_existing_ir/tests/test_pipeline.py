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

    def test_a_conditional_is_refused_by_the_lowering_until_it_becomes_a_select(self):
        # canonicalize folds the arms and turns the scf.if into arith.select; without it there is no branch on a qubit
        with self.assertRaises(Exception):
            simulate.optimise(program(CONDITIONAL, "v2"), "--lower-to-qc")


if __name__ == "__main__":
    unittest.main()
