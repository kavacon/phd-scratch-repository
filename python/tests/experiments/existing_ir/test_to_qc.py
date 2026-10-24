import unittest

from xdsl.passes import PassPipeline
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.dead_code_elimination import DeadCodeElimination

from checking.simulate import simulate
from compiler.parser.parser import parse as parse_source
from experiments.existing_ir.ir import context
from experiments.existing_ir.ranges import Range, ranges
from experiments.existing_ir.rewrites.to_qc import LowerToQC
from experiments.existing_ir.simulate import run as simulated
from experiments.existing_ir.translate import translate

PROGRAMS = (
    "x = 5; y = x + 3; y = y - 1; return y;",
    "x = 2; y = x - 9; return y;",  # negative result
    "a = 100; b = a + a; c = b - 3; return c;",  # a result wider than its operands
    "x = 5; y = x - 5; return y;",  # a result narrower than its operands
    "x = 0; return x;",
)
STANDARD = (CanonicalizePass(), DeadCodeElimination())


def lowered(source: str, *passes, width: int = 32):
    module = translate(parse_source(source), width)
    PassPipeline((*passes, LowerToQC())).apply(context(), module)
    return module


def names(module) -> list:
    return [op.name for op in module.walk()]


class ToQC(unittest.TestCase):
    def test_the_lowered_program_gives_the_value_of_the_source_interpreter_with_and_without_the_standard_passes(self):
        for source in PROGRAMS:
            for passes in ((), STANDARD):
                self.assertEqual(simulated(lowered(source, *passes)), simulate(parse_source(source)).value,
                                 (source, passes))

    def test_arithmetic_wraps_modulo_the_type_width(self):
        # 7 + 7 is 14, which is -2 in four bits of two's complement
        self.assertEqual(simulated(lowered("a = 7; b = a + a; return b;", width=4)), -2)

    def test_optimising_the_source_first_removes_the_arithmetic_from_the_circuit(self):
        plain, optimised = lowered(PROGRAMS[0]), lowered(PROGRAMS[0], *STANDARD)
        self.assertIn("qc.call", names(plain))
        self.assertEqual([n for n in names(optimised) if n.startswith("qc.") or n == "memref.alloc"],
                         ["memref.alloc", "qc.x", "qc.x", "qc.x", "qc.measure", "qc.measure", "qc.measure",
                          "qc.measure"])  # 7 is 0111, in one register of four qubits


class Ranges(unittest.TestCase):
    def test_ranges_follow_the_arithmetic_and_give_the_narrowest_width(self):
        module = translate(parse_source(PROGRAMS[0]))
        found = [(op.name, ranges(module)[op.results[0]]) for op in module.walk() if op.results]
        self.assertEqual(found, [("arith.constant", Range(5, 5)), ("arith.constant", Range(3, 3)),
                                 ("arith.addi", Range(8, 8)), ("arith.constant", Range(1, 1)),
                                 ("arith.subi", Range(7, 7))])
        self.assertEqual([Range(*r).width for r in ((5, 5), (7, 7), (8, 8), (-1, -1), (-8, 7), (-9, 7))], [4, 4, 5, 1, 4, 5])


if __name__ == "__main__":
    unittest.main()
