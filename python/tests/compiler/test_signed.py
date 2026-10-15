import itertools
import unittest

from checking.circuits import run_gates, to_signed, value_of
from checking.simulate import simulate
from compiler.compiler import Compiler
from compiler.parser.parser import parse
from compiler.passes.allocate import allocate
from compiler.passes.balance import balance
from compiler.passes.flatten import flatten
from compiler.passes.inline import inline
from compiler.passes.isa import lower_to_isa
from compiler.passes.qasm import measured_qubits, to_openqasm, x_gates
from compiler.passes.rename import rename
from compiler.passes.unroll import unroll
from compiler.system import System

W = 3
RANGE = range(-4, 4)


def run(source, system):
    compiler = Compiler(system).load(inline, unroll, flatten, balance, rename, lower_to_isa, allocate, to_openqasm)
    circuit = compiler.run(source).circuit
    state = run_gates(x_gates(circuit), circuit.num_qubits)
    value = value_of(state, measured_qubits(circuit))
    return to_signed(value, W), []


class ReferenceMachineTestCase(unittest.TestCase):
    def test_without_a_system_integers_are_unbounded(self):
        self.assertEqual(18, simulate(parse("a = 3; b = 6; return a * b;")).value)

    def test_with_a_system_they_wrap_to_the_register_width_as_two_s_complement(self):
        system = System(W, 100)
        self.assertEqual(2, simulate(parse("a = 3; b = 6; return a * b;"), system).value)    # 18 mod 8 is 2
        self.assertEqual(-2, simulate(parse("a = 3; b = 3; return a + b;"), system).value)   # 6 is -2
        self.assertEqual(-4, simulate(parse("a = 3; b = 1; return a + b;"), system).value)   # 4 is -4, the most negative

    def test_a_literal_outside_the_range_wraps_too(self):
        self.assertEqual(-3, simulate(parse("return 5;"), System(W, 100)).value)

    def test_comparison_is_on_the_wrapped_values(self):
        # 6 + 1 is 7 in the register, which is -1, so -1 < 3 is true
        self.assertEqual(True, simulate(parse("a = 6; b = a + 1; c = b < 3; return c;"), System(W, 100)).value)


class CircuitsTestCase(unittest.TestCase):
    """Every two-register and register-and-constant case at width 3, against the reference machine."""

    def check(self, template, system, values):
        for a, b in itertools.product(values, repeat=2):
            source = template.format(a=a, b=b)
            value, left = run(source, system)
            self.assertEqual(simulate(parse(source), system).value, value, source)

    def test_signed_comparison_of_two_registers(self):
        system = System(W, 200)
        for op in ("<", ">", "==", "!="):
            self.check("a = {a}; b = {b}; c = a " + op + " b; return c;", system, RANGE)

    def test_comparing_two_literals_is_decided_when_the_program_is_compiled(self):
        system = System(W, 200)
        for op in ("<", ">", "==", "!="):
            for a, b in itertools.product(RANGE, repeat=2):
                source = f"c = {a} {op} {b}; return c;"
                self.assertEqual(simulate(parse(source), system).value, run(source, system)[0], source)

    def test_signed_comparison_with_a_constant_on_either_side(self):
        system = System(W, 200)
        for op in ("<", ">"):
            for k in RANGE:
                self.check("a = {a}; b = {b}; c = a " + op + f" {k}; d = {k} " + op + " b; e = c && d; return e;", system, RANGE)

    def test_signed_arithmetic_wraps_like_two_s_complement(self):
        system = System(W, 200)
        for op in ("+", "-", "*"):
            self.check("a = {a}; b = {b}; c = a " + op + " b; return c;", system, RANGE)

    def test_negation_and_negative_literals(self):
        system = System(W, 200)
        self.check("a = {a}; b = {b}; c = -a; d = c + -1; e = d - b; return e;", system, RANGE)

    def test_conditionals_on_a_signed_comparison(self):
        system = System(W, 200)
        self.check("a = {a}; b = {b}; c = a < b; r = 0; if c {{ r = b - a; }} else {{ r = a - b; }} return r;", system, RANGE)


if __name__ == '__main__':
    unittest.main()
