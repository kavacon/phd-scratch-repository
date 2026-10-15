import unittest

from checking.circuits import run_gates, to_signed, value_of
from checking.simulate import simulate, simulate_isa
from compiler.compiler import Compiler, PassError
from compiler.parser.instructions import Alloc, Reg, Update, inverse
from compiler.parser.parser import parse
from compiler.passes.allocate import allocate
from compiler.passes.balance import balance
from compiler.passes.flatten import flatten
from compiler.passes.inline import inline
from compiler.passes.isa import IsaError, lower_to_isa
from compiler.passes.qasm import measured_qubits, to_openqasm, x_gates
from compiler.passes.rename import rename
from compiler.passes.unroll import unroll
from compiler.printer import unparse, unparse_isa
from compiler.system import System
from experiments.allocation_poc.passes.allocation import ReleaseError, lower_allocation

FRONT = (inline, unroll, flatten, balance, rename)
W = 3


def isa_of(source, system=None):
    return Compiler(system).load(*FRONT, lower_to_isa).run(source).output


def circuit_of(source, system):
    return Compiler(system).load(*FRONT, lower_to_isa, allocate, to_openqasm).run(source).circuit


def run_circuit(source, system):
    circuit = circuit_of(source, system)
    state = run_gates(x_gates(circuit), circuit.num_qubits)
    value = value_of(state, measured_qubits(circuit))
    return to_signed(value, system.register_width), []


PROGRAMS = [
    "a = {a}; b = {b}; a += b; a -= 2; return a;",
    "a = {a}; b = {b}; a += b * 2; return a;",
    "a = {a}; b = {b}; c = a < b; if c {{ a += 3; }} return a;",
    "a = {a}; b = {b}; c = a < b; if c {{ a += 1; }} else {{ a -= 1; }} return a;",
    "a = {a}; b = {b}; a += -3; b -= a; return b;",
    "a = {a}; b = {b}; c = a == b; if c {{ a += 2; }} a += 1; a -= b; return a;",
]


class SyntaxAndLoweringTestCase(unittest.TestCase):
    def test_the_update_has_no_register_of_its_own(self):
        source = "a = 3; b = 4; a += b; a -= 1; return a;"
        program = isa_of(source)
        self.assertEqual(["a#1", "b#1"], [i.register for i in program.body if isinstance(i, Alloc)])  # nothing for a#2 or a#3
        self.assertEqual(["ADD=", "SUB="], [unparse_isa(program).split()[i] for i, t in enumerate(unparse_isa(program).split()) if t.endswith("=")])
        self.assertEqual([("add", "a#1"), ("sub", "a#1")], [(u.opcode, u.dst) for u in program.body if isinstance(u, Update)])
        self.assertEqual(Reg("a#1"), program.result)

    def test_an_update_uses_one_register_fewer_than_the_same_program_out_of_place(self):
        inplace = isa_of("a = 3; b = 4; a += b; return a;")
        copied = isa_of("a = 3; b = 4; a = a + b; return a;")
        self.assertEqual(2, simulate_isa(inplace).peak["<program>"])
        self.assertEqual(3, simulate_isa(copied).peak["<program>"])

    def test_the_printed_forms(self):
        raw = parse("a = 3; a += 4; a -= b; return a;")
        self.assertEqual("a = 3;\na += 4;\na -= b;\nreturn a;", unparse(raw))
        renamed = Compiler().load(*FRONT).run("a = 3; a += 4; return a;").output
        self.assertIn("a#2 = a#1 + 4;  // in place", unparse(renamed))

    def test_the_inverse_of_an_update_is_the_other_one(self):
        add = Update("add", "x", Reg("y"))
        self.assertEqual(Update("sub", "x", Reg("y")), inverse(add))
        self.assertEqual(add, inverse(inverse(add)))

    def test_a_branch_that_leaves_the_variable_alone_needs_no_copy(self):
        program = isa_of("a = 3; c = a < 5; if c { a += 1; } return a;")
        self.assertEqual(["a#1", "c#1"], [i.register for i in program.body if isinstance(i, Alloc)])

    def test_updating_in_one_branch_and_reassigning_in_the_other_is_refused(self):
        with self.assertRaises(PassError) as caught:
            isa_of("a = 3; c = a < 5; if c { a += 1; } else { a = a * 2; } return a;")
        self.assertIsInstance(caught.exception.cause, IsaError)
        self.assertIn("in place in one branch", str(caught.exception))

    def test_a_register_cannot_be_updated_with_itself(self):
        with self.assertRaises(PassError) as caught:
            isa_of("a = 3; a += a; return a;")
        self.assertIn("with itself", str(caught.exception))

    def test_release_insertion_refuses_a_program_with_updates(self):
        with self.assertRaises(ReleaseError):
            lower_allocation(Compiler().load(*FRONT).run("a = 3; a += 1; return a;").output)

    def test_a_parameter_the_callee_updates_is_copied_so_the_callers_variable_is_untouched(self):
        source = "function f(p: int): int { p += 5; return p; } x = 2; y = f(x); return x + y;"
        reference = simulate(parse(source)).value
        circuit_value, _ = run_circuit(source, System(W, 100))
        self.assertEqual((2 + 7) % 8, reference % 8)
        self.assertEqual(reference % 8, circuit_value)


class ResultsTestCase(unittest.TestCase):
    def test_circuits_agree_with_the_reference_for_every_input(self):
        system = System(W, 200)
        for template in PROGRAMS:
            for a in range(-4, 4):
                for b in range(-4, 4):
                    source = template.format(a=a, b=b)
                    value, left = run_circuit(source, system)
                    self.assertEqual(simulate(parse(source), system).value, value, source)

    def test_an_update_on_the_instruction_machine_matches_the_program_machine(self):
        system = System(W, 200)
        for template in PROGRAMS:
            source = template.format(a=3, b=-2)
            self.assertEqual(simulate(parse(source), system).value, simulate_isa(isa_of(source), system).value, source)


if __name__ == '__main__':
    unittest.main()
