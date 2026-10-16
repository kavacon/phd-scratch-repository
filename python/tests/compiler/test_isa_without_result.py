import unittest

from checking.simulate import simulate_isa
from compiler.parser.instructions import Alloc, Const, IsaProgram, Op, Reg, Scratch
from compiler.passes.allocate import allocate
from compiler.passes.qasm import to_openqasm
from compiler.printer import unparse_isa
from compiler.system import System

SYSTEM = System(3, 100)
BODY = (Scratch("R2"), Alloc("R0"), Alloc("R1"), Op("copy", "R0", (Const(2),)), Op("add", "R1", (Reg("R0"), Const(1))))


class IsaWithoutResultTestCase(unittest.TestCase):
    def test_a_program_may_have_no_result(self):
        self.assertIsNone(IsaProgram(BODY).result)

    def test_it_prints_without_a_return_line(self):
        self.assertNotIn("RETURN", unparse_isa(IsaProgram(BODY)))
        self.assertIn("RETURN R1", unparse_isa(IsaProgram(BODY, Reg("R1"))))

    def test_the_reference_machine_runs_it_and_returns_no_value(self):
        result = simulate_isa(IsaProgram(BODY), SYSTEM)
        self.assertIsNone(result.value)
        self.assertEqual({"R0", "R1"}, result.leaks["<program>"])  # nothing is the result, so both are left held

    def test_the_circuit_is_just_its_gates(self):
        bare = to_openqasm(IsaProgram(BODY), SYSTEM)
        read = to_openqasm(IsaProgram(BODY, Reg("R1")), SYSTEM)
        self.assertEqual(0, bare.num_clbits)
        self.assertNotIn("measure", bare.count_ops())
        self.assertEqual(3, read.num_clbits)
        self.assertEqual(read.size() - 3, bare.size())

    def test_allocation_keeps_it_without_a_result(self):
        body = (Op("copy", "a", (Const(2),)), Op("add", "b", (Reg("a"), Const(1))))
        program = IsaProgram((Alloc("a"), Alloc("b")) + body)
        self.assertIsNone(allocate(program, SYSTEM).result)


if __name__ == '__main__':
    unittest.main()
