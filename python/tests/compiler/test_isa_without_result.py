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
    def test_a_program_with_no_result_prints_runs_and_lowers_without_any_readout(self):
        program = IsaProgram(BODY)
        self.assertNotIn("RETURN", unparse_isa(program))
        self.assertIsNone(simulate_isa(program, SYSTEM).value)
        bare, read = to_openqasm(program, SYSTEM), to_openqasm(IsaProgram(BODY, Reg("R1")), SYSTEM)
        self.assertEqual(0, bare.num_clbits)
        self.assertEqual(read.size() - 3, bare.size())  # the three measurements are all that is left out

    def test_allocation_keeps_it_without_a_result(self):
        body = (Alloc("a"), Alloc("b"), Op("copy", "a", (Const(2),)), Op("add", "b", (Reg("a"), Const(1))))
        self.assertIsNone(allocate(IsaProgram(body), SYSTEM).result)


if __name__ == '__main__':
    unittest.main()
