import textwrap
import unittest

from checking.simulate import simulate_isa
from compiler.parser.instructions import Alloc, Scratch
from compiler.passes.allocate import WidthExceeded, allocate
from compiler.passes.isa import lower_to_isa
from compiler.printer import unparse_isa
from compiler.system import System
from experiments.allocation_poc.experiment import compile_source
from experiments.allocation_poc.programs import PROGRAMS

IF_ELSE = "x = 2; a = x + 1; y = 0; if a < 5 { t = a * 2; y = t + 1; } else { y = a - 1; } r = y + 1; return r;"
CALLS = ("function g(n: int): int { m = n * 2; return m + 1; } function h(n: int): int { a = g(n); b = g(a); return b; } "
         "x = 3; return h(x);")


def isa(source):
    return lower_to_isa(compile_source(source))


def sources():
    return [p["source"] for p in PROGRAMS] + [IF_ELSE, CALLS]


def scratch_of(program):
    return next(i.register for i in program.body if isinstance(i, Scratch))


def slots_used(program):
    """The scratch register follows the slots, so its number is how many slots the program uses."""
    return int(scratch_of(program)[1:])


class PackingTestCase(unittest.TestCase):
    def test_registers_are_renamed_to_slots_and_the_scratch_register_is_declared_after_them(self):
        result = allocate(isa("function area(w: int, h: int): int { p = w * h; q = p + 1; return q; } x = 3; y = 4; return area(x, y);"), System(1, 10))
        self.assertEqual(textwrap.dedent("""
            SCRATCH R4
            ALLOC R0
            COPY  R0, 3
            ALLOC R1
            COPY  R1, 4
            ALLOC R2
            MUL   R2, R0, R1
            ALLOC R3
            ADD   R3, R2, 1
            IMUL  R2, R0, R1
            FREE  R2
            ICOPY R1, 4
            FREE  R1
            ICOPY R0, 3
            FREE  R0
            RETURN R3
        """).strip(), unparse_isa(result))

    def test_a_freed_slot_is_taken_by_the_next_register(self):
        # y#1 is freed before $t1#1 is allocated, so they share a slot: 7 logical registers on 6 slots
        program = isa(IF_ELSE)
        logical = [i.register for i in program.body if isinstance(i, Alloc)]
        result = allocate(program, System(1, 8))
        physical = [i.register for i in result.body if isinstance(i, Alloc)]
        self.assertEqual(7, len(logical))
        self.assertEqual(6, slots_used(result))
        slots = dict(zip(logical, physical))
        self.assertEqual(slots["y#1"], slots["$t1#1"])
        self.assertEqual(6, len(set(physical)))

    def test_a_program_that_needs_more_slots_than_the_machine_has_is_refused_with_the_packing_it_needed(self):
        program = isa(IF_ELSE)
        with self.assertRaises(WidthExceeded) as caught:
            allocate(program, System(4, 16))  # 16 // 4 - 1 = 3 slots, the program needs 6
        self.assertEqual((6, 3), (caught.exception.needed, caught.exception.available))
        self.assertEqual(6, slots_used(caught.exception.partial))  # the packing it would have needed is still there
        self.assertIn("needs 6 register slots but only 3", str(caught.exception))
        with self.assertRaises(WidthExceeded):  # the scratch register is what tips it over
            allocate(program, System(1, 6))


class BehaviourTestCase(unittest.TestCase):
    def test_the_physical_program_gives_the_same_result_and_leaves_nothing_held(self):
        for source in sources():
            with self.subTest(source=source[:40]):
                program = isa(source)
                after = simulate_isa(allocate(program, System(3, 3000)))
                self.assertEqual(simulate_isa(program).value, after.value)
                self.assertEqual({}, {s: held for s, held in after.leaks.items() if held})


if __name__ == '__main__':
    unittest.main()
