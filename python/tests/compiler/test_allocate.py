import textwrap
import unittest

from checking.simulate import SimulationError, simulate_isa
from compiler.parser.instructions import Alloc, Const, IsaProgram, Scratch
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


def without_scratch(program):
    return [i for i in program.body if not isinstance(i, Scratch)]


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

    def test_the_result_has_the_shape_of_its_input_with_one_declaration_added(self):
        program = isa(IF_ELSE)
        result = allocate(program, System(1, 100))
        self.assertIsInstance(result.body[0], Scratch)
        self.assertEqual(len(program.body), len(without_scratch(result)))
        for before, after in zip(program.body, without_scratch(result)):
            self.assertIs(type(before), type(after))

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

    def test_the_lowest_free_slot_is_taken_first(self):
        result = allocate(isa("a = 1; b = 2; c = a + b; return c;"), System(1, 8))
        self.assertEqual(["R0", "R1", "R2"], [i.register for i in result.body if isinstance(i, Alloc)])

    def test_the_scratch_register_is_never_a_slot(self):
        result = allocate(isa(IF_ELSE), System(3, 40))
        slots = {i.register for i in result.body if isinstance(i, Alloc)}
        self.assertNotIn(scratch_of(result), slots)
        self.assertEqual({f"R{i}" for i in range(slots_used(result))}, slots)

    def test_a_program_fits_exactly_when_the_slots_it_needs_are_the_systems_slots(self):
        program = isa(IF_ELSE)  # needs 6 slots
        allocate(program, System(4, 28))  # 28 // 4 - 1 = 6
        with self.assertRaises(WidthExceeded) as caught:
            allocate(program, System(4, 24))  # 5 slots
        self.assertEqual((6, 5), (caught.exception.needed, caught.exception.available))

    def test_a_program_that_needs_more_slots_than_the_machine_has_is_refused_with_the_packing_it_needed(self):
        program = isa(IF_ELSE)
        with self.assertRaises(WidthExceeded) as caught:
            allocate(program, System(4, 16))  # 16 // 4 - 1 = 3 slots, the program needs 6
        self.assertEqual((6, 3), (caught.exception.needed, caught.exception.available))
        self.assertEqual(6, slots_used(caught.exception.partial))  # the packing it would have needed is still there
        self.assertIn("needs 6 register slots but only 3", str(caught.exception))
        with self.assertRaises(WidthExceeded):  # the scratch register is what tips it over
            allocate(program, System(1, 6))

    def test_a_machine_too_small_for_even_the_scratch_register_has_no_slots(self):
        with self.assertRaises(WidthExceeded) as caught:
            allocate(isa("a = 1; return a;"), System(4, 3))
        self.assertEqual(0, caught.exception.available)

    def test_the_slots_needed_are_the_peak_of_the_program(self):
        for source in sources():
            with self.subTest(source=source[:40]):
                program = isa(source)
                self.assertEqual(max(simulate_isa(program).peak.values()), slots_used(allocate(program, System(1, 1000))))


class BehaviourTestCase(unittest.TestCase):
    def test_the_physical_program_gives_the_same_result_and_leaves_nothing_held(self):
        for source in sources():
            with self.subTest(source=source[:40]):
                program = isa(source)
                after = simulate_isa(allocate(program, System(3, 3000)))
                self.assertEqual(simulate_isa(program).value, after.value)
                self.assertEqual({}, {s: held for s, held in after.leaks.items() if held})

    def test_the_machine_catches_two_live_registers_sharing_a_slot(self):
        with self.assertRaises(SimulationError):
            simulate_isa(IsaProgram((Alloc("R0"), Alloc("R0")), Const(0)))

    def test_the_peak_is_unchanged_by_renaming_and_does_not_count_the_scratch_register(self):
        for source in sources():
            program = isa(source)
            self.assertEqual(simulate_isa(program).peak, simulate_isa(allocate(program, System(1, 1000))).peak)

    def test_inlined_callee_registers_were_allocated_under_their_own_names(self):
        program = isa(CALLS)
        names = {i.register for i in program.body if isinstance(i, Alloc)}
        self.assertTrue(any(n.startswith("h1.g1.") for n in names))
        self.assertTrue(any(n.startswith("h1.g2.") for n in names))


if __name__ == '__main__':
    unittest.main()
