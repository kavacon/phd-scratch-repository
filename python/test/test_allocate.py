import textwrap
import unittest

from allocation_poc.printer import unparse_isa
from allocation_poc.programs import PROGRAMS
from allocation_poc.simulate import SimulationError, simulate_isa
from parser.instructions import Alloc, Const, Free, IsaProgram
from parser.parser import parse
from passes.allocate import WidthExceeded, allocate
from passes.isa import lower_to_isa

IF_ELSE = "x = 2; a = x + 1; y = 0; if a < 5 { t = a * 2; y = t + 1; } else { y = a - 1; } r = y + 1; return r;"
CALLS = ("function g(n: int): int { m = n * 2; return m + 1; } function h(n: int): int { a = g(n); b = g(a); return b; } "
         "x = 3; return h(x);")


def isa(source):
    return lower_to_isa(parse(source))


def sources():
    return [p["source"] for p in PROGRAMS] + [IF_ELSE, CALLS]


class PackingTestCase(unittest.TestCase):
    def test_registers_are_renamed_to_slots(self):
        result = allocate(isa("function area(w: int, h: int): int { p = w * h; q = p + 1; return q; } x = 3; y = 4; return area(x, y);"), 1, 10)
        self.assertEqual(textwrap.dedent("""
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
        """).strip(), unparse_isa(result.program))

    def test_a_freed_slot_is_taken_by_the_next_register(self):
        # y#1 is freed before $t1#1 is allocated, so they share a slot: 7 logical registers on 6 slots
        program = isa(IF_ELSE)
        logical = {i.register for i in program.body if isinstance(i, Alloc)}
        result = allocate(program, 1, 7)
        self.assertEqual(7, len(logical))
        self.assertEqual(6, result.needed)
        slots = {s.logical.register: s.physical.register for s in result.trace if isinstance(s.logical, Alloc)}
        self.assertEqual(slots["y#1"], slots["$t1#1"])
        self.assertEqual(6, len(set(slots.values())))

    def test_the_lowest_free_slot_is_taken_first(self):
        result = allocate(isa("a = 1; b = 2; c = a + b; return c;"), 1, 8)
        taken = [s.physical.register for s in result.trace if isinstance(s.logical, Alloc)]
        self.assertEqual(["R0", "R1", "R2"], taken)

    def test_slots_come_from_the_two_parameters_less_one_register_of_scratch(self):
        program = isa(IF_ELSE)
        self.assertEqual(6, allocate(program, 1, 7).slots)  # 7 // 1 - 1
        self.assertEqual(6, allocate(program, 2, 15).slots)  # 15 // 2 - 1
        self.assertEqual(8, allocate(program, 4, 36).slots)
        result = allocate(program, 4, 28)
        self.assertEqual((4, 28, 6, 6), (result.register_width, result.qubits, result.slots, result.needed))
        self.assertEqual((4, 28), (result.scratch_qubits, result.qubits_used))  # 6 registers of 4, and 4 for scratch

    def test_the_scratch_register_is_as_wide_as_a_register_and_is_never_a_slot(self):
        result = allocate(isa(IF_ELSE), 3, 40)
        self.assertEqual(3, result.scratch_qubits)
        self.assertEqual(40 // 3 - 1, result.slots)
        self.assertEqual({f"R{i}" for i in range(result.needed)}, {s.physical.register for s in result.trace if isinstance(s.logical, Alloc)})

    def test_a_program_that_needs_more_slots_than_the_machine_has_is_refused(self):
        program = isa(IF_ELSE)
        with self.assertRaises(WidthExceeded) as caught:
            allocate(program, 4, 16)  # 16 // 4 - 1 = 3 slots, the program needs 6
        self.assertEqual((6, 3), (caught.exception.needed, caught.exception.available))
        self.assertFalse(caught.exception.allocation.fits)
        self.assertEqual(6, caught.exception.allocation.needed)  # the packing it would have needed is still there
        self.assertIn("needs 6 register slots but only 3", str(caught.exception))
        with self.assertRaises(WidthExceeded):  # the scratch register is what tips it over
            allocate(program, 1, 6)

    def test_a_machine_too_small_for_even_the_scratch_register_has_no_slots(self):
        with self.assertRaises(WidthExceeded) as caught:
            allocate(isa("a = 1; return a;"), 4, 3)
        self.assertEqual(0, caught.exception.available)

    def test_the_slots_needed_are_the_peak_of_the_program(self):
        for source in sources():
            with self.subTest(source=source[:40]):
                program = isa(source)
                self.assertEqual(max(simulate_isa(program).peak.values()), allocate(program, 1, 1000).needed)

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            allocate(isa(IF_ELSE), 0, 10)
        with self.assertRaises(ValueError):
            allocate(isa(IF_ELSE), 1, -1)


class BehaviourTestCase(unittest.TestCase):
    def test_the_physical_program_gives_the_same_result_and_leaves_nothing_held(self):
        for source in sources():
            with self.subTest(source=source[:40]):
                program = isa(source)
                after = simulate_isa(allocate(program, 3, 3000).program)
                self.assertEqual(simulate_isa(program).value, after.value)
                self.assertEqual({}, {s: held for s, held in after.leaks.items() if held})

    def test_the_machine_catches_two_live_registers_sharing_a_slot(self):
        with self.assertRaises(SimulationError):
            simulate_isa(IsaProgram((Alloc("R0"), Alloc("R0")), Const(0)))

    def test_the_peak_is_unchanged_by_renaming(self):
        for source in sources():
            program = isa(source)
            self.assertEqual(simulate_isa(program).peak, simulate_isa(allocate(program, 1, 1000).program).peak)


class TraceTestCase(unittest.TestCase):
    def test_the_trace_shows_what_every_slot_held_after_each_instruction(self):
        result = allocate(isa("a = 3; b = a + 1; return b;"), 1, 4)
        first, *_, last = result.trace
        self.assertIsInstance(first.logical, Alloc)
        self.assertEqual([("a#1", 0)], [(h.register, h.slot) for h in first.held])
        self.assertEqual(0, first.changed)
        self.assertIn("b#1", {h.register for h in last.held})  # the returned register is still held at the end
        freed = [s for s in result.trace if isinstance(s.logical, Free)]
        self.assertTrue(freed and all(s.changed is not None for s in freed))
        self.assertEqual(len(isa("a = 3; b = a + 1; return b;").body), len(result.trace))

    def test_no_step_holds_two_registers_in_one_slot(self):
        for source in sources():
            for step in allocate(isa(source), 1, 1000).trace:
                slots = [h.slot for h in step.held]
                self.assertEqual(len(slots), len(set(slots)))

    def test_inlined_callee_registers_are_held_under_their_own_names(self):
        result = allocate(isa(CALLS), 1, 100)
        names = {h.register for s in result.trace for h in s.held}
        self.assertTrue(any(n.startswith("h1.g1.") for n in names))
        self.assertTrue(any(n.startswith("h1.g2.") for n in names))


if __name__ == '__main__':
    unittest.main()
