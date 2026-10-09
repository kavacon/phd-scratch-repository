import textwrap
import unittest

from allocation_poc.printer import unparse_isa
from allocation_poc.programs import PROGRAMS
from allocation_poc.simulate import SimulationError, simulate, simulate_isa
from parser.instructions import (
    Alloc, Call, Const, Ctrl, EndCtrl, Free, InstructionSet, InvCtrl, IsaProgram, Op, Reg, inverse, inverse_block,
)
from parser.parser import parse
from passes.balance import balance
from passes.flatten import flatten
from passes.isa import IsaError, lower_to_isa
from passes.rename import rename
from passes.unroll import unroll


def lowered(source):
    return lower_to_isa(parse(source))


def expected(text):
    return textwrap.dedent(text).strip()


class LoweringExamplesTestCase(unittest.TestCase):
    def check(self, source, text):
        self.assertEqual(expected(text), unparse_isa(lowered(source)))

    def test_each_function_and_main_is_its_own_instruction_set_and_a_call_stays_a_call(self):
        self.check('function area(w: int, h: int): int { p = w * h; q = p + 1; return q; } x = 3; y = 4; return area(x, y);', """
            area(w#0, h#0):
                ALLOC p#1
                MUL   p#1, w#0, h#0
                ALLOC q#1
                ADD   q#1, p#1, 1
                MUL   p#1, w#0, h#0
                FREE  p#1
                RETURN q#1

            main:
                ALLOC x#1
                COPY  x#1, 3
                ALLOC y#1
                COPY  y#1, 4
                ALLOC $t1#1
                CALL  $t1#1, area(x#1, y#1)
                COPY  y#1, 4
                FREE  y#1
                COPY  x#1, 3
                FREE  x#1
                RETURN $t1#1
        """)

    def test_a_function_that_calls_another_keeps_the_call_and_releasing_its_result_is_the_same_call_again(self):
        self.check('function g(n: int): int { m = n * 2; return m + 1; } function h(n: int): int { a = g(n); b = g(a); return b; } x = 3; return h(x);', """
            g(n#0):
                ALLOC m#1
                MUL   m#1, n#0, 2
                ALLOC $t1#1
                ADD   $t1#1, m#1, 1
                MUL   m#1, n#0, 2
                FREE  m#1
                RETURN $t1#1

            h(n#0):
                ALLOC a#1
                CALL  a#1, g(n#0)
                ALLOC b#1
                CALL  b#1, g(a#1)
                CALL  a#1, g(n#0)
                FREE  a#1
                RETURN b#1

            main:
                ALLOC x#1
                COPY  x#1, 3
                ALLOC $t1#1
                CALL  $t1#1, h(x#1)
                COPY  x#1, 3
                FREE  x#1
                RETURN $t1#1
        """)

    def test_a_call_inside_an_if_is_an_ordinary_instruction_in_the_controlled_block(self):
        self.check('function g(n: int): int { m = n * 2; return m + 1; } x = 3; y = 0; if x < 5 { y = g(x); } else { y = 1; } return y;', """
            g(n#0):
                ALLOC m#1
                MUL   m#1, n#0, 2
                ALLOC $t1#1
                ADD   $t1#1, m#1, 1
                MUL   m#1, n#0, 2
                FREE  m#1
                RETURN $t1#1

            main:
                ALLOC x#1
                COPY  x#1, 3
                ALLOC y#1
                COPY  y#1, 0
                COPY  y#1, 0
                FREE  y#1
                ALLOC $t1#1
                LT    $t1#1, x#1, 5
                ALLOC y#2
                CTRL    $t1#1
                    CALL  y#2, g(x#1)
                ENDCTRL
                INVCTRL $t1#1
                    COPY  y#2, 1
                ENDCTRL
                LT    $t1#1, x#1, 5
                FREE  $t1#1
                COPY  x#1, 3
                FREE  x#1
                RETURN y#2
        """)

    def test_an_if_is_two_controlled_blocks_with_allocation_before_and_free_after(self):
        # t#1 and y#2 are allocated before the controlled blocks; the release is the same two blocks with the
        # same controls, in reverse, and the frees come after it. Nothing is allocated or freed inside a block.
        self.check("x = 2; a = x + 1; y = 0; if a < 5 { t = a * 2; y = t + 1; } else { y = a - 1; } r = y + 1; return r;", """
            main:
                ALLOC x#1
                COPY  x#1, 2
                ALLOC a#1
                ADD   a#1, x#1, 1
                ALLOC y#1
                COPY  y#1, 0
                COPY  y#1, 0
                FREE  y#1
                ALLOC $t1#1
                LT    $t1#1, a#1, 5
                ALLOC t#1
                ALLOC y#2
                CTRL    $t1#1
                    MUL   t#1, a#1, 2
                    ADD   y#2, t#1, 1
                ENDCTRL
                INVCTRL $t1#1
                    SUB   y#2, a#1, 1
                ENDCTRL
                ALLOC r#1
                ADD   r#1, y#2, 1
                CTRL    $t1#1
                    ADD   y#2, t#1, 1
                    MUL   t#1, a#1, 2
                ENDCTRL
                INVCTRL $t1#1
                    SUB   y#2, a#1, 1
                ENDCTRL
                FREE  y#2
                FREE  t#1
                LT    $t1#1, a#1, 5
                FREE  $t1#1
                ADD   a#1, x#1, 1
                FREE  a#1
                COPY  x#1, 2
                FREE  x#1
                RETURN r#1
        """)

    def test_an_if_on_a_constant_is_resolved_and_only_the_branch_that_runs_is_emitted(self):
        self.check("x = 1; if true { x = 2; } else { x = 3; } return x;", """
            main:
                ALLOC x#1
                COPY  x#1, 1
                COPY  x#1, 1
                FREE  x#1
                ALLOC x#2
                COPY  x#2, 2
                RETURN x#2
        """)

    def test_each_instruction_set_is_a_flat_list_and_nothing_is_inlined(self):
        for program_source in (p["source"] for p in PROGRAMS):
            program = lowered(program_source)
            names = [f.name for f in program.functions] + ["main"]
            self.assertEqual(len(names), len(set(names)))
            calls = [i for f in list(program.functions) + [program.main] for i in walk_instructions(f.body) if isinstance(i, Call)]
            self.assertTrue(all(c.function in names for c in calls))  # a call names the set it runs

    def test_a_program_that_is_not_flat_is_rejected(self):
        with self.assertRaises(IsaError):
            lower_to_isa(parse("a = 1; b = (a + 1) * 2; return b;", passes=()))


def walk_instructions(block):
    return iter(block)


class LoweringBehaviourTestCase(unittest.TestCase):
    EXTRA = [
        "function g(n: int): int { m = n * 2; return m + 1; } function h(n: int): int { a = g(n); b = g(a); return b; } return h(@N@);",
        "function g(n: int): int { m = n * 2; return m + 1; } function f(n: int): int { y = 0; if n < 3 { y = g(n); } else { y = n; } "
        "r = y + 1; return r; } return f(@N@);",
        "function g(n: int): int { r = 0; if n < 2 { r = n + 1; } else { r = n * 2; } return r; } x = 3; y = g(x); z = g(y); return z + g(@N@);",
        "function f(n: int): int { x = 0; y = 0; if n < 3 { t = n * 2; x = t + 1; y = t + 2; } else { x = n - 1; y = n - 2; } "
        "z = y + 1; return x + z; } return f(@N@);",
        "function f(n: int): int { x = n; if n < 2 { x = n + 10; } else { if n < 4 { t = n * 2; x = t + 1; } else { x = n - 1; } } "
        "r = x + n; return r; } return f(@N@);",
        "function g(n: int): int { s = n; for i in 0 .. 3 { t = s + i; s = t * 2 + 1; } return s; } return g(@N@);",
        "function f(a: int): int { x = a == 2 == a; y = !x || a > 1; return y; } return f(@N@);",
    ]

    def sources(self):
        programs = [p["source"] for p in PROGRAMS]
        return programs + [e.replace("@N@", str(n)) for e in self.EXTRA for n in range(6)]

    def test_the_result_is_unchanged_and_every_free_finds_zero(self):
        for source in self.sources():
            with self.subTest(source=source[:50]):
                before = simulate(parse(source, passes=())).value
                after = simulate_isa(lowered(source))
                self.assertEqual(before, after.value)
                self.assertEqual({}, {s: l for s, l in after.leaks.items() if l}, "registers left held")

    def test_nothing_is_allocated_or_freed_inside_a_controlled_block(self):
        for source in self.sources():
            program = lowered(source)
            for function in list(program.functions) + [program.main]:
                open_regions = 0
                for instruction in function.body:
                    if isinstance(instruction, (Ctrl, InvCtrl)):
                        open_regions += 1
                    elif isinstance(instruction, EndCtrl):
                        open_regions -= 1
                    elif open_regions:
                        self.assertNotIsInstance(instruction, (Alloc, Free), source[:50])
                self.assertEqual(0, open_regions, source[:50])

    def test_a_free_that_finds_a_non_zero_register_is_caught(self):
        main = InstructionSet("main", (), (Alloc("p"), Op("copy", "p", (Const(5),)), Free("p")), Const(0))
        with self.assertRaises(SimulationError):
            simulate_isa(IsaProgram((), main))


class InverseTestCase(unittest.TestCase):
    def test_each_instruction_has_an_inverse_and_the_inverse_of_the_inverse_is_itself(self):
        block = (
            Alloc("p"), Op("add", "p", (Reg("w"), Const(1))), Call("f", "p", (Reg("w"),)),
            InvCtrl(Reg("c")), Op("mul", "p", (Reg("w"), Reg("w"))), Op("copy", "q", (Const(True),)), EndCtrl(),
            Free("p"),
        )
        self.assertEqual(Free("p"), inverse(Alloc("p")))
        self.assertEqual(Alloc("p"), inverse(Free("p")))
        self.assertEqual(block[1], inverse(block[1]))  # an operation XORs into a register, so it undoes itself
        self.assertEqual(block, inverse_block(inverse_block(block)))

    def test_a_controlled_region_is_undone_with_the_same_control_and_reversed_body(self):
        a, b = Op("add", "y", (Reg("a"), Const(1))), Op("mul", "t", (Reg("a"), Const(2)))
        self.assertEqual((Ctrl(Reg("c")), b, a, EndCtrl()), inverse_block((Ctrl(Reg("c")), a, b, EndCtrl())))

    def test_nested_regions_are_undone_innermost_last_opened_first(self):
        a, b = Op("add", "y", (Reg("a"), Const(1))), Op("mul", "t", (Reg("a"), Const(2)))
        block = (Ctrl(Reg("c")), a, InvCtrl(Reg("d")), b, EndCtrl(), EndCtrl())
        self.assertEqual((Ctrl(Reg("c")), InvCtrl(Reg("d")), b, EndCtrl(), a, EndCtrl()), inverse_block(block))

    def test_unbalanced_regions_are_rejected(self):
        with self.assertRaises(ValueError):
            inverse_block((EndCtrl(),))
        with self.assertRaises(ValueError):
            inverse_block((Ctrl(Reg("c")),))

    def test_the_machine_applies_every_open_control_and_polarity(self):
        # y ^= 1 only where c holds and d does not
        def run(c, d):
            body = (Alloc("c"), Op("copy", "c", (Const(c),)), Alloc("d"), Op("copy", "d", (Const(d),)), Alloc("y"),
                    Ctrl(Reg("c")), InvCtrl(Reg("d")), Op("copy", "y", (Const(1),)), EndCtrl(), EndCtrl())
            return simulate_isa(IsaProgram((), InstructionSet("main", (), body, Reg("y")))).value
        self.assertEqual([0, 1, 0, 0], [run(0, 0), run(1, 0), run(0, 1), run(1, 1)])

    PROGRAMS = [
        "a = 3; b = a * 2; c = b + a; return c;",
        "a = 3; y = 0; if a < 5 { t = a * 2; y = t + 1; } else { y = a - 1; } r = y + 1; return r;",
        "a = 7; y = 0; if a < 5 { t = a * 2; y = t + 1; } else { y = a - 1; } r = y + 1; return r;",
        "a = 3; x = 0; if a < 5 { if a < 2 { x = 1; } else { x = 2; } } else { x = 3; } return x;",
    ]

    def test_running_a_program_and_then_its_inverse_leaves_nothing_held(self):
        # before releases are inserted, so the inverse of the whole body is what cleans up every register
        for source in self.PROGRAMS:
            with self.subTest(source=source):
                program = lower_to_isa(parse(source, passes=(unroll, flatten, balance, rename)))
                body = program.main.body
                there_and_back = InstructionSet("main", (), body + inverse_block(body), Const(0))
                result = simulate_isa(IsaProgram(program.functions, there_and_back))
                self.assertEqual({}, {s: l for s, l in result.leaks.items() if l})


if __name__ == '__main__':
    unittest.main()
