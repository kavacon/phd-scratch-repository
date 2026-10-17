import textwrap
import unittest

from compiler.printer import unparse_isa
from experiments.allocation_poc.programs import PROGRAMS
from checking.simulate import SimulationError, simulate, simulate_isa
from compiler.parser.instructions import (
    Alloc,
    Const,
    Ctrl,
    EndCtrl,
    Free,
    ICtrl,
    IOp,
    IsaProgram,
    Op,
    Reg,
    inverse,
    inverse_block,
)
from compiler.parser.parser import parse
from experiments.allocation_poc.experiment import compile_source
from compiler.passes.balance import balance
from compiler.passes.flatten import flatten
from compiler.passes.isa import lower_to_isa
from compiler.passes.rename import rename
from compiler.passes.unroll import unroll


def lowered(source):
    return lower_to_isa(compile_source(source))


def expected(text):
    return textwrap.dedent(text).strip()


class LoweringExamplesTestCase(unittest.TestCase):
    def check(self, source, text):
        self.assertEqual(expected(text), unparse_isa(lowered(source)))

    def test_a_call_is_inlined_so_the_program_is_one_flat_list_with_the_callee_registers_under_its_own_names(self):
        self.check('function area(w: int, h: int): int { p = w * h; q = p + 1; return q; } x = 3; y = 4; return area(x, y);', """
            ALLOC x#1
            COPY  x#1, 3
            ALLOC y#1
            COPY  y#1, 4
            ALLOC area1.p#1
            MUL   area1.p#1, x#1, y#1
            ALLOC area1.q#1
            ADD   area1.q#1, area1.p#1, 1
            IMUL  area1.p#1, x#1, y#1
            FREE  area1.p#1
            ICOPY y#1, 4
            FREE  y#1
            ICOPY x#1, 3
            FREE  x#1
            RETURN area1.q#1
        """)

    def test_an_if_is_two_controlled_blocks_with_allocation_before_and_free_after(self):
        # t#1 and y#2 are allocated before the controlled blocks; the release is the same two blocks with the
        # same controls, in reverse, and the frees come after it. Nothing is allocated or freed inside a block.
        self.check("x = 2; a = x + 1; y = 0; if a < 5 { t = a * 2; y = t + 1; } else { y = a - 1; } r = y + 1; return r;", """
            ALLOC x#1
            COPY  x#1, 2
            ALLOC a#1
            ADD   a#1, x#1, 1
            ALLOC y#1
            COPY  y#1, 0
            ICOPY y#1, 0
            FREE  y#1
            ALLOC $t1#1
            LT    $t1#1, a#1, 5
            ALLOC t#1
            ALLOC y#2
            CTRL    $t1#1
                MUL   t#1, a#1, 2
                ADD   y#2, t#1, 1
            ENDCTRL
            ICTRL   $t1#1
                SUB   y#2, a#1, 1
            ENDCTRL
            ALLOC r#1
            ADD   r#1, y#2, 1
            CTRL    $t1#1
                IADD  y#2, t#1, 1
                IMUL  t#1, a#1, 2
            ENDCTRL
            ICTRL   $t1#1
                ISUB  y#2, a#1, 1
            ENDCTRL
            FREE  y#2
            FREE  t#1
            ILT   $t1#1, a#1, 5
            FREE  $t1#1
            IADD  a#1, x#1, 1
            FREE  a#1
            ICOPY x#1, 2
            FREE  x#1
            RETURN r#1
        """)


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
                before = simulate(parse(source)).value
                after = simulate_isa(lowered(source))
                self.assertEqual(before, after.value)
                self.assertEqual({}, {s: held for s, held in after.leaks.items() if held}, "registers left held")

    def test_nothing_is_allocated_or_freed_inside_a_controlled_block(self):
        for source in self.sources():
            program = lowered(source)
            open_regions = 0
            for instruction in program.body:
                if isinstance(instruction, (Ctrl, ICtrl)):
                    open_regions += 1
                elif isinstance(instruction, EndCtrl):
                    open_regions -= 1
                elif open_regions:
                    self.assertNotIsInstance(instruction, (Alloc, Free), source[:50])
            self.assertEqual(0, open_regions, source[:50])


class PreconditionTestCase(unittest.TestCase):
    def run_body(self, *body, result=None):
        return simulate_isa(IsaProgram(tuple(body), Const(0) if result is None else result))

    def test_an_op_needs_a_zero_register_and_its_inverse_needs_the_value_it_computes(self):
        add = Op("add", "p", (Const(2), Const(3)))
        self.assertEqual(0, self.run_body(Alloc("p"), add, inverse(add), Free("p")).value)
        with self.assertRaises(SimulationError):  # writing into a register that already holds a value
            self.run_body(Alloc("p"), add, add, inverse(add), Free("p"))
        with self.assertRaises(SimulationError):  # undoing with the wrong operands
            self.run_body(Alloc("p"), add, IOp("add", "p", (Const(2), Const(4))), Free("p"))
        with self.assertRaises(SimulationError):  # undoing something that was never computed
            self.run_body(Alloc("p"), inverse(add), Free("p"))


class InverseTestCase(unittest.TestCase):
    def test_each_instruction_has_an_explicit_inverse_and_the_inverse_of_the_inverse_is_itself(self):
        block = (
            Alloc("p"), Op("add", "p", (Reg("w"), Const(1))),
            ICtrl(Reg("c")), Op("mul", "p", (Reg("w"), Reg("w"))), IOp("copy", "q", (Const(True),)), EndCtrl(),
            Free("p"),
        )
        self.assertEqual(Free("p"), inverse(Alloc("p")))
        self.assertEqual(Alloc("p"), inverse(Free("p")))
        self.assertEqual(IOp("add", "p", (Reg("w"), Const(1))), inverse(block[1]))
        self.assertEqual(block[1], inverse(inverse(block[1])))
        self.assertEqual(block, inverse_block(inverse_block(block)))

    def test_the_machine_applies_every_open_control_and_polarity(self):
        # y ^= 1 only where c holds and d does not
        def run(c, d):
            body = (Alloc("c"), Op("copy", "c", (Const(c),)), Alloc("d"), Op("copy", "d", (Const(d),)), Alloc("y"),
                    Ctrl(Reg("c")), ICtrl(Reg("d")), Op("copy", "y", (Const(1),)), EndCtrl(), EndCtrl())
            return simulate_isa(IsaProgram(body, Reg("y"))).value
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
                program = lower_to_isa(compile_source(source, passes=(unroll, flatten, balance, rename)))
                result = simulate_isa(IsaProgram(program.body + inverse_block(program.body), Const(0)))
                self.assertEqual({}, {s: held for s, held in result.leaks.items() if held})


if __name__ == '__main__':
    unittest.main()
