import textwrap
import unittest

from tests.compiler.test_flatten import is_flat, is_plain

from compiler.printer import unparse
from checking.simulate import SimulationError, simulate
from compiler.parser.ast_tools import walk
from compiler.parser.grammar import If, Unassign
from compiler.parser.parser import parse
from experiments.allocation_poc.experiment import compile_source
from experiments.allocation_poc.passes.allocation import lower_allocation
from compiler.passes.balance import balance
from compiler.passes.flatten import flatten
from compiler.passes.rename import rename
from compiler.passes.unroll import unroll


def compiled(source):
    """The default pipeline without inlining, so functions stay and the release pass is tested on them as scopes."""
    return compile_source(source, passes=(unroll, flatten, balance, rename, lower_allocation))


def expected(text):
    return textwrap.dedent(text).strip()


class InsertionExamplesTestCase(unittest.TestCase):
    def check(self, source, text):
        self.assertEqual(expected(text), unparse(compiled(source)))

    def test_temporary_is_released_after_its_last_use(self):
        self.check("function area(w: int, h: int): int { p = w * h; q = p + 1; return q; } return area(2, 3);", """
            function area(w#0: int, h#0: int): int {
                p#1 = w#0 * h#0;
                q#1 = p#1 + 1;
                p#1 ~= w#0 * h#0;
                return q#1;
            }
            $t1#1 = area(2, 3);
            return $t1#1;
        """)

    def test_what_an_if_produced_is_released_together_with_the_temporaries_it_was_built_from(self):
        # t#1 feeds y#2, so it stays alive after the if. The release if re-tests the same condition register and
        # undoes y#2 and then t#1 in the then branch, one operation at a time. Nothing in it is nested.
        self.check("""
            function f(x: int): int {
                a = x + 1; y = 0;
                if a < 5 { t = a * 2; y = t + 1; } else { y = a - 1; }
                r = y + 1;
                return r;
            }
            return f(1);
        """, """
            function f(x#0: int): int {
                a#1 = x#0 + 1;
                y#1 = 0;
                y#1 ~= 0;
                $t1#1 = a#1 < 5;
                if $t1#1 {
                    t#1 = a#1 * 2;
                    y#2 = t#1 + 1;
                } else {
                    y#2 = a#1 - 1;
                }
                r#1 = y#2 + 1;
                if $t1#1 {
                    y#2 ~= t#1 + 1;
                    t#1 ~= a#1 * 2;
                } else {
                    y#2 ~= a#1 - 1;
                }
                $t1#1 ~= a#1 < 5;
                a#1 ~= x#0 + 1;
                return r#1;
            }
            $t1#1 = f(1);
            return $t1#1;
        """)


class InsertionBehaviourTestCase(unittest.TestCase):
    """Run the programs: values are unchanged, every witness is right, and nothing is read after release."""

    PROGRAMS = {
        "chain": "function f(x: int): int { a = x + 1; b = a * 2; c = b + a; return c; } return f(@N@);",
        "if else": """
            function f(x: int): int {
                a = x + 1; y = 0;
                if a < 5 { t = a * 2; y = t + 1; } else { y = a - 1; }
                r = y + 1;
                return r;
            }
            return f(@N@);
        """,
        "one-armed if": "function f(a: int): int { x = a + 1; if x < 4 { x = x + 5; } return x; } return f(@N@);",
        "nested ifs": """
            function f(a: int): int {
                x = a;
                if a < 5 { if a < 3 { x = a + 1; } else { x = a + 2; } x = x * 2; } else { x = a - 1; }
                r = x + a;
                return r;
            }
            return f(@N@);
        """,
        "loop with a temporary": "function g(n: int): int { s = n; for i in 0 .. 3 { t = s + i; s = t * 2 + 1; } return s; } return g(@N@);",
        "functions calling functions": """
            function double(x: int): int { y = x * 2; return y + 0; }
            function quad(x: int): int { d = double(x); return double(d); }
            return quad(@N@);
        """,
    }

    def test_values_are_unchanged_and_releases_are_valid(self):
        for name, template in self.PROGRAMS.items():
            for n in range(6):
                source = template.replace("@N@", str(n))
                with self.subTest(program=name, n=n):
                    before = simulate(parse(source)).value  # the program as written, loops and all
                    after = simulate(compiled(source))
                    self.assertEqual(before, after.value)
                    self.assertEqual({}, {s: held for s, held in after.leaks.items() if held}, "registers left behind")

    def test_every_release_is_one_operation_and_every_condition_is_a_register(self):
        # including the releases of what an if produced, which used to re-test a nested expression inline
        for name, template in self.PROGRAMS.items():
            with self.subTest(program=name):
                for statement in compiled(template.replace("@N@", "2")):
                    for node in walk(statement):
                        if isinstance(node, Unassign):
                            self.assertTrue(is_flat(node.witness), f"{node.name} ~= ...")
                        elif isinstance(node, If):
                            self.assertTrue(is_plain(node.condition))

    def test_a_wrong_witness_is_caught_by_the_simulation(self):
        broken = compile_source("function f(a: int): int { t = a + 1; t ~= a + 2; return a; } return f(1);")
        with self.assertRaises(SimulationError):
            simulate(broken)


if __name__ == '__main__':
    unittest.main()
