import textwrap
import unittest

from compiler.printer import unparse
from experiments.allocation_poc.programs import PROGRAMS
from checking.simulate import simulate
from compiler.parser.ast_tools import walk
from compiler.parser.grammar import Call, FunctionDef
from compiler.parser.parser import parse
from experiments.allocation_poc.experiment import compile_source
from compiler.passes.inline import InlineError, inline

AREA = "function area(w: int, h: int): int { p = w * h; q = p + 1; return q; }"


def inlined(source):
    return compile_source(source, passes=(inline,))


def expected(text):
    return textwrap.dedent(text).strip()


class InlineExamplesTestCase(unittest.TestCase):
    def check(self, source, text):
        self.assertEqual(expected(text), unparse(inlined(source)))

    def test_a_call_is_replaced_by_the_callee_body_and_the_definition_is_dropped(self):
        self.check(AREA + " x = 3; y = 4; return area(x, y);", """
            x = 3;
            y = 4;
            area1.p = x * y;
            area1.q = area1.p + 1;
            return area1.q;
        """)

    def test_a_parameter_the_callee_assigns_to_is_copied_so_the_callers_variable_is_unchanged(self):
        self.check("function bump(n: int): int { n = n + 1; return n; } x = 3; y = bump(x); return x + y;", """
            x = 3;
            bump1.n = x;
            bump1.n = bump1.n + 1;
            y = bump1.n;
            return x + y;
        """)

    def test_calls_inside_a_callee_are_inlined_first_and_their_names_nest(self):
        self.check("function g(n: int): int { m = n * 2; return m + 1; } function h(n: int): int { a = g(n); return g(a); } x = 3; return h(x);", """
            x = 3;
            h1.g1.m = x * 2;
            h1.a = h1.g1.m + 1;
            h1.g2.m = h1.a * 2;
            return h1.g2.m + 1;
        """)


class InlineErrorsTestCase(unittest.TestCase):
    def test_recursion_is_rejected(self):
        with self.assertRaises(InlineError):
            inlined("function f(n: int): int { a = f(n); return a; } return f(1);")
        with self.assertRaises(InlineError):
            inlined("function f(n: int): int { a = g(n); return a; } function g(n: int): int { b = f(n); return b; } return f(1);")


class InlineBehaviourTestCase(unittest.TestCase):
    EXTRA = [
        "function g(n: int): int { s = n; for i in 0 .. 3 { t = s + i; s = t * 2 + 1; } return s; } return g(@N@);",
        "function bump(n: int): int { n = n + 1; return n; } x = @N@; y = bump(x); z = bump(y); return x + y + z;",
        "function g(n: int): int { if n < 2 { r = n + 1; } else { r = n * 2; } return r; } x = @N@; y = g(x); z = g(y + 1); return z;",
        "function g(n: int): int { m = n * 2; return m + 1; } function h(n: int): int { a = g(n); b = g(a); return b + a; } return h(@N@);",
    ]

    def test_the_result_is_unchanged_and_no_function_or_call_is_left(self):
        sources = [p["source"] for p in PROGRAMS] + [e.replace("@N@", str(n)) for e in self.EXTRA for n in range(5)]
        for source in sources:
            with self.subTest(source=source[:50]):
                before = parse(source)
                after = inlined(source)
                self.assertEqual(simulate(before).value, simulate(after).value)
                nodes = [n for statement in after for n in walk(statement)]
                self.assertFalse(any(isinstance(n, (Call, FunctionDef)) for n in nodes))


if __name__ == '__main__':
    unittest.main()
