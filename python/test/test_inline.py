import textwrap
import unittest

from allocation_poc.printer import unparse
from allocation_poc.programs import PROGRAMS
from allocation_poc.simulate import simulate
from parser.grammar import Call, FunctionDef
from parser.parser import parse, walk
from passes.inline import InlineError, inline

AREA = "function area(w: int, h: int): int { p = w * h; q = p + 1; return q; }"


def inlined(source):
    return parse(source, passes=(inline,))


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

    def test_a_variable_or_literal_argument_replaces_its_parameter_with_no_copy(self):
        self.check(AREA + " x = 3; return area(x, 5);", """
            x = 3;
            area1.p = x * 5;
            area1.q = area1.p + 1;
            return area1.q;
        """)

    def test_an_argument_that_is_an_expression_is_computed_once_into_a_local(self):
        self.check("function twice(n: int): int { return n + n; } x = 3; return twice(x * 2);", """
            x = 3;
            twice1.n = x * 2;
            return twice1.n + twice1.n;
        """)

    def test_a_parameter_the_callee_assigns_to_is_copied_so_the_callers_variable_is_unchanged(self):
        self.check("function bump(n: int): int { n = n + 1; return n; } x = 3; y = bump(x); return x + y;", """
            x = 3;
            bump1.n = x;
            bump1.n = bump1.n + 1;
            y = bump1.n;
            return x + y;
        """)

    def test_two_calls_get_their_own_names(self):
        self.check(AREA + " x = 3; a = area(x, 2); b = area(x, 4); return a + b;", """
            x = 3;
            area1.p = x * 2;
            area1.q = area1.p + 1;
            a = area1.q;
            area2.p = x * 4;
            area2.q = area2.p + 1;
            b = area2.q;
            return a + b;
        """)

    def test_calls_inside_a_callee_are_inlined_first_and_their_names_nest(self):
        self.check("function g(n: int): int { m = n * 2; return m + 1; } function h(n: int): int { a = g(n); return g(a); } x = 3; return h(x);", """
            x = 3;
            h1.g1.m = x * 2;
            h1.a = h1.g1.m + 1;
            h1.g2.m = h1.a * 2;
            return h1.g2.m + 1;
        """)

    def test_a_call_in_an_if_condition_runs_before_the_if_and_a_call_in_a_branch_stays_in_it(self):
        self.check("function g(n: int): int { m = n * 2; return m + 1; } x = 3; y = 0; if g(x) < 5 { y = g(x + 1); } else { y = 1; } return y;", """
            x = 3;
            y = 0;
            g1.m = x * 2;
            if (g1.m + 1) < 5 {
                g2.n = x + 1;
                g2.m = g2.n * 2;
                y = g2.m + 1;
            } else {
                y = 1;
            }
            return y;
        """)

    def test_a_call_in_a_loop_body_stays_in_the_loop(self):
        self.check("function g(n: int): int { m = n * 2; return m + 1; } s = 0; for i in 0 .. 3 { s = s + g(i); } return s;", """
            s = 0;
            for i in 0 .. 3 {
                g1.m = i * 2;
                s = s + (g1.m + 1);
            }
            return s;
        """)

    def test_a_call_whose_value_is_discarded_still_runs_its_body(self):
        self.check("function g(n: int): int { m = n * 2; return m; } x = 3; g(x); return x;", """
            x = 3;
            g1.m = x * 2;
            return x;
        """)

    def test_an_uncalled_function_is_dropped(self):
        self.check("function g(n: int): int { return n; } return 1;", "return 1;")


class InlineErrorsTestCase(unittest.TestCase):
    def test_recursion_is_rejected(self):
        with self.assertRaises(InlineError):
            inlined("function f(n: int): int { a = f(n); return a; } return f(1);")
        with self.assertRaises(InlineError):
            inlined("function f(n: int): int { a = g(n); return a; } function g(n: int): int { b = f(n); return b; } return f(1);")

    def test_an_undefined_function_and_a_wrong_argument_count_are_rejected(self):
        with self.assertRaises(InlineError):
            inlined("return f(1);")
        with self.assertRaises(InlineError):
            inlined(AREA + " return area(1);")


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
                before = parse(source, passes=())
                after = inlined(source)
                self.assertEqual(simulate(before).value, simulate(after).value)
                nodes = [n for statement in after for n in walk(statement)]
                self.assertFalse(any(isinstance(n, (Call, FunctionDef)) for n in nodes))

    def test_the_whole_pipeline_releases_the_callees_locals(self):
        for program in PROGRAMS:
            with self.subTest(program=program["id"]):
                result = simulate(parse(program["source"]))
                self.assertEqual({}, {s: l for s, l in result.leaks.items() if l})


if __name__ == '__main__':
    unittest.main()
