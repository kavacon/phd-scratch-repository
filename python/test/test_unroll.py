import textwrap
import unittest

from allocation_poc.printer import unparse
from allocation_poc.simulate import simulate
from parser.parser import parse
from passes.rename import RenameError
from passes.unroll import UnrollError, unroll


def unrolled(source):
    return unparse(parse(source, passes=(unroll,)))


def expected(text):
    return textwrap.dedent(text).strip()


class UnrollExamplesTestCase(unittest.TestCase):
    def check(self, source, text):
        self.assertEqual(expected(text), unrolled(source))

    def test_each_iteration_gets_a_copy_with_the_loop_variable_replaced(self):
        self.check("s = 1; for i in 0 .. 3 { s = s + i; } return s;", """
            s = 1;
            s = s + 0;
            s = s + 1;
            s = s + 2;
            return s;
        """)

    def test_names_local_to_the_body_get_a_suffix_per_iteration(self):
        self.check("s = 1; for i in 1 .. 3 { t = s * i; s = t + 1; } return s;", """
            s = 1;
            t@1 = s * 1;
            s = t@1 + 1;
            t@2 = s * 2;
            s = t@2 + 1;
            return s;
        """)

    def test_loops_that_run_no_times_disappear(self):
        self.check("x = 1; for i in 2 .. 2 { x = 5; } for i in 3 .. 1 { x = 6; } return x;", """
            x = 1;
            return x;
        """)

    def test_nested_loops_expand_inner_first(self):
        self.check("s = 0; for i in 0 .. 2 { for j in 0 .. 2 { u = i * j; s = s + u; } } return s;", """
            s = 0;
            u@0@0 = 0 * 0;
            s = s + u@0@0;
            u@1@0 = 0 * 1;
            s = s + u@1@0;
            u@0@1 = 1 * 0;
            s = s + u@0@1;
            u@1@1 = 1 * 1;
            s = s + u@1@1;
            return s;
        """)

    def test_a_loop_variable_only_exists_inside_its_loop(self):
        self.check("i = 5; for i in 0 .. 2 { x = i; } return i;", """
            i = 5;
            x@0 = 0;
            x@1 = 1;
            return i;
        """)

    def test_loops_inside_ifs_and_functions_are_expanded(self):
        self.check("function f(a: int): int { x = a; if a < 2 { for i in 0 .. 2 { x = x + i; } } return x; } return f(1);", """
            function f(a: int): int {
                x = a;
                if a < 2 {
                    x = x + 0;
                    x = x + 1;
                }
                return x;
            }
            return f(1);
        """)

    def test_no_loops_are_left(self):
        source = "s = 0; for i in 0 .. 2 { for j in 0 .. 2 { if s < 3 { s = s + 1; } } } return s;"
        self.assertNotIn("for ", unrolled(source))

    def test_the_original_program_is_untouched(self):
        raw = parse("s = 0; for i in 0 .. 2 { s = s + i; } return s;", passes=())
        unroll(raw)
        self.assertEqual(parse("s = 0; for i in 0 .. 2 { s = s + i; } return s;", passes=()), raw)


class UnrollRulesTestCase(unittest.TestCase):
    def test_body_locals_do_not_escape_the_loop(self):
        with self.assertRaises(RenameError):
            parse("for i in 0 .. 2 { t = i; } x = t; return x;")

    def test_the_loop_variable_cannot_be_assigned(self):
        with self.assertRaises(UnrollError):
            parse("for i in 0 .. 2 { i = 1; } return 1;", passes=(unroll,))

    def test_loops_that_would_expand_too_far_are_rejected(self):
        for source in ("for i in 0 .. 100000 { x = 1; } return 1;",
                       "for i in 0 .. 200 { for j in 0 .. 200 { x = 1; } } return 1;"):
            with self.assertRaises(UnrollError, msg=source):
                parse(source, passes=(unroll,))


class UnrollBehaviourTestCase(unittest.TestCase):
    PROGRAMS = [
        "function f(n: int): int { s = n; for i in 0 .. 4 { t = s + i; s = t * 2 + 1; } return s; } return f(@N@);",
        "function f(n: int): int { s = 0; for i in 0 .. 3 { for j in 0 .. 3 { u = i * j + n; s = s + u; } } return s; } return f(@N@);",
        "function f(n: int): int { a = n; for i in 0 .. 5 { if a < 20 { t = a + i; a = t * 2; } } return a; } return f(@N@);",
        "function f(n: int): int { a = n; b = 1; for i in 1 .. 4 { c = a + b; b = a; a = c; } return a; } return f(@N@);",
    ]

    def test_the_result_is_unchanged_and_nothing_is_left_unreleased(self):
        for source in self.PROGRAMS:
            for n in range(5):
                with self.subTest(source=source, n=n):
                    text = source.replace("@N@", str(n))
                    before = simulate(parse(text, passes=()))
                    after = simulate(parse(text))
                    self.assertEqual(before.value, after.value)
                    self.assertEqual({}, {s: l for s, l in after.leaks.items() if l})


if __name__ == '__main__':
    unittest.main()
