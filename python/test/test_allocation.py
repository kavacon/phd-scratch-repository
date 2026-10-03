import textwrap
import unittest
import warnings

from parser.parser import parse
from passes.allocation import insert_releases, lower_allocation
from passes.balance import balance
from passes.rename import rename
from allocation_poc.printer import unparse
from allocation_poc.simulate import SimulationError, simulate


def compiled(source):
    """The default pipeline: balance, rename, then insert the releases."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse(source)


def prepared(source):
    """What the insertion pass starts from."""
    return parse(source, passes=(balance, rename))


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
            return area(2, 3);
        """)

    def test_a_chain_is_released_in_reverse_creation_order(self):
        self.check("function f(x: int): int { a = x + 1; b = a * 2; c = b + a; return c; } return f(1);", """
            function f(x#0: int): int {
                a#1 = x#0 + 1;
                b#1 = a#1 * 2;
                c#1 = b#1 + a#1;
                b#1 ~= a#1 * 2;
                a#1 ~= x#0 + 1;
                return c#1;
            }
            return f(1);
        """)

    def test_a_returned_expression_is_named_so_what_it_reads_can_be_released(self):
        self.check("function f(a: int): int { t = a + 1; return t * 2; } return f(1);", """
            function f(a#0: int): int {
                t#1 = a#0 + 1;
                return#1 = t#1 * 2;
                t#1 ~= a#0 + 1;
                return return#1;
            }
            return f(1);
        """)

    def test_a_value_that_is_overwritten_before_use_is_released_straight_away(self):
        self.check("function f(a: int): int { y = 0; y = a + 1; return y; } return f(1);", """
            function f(a#0: int): int {
                y#1 = 0;
                y#1 ~= 0;
                y#2 = a#0 + 1;
                return y#2;
            }
            return f(1);
        """)

    def test_branch_temporaries_are_released_inside_the_branch_and_the_result_after_the_if(self):
        # y#2 is produced by both branches, so its release is an if on the same condition. Its witnesses use
        # a#1 only: the branch temporary t#1 is gone by then, so its definition is substituted in.
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
                if a#1 < 5 {
                    t#1 = a#1 * 2;
                    y#2 = t#1 + 1;
                    t#1 ~= a#1 * 2;
                } else {
                    y#2 = a#1 - 1;
                }
                r#1 = y#2 + 1;
                if a#1 < 5 {
                    y#2 ~= (a#1 * 2) + 1;
                } else {
                    y#2 ~= a#1 - 1;
                }
                a#1 ~= x#0 + 1;
                return r#1;
            }
            return f(1);
        """)

    def test_nested_ifs_nest_the_release(self):
        self.check("""
            function f(a: int): int {
                x = a;
                if a < 5 { if a < 3 { x = a + 1; } else { x = a + 2; } x = x * 2; } else { x = a - 1; }
                return a;
            }
            return f(1);
        """, """
            function f(a#0: int): int {
                x#1 = a#0;
                x#1 ~= a#0;
                if a#0 < 5 {
                    if a#0 < 3 {
                        x#2 = a#0 + 1;
                    } else {
                        x#2 = a#0 + 2;
                    }
                    x#4 = x#2 * 2;
                    if a#0 < 3 {
                        x#2 ~= a#0 + 1;
                    } else {
                        x#2 ~= a#0 + 2;
                    }
                } else {
                    x#4 = a#0 - 1;
                }
                if a#0 < 5 {
                    if a#0 < 3 {
                        x#4 ~= (a#0 + 1) * 2;
                    } else {
                        x#4 ~= (a#0 + 2) * 2;
                    }
                } else {
                    x#4 ~= a#0 - 1;
                }
                return a#0;
            }
            return f(1);
        """)

    def test_loop_body_temporaries_are_released_every_iteration(self):
        self.check("function g(x: int): int { s = x; for i in 0 .. 3 { t = s + i; s = t * 2; } return s; } return g(3);", """
            function g(x#0: int): int {
                s#1 = x#0;
                // header: s#2 = phi(s#1, s#3)
                for i#1 in 0 .. 3 {
                    t#1 = s#2 + i#1;
                    s#3 = t#1 * 2;
                    t#1 ~= s#2 + i#1;
                }
                return s#2;
            }
            return g(3);
        """)

    def test_a_value_carried_round_a_loop_is_reported_and_left_alone(self):
        source = "function g(n: int): int { s = 0; for i in 0 .. 3 { s = s + i; } return n; } return g(3);"
        _, unreleased = insert_releases(prepared(source))
        self.assertEqual(["s#2"], unreleased)
        with self.assertWarns(UserWarning):
            lower_allocation(prepared(source))


class InsertionBehaviourTestCase(unittest.TestCase):
    """Run the programs: values are unchanged, every witness is right, and nothing is read after release."""

    PROGRAMS = {
        "chain": "function f(x: int): int { a = x + 1; b = a * 2; c = b + a; return c; } return f(@N@);",
        "returned expression": "function f(a: int): int { t = a + 1; u = t * 3; return t + u; } return f(@N@);",
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
        "branch assigns a variable twice": """
            function f(a: int): int {
                x = a + 1;
                if a < 3 { x = x * 2; x = x + 7; } else { x = x - 1; }
                return x;
            }
            return f(@N@);
        """,
        "loop with a temporary": "function g(n: int): int { s = n; for i in 0 .. 3 { t = s + i; s = t * 2 + 1; } return s; } return g(@N@);",
        "if inside a loop": """
            function g(n: int): int {
                a = n;
                for i in 0 .. 3 { if a < 20 { t = a + i; a = t * 2; } }
                return a;
            }
            return g(@N@);
        """,
        "functions calling functions": """
            function double(x: int): int { y = x * 2; return y + 0; }
            function quad(x: int): int { d = double(x); return double(d); }
            return quad(@N@);
        """,
    }
    # loops keep their carried values (the entry value and the last iteration's), which are not released
    LOOP_LEAKS = {"loop with a temporary", "if inside a loop"}

    def test_values_are_unchanged_and_releases_are_valid(self):
        for name, template in self.PROGRAMS.items():
            for n in range(6):
                source = template.replace("@N@", str(n))
                with self.subTest(program=name, n=n):
                    before = simulate(prepared(source)).value
                    after = simulate(compiled(source))
                    self.assertEqual(before, after.value)
                    if name not in self.LOOP_LEAKS:
                        self.assertEqual({}, {s: l for s, l in after.leaks.items() if l}, "registers left behind")

    def test_a_wrong_witness_is_caught_by_the_simulation(self):
        broken = parse("function f(a: int): int { t = a + 1; t ~= a + 2; return a; } return f(1);")
        with self.assertRaises(SimulationError):
            simulate(broken)

    def test_inserting_twice_adds_nothing(self):
        for name, template in self.PROGRAMS.items():
            with self.subTest(program=name):
                once = compiled(template.replace("@N@", "2"))
                self.assertEqual(once, lower_allocation(once))


if __name__ == '__main__':
    unittest.main()
