import unittest

from compiler.parser.parser import parse
from experiment_existing_ir.reference import ReferenceError_, law


def of(source: str, grammar: str = "v2"):
    return law(parse(source, grammar))


def close(actual, expected):
    return set(actual) == set(expected) and all(abs(actual[v] - expected[v]) < 1e-9 for v in expected)


class Reference(unittest.TestCase):
    """The checker is pinned to the values worked out by hand when the semantics were decided."""

    def test_one_roll_per_expression_and_an_alias_is_the_same_object(self):
        self.assertEqual(of("a = 0..4; return a - a;"), {0: 1.0})
        self.assertTrue(close(of("a = 0..4; return a + a;"), {0: .25, 2: .25, 4: .25, 6: .25}))
        self.assertTrue(close(of("a = 0..4; b = a; b += 1; return a;"), {1: .25, 2: .25, 3: .25, 4: .25}))

    def test_a_computed_assignment_is_independent_and_extracting_a_temporary_changes_the_result(self):
        triangle = {-2: 1 / 16, -1: 2 / 16, 0: 3 / 16, 1: 4 / 16, 2: 3 / 16, 3: 2 / 16, 4: 1 / 16}
        self.assertTrue(close(of("a = 0..4; c = a + 1; return c - a;"), triangle))
        self.assertTrue(close(of("a = 0..4; c = (a + 1) + a; return c;"), {1: .25, 3: .25, 5: .25, 7: .25}))
        self.assertTrue(close(of("a = 0..4; t = a + 1; c = t + a; return c;"),
                              {1: 1 / 16, 2: 2 / 16, 3: 3 / 16, 4: 4 / 16, 5: 3 / 16, 6: 2 / 16, 7: 1 / 16}))

    def test_a_conditional_on_a_distribution_ties_what_it_assigns(self):
        self.assertTrue(close(of("a = 0..4; x = 2; if a < 2 { x += 3; } else { x -= 1; } return x;"), {5: .5, 1: .5}))
        self.assertTrue(close(of("a = 0..4; b = 0..2; x = 2; if b < 1 { x += a; } return x;"),
                              {2: 5 / 8, 3: 1 / 8, 4: 1 / 8, 5: 1 / 8}))

    def test_integers_copy_and_distributions_alias_across_calls_and_branches(self):
        self.assertEqual(of("function f(p) { p += 3; return p; }\nx = 2; y = f(x); return x + y;"), {7: 1.0})
        self.assertTrue(close(of("function bump(p) { p += 1; return p; }\na = 0..4; b = bump(a); b += 10; return a;"),
                              {11: .25, 12: .25, 13: .25, 14: .25}))
        self.assertEqual(of("a = 0..4; x = 2; k = 9; for i in 0..2 { if k < 3 { x += a; } } y = x; y += 1; return x;"),
                         {2: 1.0})  # the branch is dead, so x is an integer and y a copy
        self.assertTrue(close(of("a = 0..4; x = 2; for i in 0..3 { x += a; } y = x; y += 1; return x;"),
                              {3: .25, 6: .25, 9: .25, 12: .25}))

    def test_what_the_semantics_forbid_is_refused(self):
        with self.assertRaises(ReferenceError_):
            of("a = 0..4; a = 5; return a;")
        with self.assertRaises(ReferenceError_):
            of("function f(p) { return f(p); }\nreturn f(1);")


if __name__ == "__main__":
    unittest.main()
