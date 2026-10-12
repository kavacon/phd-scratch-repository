import textwrap
import unittest

from allocation_poc.printer import unparse
from allocation_poc.programs import PROGRAMS
from allocation_poc.simulate import simulate
from parser.ast_tools import walk
from parser.grammar import Assign, BinOp, Boolean, Call, Equals, ExprStmt, If, Not, Number, Return, Unassign, Variable
from parser.parser import parse
from passes.flatten import FlattenError, flatten
from passes.unroll import unroll


def flattened(source):
    return unparse(parse(source, passes=(unroll, flatten)))


def expected(text):
    return textwrap.dedent(text).strip()


class FlattenExamplesTestCase(unittest.TestCase):
    def check(self, source, text):
        self.assertEqual(expected(text), flattened(source))

    def test_a_nested_operation_goes_into_a_temporary_and_the_outer_one_keeps_the_name(self):
        self.check("a = 1; y = (a * 2) + 1; return y;", """
            a = 1;
            $t1 = a * 2;
            y = $t1 + 1;
            return y;
        """)

    def test_temporaries_are_numbered_in_the_order_they_are_computed(self):
        self.check("a = 1; b = 2; y = (a * 2) + (b * 3); return y;", """
            a = 1;
            b = 2;
            $t1 = a * 2;
            $t2 = b * 3;
            y = $t1 + $t2;
            return y;
        """)

    def test_deeper_nesting_and_not(self):
        self.check("a = 1; b = 2; y = !((a + 1) < (b * 2)); return y;", """
            a = 1;
            b = 2;
            $t1 = a + 1;
            $t2 = b * 2;
            $t3 = $t1 < $t2;
            y = !$t3;
            return y;
        """)

    def test_already_flat_statements_are_left_alone(self):
        source = "a = 1; b = a + 2; c = b; d = true; return d;"
        self.assertEqual(unparse(parse(source, passes=(unroll,))), flattened(source))

    def test_a_chain_of_equalities_becomes_pairwise_comparisons(self):
        self.check("a = 1; b = 1; c = 1; z = a == b == c; return z;", """
            a = 1;
            b = 1;
            c = 1;
            $t1 = a == b;
            $t2 = b == c;
            z = $t1 && $t2;
            return z;
        """)
        self.check("a = 1; z = a == a == a == a; return z;", """
            a = 1;
            $t1 = a == a;
            $t2 = a == a;
            $t3 = a == a;
            $t4 = $t1 && $t2;
            z = $t4 && $t3;
            return z;
        """)

    def test_a_returned_expression_or_literal_is_named_but_a_returned_variable_is_not(self):
        self.check("function f(a: int): int { return a * 2 + 1; } return f(1);", """
            function f(a: int): int {
                $t1 = a * 2;
                $t2 = $t1 + 1;
                return $t2;
            }
            $t1 = f(1);
            return $t1;
        """)
        self.check("function f(a: int): int { return a; } return 7;", """
            function f(a: int): int {
                return a;
            }
            $t1 = 7;
            return $t1;
        """)

    def test_an_if_condition_is_computed_into_a_register_before_the_if(self):
        self.check("a = 1; b = 2; x = 0; if (a + 1) < (b * 2) { x = 1; } else { x = 2; } return x;", """
            a = 1;
            b = 2;
            x = 0;
            $t1 = a + 1;
            $t2 = b * 2;
            $t3 = $t1 < $t2;
            if $t3 {
                x = 1;
            } else {
                x = 2;
            }
            return x;
        """)

    def test_a_plain_condition_needs_no_register(self):
        self.check("a = true; x = 0; if a { x = 1; } return x;", """
            a = true;
            x = 0;
            if a {
                x = 1;
            }
            return x;
        """)

    def test_temporaries_made_inside_a_branch_stay_in_that_branch(self):
        self.check("a = 1; x = 0; if a < 5 { x = (a + 1) * 2; } else { x = a; } return x;", """
            a = 1;
            x = 0;
            $t1 = a < 5;
            if $t1 {
                $t2 = a + 1;
                x = $t2 * 2;
            } else {
                x = a;
            }
            return x;
        """)

    def test_call_arguments_and_nested_calls(self):
        self.check("function g(a: int): int { return a; } a = 1; y = g(a + 1) + g(g(2) * 3); return y;", """
            function g(a: int): int {
                return a;
            }
            a = 1;
            $t1 = a + 1;
            $t2 = g($t1);
            $t3 = g(2);
            $t4 = $t3 * 3;
            $t5 = g($t4);
            y = $t2 + $t5;
            return y;
        """)

    def test_a_call_statement_keeps_its_shape_with_flat_arguments(self):
        self.check("function g(a: int): int { return a; } a = 1; g(a * 2); return a;", """
            function g(a: int): int {
                return a;
            }
            a = 1;
            $t1 = a * 2;
            g($t1);
            return a;
        """)

    def test_a_hand_written_witness_is_flattened_too(self):
        self.check("a = 1; x = a * 2 + 1; x ~= a * 2 + 1; return a;", """
            a = 1;
            $t1 = a * 2;
            x = $t1 + 1;
            $t2 = a * 2;
            x ~= $t2 + 1;
            return a;
        """)

    def test_a_returned_literal_is_named_so_every_program_returns_a_variable(self):
        self.check("return 2;", """
            $t1 = 2;
            return $t1;
        """)
        self.check("x = 1; return true;", """
            x = 1;
            $t1 = true;
            return $t1;
        """)
        self.check("function f(a: int): int { return 3; } return a;" if False else "function f(a: int): int { return 3; } return 1;", """
            function f(a: int): int {
                $t1 = 3;
                return $t1;
            }
            $t1 = 1;
            return $t1;
        """)

    def test_each_function_numbers_its_own_temporaries(self):
        self.check("function f(a: int): int { return a * 2 + 1; } function g(a: int): int { return a * 3 + 1; } return 1;", """
            function f(a: int): int {
                $t1 = a * 2;
                $t2 = $t1 + 1;
                return $t2;
            }
            function g(a: int): int {
                $t1 = a * 3;
                $t2 = $t1 + 1;
                return $t2;
            }
            $t1 = 1;
            return $t1;
        """)

    def test_loops_must_be_unrolled_first(self):
        with self.assertRaises(FlattenError):
            parse("s = 0; for i in 0 .. 2 { s = s + i; } return s;", passes=(flatten,))

    def test_the_original_program_is_untouched(self):
        raw = parse("a = 1; y = (a * 2) + 1; return y;", passes=())
        flatten(raw)
        self.assertEqual(parse("a = 1; y = (a * 2) + 1; return y;", passes=()), raw)


def operands(node):
    if isinstance(node, BinOp):
        return [node.left, node.right]
    if isinstance(node, Not):
        return [node.operand]
    if isinstance(node, (Equals, Call)):
        return list(node.operands if isinstance(node, Equals) else node.args)
    return []


def is_plain(node):
    return isinstance(node, (Number, Boolean, Variable))


def is_flat(node):
    return is_plain(node) or (type(node) in (BinOp, Not, Equals, Call) and all(is_plain(o) for o in operands(node)))


class FlattenInvariantsTestCase(unittest.TestCase):
    SOURCES = [p["source"] for p in PROGRAMS] + [
        "function f(a: int, b: int): int { y = (a * 2) + (b * 3) * (a - b); z = a == b == y; "
        "if (a + 1) < (b * 2) && !(a == b) { y = y + (z + 1); } else { y = y - (a * b); } return y + z * 2; } return f(1, 2);",
    ]

    def test_every_statement_is_one_operation_on_plain_values(self):
        for source in self.SOURCES:
            program = parse(source, passes=(unroll, flatten))
            for node in (n for statement in program for n in walk(statement)):
                if isinstance(node, Assign):
                    self.assertTrue(is_flat(node.value), unparse([node]))
                elif isinstance(node, (Return, ExprStmt, Unassign)):
                    inner = node.value if isinstance(node, Return) else node.witness if isinstance(node, Unassign) else node.expr
                    self.assertTrue(is_plain(inner) if isinstance(node, Return) else is_flat(inner), unparse([node]))
                elif isinstance(node, If):
                    self.assertTrue(is_plain(node.condition), unparse([node]))
                elif isinstance(node, Equals):
                    self.assertEqual(2, len(node.operands))

    def test_flattening_twice_changes_nothing(self):
        for source in self.SOURCES:
            once = parse(source, passes=(unroll, flatten))
            self.assertEqual(once, flatten(once))

    def test_the_result_is_unchanged(self):
        for source in self.SOURCES:
            with self.subTest(source=source[:40]):
                before = simulate(parse(source, passes=())).value
                after = simulate(parse(source, passes=(unroll, flatten))).value
                self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
