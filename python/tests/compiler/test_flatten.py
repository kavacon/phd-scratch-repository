import textwrap
import unittest

from compiler.printer import unparse
from experiments.allocation_poc.programs import PROGRAMS
from checking.simulate import simulate
from compiler.parser.ast_tools import walk
from compiler.parser.grammar import Assign, BinOp, Boolean, Call, Equals, ExprStmt, If, Not, Number, Return, Unassign, Variable
from compiler.parser.parser import parse
from experiments.allocation_poc.experiment import compile_source
from compiler.passes.flatten import flatten
from compiler.passes.unroll import unroll


def flattened(source):
    return unparse(compile_source(source, passes=(unroll, flatten)))


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
            program = compile_source(source, passes=(unroll, flatten))
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

    def test_the_result_is_unchanged(self):
        for source in self.SOURCES:
            with self.subTest(source=source[:40]):
                before = simulate(parse(source)).value
                after = simulate(compile_source(source, passes=(unroll, flatten))).value
                self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
