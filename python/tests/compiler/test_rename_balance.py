import textwrap
import unittest

from compiler.printer import unparse
from compiler.parser.grammar import Assign, If
from compiler.parser.parser import parse
from experiments.allocation_poc.experiment import compile_source
from compiler.passes.balance import balance
from compiler.passes.rename import RenameError, rename

def renamed(source):
    """Rename only, so these tests are about rename and not the passes around it."""
    return unparse(rename(parse(source)))


def balanced(source):
    return unparse(balance(parse(source)))


def compiled(source):
    """Balance, then rename (the default pipeline minus release insertion)."""
    return unparse(compile_source(source, passes=(balance, rename)))


def expected(text):
    return textwrap.dedent(text).strip()


class RenameExamplesTestCase(unittest.TestCase):
    def check(self, source, text):
        self.assertEqual(expected(text), renamed(source))

    def test_reassignment_creates_new_versions(self):
        self.check("x = 1; y = x + 2; x = y * 3; z = x; return z;", """
            x#1 = 1;
            y#1 = x#1 + 2;
            x#2 = y#1 * 3;
            z#1 = x#2;
            return z#1;
        """)

    def test_function_params_are_version_zero_and_scopes_are_independent(self):
        self.check("""
            function f(a: int, b: int): int { t = a + b; t = t * 2; return t; }
            t = 5;
            r = f(t, 1);
            return r;
        """, """
            function f(a#0: int, b#0: int): int {
                t#1 = a#0 + b#0;
                t#2 = t#1 * 2;
                return t#2;
            }
            t#1 = 5;
            r#1 = f(t#1, 1);
            return r#1;
        """)

    def test_if_else_branches_share_the_final_version(self):
        # the branches are mutually exclusive, so both define x#2 and nothing needs merging afterwards
        self.check("x = 1; if x < 2 { x = x + 1; } else { x = x * 2; } y = x; return y;", """
            x#1 = 1;
            if x#1 < 2 {
                x#2 = x#1 + 1;
            } else {
                x#2 = x#1 * 2;
            }
            y#1 = x#2;
            return y#1;
        """)

    def test_unbalanced_if_is_rejected_by_rename_alone(self):
        with self.assertRaises(RenameError):
            renamed("x = 1; if x < 2 { x = 5; } return x;")

    def test_unassign_targets_the_current_version_and_renames_its_witness(self):
        self.check("x = 1; x = x + 1; y = x * 2; x ~= y / 2; return y;", """
            x#1 = 1;
            x#2 = x#1 + 1;
            y#1 = x#2 * 2;
            x#2 ~= y#1 / 2;
            return y#1;
        """)


def paths(block):
    """The names defined along every path through a block (branches are alternatives, not both taken)."""
    result = [[]]
    for statement in block:
        if isinstance(statement, Assign):
            result = [p + [statement.name] for p in result]
        elif isinstance(statement, If):
            result = [p + q for p in result for q in paths(statement.body) + paths(statement.orelse)]
    return result


class RenameInvariantsTestCase(unittest.TestCase):
    def test_every_name_is_assigned_once_along_any_path(self):
        program = compile_source("""
            function f(a: int): int { a = a + 1; if a < 3 { a = a * 2; } return a; }
            x = 0;
            for j in 0 .. 10 { x = x + 1; if x == 5 { x = x + 2; } }
            for i in 0 .. 4 { x = x + i; }
            return x;
        """)
        for statement in [program] + [s.body for s in program if hasattr(s, "params")]:
            for path in paths(statement):
                self.assertEqual(len(path), len(set(path)), path)

    def test_undefined_names_are_rejected(self):
        with self.assertRaises(RenameError):
            compile_source("x = y; return x;")
        with self.assertRaises(RenameError):
            compile_source("x = 1; function f(): int { return x; } y = f(); return y;")  # functions cannot see outer variables


class BalanceExamplesTestCase(unittest.TestCase):
    """Balance runs before rename, on the grammar-level program."""

    def check(self, source, text):
        self.assertEqual(expected(text), balanced(source))

    def test_one_armed_if_gets_an_identity_assignment_in_the_missing_else(self):
        self.check("x = 1; if x < 2 { x = 5; } return x;", """
            x = 1;
            if x < 2 {
                x = 5;
            } else {
                x = x;
            }
            return x;
        """)

    def test_an_existing_else_that_assigns_different_variables_is_balanced(self):
        self.check("a = 1; b = 2; if a < b { a = 3; } else { b = 4; } return a + b;", """
            a = 1;
            b = 2;
            if a < b {
                a = 3;
                b = b;
            } else {
                b = 4;
                a = a;
            }
            return a + b;
        """)

    def test_nested_ifs_are_balanced(self):
        self.check("""
            a = 1;
            if a < 5 { if a < 3 { a = a + 1; } }
            return a;
        """, """
            a = 1;
            if a < 5 {
                if a < 3 {
                    a = a + 1;
                } else {
                    a = a;
                }
            } else {
                a = a;
            }
            return a;
        """)

    def test_balance_then_rename_gives_the_branches_one_shared_version(self):
        self.assertEqual(expected("""
            x#1 = 1;
            if x#1 < 2 {
                x#2 = 5;
            } else {
                x#2 = x#1;
            }
            return x#2;
        """), compiled("x = 1; if x < 2 { x = 5; } return x;"))


if __name__ == '__main__':
    unittest.main()
