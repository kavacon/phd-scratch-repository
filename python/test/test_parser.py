import textwrap
import unittest

from parser.grammar import Assign, ForLoop, If
from parser.parser import _parser, parse, walk
from passes.balance import balance
from passes.rename import RenameError, rename
from allocation_poc.printer import unparse

INPUT = """
x = 1;
y = x + 2;

function f(x: int): bool {
    y = x + 1;
    return y;
}

for x in 0 .. 10 {
    y = y + x;
}

return y;
"""

class ParserTestCase(unittest.TestCase):
    def test_parse_ast(self):
        ast = _parser.parse(INPUT)
        variable_names = [x.value for x in ast.find_token("NAME")]

        self.assertIn("x", variable_names)
        self.assertIn("y", variable_names)
        self.assertIn("f", variable_names)
        self.assertEqual(1, len(list(ast.find_data("for_loop"))))
        self.assertEqual(1, len(list(ast.find_data("function_def"))))

def renamed(source):
    """Rename only, so these tests are about rename and not the passes around it."""
    return unparse(rename(parse(source, passes=())))


def balanced(source):
    return unparse(balance(parse(source, passes=())))


def compiled(source):
    """Balance, then rename (the default pipeline minus release insertion)."""
    return unparse(parse(source, passes=(balance, rename)))


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

    def test_self_reference_reads_the_previous_version(self):
        self.check("x = 1; x = x + 1; x = x + 1; return x;", """
            x#1 = 1;
            x#2 = x#1 + 1;
            x#3 = x#2 + 1;
            return x#3;
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

    def test_a_branch_that_assigns_twice_shares_only_its_final_version(self):
        self.check("x = 1; if x < 2 { x = x + 1; x = x * 3; } else { x = x * 2; } return x;", """
            x#1 = 1;
            if x#1 < 2 {
                x#2 = x#1 + 1;
                x#3 = x#2 * 3;
            } else {
                x#3 = x#1 * 2;
            }
            return x#3;
        """)

    def test_each_branch_may_use_its_own_intermediate_versions(self):
        self.check("x = 1; if x < 2 { x = x + 1; } else { x = x + 2; x = x * 2; } return x;", """
            x#1 = 1;
            if x#1 < 2 {
                x#2 = x#1 + 1;
            } else {
                x#3 = x#1 + 2;
                x#2 = x#3 * 2;
            }
            return x#2;
        """)

    def test_unbalanced_if_is_rejected_by_rename_alone(self):
        with self.assertRaises(RenameError):
            renamed("x = 1; if x < 2 { x = 5; } return x;")

    def test_block_local_variables_need_no_sharing(self):
        self.check("x = 1; y = 2; if y < 3 { z = x; } return y;", """
            x#1 = 1;
            y#1 = 2;
            if y#1 < 3 {
                z#1 = x#1;
            }
            return y#1;
        """)

    def test_for_loop_header_and_loop_variable(self):
        self.check("s = 0; for i in 0 .. 3 { s = s + i; } r = s; return r;", """
            s#1 = 0;
            // header: s#2 = phi(s#1, s#3)
            for i#1 in 0 .. 3 {
                s#3 = s#2 + i#1;
            }
            r#1 = s#2;
            return r#1;
        """)

    def test_nested_control_flow(self):
        self.check("""
            a = 1; b = 2;
            for i in 0 .. 3 {
                if a < b { t = a + i; a = t; } else { a = a; }
                b = b + a;
            }
            return a;
        """, """
            a#1 = 1;
            b#1 = 2;
            // header: a#2 = phi(a#1, a#3)
            // header: b#2 = phi(b#1, b#3)
            for i#1 in 0 .. 3 {
                if a#2 < b#2 {
                    t#1 = a#2 + i#1;
                    a#3 = t#1;
                } else {
                    a#3 = a#2;
                }
                b#3 = b#2 + a#3;
            }
            return a#2;
        """)

    def test_unassign_targets_the_current_version_and_renames_its_witness(self):
        self.check("x = 1; x = x + 1; y = x * 2; x ~= y / 2; return y;", """
            x#1 = 1;
            x#2 = x#1 + 1;
            y#1 = x#2 * 2;
            x#2 ~= y#1 / 2;
            return y#1;
        """)

    def test_unassign_accepts_a_literal_witness(self):
        self.check("x = 2; x ~= 2; return 1;", """
            x#1 = 2;
            x#1 ~= 2;
            return 1;
        """)

    def test_explicit_unassign_in_a_function(self):
        # hand-written deallocation: the same program the allocation-only version should compile to
        self.check("""
            function area(w: int, h: int): int {
                p = w * h;
                q = p + 1;
                p ~= w * h;
                return q;
            }
            return area(2, 3);
        """, """
            function area(w#0: int, h#0: int): int {
                p#1 = w#0 * h#0;
                q#1 = p#1 + 1;
                p#1 ~= w#0 * h#0;
                return q#1;
            }
            return area(2, 3);
        """)

    def test_explicit_unassign_of_a_chain_goes_in_reverse_creation_order(self):
        self.check("""
            function f(x: int): int {
                a = x + 1;
                b = a * 2;
                c = b + a;
                b ~= a * 2;
                a ~= x + 1;
                return c;
            }
            return f(1);
        """, """
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


def paths(block):
    """The names defined along every path through a block (branches are alternatives, not both taken)."""
    result = [[]]
    for statement in block:
        if isinstance(statement, Assign):
            result = [p + [statement.name] for p in result]
        elif isinstance(statement, If):
            result = [p + q for p in result for q in paths(statement.body) + paths(statement.orelse)]
        elif isinstance(statement, ForLoop):
            header = [a.name for a in statement.header]
            result = [p + header + q for p in result for q in paths(statement.body)]
    return result


class RenameInvariantsTestCase(unittest.TestCase):
    def test_every_name_is_assigned_once_along_any_path(self):
        program = parse("""
            function f(a: int): int { a = a + 1; if a < 3 { a = a * 2; } return a; }
            x = 0;
            for j in 0 .. 10 { x = x + 1; if x == 5 { x = x + 2; } }
            for i in 0 .. 4 { x = x + i; }
            return x;
        """)
        for statement in [program] + [s.body for s in program if hasattr(s, "params")]:
            for path in paths(statement):
                self.assertEqual(len(path), len(set(path)), path)

    def test_block_locals_do_not_escape(self):
        with self.assertRaises(RenameError):
            parse("if true { t = 1; } y = t; return y;")

    def test_use_after_unassign_and_unassign_in_inner_block_are_rejected(self):
        with self.assertRaises(RenameError):
            parse("x = 1; x ~= 1; y = x; return y;")
        with self.assertRaises(RenameError):
            parse("x = 1; if true { x ~= 1; } return 2;")
        with self.assertRaises(RenameError):
            parse("x = 1; n = 0; for i in 0 .. 2 { x ~= 1; n = n + 1; } return n;")

    def test_undefined_names_are_rejected(self):
        with self.assertRaises(RenameError):
            parse("x = y; return x;")
        with self.assertRaises(RenameError):
            parse("x = 1; function f(): int { return x; } return x;")  # functions cannot see outer variables

    def test_original_program_is_untouched(self):
        source = "x = 1; x = x + 1; return x;"
        original = parse(source, passes=())
        parse(source)
        self.assertEqual(parse(source, passes=()), original)


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

    def test_missing_branch_can_be_the_then_branch(self):
        self.check("x = 1; if x < 2 { } else { x = 5; } return x;", """
            x = 1;
            if x < 2 {
                x = x;
            } else {
                x = 5;
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

    def test_variables_local_to_a_branch_are_not_copied(self):
        self.check("x = 1; if x < 2 { t = 1; } return x;", """
            x = 1;
            if x < 2 {
                t = 1;
            }
            return x;
        """)

    def test_balanced_ifs_are_left_alone_and_the_pass_is_idempotent(self):
        both = "x = 1; if x < 2 { x = 5; } else { x = 6; } return x;"
        self.assertEqual(parse(both, passes=()), balance(parse(both, passes=())))
        once = balance(parse("x = 1; if x < 2 { x = 5; } return x;", passes=()))
        self.assertEqual(once, balance(once))

    def test_nested_ifs_and_loop_bodies_are_balanced(self):
        self.check("""
            a = 1;
            for i in 0 .. 2 {
                if a < 5 { if a < 3 { a = a + 1; } }
            }
            return a;
        """, """
            a = 1;
            for i in 0 .. 2 {
                if a < 5 {
                    if a < 3 {
                        a = a + 1;
                    } else {
                        a = a;
                    }
                } else {
                    a = a;
                }
            }
            return a;
        """)

    def test_function_parameters_count_as_existing_variables(self):
        out = balanced("function f(a: int): int { if a < 3 { a = a + 1; } return a; } return f(1);")
        self.assertIn("} else {\n        a = a;\n    }", out)

    def test_functions_do_not_see_outer_variables(self):
        # x is not visible inside f, so the assignment is local to the branch and not copied
        out = balanced("x = 1; function f(): int { if true { x = 2; } return 1; } return x;")
        self.assertNotIn("else", out)

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
