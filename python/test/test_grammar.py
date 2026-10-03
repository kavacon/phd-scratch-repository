import unittest

from lark.exceptions import UnexpectedInput

from parser.grammar import (
    Assign, BinOp, Boolean, Call, Equals, ExprStmt, ForLoop, FunctionDef, If, Not, Number, Param, Return, Unassign, Variable,
)
from parser.parser import parse, rewrite, walk


def raw(source):
    """Parse without any passes, so tests see the grammar-level AST."""
    return parse(source, passes=())


def stmts(source):
    """Statements of a fragment. The grammar needs a final return, so one is appended and dropped again."""
    return raw(source + " return 0;")[:-1]


class ParseTestCase(unittest.TestCase):
    def test_statement_types(self):
        program = raw("""
        x = 1;
        function f(a: int, b: int): int { t = a + b; return t; }
        for i in 0 .. 3 { x = x + i; }
        f(1, 2);
        return x;
        """)
        self.assertEqual(
            [Assign, FunctionDef, ForLoop, ExprStmt, Return], [type(s) for s in program]
        )
        self.assertEqual((Param("a", "int"), Param("b", "int")), program[1].params)
        self.assertEqual("int", program[1].return_type)
        self.assertEqual(Return(Variable("t")), program[1].body[-1])
        self.assertEqual(Call("f", (Number(1), Number(2))), program[3].expr)

    def test_function_without_params_or_return_type(self):
        function = raw("function f() { return 1; } return 1;")[0]
        self.assertEqual(((), None), (function.params, function.return_type))

    def test_precedence_and_equality(self):
        [statement] = stmts("x = 1 + 2 * 3 == 7.5;")
        self.assertEqual(
            Equals((BinOp("+", Number(1), BinOp("*", Number(2), Number(3))), Number(7.5))),
            statement.value,
        )

    def test_comparison_comments_and_restrictions(self):
        [statement] = stmts("x = 1 < 2; // trailing comment\n")
        self.assertEqual(BinOp("<", Number(1), Number(2)), statement.value)
        self.assertEqual(BinOp(">", Number(1), Number(2)), stmts("x = 1 > 2;")[0].value)
        for bad in ('x = "a";', "x = -1;", "x = 1 < 2 < 3;", "function f() { function g() { return 1; } return 2; }"):
            with self.assertRaises(UnexpectedInput, msg=bad):
                stmts(bad)

    def test_conditionals(self):
        [plain, with_else] = stmts("if x < 1 { y = 1; } if x < 1 { y = 1; } else { y = 3; }")
        self.assertEqual(If(BinOp("<", Variable("x"), Number(1)), (Assign("y", Number(1)),)), plain)
        self.assertEqual(((Assign("y", Number(1)),), (Assign("y", Number(3)),)), (with_else.body, with_else.orelse))
        with self.assertRaises(UnexpectedInput):  # `else if` is not part of the language
            stmts("if x < 1 { y = 1; } else if x > 1 { y = 2; }")

    def test_for_range(self):
        [loop] = stmts("for i in 0 .. n + 1 { x = i; }")
        self.assertEqual(("i", Number(0), BinOp("+", Variable("n"), Number(1))), (loop.var, loop.start, loop.stop))
        self.assertEqual(Number(2.5), stmts("x = 2.5;")[0].value)
        for bad in ("for i to 3 { }", "for i in 0..3 { }"):
            with self.assertRaises(UnexpectedInput, msg=bad):
                stmts(bad)

    def test_logical_operators(self):
        # C-style precedence: ! binds tightest, then &&, then ||
        [statement] = stmts("x = !a && b || c == d;")
        self.assertEqual(
            BinOp("||", BinOp("&&", Not(Variable("a")), Variable("b")), Equals((Variable("c"), Variable("d")))),
            statement.value,
        )
        self.assertEqual(Not(Not(Boolean(True))), stmts("x = !!true;")[0].value)
        self.assertEqual(BinOp("*", Not(Variable("a")), Number(2)), stmts("x = !a * 2;")[0].value)

    def test_not_equals(self):
        self.assertEqual(
            BinOp("&&", BinOp("!=", Variable("a"), Number(1)), Variable("b")),
            stmts("x = a != 1 && b;")[0].value,
        )
        with self.assertRaises(UnexpectedInput):
            stmts("x = a == b != c;")

    def test_booleans(self):
        [statement, other] = stmts("x = true; y = falsey;")
        self.assertEqual(Boolean(True), statement.value)
        self.assertEqual(Variable("falsey"), other.value)

    def test_while_is_not_part_of_the_language(self):
        with self.assertRaises(UnexpectedInput):
            stmts("while true { x = 1; }")

    def test_unassign(self):
        self.assertEqual(
            [Unassign("x", BinOp("+", Variable("y"), Number(1))), Return(Number(1))],
            raw("x ~= y + 1; return 1;"),
        )
        with self.assertRaises(UnexpectedInput):
            stmts("free x;")


class ReturnRulesTestCase(unittest.TestCase):
    def test_program_and_functions_end_with_a_valued_return(self):
        self.assertEqual(
            [FunctionDef("f", (), "int", (Return(Number(1)),)), Return(Call("f"))],
            raw("function f(): int { return 1; } return f();"),
        )

    def test_missing_misplaced_or_empty_returns_are_rejected(self):
        for bad in (
            "", "x = 1;",                                      # program must end with a return
            "return 1; x = 2; return 2;",                      # return only at the very end
            "if true { return 1; } return 2;",                 # not inside blocks
            "for i in 0 .. 2 { return 1; } return 2;",
            "function f() { x = 1; } return 1;",               # functions must end with a return
            "function f() { return 1; x = 2; } return 1;",
            "return;",                                         # return always carries a value
        ):
            with self.assertRaises(UnexpectedInput, msg=bad):
                raw(bad)


class PassesTestCase(unittest.TestCase):
    def test_passes(self):
        source = "x = 1; return x;"
        self.assertEqual([], parse(source, passes=[lambda p: []]))
        self.assertEqual(raw(source), parse(source, passes=()))
        self.assertEqual(Assign("x#1", Number(1)), parse(source)[0])  # default passes include rename


class TraversalTestCase(unittest.TestCase):
    def test_walk(self):
        [statement] = stmts("x = 1 + 2;")
        self.assertEqual(["Assign", "BinOp", "Number", "Number"], [type(n).__name__ for n in walk(statement)])

    def test_rewrite_is_pure(self):
        program = stmts("x = 1; for i in 0 .. 2 { x = x + 1; }")

        def double(node):
            return Number(node.value * 2) if isinstance(node, Number) else node

        def rename(node):
            return Variable("y") if node == Variable("x") else node

        rewritten = [rewrite(rewrite(s, double), rename) for s in program]
        self.assertEqual(Number(1), program[0].value)
        self.assertEqual(Number(2), rewritten[0].value)
        self.assertEqual(Variable("y"), rewritten[1].body[0].value.left)


if __name__ == '__main__':
    unittest.main()
