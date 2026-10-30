import re
import unittest
from dataclasses import replace

from lark.exceptions import UnexpectedInput, VisitError

from compiler.parser.grammar import Assign, BinOp, Distribution, FunctionDef, Number, Param
from compiler.parser.parser import parse
from compiler.printer import unparse
from experiment_allocation_poc.programs import PROGRAMS


def without_types(program):
    """The program with the types of its functions' parameters and results removed, as grammar v2 reads them."""
    return [replace(node, params=tuple(Param(p.name) for p in node.params), return_type=None)
            if isinstance(node, FunctionDef) else node for node in program]


class DistributionSyntaxTestCase(unittest.TestCase):
    def test_a_range_assigned_to_a_name_is_a_distribution_in_grammar_v2_only(self):
        [statement, _] = parse("x = -2..3;\nreturn x;", "v2")
        self.assertEqual(Assign("x", Distribution(Number(-2), Number(3))), statement)
        self.assertEqual("x = -2..3;", unparse([statement]))
        with self.assertRaises(UnexpectedInput):
            parse("x = 0..4;\nreturn x;", "v1")

    def test_a_function_declares_no_types_in_grammar_v2(self):
        [function, _] = parse("function bump(p) { p += 1; return p; }\nreturn 0;", "v2")
        self.assertEqual(("p",), tuple(p.name for p in function.params))
        self.assertEqual((None, None), (function.params[0].type, function.return_type))
        with self.assertRaises(UnexpectedInput):
            parse("function bump(p: int): int { return p; }\nreturn 0;", "v2")

    def test_a_distribution_is_an_expression_with_whole_number_bounds_and_at_least_one_value(self):
        [statement, _] = parse("x = 0..4 + 1;\nreturn x;", "v2")
        self.assertEqual(Assign("x", BinOp("+", Distribution(Number(0), Number(4)), Number(1))), statement)
        with self.assertRaises(UnexpectedInput):
            parse("x = 0..1.5;\nreturn x;", "v2")
        for source in ("x = 3..3;", "x = 4..0;"):
            with self.assertRaises(VisitError, msg=source):
                parse(source + "\nreturn x;", "v2")

    def test_grammar_v2_reads_every_existing_program_as_v1_does_apart_from_function_types(self):
        for program in PROGRAMS:
            untyped = re.sub(r"(\w+)\s*:\s*\w+(?=\s*[,)])", r"\1", re.sub(r"\)\s*:\s*\w+\s*\{", ") {", program["source"]))
            self.assertEqual(without_types(parse(program["source"])), parse(untyped, "v2"), program["id"])
        self.assertEqual(parse("x = 2.5 - -3;\nreturn x;"), parse("x = 2.5 - -3;\nreturn x;", "v2"))  # decimals and signs still read


if __name__ == '__main__':
    unittest.main()
